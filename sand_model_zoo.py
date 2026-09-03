"""
Which SIMPLE models fit sand push dynamics, and how much does each buy?

`sand_foresight.py` showed the paper's switched-linear operator explains ~45% of
what a push does on sand. This asks what else in the same complexity class --
closed-form or convex, no training loop, interpretable -- does as well or better,
and, more importantly, which cheaper model it has to beat to be worth anything.

All models predict the next canonical-frame image from the current one, are
fitted on the same train episodes, and are scored by the same swept-region
metric, so the numbers are directly comparable to `sand_foresight.py`.

The families
------------
mean-delta        y' = y + mean(dy). ZERO parameters. The canonical frame has
                  already normalised the action away, so if a stereotyped
                  average displacement captures most of the effect then the
                  operator is not doing state-dependent work and the headline
                  result means much less than it appears to. This is the
                  baseline that can invalidate the others, which is why it is
                  first.
linear            A y, non-negative, shrunk toward I. The paper's model.
affine            A y + b. One extra vector, absorbing systematic drift that the
                  linear part would otherwise have to encode.
reduced-rank      rank-r truncation of the ridge solution, via one SVD. Gives an
                  accuracy-vs-rank curve, i.e. the effective dimensionality of
                  sand transport, and is the principled response to a fit that
                  is underdetermined (M/D = 0.27 at the best configuration).
col-stochastic    A with non-negative columns summing to 1: a true transport
                  map, where every unit of input mass is redistributed and none
                  is created or destroyed. Justified by measurement -- sand mass
                  is conserved to 1.0000 -- and it is the variant Suh & Tedrake
                  explicitly could not solve, because it breaks the row
                  decomposition their tractability argument rests on. Projected
                  gradient with a per-column simplex projection does solve it.
knn               non-parametric: copy the delta of the most similar training
                  state. Tests whether a parametric model is needed at all.

    python sand_model_zoo.py --crop 1.0 --res 64 --blur 0
"""

from __future__ import annotations

import argparse
import time

import torch

from fit_linear_foresight import (
    canonicalise, fit_operator, fit_operator_nonneg, metrics, predict_world,
    swept_region_mask,
)
from sand_foresight import BOUNDS, DEFAULT_GLOB, load_sand_arrays
from fit_linear_foresight import actions_to_pixels


# --------------------------------------------------------------------------
# Model families. Each returns a callable canonical -> canonical prediction,
# so `predict_world` can wrap them identically.
# --------------------------------------------------------------------------

def fit_mean_delta(Y0, Y1):
    """y' = y + mean(dy). Zero parameters."""
    delta = (Y1 - Y0).mean(dim=1, keepdim=True)
    eye = torch.eye(Y0.shape[0], device=Y0.device)
    # Expressed as an affine map so it goes through the same predict path.
    return eye, delta


def fit_affine(Y0, Y1, ridge=1.0):
    """[A | b] by least squares on the input augmented with a constant row.

    The bias is NOT regularised toward anything: it is a free offset, and
    shrinking it would just push its job back onto A.
    """
    D, M = Y0.shape
    ones = torch.ones((1, M), dtype=Y0.dtype, device=Y0.device)
    Xa = torch.cat([Y0, ones], dim=0)                       # (D+1, M)
    G = Xa @ Xa.T
    reg = torch.eye(D + 1, dtype=G.dtype, device=G.device) * ridge
    reg[D, D] = 0.0                                          # free bias
    C = Y1 @ Xa.T
    # Shrink the linear block toward identity, as elsewhere: an underdetermined
    # transport operator should decay to "nothing moved", not to "mass vanishes".
    C[:, :D] = C[:, :D] + ridge * torch.eye(D, dtype=C.dtype, device=C.device)
    W = torch.linalg.solve((G + reg).T, C.T).T
    return W[:, :D].contiguous(), W[:, D:].contiguous()


def fit_reduced_rank(Y0, Y1, rank, ridge=1.0):
    """Rank-r truncation of the ridge solution (reduced-rank regression).

    The truncation is applied to the *predicted* subspace rather than to A
    directly, which is the standard formulation and the one that actually
    minimises error at a given rank.
    """
    A = fit_operator(Y0, Y1, ridge, toward_identity=True)
    P = A @ Y0                                               # fitted values
    U, S, _ = torch.linalg.svd(P @ P.T)
    Ur = U[:, :rank]
    return Ur @ (Ur.T @ A)


def _project_columns_to_simplex(A):
    """Project each column of A onto {x >= 0, sum x = 1}.

    Standard sort-based Euclidean projection, vectorised over columns. This is
    what makes a mass-conserving operator tractable: the constraint couples the
    entries of a COLUMN, whereas ordinary least squares decomposes by ROW, which
    is exactly why the paper could not solve this variant with its per-row QP.
    """
    D = A.shape[0]
    u, _ = torch.sort(A, dim=0, descending=True)
    css = u.cumsum(dim=0) - 1.0
    idx = torch.arange(1, D + 1, device=A.device, dtype=A.dtype).unsqueeze(1)
    cond = u - css / idx > 0
    rho = cond.to(A.dtype).cumsum(dim=0).argmax(dim=0)       # last True per column
    theta = css.gather(0, rho.unsqueeze(0)) / (rho + 1).to(A.dtype).unsqueeze(0)
    return (A - theta).clamp_min(0.0)


def fit_col_stochastic(Y0, Y1, max_iter=800, tol=1e-7):
    """Mass-conserving transport operator: A >= 0 with unit column sums.

    Projected gradient. Starts from the identity, which is already feasible (a
    permutation matrix is column-stochastic) and is the right prior: where the
    data cannot pin the operator down it should fall back to "nothing moved".
    """
    G = Y0 @ Y0.T
    C = Y1 @ Y0.T
    L = 2.0 * float(torch.linalg.eigvalsh(G)[-1])
    A = torch.eye(Y0.shape[0], device=Y0.device, dtype=Y0.dtype)
    Z, t = A.clone(), 1.0
    for _ in range(max_iter):
        A_new = _project_columns_to_simplex(Z - (2.0 / L) * (Z @ G - C))
        t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
        Z = A_new + ((t - 1.0) / t_new) * (A_new - A)
        step = float((A_new - A).norm() / A_new.norm().clamp_min(1e-12))
        A, t = A_new, t_new
        if step < tol:
            break
    return A


def predict_knn(Y0_tr, Y1_tr, Y0_te, k=1):
    """Copy the mean delta of the k nearest training states (canonical frame)."""
    d = torch.cdist(Y0_te.T, Y0_tr.T)                        # (M_te, M_tr)
    idx = d.topk(k, largest=False).indices                   # (M_te, k)
    delta = (Y1_tr - Y0_tr).T                                # (M_tr, D)
    return (Y0_te.T + delta[idx].mean(dim=1)).T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--crop", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--normalize", default="mean")
    ap.add_argument("--view", default="density", choices=["density", "mask", "height"])
    ap.add_argument("--min-grains", type=float, default=2.0)
    ap.add_argument("--cube-size", type=float, default=None,
                    help="cube edge, metres -- set for cube-spectrum datasets "
                         "so the mask view uses each cube's real footprint")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="use only the first N episode files, for size-matched "
                         "cross-dataset comparisons")
    ap.add_argument("--min-push-mm", type=float, default=19.9)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = None if args.normalize == "none" else args.normalize
    occ_t, occ_t1, actions, ep, s0, s1 = load_sand_arrays(
        args.glob, args.grid, args.blur, norm, args.min_push_mm, dev,
        view=args.view, min_grains=args.min_grains, cube_size=args.cube_size,
        max_episodes=args.max_episodes)
    print(f"view = {args.view}"
          + (f" (>= {args.min_grains:g} grains/cell)" if args.view == "mask" else ""))
    H = W = args.grid
    R, CR = args.res, args.crop

    eps = ep.unique()
    g = torch.Generator().manual_seed(args.seed)
    perm = eps[torch.randperm(len(eps), generator=g).to(eps.device)]
    val = set(perm[:max(1, int(len(eps) * args.val_frac))].tolist())
    te = torch.tensor([int(e) in val for e in ep], device=dev)
    tr = ~te
    print(f"split: {int(tr.sum())} train / {int(te.sum())} val transitions "
          f"({len(eps) - len(val)}/{len(val)} episodes)")

    s_px, e_px = actions_to_pixels(actions, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    s_px, e_px = s_px.to(dev), e_px.to(dev)

    Y0 = canonicalise(occ_t[tr], s_px[tr], e_px[tr], R, CR).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(occ_t1[tr], s_px[tr], e_px[tr], R, CR).reshape(int(tr.sum()), -1).T
    D, M = Y0.shape
    print(f"operator {D}x{D}, M={M}, M/D={M / D:.2f}\n")

    ote, o1te, ste, ete = occ_t[te], occ_t1[te], s_px[te], e_px[te]
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W),
                               half_width_px=0.5 * plate_px + 2.0,
                               pad_px=0.5 * plate_px)

    def world(A, b=None):
        return predict_world(A, ote, ste, ete, R, (H, W), CR) if b is None else \
            _predict_affine(A, b, ote, ste, ete, R, (H, W), CR)

    preds, timing = {}, {}
    preds["persistence"] = ote

    t = time.time(); Am, bm = fit_mean_delta(Y0, Y1)
    preds["mean-delta (0 params)"] = world(Am, bm); timing["mean-delta (0 params)"] = time.time() - t

    t = time.time(); A = fit_operator_nonneg(Y0, Y1, max_iter=args.iters, ridge=args.ridge)
    preds["linear-nonneg"] = world(A); timing["linear-nonneg"] = time.time() - t

    t = time.time(); Aa, ba = fit_affine(Y0, Y1, args.ridge)
    preds["affine (Ay+b)"] = world(Aa, ba); timing["affine (Ay+b)"] = time.time() - t

    t = time.time(); Ac = fit_col_stochastic(Y0, Y1, max_iter=args.iters)
    preds["col-stochastic"] = world(Ac); timing["col-stochastic"] = time.time() - t
    print(f"  col-stochastic: column sums {float(Ac.sum(0).mean()):.4f} "
          f"+- {float(Ac.sum(0).std()):.4f} (target 1.0)")

    for r in args.ranks:
        if r >= D:
            continue
        t = time.time(); Ar = fit_reduced_rank(Y0, Y1, r, args.ridge)
        preds[f"reduced-rank r={r}"] = world(Ar); timing[f"reduced-rank r={r}"] = time.time() - t

    t = time.time()
    Y0te = canonicalise(ote, ste, ete, R, CR).reshape(int(te.sum()), -1).T
    knn = predict_knn(Y0, Y1, Y0te, k=1).T.reshape(-1, R, R)
    from transforms.functional import blend_push_prediction, from_push_frame, push_frame_validity_mask
    back = from_push_frame(knn, ste, ete, (H, W), CR)
    mask = push_frame_validity_mask(ste, ete, (H, W), (R, R), CR)
    preds["knn (k=1, retrieval)"] = blend_push_prediction(back, ote, mask).clamp_min(0.0)
    timing["knn (k=1, retrieval)"] = time.time() - t

    rows = {k: metrics(p, o1te, ote, region=region) for k, p in preds.items()}
    base = rows["persistence"]["rms"]
    print(f"\n=== validation, SWEPT REGION ({int(te.sum())} transitions) ===")
    hdr = f"{'model':24s} {'rms':>9s} {'% of change':>12s} {'explained':>10s} {'fit s':>7s}"
    print(hdr); print("-" * len(hdr))
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]["rms"]):
        print(f"{k:24s} {v['rms']:9.5f} {100 * v['rms'] / base:11.1f}% "
              f"{v['explained']:10.4f} {timing.get(k, 0.0):7.1f}")
    print("\n100% = no better than predicting nothing moved. The number to beat "
          "is NOT persistence\nbut mean-delta: if a zero-parameter constant "
          "displacement gets most of the way, the\noperator is not doing "
          "state-dependent work.")


def _predict_affine(A, b, occ, s_px, e_px, res, grid_res, scale, batch=256):
    """`predict_world` with a bias term."""
    from transforms.functional import (blend_push_prediction, from_push_frame,
                                       push_frame_validity_mask, to_push_frame)
    Hh, Ww = grid_res
    outs = []
    for i in range(0, occ.shape[0], batch):
        sl = slice(i, i + batch)
        o, s, e = occ[sl], s_px[sl], e_px[sl]
        canon = to_push_frame(o, s, e, (res, res), scale)
        flat = canon.reshape(canon.shape[0], -1).T
        pred = (A @ flat + b).T.reshape(-1, res, res)
        back = from_push_frame(pred, s, e, (Hh, Ww), scale)
        mask = push_frame_validity_mask(s, e, (Hh, Ww), (res, res), scale)
        outs.append(blend_push_prediction(back, o, mask).clamp_min(0.0))
    return torch.cat(outs, dim=0)


if __name__ == "__main__":
    main()
