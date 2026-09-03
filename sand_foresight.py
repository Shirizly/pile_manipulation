"""
Fit the switched-linear visual-foresight operator on SAND, and validate it.

This is the continuum test the cube work pointed at: on rigid cubes the paper's
per-pixel linear operator never beat "predict nothing moved", and the leading
explanation (docs/linear_foresight_findings.md §4, H1) is that its carrots formed
a near-continuous mass while 30-80 cubes are a discrete set moving in threshold
events. Sand is that continuum.

Everything about the fit is reused from `fit_linear_foresight.py` -- the same
SE(2) canonical push frame, the same non-negative FISTA solve, the same
identity-operator control, the same swept-region metric -- so a difference in the
result is a difference in the *material*, not in the method. Only the input
representation changes: sand goes through `transforms/sand_occupancy.py`
(unclamped column mass) instead of the binary silhouette
`particles_to_occupancy` produces.

Two dataset-specific choices, both forced by measurement:

* **Only full-length pushes are fitted.** 91.6% of collected pushes travel the
  requested 20 mm; the rest are truncated because the pile spreads over an
  episode until the blade's clamped start has no room (97.5% full on push 1,
  falling to 88% by push 4). A single-operator fit must not mix lengths, so the
  short ones are dropped rather than silently averaged in.
* **Train/validation split is by EPISODE, never by transition.** Five sequential
  pushes on one pile are strongly correlated -- s' of push k is s of push k+1 --
  so a transition-level split leaks the answer across the boundary.

    python sand_foresight.py --res 32 --crop 0.5 --blur 1.0
"""

from __future__ import annotations

import argparse
import glob as _glob
import time

import torch

from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, fit_operator_nonneg,
    metrics, predict_world, swept_region_mask,
)
from transforms.sand_occupancy import sand_mass, sand_to_density, sand_to_mask

DEFAULT_GLOB = "Genesis/data/sand/pile20/**/*_data.pt"
# The tray, and therefore the grid extent, matches the cube datasets exactly.
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}


def load_sand_arrays(pattern: str, grid: int, sigma: float, normalize: str,
                     min_push_mm: float, device: str,
                     view: str = "density", min_grains: float = 2.0,
                     min_height: float | None = None, floor_z: float = 0.010):
    """Sand transitions -> density maps, actions, and episode ids.

    Returns positions as well as maps: the physical-units reporting works on
    grains (millimetres of displacement), which is far easier to interpret than
    occupancy-per-pixel, and the cube work showed that ungrounded pixel numbers
    are actively misleading.
    """
    files = sorted(_glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no sand data matches {pattern}")

    S0, S1, PS, PE, EP = [], [], [], [], []
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=False)
        S0.append(d["states"][..., :3])
        S1.append(d["states_"][..., :3])
        PS.append(d["p_starts"])
        PE.append(d["p_stops"])
        EP.append(torch.full((d["states"].shape[0],), i, dtype=torch.long))
    s0 = torch.cat(S0).to(device)
    s1 = torch.cat(S1).to(device)
    ps = torch.cat(PS).to(device)
    pe = torch.cat(PE).to(device)
    ep = torch.cat(EP).to(device)

    length_mm = (pe[:, :2] - ps[:, :2]).norm(dim=-1) * 1000.0
    keep = length_mm >= min_push_mm
    n_drop = int((~keep).sum())
    s0, s1, ps, pe, ep = s0[keep], s1[keep], ps[keep], pe[keep], ep[keep]
    print(f"loaded {len(files)} episodes; kept {int(keep.sum())} full-length "
          f"pushes, dropped {n_drop} truncated "
          f"({100 * n_drop / max(len(keep), 1):.1f}%)")

    if view == "mask":
        # What an overhead camera sees: a binary silhouette, not a depth map.
        # This is also the like-for-like comparison with the rigid-cube
        # datasets, which were binary throughout.
        proj = lambda x: sand_to_mask(x, BOUNDS, (grid, grid),
                                      min_grains=min_grains, min_height=min_height,
                                      floor_z=floor_z, sigma=sigma)
    elif view == "height":
        from transforms.sand_occupancy import sand_to_heightmap
        proj = lambda x: sand_to_heightmap(x, BOUNDS, (grid, grid),
                                           floor_z=floor_z, sigma=sigma)
    else:
        proj = lambda x: sand_to_density(x, BOUNDS, (grid, grid),
                                         sigma=sigma, normalize=normalize)
    occ_t, occ_t1 = proj(s0), proj(s1)
    actions = torch.cat([ps[:, :2], pe[:, :2]], dim=-1)      # [sx, sy, ex, ey]
    return occ_t, occ_t1, actions, ep, s0, s1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--grid", type=int, default=64, help="density-map resolution")
    ap.add_argument("--res", type=int, default=32, help="canonical-frame resolution")
    ap.add_argument("--crop", type=float, default=0.5,
                    help="fraction of the image the canonical window spans")
    ap.add_argument("--blur", type=float, default=1.0,
                    help="Gaussian sigma on the density map, in cells")
    ap.add_argument("--normalize", default="mean", choices=["mean", "max", "none"])
    ap.add_argument("--view", default="density", choices=["density", "mask", "height"],
                    help="'density' = column mass (depth-sensing); 'mask' = binary "
                         "silhouette, i.e. what a plain overhead camera sees and "
                         "the like-for-like comparison with the binary cube data; "
                         "'height' = surface height map.")
    ap.add_argument("--min-grains", type=float, default=2.0,
                    help="mask threshold in grains/cell. 2 keeps 83.4%% of the "
                         "mass over 13.1%% of the grid on this data; 1 keeps 100%% "
                         "over 20.9%% (no detection floor).")
    ap.add_argument("--min-height", type=float, default=None,
                    help="mask on height (m) instead of column mass")
    ap.add_argument("--min-push-mm", type=float, default=19.9)
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fraction of EPISODES held out (never transitions)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = None if args.normalize == "none" else args.normalize

    occ_t, occ_t1, actions, ep, s0, s1 = load_sand_arrays(
        args.glob, args.grid, args.blur, norm, args.min_push_mm, dev,
        view=args.view, min_grains=args.min_grains, min_height=args.min_height)
    print(f"view = {args.view}" + (f" (>= {args.min_grains:g} grains/cell)"
                                   if args.view == "mask" and args.min_height is None
                                   else ""))
    H = W = args.grid
    R, CR = args.res, args.crop

    # ---- split by episode -------------------------------------------------
    eps = ep.unique()
    g = torch.Generator().manual_seed(args.seed)
    perm = eps[torch.randperm(len(eps), generator=g).to(eps.device)]
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = set(perm[:n_val].tolist())
    is_val = torch.tensor([int(e) in val_eps for e in ep], device=dev)
    tr, te = ~is_val, is_val
    print(f"split: {int(tr.sum())} train / {int(te.sum())} validation "
          f"transitions, {len(eps) - n_val}/{n_val} episodes")

    s_px, e_px = actions_to_pixels(actions, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    s_px, e_px = s_px.to(dev), e_px.to(dev)

    # ---- what one push actually does, in grains ---------------------------
    disp = (s1 - s0)[..., :2].norm(dim=-1) * 1000.0
    print(f"\none push moves grains: mean {float(disp.mean()):.2f} mm, "
          f"p95 {float(disp.quantile(0.95)):.2f}, max {float(disp.max()):.2f} "
          f"(push = {float((actions[:, 2:] - actions[:, :2]).norm(dim=-1).mean() * 1000):.1f} mm)")
    print(f"mass in tray: {float(sand_mass(s0, BOUNDS).mean()):.4f} -> "
          f"{float(sand_mass(s1, BOUNDS).mean()):.4f}")

    # ---- fit --------------------------------------------------------------
    Y0 = canonicalise(occ_t[tr], s_px[tr], e_px[tr], R, CR).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(occ_t1[tr], s_px[tr], e_px[tr], R, CR).reshape(int(tr.sum()), -1).T
    D, M = Y0.shape
    print(f"\noperator {D}x{D} from M={M} pairs -> M/D={M / D:.2f} "
          f"({'OVER' if M > D else 'UNDER'}determined per row)")

    t0 = time.time()
    A_nn = fit_operator_nonneg(Y0, Y1, max_iter=args.iters, ridge=args.ridge)
    print(f"  non-negative FISTA: {time.time() - t0:.1f}s")
    A_ridge = fit_operator(Y0, Y1, args.ridge, toward_identity=True)

    # ---- validate ---------------------------------------------------------
    ote, o1te = occ_t[te], occ_t1[te]
    ste, ete = s_px[te], e_px[te]
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W),
                               half_width_px=0.5 * plate_px + 2.0,
                               pad_px=0.5 * plate_px)
    eye = torch.eye(R * R, device=dev)

    preds = {
        "persistence": ote,
        "identity (warp only)": predict_world(eye, ote, ste, ete, R, (H, W), CR),
        "linear-nonneg": predict_world(A_nn, ote, ste, ete, R, (H, W), CR),
        "linear-ridge->I": predict_world(A_ridge, ote, ste, ete, R, (H, W), CR),
    }
    try:
        from fit_linear_foresight import predict_heuristic
        preds["heur-cumulative"] = predict_heuristic(
            "cumulative", ote.cpu(), ste.cpu(), ete.cpu()).to(dev)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  (heuristic baseline unavailable: {exc})")

    for title, reg in (("WHOLE IMAGE", None), ("SWEPT REGION", region)):
        print(f"\n=== validation, {title} ({int(te.sum())} transitions) ===")
        hdr = (f"{'model':22s} {'rms':>9s} {'% of the change':>16s} "
               f"{'softIoU':>8s} {'explained':>10s}")
        print(hdr)
        print("-" * len(hdr))
        rows = {k: metrics(p, o1te, ote, region=reg) for k, p in preds.items()}
        base = rows["persistence"]["rms"]
        for k, v in sorted(rows.items(), key=lambda kv: kv[1]["rms"]):
            print(f"{k:22s} {v['rms']:9.5f} {100 * v['rms'] / max(base, 1e-12):15.1f}% "
                  f"{v['soft_iou']:8.4f} {v['explained']:10.4f}")

    print("\n'% of the change' is relative to persistence, whose error IS the "
          "change that occurred.\n100% = no better than predicting nothing moved; "
          "below 100% = the model helps.")


if __name__ == "__main__":
    main()
