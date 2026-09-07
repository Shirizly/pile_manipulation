"""Is the operator good enough to CONTROL with, even if its images are poor?

Motivation. Suh & Tedrake's headline result is a control result: greedy descent
on an image-space Lyapunov function drives the pile into a target set. Their
prediction evidence is comparative and narrow -- Table 1 gives 1.858 for the
linear model against 2.062 for the deep one, ~10% apart -- and they never report
a persistence baseline at all. So "the operator's per-pixel error is 96% of the
change" does not by itself contradict the paper. A greedy controller never needs
an accurate image; it needs the *ordering* of candidate actions to be right.

This measures exactly that, on the same held-out transitions:

    V(I) = d^T y / ||y||_1        (their eq. 4: distance-transform-weighted mass)
    dV_true = V(I_1) - V(I_0)     what the push actually achieved
    dV_pred = V(I_hat_1) - V(I_0) what the model expected

and reports, for each model:

  * correlation between predicted and actual dV -- can it rank actions?
  * sign agreement -- can it tell a helpful push from a harmful one?
  * regret of picking the model's best action out of a random slate, in units of
    the achievable dV spread, against an oracle that knows dV_true.

Persistence predicts dV_pred = 0 for every action, so it cannot rank at all: it
is the floor here by construction, which makes this a fairer test of the
operator than one-step image error was.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dmdc_baseline import load_transition_arrays, split_by_episode
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator_nonneg, predict_heuristic,
    predict_world,
)
from loro_foresight import gaussian_blur


def pile_centroid_and_support(occ, thresh=1e-6):
    """Mass centroid (row, col) and bounding box of a pooled occupancy set.

    Coordinator correction, 2026-09-07: `ind-square8`'s degeneracy was
    mis-diagnosed as "a single push never crosses this small a boundary"
    when the real cause is that its FIXED, corner-relative placement never
    overlaps the pile's support at all -- a placement bug (C-040's failure
    mode wearing new clothes: a saturated indicator, here saturated at 1
    rather than 0 because the target is always empty instead of always
    full). This computes where the pile ACTUALLY is, from the data, so a
    pile-relative goal's target can be placed to genuinely intersect it.

    `occ` is (N, ..., H, W) -- any leading dims are pooled. Returns
    ((cy, cx), (r0, r1, c0, c1)) where the bounding box is the tight range
    of rows/cols whose pooled mass exceeds `thresh` (matching the threshold
    `functional_degeneracy_screen.py`-style checks use elsewhere in this
    file for "is there any signal here").
    """
    flat = occ.reshape(-1, occ.shape[-2], occ.shape[-1])
    mean_field = flat.mean(dim=0)
    H, W = mean_field.shape
    rows = (mean_field.sum(dim=1) > thresh).nonzero(as_tuple=True)[0]
    cols = (mean_field.sum(dim=0) > thresh).nonzero(as_tuple=True)[0]
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                             torch.arange(W, dtype=torch.float32), indexing="ij")
    tot = mean_field.sum().clamp_min(1e-12)
    cy = float((mean_field * yy).sum() / tot)
    cx = float((mean_field * xx).sum() / tot)
    return (cy, cx), (r0, r1, c0, c1)


def lyapunov_weights(grid_res, goal, device, pile_center=None):
    """Weight field `d` (or `w`) for a target set, normalised to [0, 1].

    `goal` is one of the three original keys, byte-identical to before:
      center  -- a centred square covering a quarter of the tray area
      corner  -- a square in one corner (their non-convex-ish harder case)
      stripe  -- a central band, so the cost only rewards one axis

    plus the functional-sharpness family added for EXP-0024_v2/EXP-0026_v2
    (see docs/experiments/METRICS.md, "weight-field functional family"),
    which vary the FUNCTIONAL FORM (distance transform vs indicator, and a
    clip radius in between) at a FIXED target, and one control that varies
    target SIZE without sharpening the functional:

      distclip-corner-r{2,4,8} -- distance transform to the corner
          half-plane (same mask as `corner`), clipped and renormalised at
          radius r pixels: min(dist_px, r) / r. r large -> smooth (like
          `corner`); r -> 0 -> an indicator. A tolerance knob at FIXED target.
      ind-corner    -- indicator (1 outside, 0 inside) of the corner
          half-plane -- the r -> 0 limit of distclip-corner-r*.
      ind-square8   -- indicator of an 8x8 px square (side = H // 8, i.e. an
          "eighth-side square"), offset one side-length in from the corner
          (rows/cols [H//8 : 2*H//8]) -- off-centre so a centred pile does
          not hit the C-040 degeneracy, and clear of the array boundary so
          the distance-transform sibling below is not itself boundary-cropped
          (measured: a corner-flush placement understates a small target's
          own spectral concentration, since most of the grid then sits on
          one side of it -- placement matters for a distance transform's
          actual values, unlike for an indicator's, whose |FFT| a discrete
          transform's implicit periodicity makes position-invariant). The
          SHARP form.
      dist-square8  -- distance transform to the SAME 8x8 px square. Isolates
          target SIZE from functional sharpness: shrinking a distance
          transform's target does not sharpen its spectrum (measured), so
          this is the control that shows size alone is not the lever.

    `ind-square8` (above) turned out to be a mis-diagnosed degeneracy, not a
    reachability limit: its fixed, corner-relative placement never overlaps
    the pile's own support (measured centroid ~(31.5, 31.5) on a 64x64 grid,
    support rows/cols 25-38 -- see `pile_centroid_and_support`), so `dV` is
    identically 0 for a trivial reason (the target is always empty) rather
    than because sharp targets are inherently hard to hit. The following
    keys require `pile_center=(cy, cx)`, computed from the ACTUAL data (never
    hard-coded), so the target genuinely intersects the pile:

      ind-square8-pile   -- 8x8 px indicator centred on `pile_center`.
      ind-square16-pile  -- 16x16 px indicator centred on `pile_center` (a
          larger sharp target, in case 8x8 gives too discrete a `dV`).
      ind-stripe-thin-pile -- a 4-px-wide indicator stripe (full image
          width) centred on `pile_center`'s row -- structured rather than
          blobby, crossing the pile rather than sitting inside it.
    """
    from scipy.ndimage import distance_transform_edt

    H, W = grid_res
    mask = np.zeros((H, W), dtype=bool)
    if goal == "center":
        a, b = H // 4, 3 * H // 4
        mask[a:b, a:b] = True
    elif goal == "corner":
        mask[: H // 2, : W // 2] = True
    elif goal == "stripe":
        mask[H // 2 - H // 8: H // 2 + H // 8, :] = True
    elif goal in ("ind-corner",) or goal.startswith("distclip-corner-r"):
        mask[: H // 2, : W // 2] = True
    elif goal in ("ind-square8", "dist-square8"):
        side_h, side_w = max(1, H // 8), max(1, W // 8)
        mask[side_h:2 * side_h, side_w:2 * side_w] = True
    elif goal in ("ind-square8-pile", "ind-square16-pile"):
        if pile_center is None:
            raise ValueError(f"{goal} requires pile_center=(cy, cx), "
                              f"computed from data via pile_centroid_and_support")
        side = 8 if goal == "ind-square8-pile" else 16
        cy, cx = pile_center
        r0 = max(0, min(H - side, int(round(cy - side / 2))))
        c0 = max(0, min(W - side, int(round(cx - side / 2))))
        mask[r0:r0 + side, c0:c0 + side] = True
    elif goal == "ind-stripe-thin-pile":
        if pile_center is None:
            raise ValueError(f"{goal} requires pile_center=(cy, cx), "
                              f"computed from data via pile_centroid_and_support")
        cy, _cx = pile_center
        thickness = 4
        r0 = max(0, min(H - thickness, int(round(cy - thickness / 2))))
        mask[r0:r0 + thickness, :] = True
    else:
        raise ValueError(goal)

    if goal.startswith("ind-"):
        # Indicator: 1 outside the target, 0 inside -- the r -> 0 limit of a
        # clipped distance transform, and the sharp form for the target-size
        # control (`ind-square8`).
        return torch.from_numpy((~mask).astype(np.float32)).to(device)

    d = distance_transform_edt(~mask).astype(np.float32)
    if goal.startswith("distclip-corner-r"):
        r = float(goal.rsplit("r", 1)[1])
        d = np.minimum(d, r) / r
        return torch.from_numpy(d).to(device)

    d /= max(float(d.max()), 1e-6)
    return torch.from_numpy(d).to(device)


def lyapunov(occ, d, eps=1e-6):
    """V = d^T y / ||y||_1 — mass-normalised mean distance to the target."""
    n = occ.shape[0]
    flat = occ.reshape(n, -1)
    return (flat * d.reshape(1, -1)).sum(1) / flat.sum(1).clamp_min(eps)


def _partial(x, y, z):
    """Correlation of x and y after linearly removing z from both.

    The slate test below draws its candidates from DIFFERENT states, because the
    data holds one action per state. That conflates "this action is good" with
    "this state was easy", and a model that only tracked state difficulty would
    score well while being useless for choosing between actions. Removing the
    state's own V_0 (and its contact score, when available) leaves the part of
    the ranking that is about the action.
    """
    def resid(v):
        Z = torch.stack([torch.ones_like(z[0])] + list(z), dim=-1)
        beta = torch.linalg.lstsq(Z, v.unsqueeze(-1)).solution
        return v - (Z @ beta).squeeze(-1)
    a, b = resid(x), resid(y)
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).mean() / (a.std().clamp_min(1e-9) * b.std().clamp_min(1e-9)))


def rank_metrics(dv_pred, dv_true, control=None):
    """Correlations, sign agreement, and a slate-selection regret."""
    p = dv_pred - dv_pred.mean()
    t = dv_true - dv_true.mean()
    pear = float((p * t).mean() / (p.std().clamp_min(1e-9) * t.std().clamp_min(1e-9)))
    rp = dv_pred.argsort().argsort().float()
    rt = dv_true.argsort().argsort().float()
    rp = (rp - rp.mean()) / rp.std().clamp_min(1e-9)
    rt = (rt - rt.mean()) / rt.std().clamp_min(1e-9)
    spear = float((rp * rt).mean())
    # Sign agreement only counts pushes that did something either way.
    live = dv_true.abs() > 1e-4
    sign = float((torch.sign(dv_pred[live]) == torch.sign(dv_true[live])).float().mean())

    # Slate regret: form random slates of K candidates, pick the one the model
    # says is best, and compare the dV actually obtained against the best in the
    # slate. Reported as a fraction of the oracle's advantage over a random
    # pick, so 1.0 = as good as the oracle and 0.0 = no better than random.
    g = torch.Generator(device='cpu').manual_seed(0)
    n = dv_true.shape[0]
    out = {}
    for K in (4, 16):
        idx = torch.randint(0, n, (2000, K), generator=g).to(dv_true.device)
        cand_true = dv_true[idx]
        cand_pred = dv_pred[idx]
        chosen = cand_true.gather(1, cand_pred.argmin(dim=1, keepdim=True)).squeeze(1)
        oracle = cand_true.min(dim=1).values
        rand = cand_true.mean(dim=1)
        denom = (rand - oracle).clamp_min(1e-9)
        out[K] = float(((rand - chosen) / denom).mean())
    part = _partial(dv_pred, dv_true, control) if control else float('nan')
    return pear, spear, sign, out, part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="configs/dataset/genesis_foresight_pile30.yaml")
    ap.add_argument("--res", type=int, default=32)
    ap.add_argument("--crop", type=float, default=0.5)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--goals", default="center,corner,stripe")
    args = ap.parse_args()

    d = load_transition_arrays(args.dataset, "train")
    H, W = d.occ_t.shape[-2:]
    d.occ_t, d.occ_t1 = (gaussian_blur(d.occ_t, args.blur),
                         gaussian_blur(d.occ_t1, args.blur))
    s, e = actions_to_pixels(d.actions, d.workspace_min, d.workspace_max, (H, W))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    o0, o1, s, e = d.occ_t.to(dev), d.occ_t1.to(dev), s.to(dev), e.to(dev)

    torch.manual_seed(0)
    mtr, mte = split_by_episode(d, 0.25)
    R, CR = args.res, args.crop
    Y0 = canonicalise(o0[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
    Y1 = canonicalise(o1[mtr], s[mtr], e[mtr], R, CR).reshape(int(mtr.sum()), -1).T
    A = fit_operator_nonneg(Y0, Y1, max_iter=args.iters, ridge=1.0)

    ote, o1te, ste, ete = o0[mte], o1[mte], s[mte], e[mte]
    preds = {
        "persistence": ote,
        "linear operator": predict_world(A, ote, ste, ete, R, (H, W), CR),
        "heur-cumulative": predict_heuristic(
            "cumulative", ote.cpu(), ste.cpu(), ete.cpu()).to(dev),
    }
    print(f"n_train={int(mtr.sum())}  n_test={int(mte.sum())}  "
          f"operator {R*R}x{R*R}, crop {CR}, blur {args.blur}")

    for goal in [g.strip() for g in args.goals.split(",") if g.strip()]:
        dw = lyapunov_weights((H, W), goal, dev)
        v0, v1 = lyapunov(ote, dw), lyapunov(o1te, dw)
        dv_true = v1 - v0
        # Controls for the state: its own cost, and how much pile the blade
        # meets (both known before acting, so a controller has them too).
        from fit_linear_foresight import contact_score, swept_region_mask
        _plate = 0.04 / 0.128 * W
        control = [v0, contact_score(ote, ste, ete, (H, W), _plate)]
        print(f"\n=== goal '{goal}' ===")
        print(f"  actual dV: mean {float(dv_true.mean()):+.5f}  sd {float(dv_true.std()):.5f}  "
              f"helpful pushes (dV<0): {100 * float((dv_true < 0).float().mean()):.0f}%")
        print(f"  {'model':18s} {'pearson':>8s} {'partial':>8s} {'spearman':>9s} "
              f"{'sign ok':>8s} {'slate4':>8s} {'slate16':>8s}")
        for nm, p in preds.items():
            dv_pred = lyapunov(p, dw) - v0
            if float(dv_pred.abs().max()) < 1e-9:
                print(f"  {nm:18s} {'--':>8s} {'--':>8s} {'--':>9s} {'--':>8s} "
                      f"{0.0:8.3f} {0.0:8.3f}   (predicts dV=0 always: cannot rank)")
                continue
            pe, sp, sg, reg, part = rank_metrics(dv_pred, dv_true, control)
            print(f"  {nm:18s} {pe:8.3f} {part:8.3f} {sp:9.3f} "
                  f"{100 * sg:7.0f}% {reg[4]:8.3f} {reg[16]:8.3f}")
    print("\nslateK = fraction of the oracle's advantage over a random pick that "
          "the model captures\nwhen choosing from K candidates (1.0 = oracle, "
          "0.0 = no better than random).\nThis is what a greedy controller "
          "actually needs; per-pixel accuracy is not.")
    print("partial = correlation after removing the state's own cost and contact "
          "score, i.e.\nthe part of the ranking that is about the ACTION rather "
          "than which state it hit.\nCandidates come from different states "
          "(the data holds one action per state), so\nthe unpartialled columns "
          "overstate what a same-state slate would give.")


if __name__ == "__main__":
    main()
