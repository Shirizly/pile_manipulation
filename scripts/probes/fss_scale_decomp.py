"""EXP-0007: test P1 (docs/ideas_log_signal_vs_detail.md sec.5) with M1 (FSS)
and M5 (scale-decomposed error) on the piled-cube spectrum (n20).

FSS(r): box-filter both fields to radius r, then
    FSS(r) = 1 - MSE(frac_pred, frac_truth) / (mean(frac_pred^2)+mean(frac_truth^2))
pooled (sum over all region-weighted pixels, not per-sample-averaged) so the
denominator never blows up on a near-empty sample.

Confound handled: FSS -> 1 for every model at large r because a smoothed field
agrees with everything, including persistence. So the reported quantity is
FSS(model) - FSS(persistence) and FSS(model) - FSS(mean-delta) at each r, not
raw FSS. A model with no real skill (persistence, or the warp-only identity
control) is included explicitly as the zero-skill reference the other curves
are measured against.

Scale decomposition: Laplacian pyramid of the ERROR field (pred - truth) and
of the SIGNAL field (truth - persistence), same pyramid depth, region-masked
at each level (mask is average-pooled down the pyramid alongside the image).
Reported as the pooled RMS ratio per band = ||error_band|| / ||signal_band||;
1.0 = no better than persistence in that band, 0 = perfect.

Usage: PYTHONPATH=. python scripts/probes/fss_scale_decomp.py [--sigma 0.0]
"""
import argparse
import torch
import torch.nn.functional as F

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (actions_to_pixels, canonicalise, fit_operator,
                                   swept_region_mask, predict_world)
from transforms.particle_fields import _gaussian_blur2d

H = W = 64
RADII = [1, 2, 3, 5, 8, 12, 16]
PYR_LEVELS = 4


def box_filter(x, r):
    if r == 0:
        return x
    k = 2 * r + 1
    return F.avg_pool2d(x.unsqueeze(1), kernel_size=k, stride=1, padding=r).squeeze(1)


def fss_pooled(pred, truth, region, radii):
    out = {}
    for r in radii:
        fp, ft = box_filter(pred, r), box_filter(truth, r)
        num = (region * (fp - ft) ** 2).sum()
        den = (region * fp ** 2).sum() + (region * ft ** 2).sum()
        out[r] = float(1.0 - num / den.clamp_min(1e-9))
    return out


def build_pyramid(x, levels):
    """Burt-Adelson-style pyramid using the repo's own separable blur.

    Returns (laps, gs): laps[0] is finest (highest-freq) band ... laps[-1] is
    the coarsest low-pass residual; gs are the Gaussian levels (for masks).
    """
    gs = [x]
    cur = x
    for _ in range(levels):
        blurred = _gaussian_blur2d(cur, 1.0)
        down = F.avg_pool2d(blurred.unsqueeze(1), 2).squeeze(1)
        gs.append(down)
        cur = down
    laps = []
    for i in range(levels):
        up = F.interpolate(gs[i + 1].unsqueeze(1), size=gs[i].shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        laps.append(gs[i] - up)
    laps.append(gs[-1])
    return laps, gs


def pyramid_mask(mask, levels):
    """Average-pool the region mask down the same pyramid (no blur: masks are
    already smooth 0/1 rectangles), matched size-for-size with build_pyramid."""
    ms = [mask]
    cur = mask
    for _ in range(levels):
        cur = F.avg_pool2d(cur.unsqueeze(1), 2).squeeze(1)
        ms.append(cur)
    return ms  # ms[i] matches gs[i] and laps[i] for i < levels; ms[-1] matches laps[-1]


def band_ratios(err, sig, mask, levels):
    err_laps, _ = build_pyramid(err, levels)
    sig_laps, _ = build_pyramid(sig, levels)
    masks = pyramid_mask(mask, levels)
    out = []
    for i in range(levels + 1):
        w = masks[i]
        num = (w * err_laps[i] ** 2).sum()
        den = (w * sig_laps[i] ** 2).sum().clamp_min(1e-9)
        out.append(float((num / den).sqrt()))
    return out


def run(sigma, res=64, crop=1.0, ridge=1.0, seed=0):
    pattern = "Genesis/data/cube_spectrum/n20/*_data.pt"
    o0, o1, act, ep, _, _ = load_transition_fields(
        pattern, H, sigma, "mean", 19.9, "cpu",
        view="mask", min_grains=1.0, cube_size=0.005)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                    (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(seed)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps) // 4)].tolist())
    te = torch.tensor([int(e) in val for e in ep])
    tr = ~te
    print(f"sigma={sigma}: {int(tr.sum())} train / {int(te.sum())} test transitions "
          f"({len(eps)} episodes, {len(val)} held out)")

    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    A = fit_operator(Y0, Y1, ridge, toward_identity=True)
    I = torch.eye(A.shape[0])
    bmd = (Y1 - Y0).mean(1, keepdim=True)  # canonical-frame mean delta, a (D,1) column

    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    n_te = int(te.sum())

    # World-frame predictions for every model in the spectrum.
    persistence = ote
    identity_warp = predict_world(I, ote, ste, ete, res, (H, W))
    # mean-delta: add the canonical bmd to every test column, then run the
    # same warp/unwarp/blend pipeline predict_world uses internally.
    from transforms.functional import (to_push_frame, from_push_frame,
                                        push_frame_validity_mask, blend_push_prediction)
    canon0 = to_push_frame(ote, ste, ete, (res, res), crop)
    canon_md = canon0 + bmd.T.reshape(1, res, res)
    back_md = from_push_frame(canon_md, ste, ete, (H, W), crop)
    mask_md = push_frame_validity_mask(ste, ete, (H, W), (res, res), crop)
    mean_delta = blend_push_prediction(back_md, ote, mask_md).clamp_(0.0, 1.0)
    linear_op = predict_world(A, ote, ste, ete, res, (H, W))

    models = {"persistence": persistence, "identity_warp": identity_warp,
              "mean_delta": mean_delta, "linear_operator": linear_op}

    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    n_region_px = float(region.sum())
    print(f"region: {n_region_px:.0f} weighted px over {n_te} test transitions "
          f"({n_region_px / n_te:.1f} px/transition)")

    # ---- M1: FSS curve, every model, plus skill relative to zero-skill refs ----
    curves = {name: fss_pooled(pred, o1te, region, RADII) for name, pred in models.items()}
    print("\n=== FSS(r), sigma=%.1f ===" % sigma)
    header = "r     " + "".join(f"{n:>16s}" for n in models)
    print(header)
    for r in RADII:
        row = f"{r:<6d}" + "".join(f"{curves[n][r]:16.4f}" for n in models)
        print(row)

    print("\n=== operator skill relative to persistence / mean-delta (FSS diff) ===")
    print("r     op-persist   op-meandelta   idwarp-persist")
    for r in RADII:
        d1 = curves["linear_operator"][r] - curves["persistence"][r]
        d2 = curves["linear_operator"][r] - curves["mean_delta"][r]
        d3 = curves["identity_warp"][r] - curves["persistence"][r]
        print(f"{r:<6d}{d1:12.4f}{d2:15.4f}{d3:17.4f}")

    # ---- M5: scale-decomposed error, as a fraction of per-band change ----
    signal = o1te - persistence  # the change persistence misses entirely
    print("\n=== M5 scale-decomposed error: ||err_band|| / ||signal_band|| (1.0 = no skill) ===")
    print("model            " + "".join(f"L{i}(fine)".rjust(10) if i == 0 else
                                         (f"L{i}".rjust(10) if i < PYR_LEVELS else "resid".rjust(10))
                                         for i in range(PYR_LEVELS + 1))
          + "  <- band sizes ~" + str([2 ** i for i in range(PYR_LEVELS)] + ["coarse"]))
    band_results = {}
    for name, pred in models.items():
        err = pred - o1te
        ratios = band_ratios(err, signal, region, PYR_LEVELS)
        band_results[name] = ratios
        print(f"{name:16s}" + "".join(f"{v:10.4f}" for v in ratios))

    return curves, band_results, region


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--also-sigma1", action="store_true")
    args = ap.parse_args()
    run(args.sigma)
    if args.also_sigma1:
        run(1.0)
