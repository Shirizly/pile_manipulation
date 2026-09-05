"""EXP-0012: re-test EXP-0008's degradation mechanism on SAME-STATE slates.

EXP-0008 (`docs/experiments/EXP-0008-degradation-spectrum-p2-p3.md`, reviewer
amendment) found that high-frequency noise catastrophically destroys the
fitted operator's action ranking (Spearman dV correlation 0.474 -> 0.089 at
matched rms) while displacement barely dents it, and gave a mechanism:
noise is drawn INDEPENDENTLY per candidate, so it does not cancel when
candidates are compared, while displacement/amplitude/blur perturb every
candidate in the batch the SAME way and so cancel. But that measurement pools
candidates from DIFFERENT start states (`Genesis/data/cube_spectrum/n20`
holds one action per state) -- so "independent noise per candidate" was
guaranteed by the data, not demonstrated to survive a real MPC loop where one
state's candidates share a prediction context.

This script reuses `scripts/probes/degradation_spectrum.py`'s exact fit (same
operator, same training data, same degradation recipes) but evaluates on
`Genesis/data/slates/<tag>` -- SAME-STATE candidate slates collected by
`Genesis/same_state_slate_collection.py` -- and computes rank_metrics WITHIN
each slate (fixed start state, ~32 candidate actions) rather than pooled
across the whole test set. If the mechanism is right, per-slate hf-noise
Spearman should still collapse relative to the undegraded operator, just by
less than EXP-0008's pooled numbers (because same-state noise partially
cancels in a slate-relative sense that pooled-across-states noise cannot).
If per-slate ranking is roughly unaffected by hf-noise, the pooled EXP-0008
result was mostly measuring cross-state confound, not within-slate variance.

Usage: PYTHONPATH=. python scripts/probes/same_state_degradation.py
"""
import argparse
import glob as _glob

import numpy as np
import torch

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import actions_to_pixels, canonicalise, fit_operator, predict_world
from transforms.particle_fields import _gaussian_blur2d
from scripts.probes.fss_scale_decomp import build_pyramid
from control_utility_test import lyapunov_weights, lyapunov, rank_metrics

H = W = 64


def finest_band_noise(shape, gen, region_std):
    raw = torch.randn(shape, generator=gen)
    laps, _ = build_pyramid(raw, 4)
    finest = laps[0]
    finest = finest - finest.mean()
    finest = finest / finest.std().clamp_min(1e-9) * region_std
    return finest


def per_slate_metrics(dv_pred, dv_true, ep, min_slate=8):
    """rank_metrics computed WITHIN each slate (group of equal ep), then
    averaged across slates. Slates smaller than min_slate are dropped (too
    few candidates for a rank correlation to mean anything)."""
    slates = {}
    for e in ep.unique().tolist():
        idx = (ep == e).nonzero(as_tuple=True)[0]
        if idx.numel() < min_slate:
            continue
        p, t = dv_pred[idx], dv_true[idx]
        if float(t.std()) < 1e-9:
            continue
        pear, spear, sign, reg, _ = rank_metrics(p, t)
        slates[int(e)] = dict(pearson=pear, spearman=spear, sign=sign, slate4=reg[4], n=int(idx.numel()))
    if not slates:
        return dict(n_slates=0)
    keys = ["pearson", "spearman", "sign", "slate4"]
    out = {"n_slates": len(slates)}
    for k in keys:
        vals = np.array([s[k] for s in slates.values()])
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-pattern", default="Genesis/data/slates/n20_heap_5mm/*_data.pt")
    ap.add_argument("--train-pattern", default="Genesis/data/cube_spectrum/n20/*_data.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-slate", type=int, default=8)
    args = ap.parse_args()

    # --- fit exactly as EXP-0008 did, on the (disjoint) cross-state n20 set ---
    o0, o1, act, ep, _, _ = load_transition_fields(
        args.train_pattern, H, 0.0, "mean", 19.9, "cpu",
        view="mask", min_grains=1.0, cube_size=0.005)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                    (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    res, crop, ridge = 64, 1.0, 1.0
    Y0 = canonicalise(o0, s_px, e_px, res, crop).reshape(o0.shape[0], -1).T
    Y1 = canonicalise(o1, s_px, e_px, res, crop).reshape(o0.shape[0], -1).T
    A = fit_operator(Y0, Y1, ridge, toward_identity=True)
    print(f"fit on {o0.shape[0]} cross-state transitions from {args.train_pattern}")

    # --- evaluate on the same-state slate set (fully held out: different
    # collection run, never seen by the fit) ---
    files = sorted(_glob.glob(args.slate_pattern))
    if not files:
        raise SystemExit(f"no slate data at {args.slate_pattern} -- run "
                          f"Genesis/same_state_slate_collection.py first")
    so0, so1, sact, sep, _, _ = load_transition_fields(
        args.slate_pattern, H, 0.0, "mean", 19.9, "cpu",
        view="mask", min_grains=1.0, cube_size=0.005)
    ss_px, se_px = actions_to_pixels(sact, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                      (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    print(f"{so0.shape[0]} slate transitions, {sep.unique().numel()} slates "
          f"(files matched: {len(files)})")

    linear_op = predict_world(A, so0, ss_px, se_px, res, (H, W))
    delta = linear_op - so0

    from fit_linear_foresight import swept_region_mask
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(ss_px, se_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    true_signal = so1 - so0
    sig_laps, _ = build_pyramid(true_signal, 4)
    finest_signal = sig_laps[0]
    mask_bool = region > 0
    region_std = float(finest_signal[mask_bool].std())
    print(f"finest-band true-signal std over region: {region_std:.5f}")

    gen = torch.Generator().manual_seed(1234)

    def make_field(delta_):
        return (so0 + delta_).clamp(0.0, 1.0)

    models = {"persistence": so0, "operator (undegraded)": linear_op}
    for k in (1, 2, 4):
        models[f"displacement k={k}"] = make_field(torch.roll(delta, shifts=k, dims=-1))
    for m in (0.5, 1.0, 2.0):
        noise = finest_band_noise(delta.shape, gen, region_std * m)
        models[f"hf-noise m={m}"] = make_field(delta + noise)

    for goal in ("center", "corner"):
        dw = lyapunov_weights((H, W), goal, "cpu")
        v0 = lyapunov(so0, dw)
        v1 = lyapunov(so1, dw)
        dv_true = v1 - v0
        print(f"\n=== goal={goal} ===  dv_true mean={float(dv_true.mean()):+.5f} "
              f"sd={float(dv_true.std()):.5f} helpful(dV<0)="
              f"{100*float((dv_true<0).float().mean()):.0f}%")
        print(f"{'model':26s}{'n_slt':>6s}{'spearman':>11s}{'(sd)':>8s}"
              f"{'sign%':>8s}{'slate4':>9s}{'(sd)':>8s}")
        for name, pred in models.items():
            dv_pred = lyapunov(pred, dw) - v0
            m = per_slate_metrics(dv_pred, dv_true, sep, min_slate=args.min_slate)
            if m["n_slates"] == 0:
                print(f"{name:26s}  (no slate had >= {args.min_slate} live candidates)")
                continue
            print(f"{name:26s}{m['n_slates']:6d}{m['spearman_mean']:11.3f}"
                  f"{m['spearman_sd']:8.3f}{100*m['sign_mean']:7.0f}%"
                  f"{m['slate4_mean']:9.3f}{m['slate4_sd']:8.3f}")

    print("\nspearman/slate4 here are computed WITHIN each same-state slate "
          "(one start state, ~n_envs candidate\nactions) and then averaged "
          "across slates -- unlike EXP-0008's pooled-across-states numbers.\n"
          "Compare operator(undegraded) vs hf-noise m=... at matched levels "
          "to EXP-0008's Table for the\ncross-state comparison.")


if __name__ == "__main__":
    main()
