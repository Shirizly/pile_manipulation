"""EXP-0025: EXP-0008's FULL degradation spectrum, re-run on SAME-STATE slates.

EXP-0008 (docs/experiments/EXP-0008-degradation-spectrum-p2-p3.md) ran the
complete spectrum -- displacement, amplitude, hf-noise, blur, wrong-physics --
on candidate slates drawn from DIFFERENT start states
(Genesis/data/cube_spectrum/n20, one action per state), which guarantees
"independent noise per candidate" by construction. EXP-0012 re-ran ONLY the
noise-vs-displacement pair on genuinely same-state slates and found the
cross-state confound inflates the noise effect about two-fold (84% -> 42%
relative Spearman drop at m=2.0, n=50 states). The rest of EXP-0008's spectrum
-- amplitude, blur, wrong-physics -- has never been re-run confound-free, and
those arms carry EXP-0008's most striking claim: amplitude and blur errors
cost control utility NOTHING.

This script reuses scripts/probes/same_state_degradation.py's fit/eval
machinery (same operator, same slate data) and scripts/probes/
degradation_spectrum.py's exact degradation recipes (same amplitude/blur/
wrong-physics constructions EXP-0008 used), extended to:
  - the full spectrum (not just displacement + hf-noise)
  - fit_linear_foresight.py::metrics' "accuracy" key, computed per-slate and
    averaged (the new standard image metric), alongside slate4/spearman.

Usage: PYTHONPATH=. python scripts/probes/full_spectrum_same_state.py
"""
import argparse
import glob as _glob

import numpy as np
import torch

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (actions_to_pixels, canonicalise, fit_operator,
                                   predict_world, swept_region_mask, metrics)
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


def per_slate(pred, so0, so1, region, dv_pred_full, dv_true, ep, min_slate=8):
    """accuracy (fit_linear_foresight.metrics) and rank_metrics, each computed
    WITHIN a slate (group of equal ep) then returned per-slate for the caller
    to average. Slates with <min_slate live candidates are dropped."""
    out = []
    for e in ep.unique().tolist():
        idx = (ep == e).nonzero(as_tuple=True)[0]
        if idx.numel() < min_slate:
            continue
        t = dv_true[idx]
        if float(t.std()) < 1e-9:
            continue
        acc = metrics(pred[idx], so1[idx], so0[idx], region[idx])["accuracy"]
        pear, spear, sign, reg, _ = rank_metrics(dv_pred_full[idx], t)
        out.append(dict(accuracy=acc, spearman=spear, slate4=reg[4], sign=sign, n=int(idx.numel())))
    return out


def summarise(rows):
    if not rows:
        return dict(n_slates=0)
    out = {"n_slates": len(rows)}
    for k in ("accuracy", "spearman", "slate4", "sign"):
        vals = np.array([r[k] for r in rows])
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_sd"] = float(np.nanstd(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slate-pattern", default="Genesis/data/slates/n20_heap_5mm/*_data.pt")
    ap.add_argument("--train-pattern", default="Genesis/data/cube_spectrum/n20/*_data.pt")
    ap.add_argument("--min-slate", type=int, default=8)
    args = ap.parse_args()

    # --- fit exactly as EXP-0008/EXP-0012 did, on the disjoint cross-state set ---
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

    # --- evaluate on the same-state slate set (disjoint collection, never
    # seen by the fit) ---
    files = sorted(_glob.glob(args.slate_pattern))
    if not files:
        raise SystemExit(f"no slate data at {args.slate_pattern}")
    so0, so1, sact, sep, _, _ = load_transition_fields(
        args.slate_pattern, H, 0.0, "mean", 19.9, "cpu",
        view="mask", min_grains=1.0, cube_size=0.005)
    ss_px, se_px = actions_to_pixels(sact, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                      (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    print(f"{so0.shape[0]} slate transitions, {sep.unique().numel()} slates "
          f"(files matched: {len(files)})")

    linear_op = predict_world(A, so0, ss_px, se_px, res, (H, W))
    delta = linear_op - so0

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

    models = {"persistence": so0, "oracle": so1, "operator (undegraded)": linear_op}
    for k in (1, 2, 4):
        models[f"displacement k={k}"] = make_field(torch.roll(delta, shifts=k, dims=-1))
    for a in (0.5, 0.75, 1.25, 1.5):
        models[f"amplitude a={a}"] = make_field(a * delta)
    for m in (0.5, 1.0, 2.0):
        noise = finest_band_noise(delta.shape, gen, region_std * m)
        models[f"hf-noise m={m}"] = make_field(delta + noise)
    for sig in (1.0, 2.0):
        models[f"blur s={sig}"] = make_field(_gaussian_blur2d(delta, sig))
    # wrong-physics: substitute the adjacent transition's predicted delta.
    # Transitions are file-ordered (file == slate), so dims=0 roll mostly
    # swaps in another CANDIDATE FROM THE SAME SLATE's delta -- a stronger,
    # same-state-consistent version of EXP-0008's cross-state substitution.
    models["wrong-physics"] = make_field(torch.roll(delta, shifts=1, dims=0))

    results = {}
    for goal in ("center", "corner"):
        dw = lyapunov_weights((H, W), goal, "cpu")
        v0 = lyapunov(so0, dw)
        v1 = lyapunov(so1, dw)
        dv_true = v1 - v0
        helpful = 100 * float((dv_true < 0).float().mean())
        print(f"\n=== goal={goal} ===  dv_true mean={float(dv_true.mean()):+.5f} "
              f"sd={float(dv_true.std()):.5f} helpful(dV<0)={helpful:.0f}%")
        print(f"{'model':26s}{'n_slt':>6s}{'accuracy':>10s}{'(sd)':>8s}"
              f"{'spearman':>10s}{'(sd)':>8s}{'slate4':>9s}{'(sd)':>8s}")
        goal_rows = {}
        for name, pred in models.items():
            dv_pred = lyapunov(pred, dw) - v0
            rows = per_slate(pred, so0, so1, region, dv_pred, dv_true, sep, args.min_slate)
            s = summarise(rows)
            goal_rows[name] = s
            if s["n_slates"] == 0:
                print(f"{name:26s}  (no slate had >= {args.min_slate} live candidates)")
                continue
            print(f"{name:26s}{s['n_slates']:6d}{s['accuracy_mean']:10.4f}"
                  f"{s['accuracy_sd']:8.4f}{s['spearman_mean']:10.3f}"
                  f"{s['spearman_sd']:8.3f}{s['slate4_mean']:9.3f}{s['slate4_sd']:8.3f}")
        results[goal] = goal_rows

    print("\naccuracy/spearman/slate4 computed WITHIN each same-state slate "
          "then averaged across slates (sd across slates reported).")
    return results


if __name__ == "__main__":
    main()
