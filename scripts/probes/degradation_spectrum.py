"""EXP-0008: synthetic-degradation spectrum for P2/P3
(docs/ideas_log_signal_vs_detail.md sec.4-5).

Take the fitted linear operator's held-out prediction on the piled-cube
spectrum (n20) and corrupt it in controlled, known-character ways. For each
degraded "model" measure (a) pixel rms and FSS(r=1) against the usable-skill
threshold, and (b) control utility -- Spearman rank-correlation of predicted
vs. true Lyapunov dV on a simple convex goal, reusing control_utility_test.py.

Usage: PYTHONPATH=. python scripts/probes/degradation_spectrum.py
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (actions_to_pixels, canonicalise, fit_operator,
                                   swept_region_mask, predict_world)
from transforms.particle_fields import _gaussian_blur2d
from scripts.probes.fss_scale_decomp import fss_pooled, build_pyramid
from control_utility_test import lyapunov_weights, lyapunov, rank_metrics

H = W = 64


def pixel_rms(pred, truth, region):
    """Pooled region-weighted rms, same pooling convention as fss_pooled."""
    num = (region * (pred - truth) ** 2).sum()
    den = region.sum().clamp_min(1e-9)
    return float((num / den).sqrt())


def finest_band_noise(shape, gen, region_std):
    """Zero-mean noise confined to the finest Laplacian band, scaled to
    region_std (the std of the finest band of the true signal, over the
    scored region) so magnitudes are physically anchored rather than
    arbitrary pixel units."""
    raw = torch.randn(shape, generator=gen)
    laps, _ = build_pyramid(raw, 4)
    finest = laps[0]
    finest = finest - finest.mean()
    finest = finest / finest.std().clamp_min(1e-9) * region_std
    return finest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--goal", default="center")
    args = ap.parse_args()

    pattern = "Genesis/data/cube_spectrum/n20/*_data.pt"
    o0, o1, act, ep, _, _ = load_transition_fields(
        pattern, H, 0.0, "mean", 19.9, "cpu",
        view="mask", min_grains=1.0, cube_size=0.005)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                    (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(args.seed)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps) // 4)].tolist())
    te = torch.tensor([int(e) in val for e in ep])
    tr = ~te
    print(f"{int(tr.sum())} train / {int(te.sum())} test transitions "
          f"({len(eps)} episodes, {len(val)} held out)")

    res, crop, ridge = 64, 1.0, 1.0
    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    A = fit_operator(Y0, Y1, ridge, toward_identity=True)

    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    n_te = int(te.sum())
    linear_op = predict_world(A, ote, ste, ete, res, (H, W))
    delta = linear_op - ote  # the predicted delta we will degrade

    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    dw = lyapunov_weights((H, W), args.goal, "cpu")
    v0 = lyapunov(ote, dw)
    v1 = lyapunov(o1te, dw)
    dv_true = v1 - v0
    print(f"goal={args.goal}  dv_true mean={float(dv_true.mean()):+.5f} "
          f"sd={float(dv_true.std()):.5f}  helpful(dV<0)="
          f"{100*float((dv_true<0).float().mean()):.0f}%")

    # Reference magnitude for hf-noise: std of the TRUE finest-band signal
    # (truth - persistence), over the scored region -- the "signal" used by
    # fss_scale_decomp's M5. Anchors noise levels to something physical
    # rather than an arbitrary pixel unit.
    true_signal = o1te - ote
    sig_laps, _ = build_pyramid(true_signal, 4)
    finest_signal = sig_laps[0]
    mask_bool = region > 0  # region: (n_te, H, W), per-transition swept band
    vals = finest_signal[mask_bool]
    region_std = float(vals.std())
    print(f"finest-band true-signal std over region: {region_std:.5f}")

    gen = torch.Generator().manual_seed(1234)

    def make_field(delta_):
        return (ote + delta_).clamp(0.0, 1.0)

    models = {}
    models["oracle"] = o1te
    models["persistence"] = ote
    models["operator (undegraded)"] = linear_op
    for k in (1, 2, 4):
        models[f"displacement k={k}"] = make_field(torch.roll(delta, shifts=k, dims=-1))
    for a in (0.5, 0.75, 1.25, 1.5):
        models[f"amplitude a={a}"] = make_field(a * delta)
    for m in (0.5, 1.0, 2.0):
        noise = finest_band_noise(delta.shape, gen, region_std * m)
        models[f"hf-noise m={m}"] = make_field(delta + noise)
    for sig in (1.0, 2.0):
        models[f"blur s={sig}"] = make_field(_gaussian_blur2d(delta, sig))
    models["wrong-physics"] = make_field(torch.roll(delta, shifts=1, dims=0))

    f0 = float((region * o1te).sum() / region.sum().clamp_min(1e-9))
    fss_useful = 0.5 + f0 / 2
    print(f"base rate f0={f0:.4f}  FSS_useful={fss_useful:.4f}")

    rows = []
    for name, pred in models.items():
        rms = pixel_rms(pred, o1te, region)
        fss1 = fss_pooled(pred, o1te, region, [1])[1]
        dv_pred = lyapunov(pred, dw) - v0
        if float(dv_pred.abs().max()) < 1e-9:
            pear = spear = sign = np.nan
            reg4 = 0.0
        else:
            pear, spear, sign, reg, part = rank_metrics(dv_pred, dv_true)
            reg4 = reg[4]
        rows.append(dict(name=name, rms=rms, fss1=fss1, pearson=pear,
                          spearman=spear, sign=sign, slate4=reg4))

    print(f"\n{'model':26s}{'rms':>8s}{'fss@1':>8s}{'>=useful':>9s}"
          f"{'pearson':>9s}{'spearman':>10s}{'sign%':>8s}{'slate4':>8s}")
    for r in rows:
        print(f"{r['name']:26s}{r['rms']:8.4f}{r['fss1']:8.4f}"
              f"{'yes' if r['fss1'] >= fss_useful else 'no':>9s}"
              f"{r['pearson']:9.3f}{r['spearman']:10.3f}"
              f"{100*r['sign']:7.0f}%{r['slate4']:8.3f}")

    # Exclude the anchors (oracle/persistence have degenerate rms=0 / dv=0)
    # from the correlation-of-metrics analysis -- they are reference points,
    # not part of the degradation spectrum being ranked.
    spectrum = [r for r in rows if r["name"] not in
                ("oracle", "persistence")]
    def spearman_xy(xs, ys):
        x = torch.tensor(xs); y = torch.tensor(ys)
        rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
        rx = (rx - rx.mean()) / rx.std().clamp_min(1e-9)
        ry = (ry - ry.mean()) / ry.std().clamp_min(1e-9)
        return float((rx * ry).mean())

    rms_vals = [r["rms"] for r in spectrum]
    fss_vals = [r["fss1"] for r in spectrum]
    ctrl_vals = [r["spearman"] for r in spectrum]
    slate_vals = [r["slate4"] for r in spectrum]
    n = len(spectrum)
    rho_rms_ctrl = spearman_xy(rms_vals, ctrl_vals)
    rho_fss_ctrl = spearman_xy(fss_vals, ctrl_vals)
    rho_rms_slate = spearman_xy(rms_vals, slate_vals)
    rho_fss_slate = spearman_xy(fss_vals, slate_vals)
    print(f"\nP3 (control utility = dV rank-correlation):")
    print(f"    spearman(rms, control) = {rho_rms_ctrl:+.3f}  (n={n})")
    print(f"    spearman(fss@1, control) = {rho_fss_ctrl:+.3f}  (n={n})")
    print(f"P3 (control utility = slate4 regret, the greedy-selection proxy):")
    print(f"    spearman(rms, slate4) = {rho_rms_slate:+.3f}  (n={n})")
    print(f"    spearman(fss@1, slate4) = {rho_fss_slate:+.3f}  (n={n})")
    print("    (rms 'better' at ranking control utility would score MORE "
          "negative here, since\n     lower rms should mean higher control "
          "utility; fss 'better' scores MORE positive.)")

    # P2: matched-rms comparison between displacement and hf-noise families.
    disp = {r["name"]: r for r in rows if r["name"].startswith("displacement")}
    noise = {r["name"]: r for r in rows if r["name"].startswith("hf-noise")}
    print("\nP2 detail -- displacement family:")
    for nm, r in disp.items():
        print(f"  {nm:20s} rms={r['rms']:.4f}  spearman={r['spearman']:.3f}  sign%={100*r['sign']:.0f}")
    print("P2 detail -- hf-noise family:")
    for nm, r in noise.items():
        print(f"  {nm:20s} rms={r['rms']:.4f}  spearman={r['spearman']:.3f}  sign%={100*r['sign']:.0f}")

    return rows, rho_rms_ctrl, rho_fss_ctrl


if __name__ == "__main__":
    main()
