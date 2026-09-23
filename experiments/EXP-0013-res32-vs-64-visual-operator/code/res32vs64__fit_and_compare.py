#!/usr/bin/env python -u
"""Fit switched-linear visual operators at res=32 and res=64 on pooled
Sean + overnight_randlen_train data, compare on accuracy
(overnight_randlen_test) and control (slates_binned n20 scatter corpus).

Reuses the cache built by build_cache.py (fast path: glob *_data.pt + direct
torch.load + particles_to_occupancy, NOT load_randlen_cell -- see
docs/CODEMAP.md's LOADER TRAP). Loads the cache ONCE; every ridge value and
resolution below re-uses the same in-memory tensors.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from fit_linear_foresight import (  # noqa: E402
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask,
)
from Baselines.LinearForesight.model import bin_index, predict_switched, push_length_m  # noqa: E402
from Baselines.common.goals import (  # noqa: E402
    letter_mask, random_quadrant_mask, dist_field_from_mask,
    mass_in_region, signed_mass_in_region, slate_n_capture,
)
from control_utility_test import lyapunov  # noqa: E402
from transforms.functional import particles_to_occupancy  # noqa: E402
from Genesis.binned_slate_dataset import BinnedSlateCorpus  # noqa: E402
from utils import git_provenance  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ROWS_PER_BIN = 50
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
OUT_DIR = Path(__file__).resolve().parent.parent

WS_MIN = torch.tensor([-0.064, -0.064])
WS_MAX = torch.tensor([0.064, 0.064])
GRID = 64  # native raster grid, fixed for every corpus (see build_cache.py)
BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH

GOAL_SHAPES = ["ring", "T", "random_quadrant"]  # ring -> letter "O"
VALUE_FNS = ["lyapunov", "mass_in_region", "signed_mass_in_region"]
K_FIXED = 32
N_RESAMPLE = 150
CONTROL_CORPUS = "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"

RIDGE_VALUES = [0.1, 1.0, 10.0]
RESOLUTIONS = [32, 64]
CROP = 1.0


def sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def load_cache():
    sean = torch.load(CACHE_DIR / "sean.pt", map_location="cpu")
    ov_train = torch.load(CACHE_DIR / "overnight_train.pt", map_location="cpu")
    ov_test = torch.load(CACHE_DIR / "overnight_test.pt", map_location="cpu")
    return sean, ov_train, ov_test


def pool_train(sean, ov_train):
    occ0 = torch.cat([sean["occ0"], ov_train["occ0"]])
    occ1 = torch.cat([sean["occ1"], ov_train["occ1"]])
    actions = torch.cat([sean["actions"], ov_train["actions"]])
    length_m = torch.cat([sean["length_m"], ov_train["length_m"]])
    n_particles = torch.cat([sean["n_particles"], ov_train["n_particles"]])
    source = (["sean"] * sean["occ0"].shape[0]) + (["overnight_train"] * ov_train["occ0"].shape[0])
    return occ0, occ1, actions, length_m, n_particles, source


def report_bin_counts(length_m, n_bins, hi):
    bin_edges = torch.linspace(0.0, hi, n_bins + 1)
    bins = bin_index(length_m, bin_edges)
    counts = {}
    for b in range(n_bins):
        lo_mm, hi_mm = float(bin_edges[b]) * 1000, float(bin_edges[b + 1]) * 1000
        counts[b] = {"range_mm": [round(lo_mm, 1), round(hi_mm, 1)],
                     "n_rows": int((bins == b).sum())}
    return bin_edges, counts


def fit_bins_at_res(occ0, occ1, s_px, e_px, length_m, bin_edges, res, ridge, device):
    n_bins = len(bin_edges) - 1
    bins = bin_index(length_m, bin_edges)
    operators, counts = [], []
    for b in range(n_bins):
        m = bins == b
        n_b = int(m.sum())
        counts.append(n_b)
        lo, hi = float(bin_edges[b]) * 1000, float(bin_edges[b + 1]) * 1000
        if n_b < MIN_ROWS_PER_BIN:
            print(f"    bin {b} [{lo:.1f},{hi:.1f}) mm: {n_b} rows (<{MIN_ROWS_PER_BIN}), identity")
            operators.append(torch.eye(res * res))
            continue
        Y0 = canonicalise(occ0[m].to(device), s_px[m].to(device), e_px[m].to(device),
                           res, CROP).reshape(n_b, -1).T
        Y1 = canonicalise(occ1[m].to(device), s_px[m].to(device), e_px[m].to(device),
                           res, CROP).reshape(n_b, -1).T
        t0 = time.time()
        A = fit_operator(Y0, Y1, ridge=ridge, toward_identity=True).cpu()
        del Y0, Y1
        torch.cuda.empty_cache() if device == "cuda" else None
        print(f"    bin {b} [{lo:.1f},{hi:.1f}) mm: {n_b} rows, ridge={ridge}, {time.time()-t0:.1f}s")
        operators.append(A)
    return operators, counts


def eval_accuracy(operators, bin_edges, res, test, H, W):
    lengths_te = push_length_m(test["actions"])
    s_te, e_te = actions_to_pixels(test["actions"], WS_MIN, WS_MAX, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_te, e_te, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    pred = predict_switched(bin_edges, operators, test["occ0"], s_te, e_te, lengths_te, res, (H, W), CROP)
    m_pers = metrics(test["occ0"], test["occ1"], test["occ0"], region=region)
    m_pred = metrics(pred, test["occ1"], test["occ0"], region=region)
    return pred, region, m_pers, m_pred, s_te, e_te


def stratified_accuracy(pred, truth, occ0, region, n_particles):
    out = {}
    for grp, lo, hi in [("n20", 15, 25), ("n50", 45, 55)]:
        m = (n_particles >= lo) & (n_particles <= hi)
        if int(m.sum()) == 0:
            continue
        out[grp] = metrics(pred[m], truth[m], occ0[m], region=region[m])
    return out


def build_goal(shape, res, seed=0):
    if shape == "ring":
        mask_np = letter_mask("O", res, res)
    elif shape == "T":
        mask_np = letter_mask("T", res, res)
    elif shape == "random_quadrant":
        mask_np, _ = random_quadrant_mask(res, res, seed=seed)
    else:
        raise ValueError(shape)
    dw = torch.from_numpy(dist_field_from_mask(mask_np)).to(DEVICE)
    mask_t = torch.from_numpy(mask_np).to(DEVICE)
    return mask_t, dw


def value_and_dv(value_fn, occ0, occ1, occ_pred, mask, dw):
    """Returns (dv_true, dv_pred, higher_is_better) where dv is always in
    'positive = model/world improved the goal' orientation."""
    if value_fn == "lyapunov":
        v0 = lyapunov(occ0, dw)
        v1_true = lyapunov(occ1, dw)
        v1_pred = lyapunov(occ_pred, dw)
        # lyapunov is a COST (lower = better) -- flip sign so dv is a benefit,
        # matching mass_in_region's orientation and slate_n_capture's
        # higher-is-better convention uniformly.
        dv_true = v0 - v1_true
        dv_pred = v0 - v1_pred
    elif value_fn == "mass_in_region":
        v0 = mass_in_region(occ0, mask)
        v1_true = mass_in_region(occ1, mask)
        v1_pred = mass_in_region(occ_pred, mask)
        dv_true = v1_true - v0
        dv_pred = v1_pred - v0
    elif value_fn == "signed_mass_in_region":
        v0 = signed_mass_in_region(occ0, mask)
        v1_true = signed_mass_in_region(occ1, mask)
        v1_pred = signed_mass_in_region(occ_pred, mask)
        dv_true = v1_true - v0
        dv_pred = v1_pred - v0
    else:
        raise ValueError(value_fn)
    return dv_true.cpu(), dv_pred.cpu()


def score_pool_kfixed(dv_true, dv_pred, slate_ids, k=K_FIXED, n_resample=N_RESAMPLE, seed=0):
    """Per-slate mean capture at a fixed K, sampled WITHOUT replacement --
    reused verbatim from experiments/EXP-0012-.../code/binned-calibration__calibrate.py's
    score_pool_kfixed, the established methodology for this exact corpus."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        n = rows.numel()
        if n < k:
            continue
        vt, vp = dv_true[rows], dv_pred[rows]
        per = []
        for _ in range(n_resample):
            idx = torch.randperm(n, generator=g)[:k]
            c = slate_n_capture(vp[idx], vt[idx], higher_is_better=True)
            if c == c:
                per.append(c)
        if per:
            out.append(float(np.mean(per)))
    return out


def wins_losses_ties(caps, tol=1e-6):
    caps = np.asarray(caps, dtype=float)
    wins = int((caps > tol).sum())
    losses = int((caps < -tol).sum())
    ties = len(caps) - wins - losses
    return wins, losses, ties


def main():
    print("=== Loading cache ===")
    t0 = time.time()
    sean, ov_train, ov_test = load_cache()
    print(f"sean={sean['occ0'].shape[0]} ov_train={ov_train['occ0'].shape[0]} "
          f"ov_test={ov_test['occ0'].shape[0]} rows, cache load {time.time()-t0:.1f}s")

    occ0, occ1, actions, length_m, n_particles, source = pool_train(sean, ov_train)
    n_total = occ0.shape[0]
    print(f"pooled train: {n_total} rows")
    hi = float(length_m.max())
    s_px, e_px = actions_to_pixels(actions, WS_MIN, WS_MAX, (GRID, GRID))

    print("\n=== Per-bin training counts (6 bins) ===")
    edges6, counts6 = report_bin_counts(length_m, 6, hi)
    for b, c in counts6.items():
        print(f"  bin {b} {c['range_mm']} mm: {c['n_rows']}")
    print("\n=== Per-bin training counts (7 bins) ===")
    edges7, counts7 = report_bin_counts(length_m, 7, hi)
    for b, c in counts7.items():
        print(f"  bin {b} {c['range_mm']} mm: {c['n_rows']}")

    n_bins = 6
    bin_edges = edges6
    print(f"\nUsing {n_bins} bins for the headline fit.")

    all_ops = {}
    for res in RESOLUTIONS:
        for ridge in RIDGE_VALUES:
            print(f"\n=== fit res={res} ridge={ridge} ===")
            ops, counts = fit_bins_at_res(occ0, occ1, s_px, e_px, length_m, bin_edges, res, ridge, DEVICE)
            all_ops[(res, ridge)] = {"operators": ops, "counts": counts}
            torch.save({"operators": ops, "bin_edges": bin_edges, "counts": counts,
                        "res": res, "ridge": ridge, "n_bins": n_bins, "crop": CROP,
                        "provenance": git_provenance()},
                       OUT_DIR / f"operators_res{res}_ridge{ridge}.pt")

    print("\n=== Evaluating accuracy on overnight_randlen_test ===")
    H, W = GRID, GRID
    accuracy = {}
    preds_by_res_ridge = {}
    for res in RESOLUTIONS:
        accuracy[res] = {}
        for ridge in RIDGE_VALUES:
            ops = all_ops[(res, ridge)]["operators"]
            pred, region, m_pers, m_pred, s_te, e_te = eval_accuracy(ops, bin_edges, res, ov_test, H, W)
            strat = stratified_accuracy(pred, ov_test["occ1"], ov_test["occ0"], region, ov_test["n_particles"])
            accuracy[res][ridge] = {"persistence": m_pers, "linear_switched": m_pred,
                                     "stratified": strat}
            preds_by_res_ridge[(res, ridge)] = (pred, region)
            print(f"  res={res} ridge={ridge}: persistence acc={m_pers['accuracy']:.4f}  "
                  f"linear-switched acc={m_pred['accuracy']:.4f}  "
                  f"n20={strat.get('n20',{}).get('accuracy',float('nan')):.4f}  "
                  f"n50={strat.get('n50',{}).get('accuracy',float('nan')):.4f}")

    print("\n=== Fairness comparison: downsample 64->32, score both on 32x32 ===")
    ds_compare = {}
    for ridge in RIDGE_VALUES:
        pred64, region64 = preds_by_res_ridge[(64, ridge)]
        n = pred64.shape[0]
        pred64_ds = F.avg_pool2d(pred64.view(n, 1, 64, 64), 2, 2).view(n, 32, 32)
        occ1_ds = F.avg_pool2d(ov_test["occ1"].view(n, 1, 64, 64), 2, 2).view(n, 32, 32)
        occ0_ds = F.avg_pool2d(ov_test["occ0"].view(n, 1, 64, 64), 2, 2).view(n, 32, 32)
        region_ds = (F.avg_pool2d(region64.view(n, 1, 64, 64), 2, 2).view(n, 32, 32) > 0).float()
        m_64ds = metrics(pred64_ds, occ1_ds, occ0_ds, region=region_ds)
        m_32 = accuracy[32][ridge]["linear_switched"]
        ds_compare[ridge] = {"res64_downsampled_to_32": m_64ds, "res32_native": m_32,
                              "accuracy_diff_64ds_minus_32": m_64ds["accuracy"] - m_32["accuracy"]}
        print(f"  ridge={ridge}: res64->32 acc={m_64ds['accuracy']:.4f}  "
              f"res32 native acc={m_32['accuracy']:.4f}  diff={m_64ds['accuracy']-m_32['accuracy']:+.4f}")

    # pick control-eval ridge: the one with best downsampled-fair accuracy delta closeness
    # -- use ridge=1.0 (established default, Baselines/LinearForesight/fit_switched.py)
    control_ridge = 1.0
    print(f"\n=== Control eval on {CONTROL_CORPUS} (ridge={control_ridge} for both resolutions) ===")
    corpus = BinnedSlateCorpus.load(CONTROL_CORPUS)
    rows = corpus.step(0)
    n_ctrl = len(rows)
    print(f"corpus: {corpus.n_slates} slates x {corpus.n_actions} candidates, "
          f"spawn={corpus.spawn_style}, n20-only, {n_ctrl} rows")

    states = rows.states[:, :, :3].float()
    states_ = rows.states_[:, :, :3].float()
    p_start = rows.p_starts[:, :2].float()
    p_stop = rows.p_stops[:, :2].float()
    actions_c = torch.cat([p_start, p_stop], dim=1)
    length_c = (p_stop - p_start).norm(dim=-1)
    slate_ids = rows.slate_idx.long()

    def chunked_occ(states_t, size=1024):
        outs = []
        for i in range(0, states_t.shape[0], size):
            outs.append(particles_to_occupancy(states_t[i:i+size].to(DEVICE), BOUNDS, (GRID, GRID),
                                                footprint_radius=RADIUS).cpu())
        return torch.cat(outs)

    occ0_c = chunked_occ(states)
    occ1_c = chunked_occ(states_)
    s_px_c, e_px_c = actions_to_pixels(actions_c, WS_MIN, WS_MAX, (GRID, GRID))

    control_preds = {}
    for res in RESOLUTIONS:
        ops = all_ops[(res, control_ridge)]["operators"]
        pred_c = predict_switched(bin_edges, ops, occ0_c, s_px_c, e_px_c, length_c, res, (GRID, GRID), CROP)
        control_preds[res] = pred_c
    control_preds["persistence"] = occ0_c
    torch.manual_seed(0)
    control_preds["random"] = torch.rand_like(occ0_c)

    control_results = {}
    for shape in GOAL_SHAPES:
        mask, dw = build_goal(shape, GRID, seed=hash(shape) % (2**31))
        control_results[shape] = {}
        for vfn in VALUE_FNS:
            control_results[shape][vfn] = {}
            dv_true_ref = None
            for model_name, occ_pred in control_preds.items():
                dv_true, dv_pred = value_and_dv(vfn, occ0_c.to(DEVICE), occ1_c.to(DEVICE),
                                                 occ_pred.to(DEVICE), mask, dw)
                if dv_true_ref is None:
                    dv_true_ref = dv_true
                caps = score_pool_kfixed(dv_true, dv_pred, slate_ids, k=K_FIXED, n_resample=N_RESAMPLE)
                w, l, t = wins_losses_ties(caps)
                frac_zero = float((dv_true.abs() < 1e-6).float().mean())
                res_entry = {
                    "mean_capture": float(np.mean(caps)) if caps else float("nan"),
                    "sem": sem(caps), "wins": w, "losses": l, "ties": t,
                    "effective_n": len(caps), "frac_dv_true_zero": frac_zero,
                }
                control_results[shape][vfn][str(model_name)] = res_entry
                print(f"  {shape}/{vfn}/{model_name}: cap={res_entry['mean_capture']:.4f} "
                      f"sem={res_entry['sem']:.4f} w/l/t={w}/{l}/{t} n_eff={len(caps)} "
                      f"fz={frac_zero:.3f}")

    # -- save everything --
    results = {
        "description": "res32 vs res64 switched-linear visual operator comparison",
        "train_rows": {"sean": sean["occ0"].shape[0], "overnight_train": ov_train["occ0"].shape[0],
                       "pooled_total": n_total},
        "bin_counts_6": counts6, "bin_counts_7": counts7,
        "bins_used_for_headline": n_bins,
        "accuracy": {str(res): {str(r): {"persistence": v["persistence"], "linear_switched": v["linear_switched"],
                                          "stratified": v["stratified"]}
                                 for r, v in d.items()}
                     for res, d in accuracy.items()},
        "fair_downsample_comparison_64_to_32": {str(r): v for r, v in ds_compare.items()},
        "control": {"corpus": CONTROL_CORPUS, "control_ridge": control_ridge,
                    "note": "n20 scatter-spawn only corpus", "results": control_results},
        "provenance": git_provenance(),
    }

    def _clean(o):
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(OUT_DIR / "results_res_compare.json", "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nwrote {OUT_DIR / 'results_res_compare.json'}")


if __name__ == "__main__":
    main()
