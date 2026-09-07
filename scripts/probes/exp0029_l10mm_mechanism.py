"""EXP-0029: why does L10mm's -0.44 image accuracy pair with a ~0.91
slateK_exact -- a prediction worse than useless as an image ranking actions
nearly as well as an oracle? Four diagnostics, each able to kill the number:

D1 -- error-projection decomposition. V = d^T y / ||y||_1 is, to first order
    at fixed mass, a LINEAR functional of the occupancy: dV_pred - dV_true =
    V(y1_hat) - V(y1) ~= (1/M0) d^T (y1_hat - y1) = (1/M0) d^T e. So the cost
    can only see the component of e = pred - truth that projects onto d;
    error orthogonal to d is invisible to dV but fully visible to `accuracy`
    (an rms over all region pixels). Measures, per push length x model, the
    fraction of swept-region error ENERGY that projects onto the (mass-
    normalised, mean-removed, per-transition) cost direction, and separately
    checks the linearisation itself against the exact whole-image dV.

D2 -- is L10mm ranking kinematic? Scores the parameter-free geometric push
    heuristics already in model/eulerian_wrapper.py (predict_heuristic) with
    the identical metric suite (accuracy + slateK_exact) at all three push
    lengths. If a heuristic with NO learned dynamics reaches ~0.9 at L10mm,
    the ranking task there is not testing dynamics knowledge.

D3 -- shuffle null. Permutes dv_pred within each slate (breaks the pairing,
    preserves both marginals) and recomputes slateK_exact; must collapse to
    ~0 for every model at every cell, or there is a pairing/leakage bug.

D4 -- the UNet-vs-linear internal inconsistency at L10mm: paired test across
    pools (mean diff, paired sem, t, win/loss/tie) at K=4 and K=128, plus the
    per-pool distribution of capture and dv_true spread (not just the mean).

Usage
-----
    PYTHONPATH=. python scripts/probes/exp0029_l10mm_mechanism.py \
        --out runs_expB/exp0029_results.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml
from scipy import stats

from control_utility_test import lyapunov, lyapunov_weights
from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_heuristic,
    predict_world, swept_region_mask,
)
from Genesis.training.dataset import PileSweepData
from pathlib import Path
from registry.dataset_registry import build_dataset
from scripts.probes.exp0009_rerun import predict_meandelta
from scripts.probes.exp0021_eval import load_unet, unet_forward
from scripts.probes.exp0026_kcurve_exact import paired, per_slate_exact, sweep
from utils import git_provenance

R, CR, RIDGE = 64, 1.0, 1.0
CELLS = ["L10mm", "L20mm", "L40mm"]
KS = [4, 16, 32, 64, 128]
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_files(eval_cfg_path: str, split: str):
    scfg = yaml.safe_load(open(eval_cfg_path).read())
    root = REPO_ROOT / "Genesis" / "data"
    files = []
    for path in scfg["paths"]:
        full = root / path
        runs = PileSweepData._collect_run_paths(PileSweepData, full)
        runs = PileSweepData._filter_split(runs, split, scfg.get("val_pct", 0),
                                           scfg.get("test_pct", 0))
        files.extend(str(d.name) for d, _ in runs)
    return files


def load_cell(cell: str):
    """Fit train-split operators; load the full step-0 eval subset (occ0,
    occ1, actions, s_px, e_px, slate id) exactly as
    scripts/probes/expB_multistep_eval.py does, so every number here is
    directly comparable to that script's own (cached) accuracy/dv numbers."""
    train_cfg = f"configs/dataset/genesis_slates_multistep_n20_{cell}_train.yaml"
    eval_cfg = f"configs/dataset/genesis_slates_multistep_n20_{cell}_eval.yaml"
    manifest_path = f"Genesis/data/slates_multistep/n20_{cell}/manifest.json"
    run_dir = f"runs_expB/unetfilm_slates_multistep_n20_{cell}"

    data_tr = load_transition_arrays(train_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ_t.shape[0]
    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)
    A_id = torch.eye(R * R, dtype=A.dtype)

    model, _run_cfg, missing = load_unet(run_dir)
    assert not missing.missing_keys and not missing.unexpected_keys, missing

    ecfg = yaml.safe_load(open(eval_cfg).read())
    wrapper = build_dataset(ecfg, "train")
    raw = wrapper.raw_dataset
    n = len(wrapper)
    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    run_idx = torch.tensor([raw.get_run_index(i) for i in range(n)])
    s_px, e_px = actions_to_pixels(actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))

    files = _run_files(eval_cfg, "train")
    manifest = json.loads(open(manifest_path).read())
    batch_lookup = {b["batch_idx"]: (b["slate_idx"], b["step_idx"]) for b in manifest["batches"]}
    run_to_batch = {}
    for r, fname in enumerate(files):
        bidx = int(fname.split("_")[1])
        run_to_batch[r] = bidx
    slate_of = torch.tensor([batch_lookup[run_to_batch[int(r)]][0] for r in run_idx.tolist()])
    step_of = torch.tensor([batch_lookup[run_to_batch[int(r)]][1] for r in run_idx.tolist()])

    X = torch.stack([wrapper[i]["input"] for i in range(n)])
    P = torch.stack([wrapper[i]["physics"] for i in range(n)])
    unet_pred_all = unet_forward(model, X, P)

    step0 = step_of == 0
    out = dict(
        occ0=occ0[step0], occ1=occ1[step0], actions=actions[step0],
        s_px=s_px[step0], e_px=e_px[step0], slate_of=slate_of[step0],
        H=H, W=W, A=A, A_id=A_id, bmd=bmd, unet_pred=unet_pred_all[step0],
        ws_min=data_tr.workspace_min, ws_max=data_tr.workspace_max,
    )
    print(f"  {cell}: loaded {int(step0.sum())} step-0 transitions, "
          f"{out['slate_of'].unique().numel()} slates, grid {H}x{W}")
    return out


def build_predictions(c: dict):
    occ0, s_px, e_px = c["occ0"], c["s_px"], c["e_px"]
    H, W = c["H"], c["W"]
    preds = {
        "persistence": occ0,
        "mean-delta": predict_meandelta(c["bmd"], occ0, s_px, e_px, R, (H, W), CR),
        "linear": predict_world(c["A"], occ0, s_px, e_px, R, (H, W), CR),
        "warp-only": predict_world(c["A_id"], occ0, s_px, e_px, R, (H, W), CR),
        "UNet": c["unet_pred"],
    }
    for h in ("cumulative", "spread"):
        try:
            preds[f"heuristic-{h}"] = predict_heuristic(h, occ0, s_px, e_px)
        except Exception as exc:  # noqa: BLE001
            print(f"  heuristic {h} failed: {exc}")
    return preds


def d1_projection(cell: str, c: dict, preds: dict, dw: torch.Tensor):
    """Per model: swept-region error energy, the fraction that projects onto
    the mass-normalised, mean-removed, PER-TRANSITION cost direction, and the
    linearised-dV vs exact-dV agreement (whole image, no region mask -- V is
    a whole-image functional)."""
    H, W = c["H"], c["W"]
    occ0, occ1 = c["occ0"], c["occ1"]
    s_px, e_px = c["s_px"], c["e_px"]
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    n = occ0.shape[0]
    d_flat = dw.reshape(1, -1)  # (1, H*W)
    M0 = occ0.reshape(n, -1).sum(dim=1).clamp_min(1e-6)
    v0 = lyapunov(occ0, dw)
    v1_true = lyapunov(occ1, dw)

    out = {}
    for name, pred in preds.items():
        if name == "persistence":
            continue
        # -- energy decomposition, swept region only --
        e_full = (pred - occ1)
        e_reg = (e_full * region).reshape(n, -1)
        reg_flat = region.reshape(n, -1)
        npix = reg_flat.sum(dim=1).clamp_min(1.0)
        mean_d_reg = (d_flat * reg_flat).sum(dim=1) / npix
        d_hat_raw = (d_flat - mean_d_reg.unsqueeze(1)) * reg_flat
        d_hat_norm = d_hat_raw.norm(dim=1).clamp_min(1e-12)
        d_hat = d_hat_raw / d_hat_norm.unsqueeze(1)
        proj = (d_hat * e_reg).sum(dim=1)  # (n,)
        e_total_energy = float((e_reg ** 2).sum())
        e_proj_energy = float((proj ** 2).sum())
        frac = e_proj_energy / e_total_energy if e_total_energy > 0 else float("nan")
        # a degenerate transition (d constant over its whole swept region,
        # so d_hat_raw is exactly 0) contributes 0 to both sums -- correct,
        # not a division problem, since it is excluded from both numerator
        # and denominator identically.
        n_degenerate = int((d_hat_norm < 1e-9).sum())

        # -- linearisation check, WHOLE IMAGE (dV is a whole-image functional) --
        e_whole = e_full.reshape(n, -1)
        ddv_lin = (d_flat * e_whole).sum(dim=1) / M0
        ddv_exact = lyapunov(pred, dw) - v1_true  # = V(pred) - V(truth) = dv_pred - dv_true
        r_lin, p_lin = stats.pearsonr(ddv_lin.numpy(), ddv_exact.numpy())
        mad = float((ddv_lin - ddv_exact).abs().mean())
        sd_exact = float(ddv_exact.std())

        out[name] = dict(
            n=n, n_degenerate_transitions=n_degenerate,
            e_total_energy=e_total_energy, e_proj_energy=e_proj_energy,
            projected_fraction=frac,
            per_transition_frac_mean=float(
                ((proj ** 2) / (e_reg ** 2).sum(dim=1).clamp_min(1e-12)).mean()),
            linearisation_pearson_r=float(r_lin), linearisation_p=float(p_lin),
            linearisation_mean_abs_diff=mad,
            linearisation_mad_over_sd=mad / sd_exact if sd_exact > 0 else float("nan"),
        )
    print(f"\n=== D1 [{cell}]: swept-region error energy projected onto cost direction ===")
    print(f"{'model':16s} {'||e||^2':>12s} {'proj E':>12s} {'frac(pooled)':>13s} "
          f"{'frac(mean)':>11s} {'lin r':>7s} {'lin MAD/sd':>11s}")
    for name, v in out.items():
        print(f"{name:16s} {v['e_total_energy']:12.4f} {v['e_proj_energy']:12.6f} "
              f"{v['projected_fraction']:13.5f} {v['per_transition_frac_mean']:11.5f} "
              f"{v['linearisation_pearson_r']:7.4f} {v['linearisation_mad_over_sd']:11.4f}")
    return out


def d1_cache_cross_check(cell: str, c: dict, preds: dict, dw: torch.Tensor, cache):
    """Sanity check: for the three models the register's cache also holds
    (mean-delta, linear, UNet), does THIS script's independent reload/refit
    reproduce the cached dv_pred (goal=corner)? Same spirit as EXP-0028 part
    (b), extended to mean-delta and UNet, not just linear."""
    v0 = lyapunov(c["occ0"], dw)
    ep = cache["ep"]
    dv = cache["dv"]["corner"]
    print(f"\n=== D1 cross-check [{cell}]: independent reload vs cached dv_pred (goal=corner) ===")
    for name in ("mean-delta", "linear", "UNet"):
        dv_pred_indep = lyapunov(preds[name], dw) - v0
        # NOTE: this script's own eval-loading loop (build_dataset over the
        # eval cfg, filtered to step_of==0) is not guaranteed to enumerate
        # rows in the same order as expB_multistep_eval.py's cache-building
        # loop, so a row-for-row diff is not attempted here (unlike EXP-0028
        # part b, which controlled for this by indexing through
        # pool_common.load_occ0_for_slate row-by-row). This is a
        # distribution-level cross-check (mean/sd) only -- large
        # disagreement would still be conspicuous.
        cached = dv[name]
        print(f"  {name:12s} indep: mean={float(dv_pred_indep.mean()):+.5f} "
              f"sd={float(dv_pred_indep.std()):.5f}  |  cached: mean={float(cached.mean()):+.5f} "
              f"sd={float(cached.std()):.5f}")


def d2_geometry(cell: str, c: dict, preds: dict, dw: torch.Tensor, recon_cache: dict):
    v0 = lyapunov(c["occ0"], dw)
    plate_px = 0.04 / 0.128 * c["W"]
    region = swept_region_mask(c["s_px"], c["e_px"], (c["H"], c["W"]),
                               0.5 * plate_px + 2.0, 0.5 * plate_px)
    print(f"\n=== D2 [{cell}]: geometry-only heuristics, same metric suite ===")
    out = {}
    for name in [k for k in preds if k.startswith("heuristic-")]:
        acc = metrics(preds[name], c["occ1"], c["occ0"], region=region)["accuracy"]
        dv_h = lyapunov(preds[name], dw) - v0
        recon_cache["dv"]["corner"][name] = dv_h
        exact, _worst, _prof = sweep(recon_cache, "corner", [name], KS, min_slate=8)[0:3]
        row = {K: float(np.mean(exact[name][K])) if exact[name][K] else float("nan") for K in KS}
        out[name] = dict(accuracy=acc, slateK_exact=row)
        print(f"  {name:20s} accuracy={acc:+.4f}  " +
              "  ".join(f"K={K}:{row[K]:.4f}" for K in KS))
    return out


def d3_shuffle(cell: str, recon_cache: dict, models: list[str], seed: int = 0,
              n_shuffles: int = 200):
    """Permute dv_pred within each slate and recompute slateK_exact.

    A SINGLE permutation draw at K=n_slate is a single realisation of a
    random variable whose expectation is 0 (w_r(K=n) puts all its weight on
    the model's rank-1 pick, so permuting dv_pred re-labels which candidate
    is 'rank 1' uniformly at random; the expected true value of a uniformly
    random candidate is mean(t), giving slateK_exact -> 0 exactly under
    expectation) -- but with only ~20 slates, one draw per slate is a noisy
    estimate of that expectation, not the expectation itself. Averaging
    `n_shuffles` independent draws PER SLATE first, then averaging across
    slates (same order as every other metric in this register: per-slate
    values averaged, never a pooled ratio of sums), gives a low-variance
    estimate of the true null rather than a single noisy draw that could be
    mistaken for a leak."""
    print(f"\n=== D3 [{cell}]: shuffle null (permute dv_pred within slate, "
          f"{n_shuffles} draws/slate) ===")
    rng = np.random.default_rng(seed)
    ep = recon_cache["ep"]
    dv = recon_cache["dv"]["corner"]
    dv_true_all = dv["dv_true"]
    out = {}
    for m in models:
        real4, real128, shuf4, shuf128, shuf4_sd, shuf128_sd = [], [], [], [], [], []
        for e in ep.unique().tolist():
            sel = (ep == e).nonzero(as_tuple=True)[0]
            if sel.numel() < 8:
                continue
            t = dv_true_all[sel]
            if float(t.std()) < 1e-9:
                continue
            p = t.clone() if m == "oracle" else dv[m][sel]
            ex_real, _w, _pr = per_slate_exact(p, t, [4, 128])
            if 4 not in ex_real:
                continue
            real4.append(ex_real.get(4, float("nan")))
            real128.append(ex_real.get(128, float("nan")))
            p_np = p.numpy().copy()
            draws4, draws128 = [], []
            for _ in range(n_shuffles):
                perm = torch.from_numpy(rng.permutation(p_np))
                ex_shuf, _w2, _pr2 = per_slate_exact(perm, t, [4, 128])
                draws4.append(ex_shuf.get(4, float("nan")))
                draws128.append(ex_shuf.get(128, float("nan")))
            shuf4.append(float(np.nanmean(draws4)))
            shuf128.append(float(np.nanmean(draws128)))
            shuf4_sd.append(float(np.nanstd(draws4)))
            shuf128_sd.append(float(np.nanstd(draws128)))
        real4, real128 = np.array(real4), np.array(real128)
        shuf4, shuf128 = np.array(shuf4), np.array(shuf128)
        n_sl = len(real4)
        # sem of the ACROSS-SLATE mean of the (already within-slate-averaged)
        # shuffled capture -- this is the uncertainty on the number reported,
        # not the single-draw spread.
        sem4 = float(np.nanstd(shuf4, ddof=1) / np.sqrt(n_sl)) if n_sl > 1 else float("nan")
        sem128 = float(np.nanstd(shuf128, ddof=1) / np.sqrt(n_sl)) if n_sl > 1 else float("nan")
        out[m] = dict(real_K4=float(np.nanmean(real4)), shuffled_K4=float(np.nanmean(shuf4)),
                      shuffled_K4_sem=sem4,
                      real_K128=float(np.nanmean(real128)), shuffled_K128=float(np.nanmean(shuf128)),
                      shuffled_K128_sem=sem128, n_slates=n_sl)
        print(f"  {m:20s} K=4  real={out[m]['real_K4']:+.4f} "
              f"shuffled={out[m]['shuffled_K4']:+.4f} (sem {sem4:.4f})   "
              f"K=128 real={out[m]['real_K128']:+.4f} "
              f"shuffled={out[m]['shuffled_K128']:+.4f} (sem {sem128:.4f})")
    return out


def d4_paired_and_pools(cell: str, recon_cache: dict):
    print(f"\n=== D4 [{cell}]: UNet vs linear paired test, and per-pool detail ===")
    ep = recon_cache["ep"]
    dv = recon_cache["dv"]["corner"]
    exact, _worst, _profile, n_slates = sweep(recon_cache, "corner",
                                              ["linear", "UNet", "oracle"], KS, min_slate=8)
    out = {"n_slates": n_slates}
    for K in (4, 128):
        st = paired(exact["UNet"][K], exact["linear"][K])
        out[f"paired_UNet_minus_linear_K{K}"] = st
        print(f"  K={K:4d}  UNet-linear mean={st['mean']:+.4f} sem={st['sem']:.4f} "
              f"t={st['t']:+.2f} wins={st['wins']}/{st['n']} losses={st['losses']} ties={st['ties']}")

    # per-pool detail: dv_true spread and per-slate capture at K=128
    spreads, cap_lin, cap_unet = [], [], []
    slates_kept = []
    for e in ep.unique().tolist():
        sel = (ep == e).nonzero(as_tuple=True)[0]
        if sel.numel() < 8:
            continue
        t = dv["dv_true"][sel]
        if float(t.std()) < 1e-9:
            continue
        slates_kept.append(int(e))
        spreads.append(float(t.max() - t.min()))
        exl, _w, _p = per_slate_exact(dv["linear"][sel], t, [128])
        exu, _w, _p = per_slate_exact(dv["UNet"][sel], t, [128])
        cap_lin.append(exl.get(128, float("nan")))
        cap_unet.append(exu.get(128, float("nan")))
    spreads = np.array(spreads); cap_lin = np.array(cap_lin); cap_unet = np.array(cap_unet)
    out["per_pool"] = dict(
        slate_id=slates_kept, dv_true_spread=spreads.tolist(),
        slateK_exact_linear_K128=cap_lin.tolist(), slateK_exact_UNet_K128=cap_unet.tolist(),
    )
    print(f"  dv_true spread (max-min) across {len(spreads)} pools: "
          f"mean={spreads.mean():.6f} median={np.median(spreads):.6f} "
          f"min={spreads.min():.6f} max={spreads.max():.6f} sd={spreads.std(ddof=1):.6f}")
    print(f"  per-pool K=128 capture -- linear: mean={np.nanmean(cap_lin):.4f} "
          f"sd={np.nanstd(cap_lin, ddof=1):.4f} min={np.nanmin(cap_lin):.4f} max={np.nanmax(cap_lin):.4f}")
    print(f"  per-pool K=128 capture -- UNet:   mean={np.nanmean(cap_unet):.4f} "
          f"sd={np.nanstd(cap_unet, ddof=1):.4f} min={np.nanmin(cap_unet):.4f} max={np.nanmax(cap_unet):.4f}")
    print("  per-pool table (slate, spread, linear_K128, UNet_K128):")
    for sid, sp, cl, cu in zip(slates_kept, spreads, cap_lin, cap_unet):
        print(f"    {sid:4d}  {sp:.6f}  {cl:+.4f}  {cu:+.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs_expB/exp0029_results.json")
    ap.add_argument("--goal", default="corner")
    args = ap.parse_args()

    prov = git_provenance()
    print(f"=== EXP-0029: L10mm mechanism diagnostics ===\nprovenance={prov}")

    results = {"provenance": prov, "cells": {}}
    for cell in CELLS:
        print(f"\n########## {cell} ##########")
        c = load_cell(cell)
        H, W = c["H"], c["W"]
        dw = lyapunov_weights((H, W), args.goal, "cpu")
        preds = build_predictions(c)
        cache = torch.load(f"runs_expB/n20_{cell}_dv_cache.pt", map_location="cpu",
                           weights_only=False)

        d1 = d1_projection(cell, c, preds, dw)
        d1_cache_cross_check(cell, c, preds, dw, cache)

        # recon_cache: self-built, same shape cache expects, for sweep()/D2/D3/D4
        v0 = lyapunov(c["occ0"], dw)
        dv_true = lyapunov(c["occ1"], dw) - v0
        recon_cache = {"ep": c["slate_of"], "dv": {"corner": {"dv_true": dv_true}}}
        for name, pred in preds.items():
            recon_cache["dv"]["corner"][name] = lyapunov(pred, dw) - v0
        recon_cache["dv"]["corner"]["oracle"] = dv_true.clone()

        d2 = d2_geometry(cell, c, preds, dw, recon_cache)
        d3 = d3_shuffle(cell, recon_cache, ["mean-delta", "linear", "UNet", "warp-only",
                                           "heuristic-cumulative", "oracle"])
        d4 = d4_paired_and_pools(cell, recon_cache) if cell == "L10mm" else None

        # image accuracy, all models, step-0 subset (for reference beside
        # the register's all-3-step numbers)
        plate_px = 0.04 / 0.128 * W
        region = swept_region_mask(c["s_px"], c["e_px"], (H, W),
                                   0.5 * plate_px + 2.0, 0.5 * plate_px)
        acc = {name: metrics(pred, c["occ1"], c["occ0"], region=region)["accuracy"]
               for name, pred in preds.items()}
        print(f"\n=== step-0-only image accuracy [{cell}] (reference; register's "
              f"headline uses all 3 steps) ===")
        for name, a in acc.items():
            print(f"  {name:20s} accuracy={a:+.4f}")

        results["cells"][cell] = dict(
            accuracy_step0=acc, d1=d1, d2=d2, d3=d3, d4=d4,
            dv_true_sd=float(dv_true.std()),
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
