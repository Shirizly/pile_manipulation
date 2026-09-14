"""Evaluate stage-2's stage2-family operators (experiments/temp/hybrid-vis-desc)
as ACTION-RANKING models across the FULL slates_multistep dataset family,
reporting slateN per experiments/METRICS.md.

Extends the original single-cell/single-dataset run (see RESULTS.md's first
section, "Run 1") to:
  - 3 more cells: hybrid14 (switched, +14-dim descriptors), hybrid94
    (switched, +94-dim descriptors), matching the accuracy-tie question
    (cells 1-3 in the task: switched-visual / +14desc / +94desc were
    accuracy-indistinguishable, 0.1697/0.1712/0.1713).
  - all 3 available slates_multistep dataset cells with a prepared eval
    split: n20_L10mm, n20_L20mm, n20_L40mm (n50_L20mm has data but NO
    genesis_slates_multistep_n50_L20mm_eval.yaml config / manifest split
    was ever prepared -- confirmed by listing configs/dataset/ -- so it is
    excluded and reported as such, not silently dropped).
  - all 3 base goals (center, corner, stripe) per dataset, each checked for
    degeneracy (fraction of dv_true==0 across all models' candidates) before
    being scored; a goal is EXCLUDED per-dataset if >=90% of its dv_true are
    exactly 0 (the same failure mode Run 1 found for 'center' on L20mm: "do
    nothing" trivially optimal, every model ties at 1.0 by construction).
  - saves the refit operators to disk this time (.pt, all bins + global,
    every cell), the previous run's documented gap.

Cells scored (same convention as Run 1): dv_pred/dv_true both Lyapunov,
lower=better; persistence dv_pred=0; random dv_pred=iid noise seed 0.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")

from control_utility_test import lyapunov, lyapunov_weights  # noqa: E402
from fit_linear_foresight import actions_to_pixels, canonicalise  # noqa: E402
from descriptors_d import push_frame_full_descriptors  # noqa: E402
from Baselines.common.data import load_cell  # noqa: E402
from Baselines.common.goals import slate_n_capture  # noqa: E402

from fit_hybrid import (  # noqa: E402  -- stage-2's OWN fit/predict path, reused unchanged
    CELLS, bin_index, fit_cell, predict_image,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAM = 1.0  # stage-2's reported-best lam
TRAIN_CACHE = ("/home/alon/Code/pile_manipulation/experiments/temp/"
               "hybrid-vis-desc/cache/train_cache.pt")
OPDIR = "/home/alon/Code/pile_manipulation/experiments/temp/stage2-slaten/operators"

DATASETS = {
    "n20_L10mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L10mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L10mm/manifest.json"),
    "n20_L20mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json"),
    "n20_L40mm": dict(
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json"),
    # n50_L20mm: NO eval-split config/manifest was ever prepared (checked
    # configs/dataset/*slates_multistep* -- only n20_L10mm/L20mm/L40mm and
    # the pooled n20_L20L40_train exist). Data exists but is unusable here
    # without building a new split, which is out of scope for this budget.
}

GOALS = ["center", "corner", "stripe"]
DEGENERACY_THRESH = 0.90  # exclude a (dataset, goal) if >=90% dv_true == 0

MODEL_CELLS = ["visual", "hybrid14", "hybrid94"]  # switched, refit + scored
K_FIXED = 32
N_RESAMPLE = 300


def fit_all_operators():
    """Refit stage-2's visual/hybrid14/hybrid94 cells (switched bin-ops +
    single global op) on the cached TRAIN data, and PERSIST every fitted
    operator to disk (the gap Run 1 left open)."""
    train = torch.load(TRAIN_CACHE, map_location="cpu")
    N = train["canon0"].shape[0]
    train["canon0f"] = train["canon0"].reshape(N, -1)
    train["canon1f"] = train["canon1"].reshape(N, -1)
    hi = float(train["length_m"].max())
    bin_edges = torch.linspace(0.0, hi, 6 + 1)

    fitted = {}
    import os
    os.makedirs(OPDIR, exist_ok=True)
    for cell_name in MODEL_CELLS:
        cell = CELLS[cell_name]
        dd = cell["desc_dim"]
        X0_tr = torch.cat([train["canon0f"], train["desc0"][:, :dd]], dim=1) if dd else train["canon0f"]
        X1_tr = torch.cat([train["canon1f"], train["desc1"][:, :dd]], dim=1) if dd else train["canon1f"]
        ops, A_single = fit_cell(cell, X0_tr, X1_tr, train["length_m"], bin_edges, LAM)
        fitted[cell_name] = dict(cell=cell, ops=ops, A_single=A_single, bin_edges=bin_edges)
        path = f"{OPDIR}/{cell_name}_lam{LAM}.pt"
        torch.save({
            "cell_name": cell_name, "cell_spec": cell, "lam": LAM,
            "bin_edges": bin_edges, "switched_ops": ops, "global_op": A_single,
            "note": (f"Ridge toward-identity operators for '{cell_name}' "
                     f"(desc_dim={dd}), fit on hybrid-vis-desc/cache/train_cache.pt, "
                     f"lam={LAM}, 6 EXP-0003-scheme push-length bins over "
                     f"[0, {hi:.4f}] m. switched_ops[b] is per-bin (D,D); "
                     f"global_op is the single unswitched (D,D) operator. "
                     f"D = 1024 (+ desc_dim). Recipe: fit_hybrid.py::fit_cell, "
                     f"toward_identity=True."),
        }, path)
        print(f"  saved {path}")
    return fitted, bin_edges


def build_predict_cache(cell_slates, ws_min, ws_max):
    """Build the dict fit_hybrid.predict_image expects, PLUS 94-dim
    descriptors computed with build_data.py's own convention
    (actions_to_pixels for both visual canon and descriptor push-frame,
    push_frame_full_descriptors for the local block) so hybrid14/hybrid94
    cells can be scored on real slates, not just the visual cell."""
    occ0 = cell_slates.occ0.float()
    H, W = occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(cell_slates.actions, ws_min, ws_max, (H, W))
    RES = 32
    canon0 = canonicalise(occ0.to(DEVICE), s_px.to(DEVICE), e_px.to(DEVICE), RES, 1.0).cpu()
    N = occ0.shape[0]
    length_m = (cell_slates.p_stop[:, :2] - cell_slates.p_start[:, :2]).norm(dim=-1)

    gm0 = occ0.sum(dim=(-2, -1)) / (H * W)
    local0 = push_frame_full_descriptors(occ0.to(DEVICE), s_px.to(DEVICE), e_px.to(DEVICE)).cpu()
    desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)  # [N,94]

    return dict(occ0=occ0, canon0f=canon0.reshape(N, -1), start_px=s_px, end_px=e_px,
                length_m=length_m, desc0=desc0)


def paired_sem(diffs: np.ndarray) -> float:
    if len(diffs) < 2:
        return float("nan")
    return float(diffs.std(ddof=1) / np.sqrt(len(diffs)))


def per_slate_stats(value_pred_by_model, value_true, slate_ids):
    out = {name: {"capture": [], "chosen": [], "n": []} for name in value_pred_by_model}
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        vt = value_true[rows]
        for name, vp_all in value_pred_by_model.items():
            vp = vp_all[rows]
            cap = slate_n_capture(vp, vt, higher_is_better=False)
            chosen_idx = int(torch.argmin(vp))
            out[name]["capture"].append(cap)
            out[name]["chosen"].append(float(vt[chosen_idx]))
            out[name]["n"].append(int(rows.numel()))
    return out


def fixed_k_capture(value_pred, value_true, slate_ids, K, n_resample, seed=0):
    g = torch.Generator().manual_seed(seed)
    per_slate_means = []
    for sid in slate_ids.unique().tolist():
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        n = rows.numel()
        if n < K:
            continue
        vt, vp = value_true[rows], value_pred[rows]
        caps = []
        for _ in range(n_resample):
            idx = torch.randperm(n, generator=g)[:K]
            sub_t, sub_p = vt[idx], vp[idx]
            chosen = float(sub_t[int(torch.argmin(sub_p))])
            best = float(sub_t.min())
            mean = float(sub_t.mean())
            denom = mean - best
            if abs(denom) < 1e-9:
                continue
            caps.append((mean - chosen) / denom)
        if caps:
            per_slate_means.append(float(np.mean(caps)))
    return per_slate_means


def wins_losses_ties(chosen_a, chosen_b, tol=1e-9):
    w = l = t = 0
    diffs = []
    for a, b in zip(chosen_a, chosen_b):
        diffs.append(a - b)
        if abs(a - b) <= tol:
            t += 1
        elif a < b:
            w += 1
        else:
            l += 1
    return w, l, t, np.array(diffs)


def score_pool(dv_pred, dv_true, slate_ids, tag):
    stats = per_slate_stats(dv_pred, dv_true, slate_ids)
    cells_out = {}
    for name in dv_pred:
        caps = np.array([c for c in stats[name]["capture"] if c == c])
        cells_out[name] = {
            "slateN_mean": float(caps.mean()) if len(caps) else float("nan"),
            "slateN_sem": paired_sem(caps),
            "n_slates": len(caps),
        }
        k32 = fixed_k_capture(dv_pred[name], dv_true, slate_ids, K_FIXED, N_RESAMPLE)
        cells_out[name]["slate32_mean"] = float(np.mean(k32)) if k32 else float("nan")
        cells_out[name]["slate32_sem"] = paired_sem(np.array(k32))
    pairs = [("visual", "hybrid14"), ("visual", "hybrid94"), ("hybrid14", "hybrid94"),
             ("visual", "global"), ("visual", "persistence"), ("visual", "random"),
             ("global", "persistence")]
    h2h = {}
    for a, b in pairs:
        w, lo, t, diffs = wins_losses_ties(stats[a]["chosen"], stats[b]["chosen"])
        cap_a, cap_b = np.array(stats[a]["capture"]), np.array(stats[b]["capture"])
        cap_diff = cap_a - cap_b
        h2h[f"{a}_vs_{b}"] = {
            "wins": w, "losses": lo, "ties": t, "n_slates": len(stats[a]["chosen"]),
            "paired_sem_dv": paired_sem(diffs), "mean_dv_diff": float(diffs.mean()),
            "paired_sem_capture": paired_sem(cap_diff),
            "mean_capture_diff": float(cap_diff.mean()) if len(cap_diff) else float("nan"),
        }
    return {"tag": tag, "cells": cells_out, "head_to_head": h2h, "stats": stats}


def main():
    t0 = time.time()
    print("[1/5] refitting + SAVING stage-2 operators (visual, hybrid14, hybrid94)...")
    fitted, bin_edges = fit_all_operators()

    results = {"lam": LAM, "goals_checked": GOALS, "degeneracy_thresh": DEGENERACY_THRESH,
               "datasets": {}, "excluded_datasets": {
                   "n50_L20mm": "no genesis_slates_multistep_n50_L20mm_eval.yaml / "
                                "manifest eval-split was ever prepared (checked "
                                "configs/dataset/*slates_multistep*); data exists "
                                "under Genesis/data/slates_multistep/n50_L20mm(_b) "
                                "but building a fresh split is out of budget here."},
               }

    # pooled accumulators, per goal, across non-degenerate datasets
    pooled_dv_pred = {g: {name: [] for name in ["visual", "hybrid14", "hybrid94", "global", "persistence", "random"]}
                       for g in GOALS}
    pooled_dv_true = {g: [] for g in GOALS}
    pooled_slate_ids = {g: [] for g in GOALS}
    pooled_offset = {g: 0 for g in GOALS}
    excluded_goal_dataset = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n[2/5] dataset {ds_name}: loading {ds_cfg['eval_cfg']}")
        slates = load_cell(ds_cfg["eval_cfg"], "train", manifest_path=ds_cfg["manifest"],
                            tag=f"{ds_name}_slaten")
        step0 = slates.step_idx == 0
        slate_ids = slates.slate_idx[step0]
        n_slates = slate_ids.unique().numel()
        n_s0 = int(step0.sum())
        print(f"  {slates.occ0.shape[0]} total transitions, step0 n={n_s0}, "
              f"{n_slates} slates ({n_s0 // max(n_slates,1)} candidates/slate)")

        ws_min, ws_max = slates.workspace_min, slates.workspace_max
        pc = build_predict_cache(slates, torch.tensor(ws_min), torch.tensor(ws_max))
        pc_s0 = {k: v[step0] for k, v in pc.items()}

        preds = {}
        for cell_name in MODEL_CELLS:
            f = fitted[cell_name]
            preds[cell_name] = predict_image(f["cell"], f["ops"], True, pc_s0, f["bin_edges"])
        # unswitched global reference is the plain 'visual' cell's global op
        preds["global"] = predict_image(fitted["visual"]["cell"], fitted["visual"]["A_single"],
                                         False, pc_s0, fitted["visual"]["bin_edges"])

        occ0_s0 = slates.occ0[step0].float()
        occ1_s0 = slates.occ1[step0].float()

        results["datasets"][ds_name] = {"n_slates": n_slates, "candidates_per_slate":
                                         n_s0 // max(n_slates, 1), "goals": {}}

        for goal in GOALS:
            dw = lyapunov_weights((64, 64), goal, DEVICE)
            v0 = lyapunov(occ0_s0.to(DEVICE), dw)
            v1_true = lyapunov(occ1_s0.to(DEVICE), dw)
            dv_true = (v1_true - v0).cpu()

            dv_pred = {}
            for name, pred in preds.items():
                dv_pred[name] = (lyapunov(pred.to(DEVICE), dw) - v0).cpu()
            dv_pred["persistence"] = torch.zeros_like(dv_true)
            g_rand = torch.Generator().manual_seed(0)
            dv_pred["random"] = torch.rand(dv_true.shape[0], generator=g_rand) * 2 - 1

            frac_zero = float((dv_true.abs() < 1e-9).float().mean())
            degenerate = frac_zero >= DEGENERACY_THRESH
            print(f"  goal={goal}: frac(dv_true==0)={frac_zero:.3f} "
                  f"{'EXCLUDED (degenerate)' if degenerate else 'OK'}")

            if degenerate:
                excluded_goal_dataset.append((ds_name, goal, frac_zero))
                results["datasets"][ds_name]["goals"][goal] = {
                    "excluded": True, "frac_dv_true_zero": frac_zero}
                continue

            scored = score_pool(dv_pred, dv_true, slate_ids, tag=f"{ds_name}/{goal}")
            scored["frac_dv_true_zero"] = frac_zero
            scored.pop("stats")
            results["datasets"][ds_name]["goals"][goal] = scored

            # accumulate for pooling (offset slate_ids so pools across
            # datasets don't collide on raw slate index)
            off = pooled_offset[goal]
            pooled_slate_ids[goal].append(slate_ids + off)
            pooled_offset[goal] = off + int(slate_ids.max()) + 1000
            pooled_dv_true[goal].append(dv_true)
            for name in dv_pred:
                pooled_dv_pred[goal][name].append(dv_pred[name])

    print("\n[3/5] pooling across datasets, per goal...")
    results["pooled"] = {}
    for goal in GOALS:
        if not pooled_dv_true[goal]:
            results["pooled"][goal] = {"excluded_everywhere": True}
            continue
        dv_true_p = torch.cat(pooled_dv_true[goal])
        slate_ids_p = torch.cat(pooled_slate_ids[goal])
        dv_pred_p = {name: torch.cat(lst) for name, lst in pooled_dv_pred[goal].items()}
        scored = score_pool(dv_pred_p, dv_true_p, slate_ids_p, tag=f"pooled/{goal}")
        scored.pop("stats")
        results["pooled"][goal] = scored
        print(f"  pooled/{goal}: n_slates={scored['cells']['visual']['n_slates']}  "
              f"visual slateN={scored['cells']['visual']['slateN_mean']:+.3f}")

    results["excluded_goal_dataset"] = [
        {"dataset": d, "goal": g, "frac_dv_true_zero": f} for d, g, f in excluded_goal_dataset]

    out_path = "/home/alon/Code/pile_manipulation/experiments/temp/stage2-slaten/results_slaten_stage2family.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[4/5] wrote {out_path}")
    print(f"[5/5] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
