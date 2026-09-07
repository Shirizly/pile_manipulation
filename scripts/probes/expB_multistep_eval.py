"""EXP-B: image accuracy + control utility on a slates_multistep cell.

Reuses EXP-0024/EXP-0026's pipeline (fit_linear_foresight, dmdc_baseline,
control_utility_test, exp0021_eval's UNet loader, exp0026_selection_pressure's
degradation-arm recipes, exp0026_kcurve / exp0026_kcurve_exact for the K-sweep)
unchanged -- the only new code here is loading the slate-level 30/20 split
(scripts/probes/prepare_slate_multistep_split.py) and the step-0-only
filtering that same-state candidate slates require in a MULTI-STEP collection
(steps 1-2 are each env's own diverged rollout, not a slate -- see
docs/plan_selection_pressure_validation.md, "Control ranking uses STEP-0
CANDIDATES ONLY").

What this does
--------------
1. Fits the linear (ridge->identity, res=64/crop=1.0) and mean-delta
   operators on the TRAIN split (30 slates x 3 steps, ~11.5k transitions).
2. Loads the UNet checkpoint (trained separately by `training.train` on the
   identical train split).
3. Scores IMAGE ACCURACY on the full EVAL split (20 slates x 3 steps, 7680
   transitions, swept-region mask) -- all three steps, per the design.
4. Scores CONTROL UTILITY restricted to STEP 0 of the eval split ONLY (20
   slates x 128 candidates = 2560 transitions, each slate a genuine
   same-state group) -- writes a dv cache in exactly the format
   `exp0026_kcurve.py` / `exp0026_kcurve_exact.py` already expect, so both are
   reused UNCHANGED for the K-sweep (K up to 128, the full slate).
5. Optionally rebuilds EXP-0025/EXP-0026's degradation arms
   (`exp0026_selection_pressure._degradation_arms`, imported not
   reimplemented) on the step-0 linear predictions.

Usage
-----
    PYTHONPATH=. python scripts/probes/expB_multistep_eval.py \
        configs/dataset/genesis_slates_multistep_n20_L20mm_train.yaml \
        configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \
        Genesis/data/slates_multistep/n20_L20mm/manifest.json \
        runs_expB/unetfilm_slates_multistep_n20_L20mm \
        --tag n20_L20mm --degradations \
        --out-prefix runs_expB/n20_L20mm
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask,
)
from Genesis.training.dataset import PileSweepData
from pathlib import Path
from registry.dataset_registry import build_dataset
from scripts.probes.exp0009_rerun import predict_meandelta
from scripts.probes.exp0021_eval import load_unet, unet_forward
from scripts.probes.exp0026_selection_pressure import _degradation_arms
from control_utility_test import lyapunov, lyapunov_weights
from utils import git_provenance

R, CR, RIDGE = 64, 1.0, 1.0  # EXP-0024/EXP-0026's fit, unchanged


def _run_files(eval_cfg_path: str, split: str):
    """Filenames of the runs `build_dataset` will load, IN THE SAME ORDER,
    so `get_run_index()` can be mapped back to (slate_idx, step_idx) via the
    source cell's manifest.json. Same approach as
    exp0026_selection_pressure.py's `_slate_files`."""
    scfg = yaml.safe_load(open(eval_cfg_path).read())
    root = Path(__file__).resolve().parent.parent.parent / "Genesis" / "data"
    files = []
    for path in scfg["paths"]:
        full = root / path
        runs = PileSweepData._collect_run_paths(PileSweepData, full)
        runs = PileSweepData._filter_split(runs, split, scfg.get("val_pct", 0),
                                           scfg.get("test_pct", 0))
        files.extend(str(d.name) for d, _ in runs)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_cfg")
    ap.add_argument("eval_cfg")
    ap.add_argument("manifest", help="source cell's manifest.json (batch_idx -> slate/step)")
    ap.add_argument("run_dir")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--degradations", action="store_true")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    prov = git_provenance()
    print(f"=== EXP-B multistep eval: {args.tag} ===\nprovenance: {prov}")

    # ---- fit on TRAIN split (30 slates x 3 steps) --------------------------
    data_tr = load_transition_arrays(args.train_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ_t.shape[0]
    print(f"train: {n_tr} transitions, grid {H}x{W}, "
          f"{data_tr.episode_ids.unique().numel()} batches")

    Y0 = canonicalise(data_tr.occ_t, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ_t1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)

    model, run_cfg, missing = load_unet(args.run_dir)
    print(f"UNet load: {missing}")

    # ---- load EVAL split (20 slates x 3 steps = 60 batches) ----------------
    ecfg = yaml.safe_load(open(args.eval_cfg).read())
    wrapper = build_dataset(ecfg, "train")   # val_pct=0/test_pct=0 -> everything
    raw = wrapper.raw_dataset
    n = len(wrapper)
    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    run_idx = torch.tensor([raw.get_run_index(i) for i in range(n)])
    s_px, e_px = actions_to_pixels(actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    print(f"eval: {n} transitions, {run_idx.unique().numel()} batches")

    files = _run_files(args.eval_cfg, "train")
    manifest = json.loads(open(args.manifest).read())
    batch_lookup = {b["batch_idx"]: (b["slate_idx"], b["step_idx"]) for b in manifest["batches"]}
    run_to_batch = {}
    for r, fname in enumerate(files):
        bidx = int(fname.split("_")[1])  # "_<batch_idx>_data.pt"
        run_to_batch[r] = bidx
    slate_of = torch.tensor([batch_lookup[run_to_batch[int(r)]][0] for r in run_idx.tolist()])
    step_of = torch.tensor([batch_lookup[run_to_batch[int(r)]][1] for r in run_idx.tolist()])
    assert set(step_of.tolist()) == {0, 1, 2}, "expected exactly 3 steps in eval split"
    print(f"  steps present: {sorted(set(step_of.tolist()))}, "
          f"step0 transitions: {int((step_of == 0).sum())}")

    X = torch.stack([wrapper[i]["input"] for i in range(n)])
    P = torch.stack([wrapper[i]["physics"] for i in range(n)])
    unet_pred = unet_forward(model, X, P)

    preds = {
        "persistence": occ0,
        "mean-delta": predict_meandelta(bmd, occ0, s_px, e_px, R, (H, W), CR),
        "linear": predict_world(A, occ0, s_px, e_px, R, (H, W), CR),
        "UNet": unet_pred,
    }

    # =========================================================================
    # (a) IMAGE ACCURACY -- all 3 steps, swept-region mask
    # =========================================================================
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    img_preds = dict(preds)
    img_preds["oracle"] = occ1
    accuracy = {}
    accuracy_by_step = {0: {}, 1: {}, 2: {}}
    for name, pred in img_preds.items():
        m = metrics(pred, occ1, occ0, region=region)
        accuracy[name] = m["accuracy"]
        for st in (0, 1, 2):
            sel = step_of == st
            m_st = metrics(pred[sel], occ1[sel], occ0[sel], region=region[sel])
            accuracy_by_step[st][name] = m_st["accuracy"]
    print("\n-- image accuracy (held-out eval slates, all 3 steps, n=%d) --" % n)
    for name, acc in accuracy.items():
        print(f"  {name:20s} accuracy={acc:+.4f}  "
              f"(step0={accuracy_by_step[0][name]:+.4f} "
              f"step1={accuracy_by_step[1][name]:+.4f} "
              f"step2={accuracy_by_step[2][name]:+.4f})")

    # =========================================================================
    # (b) CONTROL UTILITY -- STEP 0 CANDIDATES ONLY
    # =========================================================================
    step0 = step_of == 0
    occ0_s0, occ1_s0 = occ0[step0], occ1[step0]
    s_px_s0, e_px_s0 = s_px[step0], e_px[step0]
    slate_s0 = slate_of[step0]
    preds_s0 = {k: v[step0] for k, v in preds.items()}
    n_s0 = int(step0.sum())
    n_slates_s0 = slate_s0.unique().numel()
    print(f"\nstep-0 (control-ranking) subset: {n_s0} transitions, "
          f"{n_slates_s0} slates ({n_s0 // max(n_slates_s0,1)} candidates/slate)")

    arms = {}
    if args.degradations:
        arms = _degradation_arms(preds_s0["linear"], occ0_s0, occ1_s0,
                                  s_px_s0, e_px_s0, (H, W))
        print(f"degradation arms: {len(arms)} added")

    cache = {"ep": slate_s0, "actions": actions[step0], "provenance": prov,
             "slate_files": [files[int(r)] for r in run_idx[step0].tolist()],
             "split": "eval_step0",
             "config": {"train_cfg": args.train_cfg, "eval_cfg": args.eval_cfg,
                        "run_dir": args.run_dir, "R": R, "crop": CR, "ridge": RIDGE},
             "dv": {}}
    for goal in ("corner", "center"):
        dw = lyapunov_weights((H, W), goal, "cpu")
        v0 = lyapunov(occ0_s0, dw)
        dv_true = lyapunov(occ1_s0, dw) - v0
        g = {"dv_true": dv_true, "v0": v0}
        for name, pred in preds_s0.items():
            g[name] = lyapunov(pred, dw) - v0
        for name, field in arms.items():
            g[name] = lyapunov(field, dw) - v0
        cache["dv"][goal] = g
        print(f"goal={goal}: dv_true mean {float(dv_true.mean()):+.5f} "
              f"sd {float(dv_true.std()):.5f} "
              f"helpful {100 * float((dv_true < 0).float().mean()):.0f}%")

    cache_path = f"{args.out_prefix}_dv_cache.pt"
    torch.save(cache, cache_path)
    print(f"wrote {cache_path}")

    acc_path = f"{args.out_prefix}_accuracy.json"
    with open(acc_path, "w") as f:
        json.dump({"tag": args.tag, "provenance": prov, "n_eval": n,
                   "accuracy": accuracy, "accuracy_by_step": accuracy_by_step,
                   "n_slates_step0": n_slates_s0, "n_step0": n_s0}, f, indent=2)
    print(f"wrote {acc_path}")


if __name__ == "__main__":
    main()
