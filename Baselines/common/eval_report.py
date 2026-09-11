"""Baselines/common/eval_report.py -- cross-corpus, multi-metric report.

Scores every model in `MODELS` against every corpus in `CORPORA`:

  * accuracy (`fit_linear_foresight.metrics`'s `accuracy` key, swept-region
    masked) -- for a GNN model, BOTH sides of the comparison (prediction
    AND ground truth) are passed through the SAME node-count bottleneck
    (`Baselines/GNN/perception.py::resample_occupancy_through_nodes`) so
    the score isolates dynamics-prediction error from the representational
    loss of using only `n_particles` nodes to describe an arbitrary pile
    -- not because it's unfair to the GNN otherwise, but because comparing
    a 20/30-node prediction against a full-fidelity many-cube ground truth
    would conflate "wrong dynamics" with "fewer nodes than cubes", two
    different things. NFD (a full-resolution grid model, no node-count
    bottleneck) is compared directly against raw ground truth.

  * `slateN`/"capture" (docs/experiments/METRICS.md) on step-0 same-state
    pools, for 3 goal shapes (`Baselines/common/goals.py`: a per-slate
    random quadrant, a centred 'O' ring, a centred 'T') x 3 value
    functions (Lyapunov distance-to-goal, mass-in-region, signed
    mass-in-region), reported per (goal, value function) and averaged
    over goals per value function.

Usage:
    PYTHONPATH=. python Baselines/common/eval_report.py \\
        --out-prefix Baselines/common/runs/cross_corpus_report
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import numpy as np
import torch

from control_utility_test import lyapunov
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask

from Baselines.common.data import load_cell
from Baselines.common.eval_baseline import _predictor_batch
from Baselines.common.goals import (
    dist_field_from_mask, letter_mask, mass_in_region, random_quadrant_mask,
    signed_mass_in_region, slate_n_capture,
)
from Baselines.common.randlen_data import load_randlen_cell
from Baselines.GNN.perception import resample_occupancy_through_nodes

MODELS = {
    "gnn_l20l40": dict(
        module="Baselines.GNN.predictor", factory="build_predictor",
        ckpt_env="GNN_CKPT", ckpt="Baselines/GNN/runs/ckpt_best.pth", is_gnn=True,
    ),
    "gnn_randlen_n30": dict(
        module="Baselines.GNN.predictor", factory="build_predictor",
        ckpt_env="GNN_CKPT", ckpt="Baselines/GNN/runs/randlen_train_all_n30/ckpt_best.pth",
        is_gnn=True,
    ),
    "nfd_randlen": dict(
        module="Baselines.NFD.predictor", factory="build_predictor",
        ckpt_env="NFD_CKPT", ckpt="Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth",
        is_gnn=False,
    ),
    # Baselines/LinearForesight (EXP-0003): switched (per-push-length-bin)
    # vs. single global pixel operator, at the repo's own res=64 convention
    # and at the paper's own res=32, all four fit on the SAME
    # overnight_randlen train_all corpus in the same `fit_switched.py` run
    # per resolution (see that module's docstring for why single/switched
    # are bundled together rather than fit separately).
    "linear_switched_res64": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res64.pt",
        is_gnn=False,
    ),
    "linear_single_res64": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor_single",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res64.pt",
        is_gnn=False,
    ),
    "linear_switched_res32": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res32.pt",
        is_gnn=False,
    ),
    "linear_single_res32": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor_single",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res32.pt",
        is_gnn=False,
    ),
}

CORPORA = {
    "L20mm": dict(
        kind="slate",
        train_cfg="configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
    ),
    "L40mm": dict(
        kind="slate",
        train_cfg="configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json",
    ),
    "randlen_test": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_all.yaml",
    ),
    # Spawn-mode-stratified variants of the above (EXP-0002): the SAME
    # held-out files, split by initialisation type instead of pooled
    # together, so a model's performance can be compared across spawn
    # modes. "mixed" has no n50 group at all (never collected -- see
    # Genesis/data/overnight_randlen's own directory listing), so its
    # config is the same n20-only one used elsewhere; piled/scattered each
    # pool their n20+n50 groups, mirroring `randlen_test`'s own pooling.
    "randlen_piled": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_piled_all.yaml",
    ),
    "randlen_scattered": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_scattered_all.yaml",
    ),
    "randlen_mixed": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_n20_mixed.yaml",
    ),
}

VALUE_FNS = ("lyapunov", "mass_in_region", "signed_mass")
GOAL_NAMES = ("random_quadrant", "ring_O", "T")


def _load_predictor(spec: dict):
    os.environ[spec["ckpt_env"]] = spec["ckpt"]
    mod = importlib.import_module(spec["module"])
    factory = getattr(mod, spec["factory"])
    return factory()


def _load_cell(corpus_spec: dict, tag: str):
    if corpus_spec["kind"] == "slate":
        # train_cfg is unused here (only needed to fit the linear/mean-delta
        # reference operators, out of scope for this report) but load_cell
        # doesn't require it as a pair -- only the eval_cfg matters.
        return load_cell(corpus_spec["eval_cfg"], "train",
                          manifest_path=corpus_spec["manifest"], tag=tag)
    return load_randlen_cell(corpus_spec["cfg"], "train", tag=tag)


def _accuracy(model_spec: dict, predictor, cell) -> float:
    H, W = cell.occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    batch = _predictor_batch(cell) if hasattr(cell, "states") else cell
    pred = predictor.predict_occ(batch).to(torch.float32)

    if model_spec["is_gnn"]:
        n_particles = predictor.n_particles
        raw = cell.raw
        occ0_np, occ1_np = cell.occ0.numpy(), cell.occ1.numpy()
        n = occ0_np.shape[0]
        truth_rs = np.stack([
            resample_occupancy_through_nodes(occ1_np[i], raw.to_pxl, raw.ctr_in_PXL,
                                              n_particles=n_particles, seed=i)
            for i in range(n)
        ])
        prev_rs = np.stack([
            resample_occupancy_through_nodes(occ0_np[i], raw.to_pxl, raw.ctr_in_PXL,
                                              n_particles=n_particles, seed=10_000_000 + i)
            for i in range(n)
        ])
        truth = torch.from_numpy(truth_rs).to(torch.float32)
        prev = torch.from_numpy(prev_rs).to(torch.float32)
    else:
        truth, prev = cell.occ1.to(torch.float32), cell.occ0.to(torch.float32)

    return metrics(pred, truth, prev, region=region)["accuracy"], pred


def _capture_report(cell, pred: torch.Tensor) -> dict:
    """Per (goal, value_fn) capture, averaged over step-0 same-state pools,
    plus a per-value-fn average over goals."""
    step0 = (cell.step_idx == 0)
    slate_ids = cell.slate_idx[step0]
    occ1_s0 = cell.occ1[step0].to(torch.float32)
    pred_s0 = pred[step0].to(torch.float32)
    unique_slates = slate_ids.unique().tolist()
    H, W = cell.occ0.shape[-2:]

    o_mask = torch.from_numpy(letter_mask("O", H, W).astype(np.float32))
    t_mask = torch.from_numpy(letter_mask("T", H, W).astype(np.float32))
    o_dist = torch.from_numpy(dist_field_from_mask(o_mask.numpy() > 0)).float()
    t_dist = torch.from_numpy(dist_field_from_mask(t_mask.numpy() > 0)).float()

    per_goal = {g: {v: [] for v in VALUE_FNS} for g in GOAL_NAMES}
    for sid in unique_slates:
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        true_pool = occ1_s0[rows]
        pred_pool = pred_s0[rows]

        q_mask_np, _ = random_quadrant_mask(H, W, seed=int(sid))
        q_mask = torch.from_numpy(q_mask_np.astype(np.float32))
        q_dist = torch.from_numpy(dist_field_from_mask(q_mask_np)).float()

        for goal, mask, dist in (("random_quadrant", q_mask, q_dist),
                                  ("ring_O", o_mask, o_dist),
                                  ("T", t_mask, t_dist)):
            v_true = lyapunov(true_pool, dist)
            v_pred = lyapunov(pred_pool, dist)
            per_goal[goal]["lyapunov"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=False))

            v_true = mass_in_region(true_pool, mask)
            v_pred = mass_in_region(pred_pool, mask)
            per_goal[goal]["mass_in_region"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=True))

            v_true = signed_mass_in_region(true_pool, mask)
            v_pred = signed_mass_in_region(pred_pool, mask)
            per_goal[goal]["signed_mass"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=True))

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return float(np.mean(xs)) if xs else float("nan")

    out = {"n_slates": len(unique_slates), "per_goal": {}, "averaged_over_goals": {}}
    for goal in GOAL_NAMES:
        out["per_goal"][goal] = {v: _mean(per_goal[goal][v]) for v in VALUE_FNS}
    for v in VALUE_FNS:
        out["averaged_over_goals"][v] = _mean(
            [out["per_goal"][g][v] for g in GOAL_NAMES])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="Baselines/common/runs/cross_corpus_report")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--corpora", default=",".join(CORPORA))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    model_names = args.models.split(",")
    corpus_names = args.corpora.split(",")

    results = {}
    for corpus_name in corpus_names:
        corpus_spec = CORPORA[corpus_name]
        t0 = time.time()
        cell = _load_cell(corpus_spec, tag=corpus_name)
        print(f"[{corpus_name}] loaded {cell.occ0.shape[0]} transitions "
              f"({time.time() - t0:.1f}s)")

        for model_name in model_names:
            model_spec = MODELS[model_name]
            t0 = time.time()
            predictor = _load_predictor(model_spec)
            acc, pred = _accuracy(model_spec, predictor, cell)
            capture = _capture_report(cell, pred)
            dt = time.time() - t0
            print(f"[{corpus_name} / {model_name}] accuracy={acc:.4f}  "
                  f"({dt:.1f}s)")
            for goal in GOAL_NAMES:
                row = capture["per_goal"][goal]
                print(f"    goal={goal:16s} "
                      f"lyapunov={row['lyapunov']:+.4f}  "
                      f"mass_in_region={row['mass_in_region']:+.4f}  "
                      f"signed_mass={row['signed_mass']:+.4f}")
            avg = capture["averaged_over_goals"]
            print(f"    {'averaged':16s} "
                  f"lyapunov={avg['lyapunov']:+.4f}  "
                  f"mass_in_region={avg['mass_in_region']:+.4f}  "
                  f"signed_mass={avg['signed_mass']:+.4f}")

            results.setdefault(corpus_name, {})[model_name] = {
                "accuracy": acc, "capture": capture,
            }

    out_path = f"{args.out_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
