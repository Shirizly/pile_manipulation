"""EXP-0030: in-distribution control-utility check on overnight_randlen's OWN
held-out test files.

`Genesis/data/overnight_randlen/DATASET.yaml` establishes that every file in
this corpus is a SAME-STATE POOL: all 128 envs start step 0 from one
identical settled state (measured spread 0/1.16e-10 m), exactly the
`slates_multistep` structure, just with randomised push length and 4 steps
instead of 3. So the held-out TEST files (disjoint from train, see
`scripts/probes/prepare_randlen_split.py`) can be scored with the SAME
`regret_dv`/`slateK_exact` machinery as the register's slate cells --
`Baselines.common.eval_baseline.run_eval` almost does this already, but it
hardcodes a manifest-based slate/step lookup this corpus does not have (no
manifest.json: it is not organised as named slates). This script derives
slate_idx/step_idx itself instead:

  * slate_idx = run_idx directly (one file == one pool, by construction);
  * step_idx  = (row-within-run) // 128 -- rows are env-major blocked
    (`DATASET.yaml`: row r = env r%128, step r//128), verified here by
    checking the step-0 subset comes out to exactly n_files*128 rows with no
    partial blocks (a `min_push_length_m` guard can drop a handful of
    near-zero-length rows elsewhere in a file without disturbing this, and is
    checked, not assumed -- see "What was actually run" in EXP-0030).

Restricted to the n20 groups (mixed/piled/scattered) -- `load_cell` hard-
assumes states shape (n,20,7), and the n50 groups cannot go through it. n20
piled/scattered/mixed already spans 3 of this corpus's 5 groups, so this row
is an average over BROADER conditions than the eval slate cells (which are
n20, piled only) but NOT the full corpus (n50 groups excluded) -- stated
plainly, not smoothed over, per the coordinator's instruction.

Control ranking uses step-0 rows ONLY (later steps are each env's own
diverged rollout, reintroducing the cross-state confound this whole
convention exists to avoid).

Usage
-----
    GNN_CKPT=Baselines/GNN/runs/gnn_randlen/ckpt_best.pth \\
    PYTHONPATH=. python Baselines/common/eval_randlen_indist.py \\
        --predictor Baselines.GNN.predictor:build_predictor \\
        --tag gnn_randlen_indist --out-prefix Baselines/GNN/runs/gnn_randlen_indist

**`GNN_CKPT`/`NFD_CKPT` must be set explicitly** (read by each predictor's own
`build_predictor()`, `Baselines/{GNN,NFD}/predictor.py`) -- both default to the
IN-DISTRIBUTION (pooled L20L40) checkpoint if unset, so an unset env var does
not fail loudly here: it silently scores the wrong model under the
`_randlen_` tag/output filenames. Caught during the spawn-mode-stratification
follow-up (2026-09-10) when a smoke-test run's own printed checkpoint path/
epoch (`Baselines/GNN/runs/ckpt_best.pth`, epoch=400) did not match the
randlen-trained checkpoint's path/epoch (`Baselines/GNN/runs/gnn_randlen/
ckpt_best.pth`, epoch=224) that every prior `_randlen_indist` run in this
record used -- re-run before anything was written to the record.
"""
from __future__ import annotations

import argparse
import importlib
import json

import torch

from control_utility_test import lyapunov, lyapunov_weights, pile_centroid_and_support
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, predict_world,
)
from scripts.probes.exp0009_rerun import predict_meandelta
from utils import git_provenance

from Baselines.common.data import _resolve_sample, load_cell
from Baselines.common.eval_baseline import _predictor_batch

TRAIN_CFG = "configs/dataset/genesis_overnight_randlen_train_n20.yaml"
TEST_CFG = "configs/dataset/genesis_overnight_randlen_test_n20.yaml"
R, CR, RIDGE = 64, 1.0, 1.0


def _load_predictor(spec: str):
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--train-cfg", default=TRAIN_CFG)
    ap.add_argument("--test-cfg", default=TEST_CFG)
    ap.add_argument("--goals", default="corner")
    args = ap.parse_args()

    prov = git_provenance()

    data_tr = load_cell(args.train_cfg, "train")
    H, W = data_tr.H, data_tr.W
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                    data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ0.shape[0]
    Y0 = canonicalise(data_tr.occ0, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)
    print(f"fit reference operators on {n_tr} train transitions "
          f"({data_tr.run_idx.unique().numel()} files)")

    cell = load_cell(args.test_cfg, "train", tag=args.tag)
    n = cell.occ0.shape[0]

    # Derive slate_idx/step_idx ourselves -- this corpus has no manifest.json
    # (see module docstring). slate_idx = run_idx (one file == one pool);
    # step_idx = (row-within-run) // 128 (env-major-blocked chain).
    step_idx = torch.empty(n, dtype=torch.long)
    for i in range(n):
        _, s = _resolve_sample(cell.raw, i)
        step_idx[i] = s // 128
    slate_idx = cell.run_idx

    step0 = step_idx == 0
    n_files = int(slate_idx.unique().numel())
    n_s0 = int(step0.sum())
    n_expect = n_files * 128
    print(f"test: {n} transitions, {n_files} held-out files; "
          f"step-0 subset: {n_s0} (expected {n_expect} if no row was filtered "
          f"out of a step-0 block)")
    assert n_s0 == n_expect, (
        f"step-0 subset is {n_s0}, expected exactly {n_expect} -- a "
        f"min_push_length_m filter must have dropped a step-0 row; the "
        f"step_idx = sample_idx // 128 derivation assumes it did not.")

    s_px, e_px = actions_to_pixels(cell.actions, data_tr.workspace_min,
                                    data_tr.workspace_max, (H, W))
    preds = {
        "persistence": cell.occ0,
        "mean-delta": predict_meandelta(bmd, cell.occ0, s_px, e_px, R, (H, W), CR),
        "linear": predict_world(A, cell.occ0, s_px, e_px, R, (H, W), CR),
    }
    predictor = _load_predictor(args.predictor)
    batch = _predictor_batch(cell)
    pred = predictor.predict_occ(batch)
    preds[predictor.name] = pred.detach().to(torch.float32)

    occ0_s0, occ1_s0 = cell.occ0[step0], cell.occ1[step0]
    slate_s0 = slate_idx[step0]
    preds_s0 = {k: v[step0] for k, v in preds.items()}

    cache = {"ep": slate_s0, "actions": cell.actions[step0], "provenance": prov,
             "slate_files": [cell.files[int(r)] for r in cell.run_idx[step0].tolist()],
             "split": "randlen_test_step0",
             "config": {"train_cfg": args.train_cfg, "test_cfg": args.test_cfg,
                        "run_dir": None, "R": R, "crop": CR, "ridge": RIDGE},
             "dv": {}}
    goal_list = [g.strip() for g in args.goals.split(",") if g.strip()]
    pile_center, pile_support = None, None
    if any(g.endswith("-pile") for g in goal_list):
        # Same "compute from the loaded data, never hard-code" rule
        # Baselines/common/eval_baseline.py follows for the slate cells
        # (goal-placement-pinned, INVARIANTS.md, is broken precisely because
        # this centroid is recomputed per run/per pool set -- documented, not
        # fixed here). Computed once over THIS call's step-0 occ0, i.e. over
        # whatever test-cfg was loaded -- pooled across all 3 n20 spawn
        # groups for the original pooled config, or over ONE spawn group's 5
        # held-out files for the per-stratum configs added for the spawn-mode
        # stratification follow-up, so each stratum gets a target on its own
        # pile rather than an average over all three.
        pile_center, pile_support = pile_centroid_and_support(occ0_s0)
        print(f"pile centroid (row,col)={pile_center}, support (r0,r1,c0,c1)={pile_support}")
    for goal in goal_list:
        dw = lyapunov_weights((H, W), goal, "cpu", pile_center=pile_center)
        v0 = lyapunov(occ0_s0, dw)
        dv_true = lyapunov(occ1_s0, dw) - v0
        g = {"dv_true": dv_true, "v0": v0}
        for name, p in preds_s0.items():
            g[name] = lyapunov(p, dw) - v0
        cache["dv"][goal] = g
        print(f"goal={goal}: dv_true mean {float(dv_true.mean()):+.5f} "
              f"sd {float(dv_true.std()):.5f} "
              f"helpful {100*float((dv_true < 0).float().mean()):.0f}%")

    cache_path = f"{args.out_prefix}_dv_cache.pt"
    torch.save(cache, cache_path)
    meta_path = f"{args.out_prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"tag": args.tag, "provenance": prov, "n_files": n_files,
                   "n_step0": n_s0, "predictor": predictor.name}, f, indent=2)
    print(f"wrote {cache_path}\nwrote {meta_path}")


if __name__ == "__main__":
    main()
