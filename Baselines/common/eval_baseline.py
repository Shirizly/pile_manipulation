"""Baselines/common/eval_baseline.py -- generalised EXP-B scorer.

A generalisation of ``scripts/probes/expB_multistep_eval.py`` that takes a
**pluggable predictor** in place of the hardcoded linear/mean-delta/UNet
trio. Reuses that script's pipeline unchanged (fit the linear/mean-delta
reference operators on TRAIN, score IMAGE ACCURACY over the full EVAL split
with the swept-region mask, score CONTROL UTILITY on STEP-0 CANDIDATES ONLY
and write a dV cache) -- the only new code is the predictor plumbing and the
particle/occupancy dual view (via ``Baselines.common.data``).

Predictor contract
-------------------
    class BaselinePredictor(Protocol):
        name: str
        def predict_occ(self, batch: PredictorBatch) -> torch.Tensor:
            ...  # -> (B, 64, 64) predicted NEXT occupancy

``PredictorBatch`` (below) carries BOTH views of the EVAL split -- ``occ0``
for a grid model that goes straight from occupancy, and
``states``/``p_start``/``p_stop``/``angle``/``actions`` for a particle-space
model that predicts next particle positions and rasterises them via
``Baselines.common.data.rasterize_particles`` (the SAME routine that
produced every occ0/occ1 in this dataset -- see that module's docstring).
It never carries ``occ1``/``states_`` (the ground truth).

Reference rows: ``persistence``, ``mean-delta`` and ``linear`` (fit on TRAIN
exactly as EXP-B does, via ``fit_linear_foresight``/``dmdc_baseline``, not
reimplemented) are ALWAYS computed and included in the output, in addition
to any ``--predictor`` supplied. This is also what ``--self-test`` checks
against ``runs_expB/n20_L20mm_accuracy.json`` (no external predictor needed
for that check).

Usage
-----
    PYTHONPATH=. python Baselines/common/eval_baseline.py \\
        configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml \\
        configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \\
        Genesis/data/slates_multistep/n20_L20mm/manifest.json \\
        --tag <yourmodel>_L20mm --out-prefix Baselines/<YOU>/runs/<yourmodel>_L20mm \\
        --predictor yourpkg.module:build_predictor

    # harness self-test (no external predictor; reproduces EXP-B's own numbers):
    PYTHONPATH=. python Baselines/common/eval_baseline.py --self-test
"""
from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass

import torch

from control_utility_test import lyapunov, lyapunov_weights, pile_centroid_and_support
from fit_linear_foresight import (
    actions_to_pixels, canonicalise, fit_operator, metrics, predict_world,
    swept_region_mask,
)
from scripts.probes.exp0009_rerun import predict_meandelta
from utils import git_provenance

from Baselines.common.data import CellData, load_cell

R, CR, RIDGE = 64, 1.0, 1.0  # EXP-0024/EXP-0026/EXP-B's fit, unchanged


@dataclass
class PredictorBatch:
    """Everything a predictor may look at. Never carries the ground truth
    (`occ1`/`states_`)."""
    occ0: torch.Tensor         # (B,H,W) current occupancy, in [0,1]
    actions: torch.Tensor      # (B,4) world [sx,sy,ex,ey] metres
    states: torch.Tensor       # (B,20,7) pre-push particle xyz+quat, world metres
    p_start: torch.Tensor      # (B,3) world metres
    p_stop: torch.Tensor       # (B,3) world metres
    angle: torch.Tensor        # (B,)
    run_idx: torch.Tensor      # (B,) long -- index into raw.configs, for rasterize_particles
    raw: object                 # the PileSweepData instance (for data.rasterize_particles)
    workspace_min: tuple
    workspace_max: tuple
    H: int
    W: int


def _predictor_batch(cell: CellData) -> PredictorBatch:
    return PredictorBatch(occ0=cell.occ0, actions=cell.actions, states=cell.states,
                           p_start=cell.p_start, p_stop=cell.p_stop, angle=cell.angle,
                           run_idx=cell.run_idx, raw=cell.raw,
                           workspace_min=cell.workspace_min, workspace_max=cell.workspace_max,
                           H=cell.H, W=cell.W)


def _load_predictor(spec: str):
    """`spec` is "module.path:factory_callable" -- imported and called with
    no arguments; must return an object with `.name` (str) and
    `.predict_occ(batch) -> (B,H,W)` tensor."""
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"--predictor must be 'module.path:factory', got {spec!r}")
    mod = importlib.import_module(mod_name)
    factory = getattr(mod, fn_name)
    predictor = factory()
    assert hasattr(predictor, "name") and hasattr(predictor, "predict_occ"), (
        f"{spec!r} did not return a BaselinePredictor (missing .name/.predict_occ)")
    return predictor


def run_eval(train_cfg: str, eval_cfg: str, manifest: str, tag: str, out_prefix: str,
             predictor_specs: list[str] | None = None, goals: str = "corner,center",
             degradations: bool = False):
    prov = git_provenance()
    print(f"=== Baselines/common/eval_baseline: {tag} ===\nprovenance: {prov}")

    # ---- fit reference operators on TRAIN (manifest-agnostic) --------------
    data_tr = load_cell(train_cfg, "train")
    H, W = data_tr.H, data_tr.W
    s_tr, e_tr = actions_to_pixels(data_tr.actions, data_tr.workspace_min,
                                    data_tr.workspace_max, (H, W))
    n_tr = data_tr.occ0.shape[0]
    print(f"train: {n_tr} transitions, grid {H}x{W}, "
          f"{data_tr.run_idx.unique().numel()} batches")

    Y0 = canonicalise(data_tr.occ0, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    Y1 = canonicalise(data_tr.occ1, s_tr, e_tr, R, CR).reshape(n_tr, -1).T
    A = fit_operator(Y0, Y1, ridge=RIDGE, toward_identity=True)
    bmd = (Y1 - Y0).mean(dim=1)

    # ---- load EVAL split, tagged with slate/step via the source manifest --
    cell = load_cell(eval_cfg, "train", manifest_path=manifest, tag=tag)
    n = cell.occ0.shape[0]
    s_px, e_px = actions_to_pixels(cell.actions, data_tr.workspace_min,
                                    data_tr.workspace_max, (H, W))
    print(f"eval: {n} transitions, {cell.run_idx.unique().numel()} batches")
    assert set(cell.step_idx.tolist()) == {0, 1, 2}, "expected exactly 3 steps in eval split"
    print(f"  steps present: {sorted(set(cell.step_idx.tolist()))}, "
          f"step0 transitions: {int((cell.step_idx == 0).sum())}")

    preds = {
        "persistence": cell.occ0,
        "mean-delta": predict_meandelta(bmd, cell.occ0, s_px, e_px, R, (H, W), CR),
        "linear": predict_world(A, cell.occ0, s_px, e_px, R, (H, W), CR),
    }

    batch = _predictor_batch(cell)
    predictors = [_load_predictor(spec) for spec in (predictor_specs or [])]
    for predictor in predictors:
        pred = predictor.predict_occ(batch)
        assert pred.shape == cell.occ0.shape, (
            f"{predictor.name}.predict_occ returned {tuple(pred.shape)}, "
            f"expected {tuple(cell.occ0.shape)}")
        preds[predictor.name] = pred.detach().to(torch.float32)

    # =========================================================================
    # (a) IMAGE ACCURACY -- all 3 steps, swept-region mask
    # =========================================================================
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    img_preds = dict(preds)
    img_preds["oracle"] = cell.occ1
    accuracy = {}
    accuracy_by_step = {0: {}, 1: {}, 2: {}}
    for name, pred in img_preds.items():
        m = metrics(pred, cell.occ1, cell.occ0, region=region)
        accuracy[name] = m["accuracy"]
        for st in (0, 1, 2):
            sel = cell.step_idx == st
            m_st = metrics(pred[sel], cell.occ1[sel], cell.occ0[sel], region=region[sel])
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
    step0 = cell.step_idx == 0
    occ0_s0, occ1_s0 = cell.occ0[step0], cell.occ1[step0]
    s_px_s0, e_px_s0 = s_px[step0], e_px[step0]
    slate_s0 = cell.slate_idx[step0]
    preds_s0 = {k: v[step0] for k, v in preds.items()}
    n_s0 = int(step0.sum())
    n_slates_s0 = slate_s0.unique().numel()
    print(f"\nstep-0 (control-ranking) subset: {n_s0} transitions, "
          f"{n_slates_s0} slates ({n_s0 // max(n_slates_s0, 1)} candidates/slate)")

    arms = {}
    if degradations:
        from scripts.probes.exp0026_selection_pressure import _degradation_arms
        arms = _degradation_arms(preds_s0["linear"], occ0_s0, occ1_s0, s_px_s0, e_px_s0, (H, W))
        print(f"degradation arms: {len(arms)} added")

    cache = {"ep": slate_s0, "actions": cell.actions[step0], "provenance": prov,
             "slate_files": [cell.files[int(r)] for r in cell.run_idx[step0].tolist()],
             "split": "eval_step0",
             "config": {"train_cfg": train_cfg, "eval_cfg": eval_cfg,
                        "run_dir": None, "R": R, "crop": CR, "ridge": RIDGE},
             "dv": {}}
    goal_list = [g.strip() for g in goals.split(",") if g.strip()]
    pile_center, pile_support = None, None
    if any(g.endswith("-pile") for g in goal_list):
        pile_center, pile_support = pile_centroid_and_support(occ0_s0)
        print(f"pile centroid (row,col)={pile_center}, support (r0,r1,c0,c1)={pile_support}")
    for goal in goal_list:
        dw = lyapunov_weights((H, W), goal, "cpu", pile_center=pile_center)
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

    cache_path = f"{out_prefix}_dv_cache.pt"
    torch.save(cache, cache_path)
    print(f"wrote {cache_path}")

    acc_path = f"{out_prefix}_accuracy.json"
    with open(acc_path, "w") as f:
        json.dump({"tag": tag, "provenance": prov, "n_eval": n,
                   "accuracy": accuracy, "accuracy_by_step": accuracy_by_step,
                   "n_slates_step0": n_slates_s0, "n_step0": n_s0}, f, indent=2)
    print(f"wrote {acc_path}")
    return accuracy, accuracy_by_step, cache_path, acc_path


def _self_test():
    import numpy as np

    ref_path = "runs_expB/n20_L20mm_accuracy.json"
    ref = json.load(open(ref_path))

    accuracy, accuracy_by_step, cache_path, acc_path = run_eval(
        train_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_train.yaml",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
        tag="selftest_n20_L20mm",
        out_prefix="/tmp/eval_baseline_selftest_n20_L20mm",
        predictor_specs=[], goals="corner,center", degradations=False,
    )

    ok = True
    for name in ("mean-delta", "linear"):
        got, want = accuracy[name], ref["accuracy"][name]
        close = abs(got - want) < 1e-3
        ok &= close
        print(f"self-test accuracy[{name}]: got {got:.5f} want {want:.5f} {'OK' if close else 'FAIL'}")
    for st in (0, 1, 2):
        for name in ("mean-delta", "linear"):
            got = accuracy_by_step[st][name]
            want = ref["accuracy_by_step"][str(st)][name]
            close = abs(got - want) < 1e-3
            ok &= close
            print(f"self-test accuracy_by_step[{st}][{name}]: got {got:.5f} want {want:.5f} "
                  f"{'OK' if close else 'FAIL'}")

    print("\n-- running exp0026_kcurve_exact.py on the fresh dv cache --")
    import subprocess
    kc_ref = json.load(open("runs_expB/n20_L20mm_kcurve_exact.json"))
    kc_out = "/tmp/eval_baseline_selftest_kcurve_exact.json"
    subprocess.run(
        ["python", "scripts/probes/exp0026_kcurve_exact.py", cache_path,
         "--goal", "corner", "--models", "persistence,mean-delta,linear,oracle",
         "--ks", ",".join(str(k) for k in kc_ref["ks"]), "--out", kc_out],
        check=True,
    )
    kc = json.load(open(kc_out))
    for name in ("mean-delta", "linear"):
        for K in kc_ref["ks"]:
            got = kc["slateK_exact"][name][str(K)]
            want = kc_ref["slateK_exact"][name][str(K)]
            close = abs(got - want) < 2e-3
            ok &= close
            if not close:
                print(f"self-test slateK_exact[{name}][K={K}]: got {got:.4f} want {want:.4f} FAIL")
    print(f"self-test slateK_exact[linear/mean-delta] over K={kc_ref['ks']}: "
          f"{'all OK' if ok else 'SOME FAILED (see above)'}")

    print("\nSELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_cfg", nargs="?")
    ap.add_argument("eval_cfg", nargs="?")
    ap.add_argument("manifest", nargs="?", help="source cell's manifest.json (batch_idx -> slate/step)")
    ap.add_argument("--tag")
    ap.add_argument("--out-prefix")
    ap.add_argument("--predictor", action="append", default=[],
                     help="'module.path:factory_callable' -- called with no args, must return "
                          "an object with .name and .predict_occ(batch). Repeatable.")
    ap.add_argument("--degradations", action="store_true")
    ap.add_argument("--goals", default="corner,center",
                     help="comma-separated lyapunov_weights goal keys (default: EXP-B's own two).")
    ap.add_argument("--self-test", action="store_true",
                     help="run the harness with only the built-in reference operators on "
                          "n20_L20mm and check it reproduces runs_expB/n20_L20mm_accuracy.json "
                          "and runs_expB/n20_L20mm_kcurve_exact.json.")
    args = ap.parse_args()

    if args.self_test:
        ok = _self_test()
        raise SystemExit(0 if ok else 1)

    missing = [n for n in ("train_cfg", "eval_cfg", "manifest", "tag", "out_prefix")
               if getattr(args, n) is None]
    if missing:
        ap.error(f"missing required arguments: {missing} (or pass --self-test)")

    run_eval(args.train_cfg, args.eval_cfg, args.manifest, args.tag, args.out_prefix,
              predictor_specs=args.predictor, goals=args.goals, degradations=args.degradations)


if __name__ == "__main__":
    main()
