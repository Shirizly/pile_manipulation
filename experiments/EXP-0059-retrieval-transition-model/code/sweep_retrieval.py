"""EXP-0059 hyperparameter sweep over `model.retrieval.distance.DistanceConfig`
/ k / aggregation, tuned on a LEAKAGE-SAFE VALIDATION split of DS-0008 (never
on DS-0009 -- see `model.retrieval.val_split`'s module docstring for why a
row-random split would leak and why DS-0009 must stay untouched during
tuning), then re-scored on the real DS-0009 test set for the shortlisted best
configs only.

Two phases, each independently checkpointed (atomic per-config JSON, safe to
resume/kill):

  1. `--phase val`  : every config in `GRID` scored on the DS-0008 validation
     chains (`accuracy_1`, `rollout_accuracy_4`; no `slateN` -- DS-0008 has no
     same-state action pools, see `val_split.py`), written to
     `results/sweep_val.json`.
  2. `--phase test` : the top `--top-n` configs by validation `accuracy_1`
     re-built on the REAL full bank (DS-0008 + DS-0010, no holdout) and
     scored on DS-0009 (`accuracy_1`, `rollout_accuracy_4`, `slateN`),
     written to `results/offline_eval_retrieval_bestconfigs.json` (a
     SEPARATE file from `results/offline_eval_retrieval.json`, which other
     scripts in this experiment also write to).

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/sweep_retrieval.py --phase val
    python -u experiments/EXP-0059-retrieval-transition-model/code/sweep_retrieval.py --phase test --top-n 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_retrieval as ER  # noqa: E402

from Baselines.common.goals import dist_field_from_mask  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402
from model.retrieval.val_split import build_tuning_bank_and_val_chains  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
VAL_RES = RESULTS_DIR / "sweep_val.json"
TEST_RES = RESULTS_DIR / "offline_eval_retrieval_bestconfigs.json"
VAL_BANK_PATH = ARTIFACTS / "val_bank.pt"
VAL_CHAINS_PATH = ARTIFACTS / "val_chains.pt"

# A "capped Chamfer + mismatch penalty + corridor weighting" baseline config,
# per the designer's crude probe (docs/experimental_design/
# retrieval_based_modeling.md's front-matter): moved cubes travel median
# 11.6mm, so a 20mm cap with a 2x-cap mismatch penalty gives room for a real
# match while still discounting a badly-mismatched cube.
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)

GRID = [
    ("v0_baseline", dict(cfg=DistanceConfig(), k=1, aggregation="nn1")),
    ("capped", dict(cfg=DistanceConfig(cap=0.02, mismatch_penalty=0.04), k=1, aggregation="nn1")),
    ("capped_corridor", dict(cfg=DistanceConfig(**_CAPPED), k=1, aggregation="nn1")),
    ("capped_corridor_sum", dict(cfg=DistanceConfig(**_CAPPED, reduction="sum"), k=1, aggregation="nn1")),
    ("capped_corridor_wall", dict(cfg=DistanceConfig(**_CAPPED, wall_weight=0.02), k=1, aggregation="nn1")),
    ("capped_corridor_tight_window", dict(
        cfg=DistanceConfig(**{**_CAPPED, "window_u": (0.0, 0.025), "window_v": (-0.025, 0.025)}),
        k=1, aggregation="nn1")),
    ("k5_cube_median", dict(cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median")),
    ("k5_cube_weighted_mean", dict(cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_weighted_mean")),
    ("k5_occ_mean", dict(cfg=DistanceConfig(**_CAPPED), k=5, aggregation="occ_mean")),
    ("k5_occ_weighted_mean", dict(cfg=DistanceConfig(**_CAPPED), k=5, aggregation="occ_weighted_mean")),
    ("capped_corridor_no_yaw", dict(cfg=DistanceConfig(**_CAPPED), k=1, aggregation="nn1", transfer_yaw=False)),
]


def _cfg_to_json(cfg: DistanceConfig) -> dict:
    return dict(window_u=list(cfg.window_u), window_v=list(cfg.window_v), cap=cfg.cap,
               mismatch_penalty=cfg.mismatch_penalty, corridor_v_halfwidth=cfg.corridor_v_halfwidth,
               corridor_u_pad=cfg.corridor_u_pad, corridor_weight=cfg.corridor_weight,
               wall_weight=cfg.wall_weight, reduction=cfg.reduction)


def _atomic_write(path: Path, obj: dict):
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def run_val_phase(val_frac=0.2, seed=0):
    if VAL_BANK_PATH.exists() and VAL_CHAINS_PATH.exists():
        bank = TransitionBank.load(str(VAL_BANK_PATH))
        val_chains = torch.load(str(VAL_CHAINS_PATH), map_location="cpu", weights_only=False)
    else:
        bank, val_chains = build_tuning_bank_and_val_chains(val_frac=val_frac, seed=seed)
        bank.save(str(VAL_BANK_PATH))
        torch.save(val_chains, str(VAL_CHAINS_PATH))
    print(f"tuning bank: {len(bank)} transitions; validation chains: "
         f"{sum(len(d['states']) for d in val_chains)} rows over {len(val_chains)} files")

    out = json.loads(VAL_RES.read_text()) if VAL_RES.exists() else {}
    for name, kw in GRID:
        if name in out:
            print("skip (already scored):", name); continue
        predictor = RetrievalPredictor(bank, **kw)
        t0 = time.time()
        r = ER.evaluate(predictor, val_chains, [], {})   # no pools on DS-0008 -> slateN=None
        r["wall_s"] = time.time() - t0
        r["config"] = dict(k=kw["k"], aggregation=kw["aggregation"],
                          transfer_yaw=kw.get("transfer_yaw", True),
                          distance=_cfg_to_json(kw["cfg"]))
        out[name] = r
        _atomic_write(VAL_RES, out)
        print(f"[val] {name:28s} acc1 {r['accuracy_1']:+.3f}  rollout4 "
             f"{r.get('rollout_accuracy_4', float('nan')):+.3f}  ({r['wall_s']:.1f}s)", flush=True)
    print("wrote", VAL_RES)


def run_test_phase(top_n=3):
    if not VAL_RES.exists():
        raise SystemExit("run --phase val first")
    val = json.loads(VAL_RES.read_text())
    ranked = sorted(val.items(), key=lambda kv: kv[1]["accuracy_1"], reverse=True)[:top_n]
    print("shortlisted (by validation accuracy_1):", [n for n, _ in ranked])

    bank = TransitionBank.load(str(ARTIFACTS / "bank.pt"))  # real full bank: DS-0008 + DS-0010, no holdout
    ch, pools = ER._load_corpus()
    Dist = {g: torch.from_numpy(dist_field_from_mask(ER.EN.goal_mask(g))).float() for g in ER.EN.GOALS}

    out = json.loads(TEST_RES.read_text()) if TEST_RES.exists() else {}
    for name, valrow in ranked:
        if name in out:
            print("skip (already scored):", name); continue
        cfgd = valrow["config"]["distance"]
        cfg = DistanceConfig(window_u=tuple(cfgd["window_u"]), window_v=tuple(cfgd["window_v"]),
                             cap=cfgd["cap"], mismatch_penalty=cfgd["mismatch_penalty"],
                             corridor_v_halfwidth=cfgd["corridor_v_halfwidth"],
                             corridor_u_pad=cfgd["corridor_u_pad"], corridor_weight=cfgd["corridor_weight"],
                             wall_weight=cfgd["wall_weight"], reduction=cfgd["reduction"])
        predictor = RetrievalPredictor(bank, cfg=cfg, k=valrow["config"]["k"],
                                       aggregation=valrow["config"]["aggregation"],
                                       transfer_yaw=valrow["config"]["transfer_yaw"])
        t0 = time.time()
        r = ER.evaluate(predictor, ch, pools, Dist)
        r["wall_s"] = time.time() - t0
        r["val_accuracy_1"] = valrow["accuracy_1"]
        r["config"] = valrow["config"]
        out[name] = r
        _atomic_write(TEST_RES, out)
        slate_str = f"{r['slateN']:.3f}" if r["slateN"] is not None else "n/a"
        print(f"[TEST/DS-0009] {name:28s} acc1 {r['accuracy_1']:+.3f}  rollout4 "
             f"{r.get('rollout_accuracy_4', float('nan')):+.3f}  slateN {slate_str}  "
             f"(val acc1 was {r['val_accuracy_1']:+.3f})", flush=True)
    print("wrote", TEST_RES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["val", "test"], required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top-n", type=int, default=3)
    a = ap.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if a.phase == "val":
        run_val_phase(a.val_frac, a.seed)
    else:
        run_test_phase(a.top_n)


if __name__ == "__main__":
    main()
