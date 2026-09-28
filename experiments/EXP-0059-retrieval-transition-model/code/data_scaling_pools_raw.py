"""Coordinator follow-up (2026-09-28, hard stop 08:10, offline/CPU only --
GPU reserved for the coder): regenerate per-pool raw {pool,vt,vp} arrays for
the 5 data-scaling bank configs so `pool_bootstrap_ci` (reused from EXP-0060's
`correlate.py`) can put a CI on each `slateN_tough` point and a PAIRED delta
CI on 25k vs 98k. `data_scaling.py` itself discarded this raw output --
this script reruns ONLY the DS-0009 POOLS loop (passing `ch=[]` to
`eval_extended.eval_particle_model`, which already guards that case cleanly,
skipping accuracy_1/rollout entirely) against the ALREADY-PERSISTED banks
(`artifacts/bank_{name}.pt`, built by `data_scaling.py`) -- no bank rebuild,
no chains work, CPU-forced via `CUDA_VISIBLE_DEVICES=""`.

Usage:
    CUDA_VISIBLE_DEVICES= python -u data_scaling_pools_raw.py
"""
import json, os, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_extended as EE  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RAW_OUT = Path(__file__).resolve().parents[1] / "results" / "data_scaling_pools_raw.json"
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)
BEST_CFG = DistanceConfig(**_CAPPED)

CONFIGS = ["orig_12k", "ds0012_only_12k", "25k", "50k", "98k_all"]


def main():
    assert not torch.cuda.is_available(), "expected CUDA hidden (CUDA_VISIBLE_DEVICES=); refusing to touch the GPU per coordinator instruction"
    out = json.loads(RAW_OUT.read_text()) if RAW_OUT.exists() else {}
    _, pools = EE._load_corpus()  # ch unused
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    Masks13 = EE._masks(EE.GOALS_13); MasksTough = EE._masks(EE.GOALS_TOUGH)
    for name in CONFIGS:
        if name in out:
            print("skip (already done):", name); continue
        bank_path = ARTIFACTS / f"bank_{name}.pt"
        bank = TransitionBank.load(str(bank_path))
        predictor = RetrievalPredictor(bank, cfg=BEST_CFG, k=5, aggregation="cube_median", device="cpu")
        t0 = time.time()
        r, raw = EE.eval_particle_model(f"k5_cube_median_{name}", bank, [], pools,
                                        Dist13, DistTough, Masks13, MasksTough,
                                        predictor_obj=predictor)
        wall = time.time() - t0
        print(f"{name:16s} bank={len(bank):6d} slateN_tough={r['slateN_tough']:.4f} ({wall:.1f}s)", flush=True)
        out[name] = {"slateN_tough_raw": raw["slateN_tough_raw"], "slateN_raw": raw["slateN_raw"],
                    "point": r["slateN_tough"], "wall_s": wall}
        tmp = str(RAW_OUT) + ".tmp"; RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).write_text(json.dumps(out)); os.replace(tmp, RAW_OUT)
    print("wrote", RAW_OUT)


if __name__ == "__main__":
    main()
