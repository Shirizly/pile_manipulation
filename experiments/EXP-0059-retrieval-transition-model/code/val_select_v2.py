"""EXP-0059 clean-data v2 pass: VAL-only retrieval hyperparameter selection
(k in {1, 5}, pre-declared, never tuned on test) and NFD checkpoint
confirmation, both scored on `Genesis/data/narrow_l20_n20/val_pools_v2`
(DS-0017, 2048/2048 clean rows -- no chains, pools only).

Bank: `artifacts/bank_train_v2_curated.pt` (built from train_v2 by
`build_bank_v2.py`, this same session). Gate distance held fixed at the
R1-best `_CAPPED` config (not re-swept, per the coordinator's "small,
pre-declared grid" instruction) -- only k/aggregation varies:
  k=1  -> NearestTransitionPredictor(bank)         (1-NN, no k-way aggregation)
  k=5  -> RetrievalPredictor(bank, cfg=_CAPPED, k=5, aggregation="cube_median")

NFD checkpoint confirmation: score `nfd_3ch_narrow_l20_v2` (best-val-loss,
epoch 57 per training) against a couple of the per-10-epoch saves on the
SAME VAL pools' slateN_tough (the actual control-relevant metric, not the
training loss the checkpoint selection already used) -- confirms best-val-
loss is also a reasonable slateN_tough choice, or names the discrepancy if
not.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/val_select_v2.py
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_extended as EE  # noqa: E402

from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import NearestTransitionPredictor, RetrievalPredictor  # noqa: E402

VAL_POOLS = REPO / "Genesis/data/narrow_l20_n20/val_pools_v2"
NO_CHAINS = REPO / "Genesis/data/narrow_l20_n20/__no_chains__"  # deliberately nonexistent
BANK_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "bank_train_v2_curated.pt"
RES = Path(__file__).resolve().parents[1] / "results" / "val_select_v2.json"
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = json.loads(RES.read_text()) if RES.exists() else {}

    def _save():
        tmp = str(RES) + ".tmp"; RES.parent.mkdir(parents=True, exist_ok=True)
        Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)

    ch, pools = EE._load_corpus(pools_dir=str(VAL_POOLS), chains_dir=str(NO_CHAINS),
                                exclude_flagged=True)
    assert ch == [], "VAL pools should carry no chains -- got some, check NO_CHAINS path"
    n_pools = sum(len(torch.unique(d["pool_idx"])) for d in pools)
    print(f"loaded VAL: {sum(d['states'].shape[0] for d in pools)} rows, {n_pools} pools")
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    Masks13 = EE._masks(EE.GOALS_13); MasksTough = EE._masks(EE.GOALS_TOUGH)

    # --- retrieval k-sweep ---
    bank = TransitionBank.load_curated(str(BANK_PATH))
    print(f"bank: {len(bank)} rows")
    retrieval_cfgs = {
        "retrieval_k1_v2": lambda: NearestTransitionPredictor(bank),
        "retrieval_k5_cube_median_v2": lambda: RetrievalPredictor(
            bank, cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median"),
    }
    for name, factory in retrieval_cfgs.items():
        if name in out:
            print("skip (already scored):", name); continue
        t0 = time.time()
        r, _raw = EE.eval_particle_model(name, bank, ch, pools, Dist13, DistTough,
                                         Masks13, MasksTough, predictor_obj=factory())
        r["wall_s"] = time.time() - t0
        out[name] = r; _save()
        print(f"{name:30s} slateN_tough {r['slateN_tough']:.4f}  slateN {r['slateN']:.4f}  "
             f"({r['wall_s']:.1f}s)", flush=True)

    # --- NFD checkpoint confirmation ---
    for model_id in ("nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_epoch10",
                     "nfd_3ch_narrow_l20_v2_epoch20", "nfd_3ch_narrow_l20_v2_epoch30",
                     "nfd_3ch_narrow_l20_v2_epoch40", "nfd_3ch_narrow_l20_v2_epoch50",
                     "nfd_3ch_narrow_l20_v2_epoch60"):
        if model_id in out:
            print("skip (already scored):", model_id); continue
        try:
            t0 = time.time()
            r, _raw = EE.eval_occ_model(model_id, dev, ch, pools, Dist13, DistTough,
                                        Masks13, MasksTough)
            r["wall_s"] = time.time() - t0
            out[model_id] = r; _save()
            print(f"{model_id:30s} slateN_tough {r['slateN_tough']:.4f}  slateN "
                 f"{r['slateN']:.4f}  ({r['wall_s']:.1f}s)", flush=True)
        except KeyError:
            print("skip (not registered):", model_id)

    print("\n=== VAL selection summary ===")
    for k, v in out.items():
        print(f"  {k:30s} slateN_tough={v['slateN_tough']:.4f}  slateN={v['slateN']:.4f}")
    print("wrote", RES)


if __name__ == "__main__":
    main()
