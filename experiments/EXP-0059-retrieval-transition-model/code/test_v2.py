"""EXP-0059 clean-data v2 pass: full TEST scoring on the ISS-010-fix corpus
(DS-0015 train / DS-0016 test / DS-0017 val), fair train/val/test comparison
of narrow NFD, LinearForesight (switched, n_bins=1), and retrieval.

Three DIFFERENT sub-corpora feed different metrics, per the data agent's note
(coordinator message, 2026-09-28) that rollout needs WHOLE clean chains, not
row-filtered ones (filtering individual rows out of an 8-step chain breaks
step-to-step continuity -- the same reasoning `reeval_ds0009.py` already
documents for the pre-fix legal-only rollout number):
  * accuracy_1 (+blur, occ_emd_swept) and moved-cube mm: `test_chains_v2`
    with `--exclude-flagged` (986/1024 rows) -- single-step, no continuity
    needed, so use every valid row for statistical power.
  * rollout_accuracy_1..4: `test_chains_v2_clean` (896 rows / 112 WHOLE
    8-step sequences, already fully clean by construction -- no further
    filtering) -- continuity-dependent.
  * slateN/slateN_tough/mass_in_goal_mae (+ their per-pool raw arrays for
    the paired bootstrap CIs): `test_pools_v2` (2048/2048 rows, 100% clean).

Each model is scored via TWO eval_extended calls (one per chain corpus) and
the results merged: call A's rollout_accuracy_* is wrong (computed from the
row-filtered, continuity-broken test_chains_v2) and is DISCARDED in favour of
call B's (from test_chains_v2_clean); call B's accuracy_1/slateN outputs are
likewise discarded in favour of call A's (call B passes pools=[] so slateN
is not even computed there).

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/test_v2.py
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
sys.path.insert(0, str(REPO / "experiments/EXP-0060-metric-correlation-study/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_extended as EE  # noqa: E402
import correlate as CO  # noqa: E402  (pool_bootstrap_ci, _capture)
from bootstrap_ci import paired_delta_ci  # noqa: E402

from model.retrieval.bank import TransitionBank, OCC_BOUNDS  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import (PersistencePredictor, NearestTransitionPredictor,  # noqa: E402
                                       RetrievalPredictor)

D = REPO / "Genesis/data/narrow_l20_n20"
BANK_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "bank_train_v2_curated.pt"
RES = Path(__file__).resolve().parents[1] / "results" / "test_v2.json"
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)

OCC_MODELS = ["persistence", "nfd_3ch_narrow_l20_v2", "linear_narrow_l20_v2_res64",
             "linear_narrow_l20_v2_res32"]
PARTICLE_MODELS = ["random", "retrieval_k5_cube_median_v2", "retrieval_k1_v2"]


class RandomPredictor:
    """`random` ranking/prediction floor: cube positions drawn i.i.d. uniform
    over the workspace, unrelated to the true outcome -- the induced v_pred
    is then unrelated to v_true, so slateN's capture formula has expectation
    EXACTLY 0 (same construction as `Baselines/common/eval_report.py::
    _random_floor_capture`, adapted to a particle-space predictor: a Monte-
    Carlo check of an analytic fact, not an independent floor to trust on
    its own)."""
    name = "random"

    def __init__(self, seed=0):
        self.gen = torch.Generator().manual_seed(seed)

    def predict_particles(self, states0, p_start, p_stop):
        out = states0.clone()
        B, n, _ = states0.shape
        xy = torch.rand(B, n, 2, generator=self.gen)
        xy[..., 0] = xy[..., 0] * (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) + OCC_BOUNDS["x_min"]
        xy[..., 1] = xy[..., 1] * (OCC_BOUNDS["y_max"] - OCC_BOUNDS["y_min"]) + OCC_BOUNDS["y_min"]
        out[:, :, :2] = xy
        return out


def _atomic(obj, path):
    tmp = str(path) + ".tmp"; path.parent.mkdir(parents=True, exist_ok=True)
    Path(tmp).write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = json.loads(RES.read_text()) if RES.exists() else {}

    ch_acc, pools = EE._load_corpus(chains_dir=str(D / "test_chains_v2"),
                                    pools_dir=str(D / "test_pools_v2"),
                                    exclude_flagged=True)
    ch_roll, _ = EE._load_corpus(chains_dir=str(D / "test_chains_v2_clean"),
                                 pools_dir=str(D / "__no_pools__"),
                                 exclude_flagged=False)
    n_pools = sum(len(torch.unique(d["pool_idx"])) for d in pools)
    print(f"test_chains_v2 (acc): {sum(d['states'].shape[0] for d in ch_acc)} rows")
    print(f"test_chains_v2_clean (rollout): {sum(d['states'].shape[0] for d in ch_roll)} rows")
    print(f"test_pools_v2: {sum(d['states'].shape[0] for d in pools)} rows, {n_pools} pools")

    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    Masks13 = EE._masks(EE.GOALS_13); MasksTough = EE._masks(EE.GOALS_TOUGH)

    bank = TransitionBank.load_curated(str(BANK_PATH))
    particle_factories = {
        "persistence": lambda: PersistencePredictor(),
        "random": lambda: RandomPredictor(seed=0),
        "retrieval_k1_v2": lambda: NearestTransitionPredictor(bank),
        "retrieval_k5_cube_median_v2": lambda: RetrievalPredictor(
            bank, cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median"),
    }

    def _save():
        _atomic(out, RES)

    for m in OCC_MODELS:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        r_acc, raw_acc = EE.eval_occ_model(m, dev, ch_acc, pools, Dist13, DistTough, Masks13, MasksTough)
        r_roll, _ = EE.eval_occ_model(m, dev, ch_roll, [], Dist13, DistTough, Masks13, MasksTough)
        for kk in range(1, 5):
            key = f"rollout_accuracy_{kk}"
            if key in r_roll:
                r_acc[key] = r_roll[key]
        r_acc["wall_s"] = time.time() - t0
        out[m] = {"metrics": r_acc, "raw": raw_acc}
        _save()
        print(f"{m:32s} acc1={r_acc.get('accuracy_1')} slateN={r_acc['slateN']:.3f} "
             f"slateN_tough={r_acc['slateN_tough']:.3f} rollout4={r_acc.get('rollout_accuracy_4')} "
             f"({r_acc['wall_s']:.1f}s)", flush=True)

    for m in PARTICLE_MODELS:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        predictor = particle_factories[m]()
        r_acc, raw_acc = EE.eval_particle_model(m, bank, ch_acc, pools, Dist13, DistTough,
                                                Masks13, MasksTough, predictor_obj=predictor)
        predictor2 = particle_factories[m]()  # fresh instance: same seed as `random`'s RNG state
        r_roll, _ = EE.eval_particle_model(m, bank, ch_roll, [], Dist13, DistTough,
                                           Masks13, MasksTough, predictor_obj=predictor2)
        for kk in range(1, 5):
            key = f"rollout_accuracy_{kk}"
            if key in r_roll:
                r_acc[key] = r_roll[key]
        r_acc["wall_s"] = time.time() - t0
        out[m] = {"metrics": r_acc, "raw": raw_acc}
        _save()
        print(f"{m:32s} acc1={r_acc.get('accuracy_1')} slateN={r_acc['slateN']:.3f} "
             f"slateN_tough={r_acc['slateN_tough']:.3f} rollout4={r_acc.get('rollout_accuracy_4')} "
             f"mm={r_acc.get('moved_cube_mm_mean')} ({r_acc['wall_s']:.1f}s)", flush=True)

    print("\n=== bootstrap CIs (slateN_tough, 32 test_pools_v2 pools as replication unit) ===")
    ci_out = {}
    for m, d in out.items():
        raw = d["raw"]["slateN_tough_raw"]
        mean, lo, hi = CO.pool_bootstrap_ci(raw)
        ci_out[m] = dict(point=d["metrics"]["slateN_tough"], boot_mean=mean, ci95=[lo, hi])
        print(f"  {m:32s} slateN_tough={d['metrics']['slateN_tough']:.4f}  boot CI95=[{lo:.4f},{hi:.4f}]")

    winner = "retrieval_k5_cube_median_v2"
    for other in ("nfd_3ch_narrow_l20_v2", "linear_narrow_l20_v2_res64"):
        if winner in out and other in out:
            point, lo, hi, frac_gt0 = paired_delta_ci(
                out[other]["raw"]["slateN_tough_raw"], out[winner]["raw"]["slateN_tough_raw"])
            key = f"paired_delta_{winner}_minus_{other}"
            ci_out[key] = dict(point=point, ci95=[lo, hi], frac_boot_gt0=frac_gt0)
            print(f"  paired delta ({winner} - {other}): {point:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
                 f"frac(boot>0)={frac_gt0:.3f}")

    out["_bootstrap_ci"] = ci_out
    _save()
    print("wrote", RES)


if __name__ == "__main__":
    main()
