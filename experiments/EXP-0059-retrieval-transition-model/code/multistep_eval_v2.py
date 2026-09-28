"""EXP-0059 clean-data v2 rung: multi-step ranking headline on DS-0018
(`seqpools_v2_clean`, the ISS-010-fix multi-step candidate-sequence set that
replaces DS-0013/`seqpools_dsB`). Reuses `multistep_eval.py`'s own
`run_occ`/`run_particle`/`_finish`/`slate_capture_pool*` verbatim (no metric
redefined here) -- only the pool loader and the model/predictor wiring are
new, since DS-0018's clean copy lives in a different directory and needs no
legality-sidecar filtering (already clean by construction: whole SEQUENCES
with a bad step were removed at collection time, `split_clean_archive.py`).

**Pools with a removed sequence rank fewer than 64 candidates** (2,042/2,048
sequences survive across 32 pools, i.e. on average ~0.19 sequences removed
per pool -- a handful of pools have 63 instead of 64) -- `run_occ`/
`run_particle` already handle a variable candidate count per pool (`n_cand =
len(r0)`, skip only if `n_cand < 2`), so no code change was needed there;
this is stated explicitly in the record per the coordinator's instruction,
not silently absorbed.

CPU-only by design (`CUDA_VISIBLE_DEVICES=` set by the launch command, not
this script) so the 2 concurrent NFD-seed GPU trainings are not starved.

Usage:
    CUDA_VISIBLE_DEVICES= python -u multistep_eval_v2.py \\
        --models persistence nfd_3ch_narrow_l20_v2 linear_narrow_l20_v2_res64 \\
                 linear_narrow_l20_v2_res32 \\
        --particle-models retrieval_k5_cube_median_v2 retrieval_k1_v2
"""
import argparse
import glob
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
import eval_narrow as EN  # noqa: E402
import eval_extended as EE  # noqa: E402
import multistep_eval as ME  # noqa: E402  (run_occ, run_particle, _finish, slate_capture_pool*)

from simple_mpc.adapters import occ_from_particles, occ_for_scoring  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import (PersistencePredictor, NearestTransitionPredictor,  # noqa: E402
                                       RetrievalPredictor)

DS0018_CLEAN_DIR = REPO / "Genesis/data/narrow_l20_n20/seqpools_v2_clean"
BANK_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "bank_train_v2_curated.pt"
RES = Path(__file__).resolve().parents[1] / "results" / "multistep_eval_v2.json"
N_STEPS = 3
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)


def load_pools_clean():
    """DS-0018 (`seqpools_v2_clean`): already clean by construction (whole
    bad SEQUENCES removed at collection time) -- no legality-sidecar
    filtering needed, unlike `multistep_eval.py::load_pools`'s DS-0013 path.
    A pool missing a sequence simply has < 64 rows at every `chain_step`
    (the whole sequence, all 3 steps, was removed together), so
    `rows_by_step[k]` naturally comes out shorter for that pool -- no special
    casing required, `run_occ`/`run_particle` already tolerate `n_cand < 64`."""
    files = sorted(glob.glob(str(DS0018_CLEAN_DIR / "_*_data.pt")))
    if not files:
        raise FileNotFoundError(f"no files matched {DS0018_CLEAN_DIR}/_*_data.pt")
    pools = []
    n_seq_total = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        rows_by_step = {}
        for k in range(N_STEPS):
            ix = torch.nonzero(d["chain_step"] == k)[:, 0]
            order = torch.argsort(d["chain_env"][ix])
            rows_by_step[k] = ix[order]
        pools.append((d, rows_by_step))
        n_seq_total += len(rows_by_step[0])
    print(f"loaded {len(pools)} pools, {n_seq_total} sequences total "
         f"({n_seq_total / (len(pools) * 64) * 100:.1f}% of the nominal 64/pool)")
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--particle-models", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev} (CPU-only expected -- set CUDA_VISIBLE_DEVICES= before launching "
         f"to keep the concurrent NFD-seed GPU trainings unstarved)")
    res_path = Path(a.out) if a.out else RES
    raw_path = res_path.with_name(res_path.stem + "_raw.json")
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    raw_out = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    res_path.parent.mkdir(parents=True, exist_ok=True)

    pools = load_pools_clean()
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)

    def _save():
        tmp = str(res_path) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, res_path)
        tmp2 = str(raw_path) + ".tmp"; Path(tmp2).write_text(json.dumps(raw_out)); os.replace(tmp2, raw_path)

    for m in a.models:
        if m in out:
            print("skip (already scored):", m); continue
        t0 = time.time()
        r, raw = ME.run_occ(m, dev, pools, Dist13, DistTough)
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw; _save()
        print(f"{m:36s} terminal slateN {r['terminal_slateN']:.3f} slateN_tough "
             f"{r['terminal_slateN_tough']:.3f} step_slateN_tough {r['step_slateN_tough']} "
             f"step_rollout {r['step_rollout_accuracy']} ({r['wall_s']:.1f}s)", flush=True)

    bank = None
    if a.particle_models:
        bank = TransitionBank.load_curated(str(BANK_PATH))
    predictor_factories = {
        "persistence": lambda: PersistencePredictor(),
        "retrieval_k1_v2": lambda: NearestTransitionPredictor(bank),
        "retrieval_k5_cube_median_v2": lambda: RetrievalPredictor(
            bank, cfg=DistanceConfig(**_CAPPED), k=5, aggregation="cube_median"),
    }
    for m in a.particle_models:
        if m in out:
            print("skip (already scored):", m); continue
        if m not in predictor_factories:
            print("skip (not registered):", m); continue
        predictor = predictor_factories[m]()
        t0 = time.time()
        r, raw = ME.run_particle(m, predictor, pools, Dist13, DistTough)
        r["wall_s"] = time.time() - t0
        out[m] = r; raw_out[m] = raw; _save()
        print(f"{m:36s} terminal slateN {r['terminal_slateN']:.3f} slateN_tough "
             f"{r['terminal_slateN_tough']:.3f} step_slateN_tough {r['step_slateN_tough']} "
             f"step_rollout {r['step_rollout_accuracy']} ({r['wall_s']:.1f}s)", flush=True)
    print("wrote", res_path)


if __name__ == "__main__":
    main()
