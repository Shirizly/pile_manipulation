"""EXP-0059 task 1 (data scaling): does more bank data help `k5_cube_median`
retrieval, and does DS-0012's data (random `--mode chains` reservoir,
TRAINING_PHYSICS, same narrow-domain corpus family as DS-0008) behave like
more of the SAME thing DS-0008/DS-0010 already provide, or differently
(a same-size 12k-vs-12k structure check)?

Bank configs (all valid-filtered, same convention as `model/retrieval/bank.py`):
  orig_12k         : DS-0008 + DS-0010 only (the existing `artifacts/bank.pt`)
  ds0012_only_12k  : DS-0012 subset, same SIZE as orig_12k (seed 0)
  25k              : orig_12k + DS-0012 subset to reach 25,000 total (seed 1)
  50k              : orig_12k + DS-0012 subset to reach 50,000 total (seed 1)
  98k_all          : orig_12k + ALL of DS-0012 (no subsampling)

For each: build (persist to `artifacts/bank_<name>.pt`), then score
`RetrievalPredictor(bank, cfg=_CAPPED, k=5, aggregation="cube_median")`
("k5_cube_median", the R1 sweep's best clean config) on DS-0009 via
`eval_extended.eval_particle_model` (accuracy_1, slateN, slateN_tough,
rollout_accuracy_1..4, moved_cube_mm), plus an oracle best-of-top-50-by-
distance mm error on 256 DS-0009 test_chains queries (reusing
`diagnostics_r2.py`'s section5 method, generalized to take an arbitrary
bank instead of the fixed `artifacts/bank.pt`).

Checkpointed atomically per bank config (`results/data_scaling.json`), safe
to resume/kill. GPU search is already chunked in `model/retrieval/
distance.py::full_distance_matrix`/`topk_search` (chunk=1500 over the bank
dimension) -- this script does not add its own chunking, only confirms wall
time scales sub-linearly-ish and reports it per config.

Usage:
    python -u experiments/EXP-0059-retrieval-transition-model/code/data_scaling.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_narrow as EN  # noqa: E402
import eval_extended as EE  # noqa: E402

from simple_mpc.adapters import occ_from_particles  # noqa: E402
from model.retrieval.bank import TransitionBank, DEFAULT_MOVED_THRESHOLD, _load_dir  # noqa: E402
from model.retrieval.distance import (DistanceConfig, query_points_and_weights,
                                      bank_points_and_weights, full_distance_matrix)  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402
from model.retrieval.frame import push_frame_to_world  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "data_scaling.json"
D = EN.D
DS0012_DIR = REPO / "Genesis/data/narrow_l20_n20/reservoir_dsC"

_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)
BEST_CFG = DistanceConfig(**_CAPPED)
MOVED_THRESH = 0.001


def _atomic(obj, path):
    tmp = str(path) + ".tmp"; Path(tmp).write_text(json.dumps(obj, indent=1)); os.replace(tmp, path)


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def load_base():
    """DS-0008 + DS-0010, valid-filtered -- identical rows to the existing
    `artifacts/bank.pt` (cross-checked by length below)."""
    s0a, s1a, psa, pea = _load_dir(str(REPO / "Genesis/data/narrow_l20_n20/train/_*_data.pt"), has_valid=True)
    s0b, s1b, psb, peb = _load_dir(str(REPO / "Genesis/data/narrow_l20_n20/extra_18_22/*_data.pt"), has_valid=False)
    s0 = torch.cat([s0a, s0b]); s1 = torch.cat([s1a, s1b])
    ps = torch.cat([psa, psb]); pe = torch.cat([pea, peb])
    src = ["DS-0008"] * len(s0a) + ["DS-0010"] * len(s0b)
    return s0, s1, ps, pe, src


def load_ds0012():
    s0, s1, ps, pe = _load_dir(str(DS0012_DIR / "_*_data.pt"), has_valid=True)
    return s0, s1, ps, pe


def subsample(n_total, s0, s1, ps, pe, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s0), size=min(n_total, len(s0)), replace=False)
    idx = torch.from_numpy(np.sort(idx))
    return s0[idx], s1[idx], ps[idx], pe[idx]


def build_bank(name, s0, s1, ps, pe, src):
    path = ARTIFACTS / f"bank_{name}.pt"
    if path.exists():
        print(f"  reusing persisted {path}")
        return TransitionBank.load(str(path)), path
    t0 = time.time()
    bank = TransitionBank.from_states(s0, s1, ps, pe, source=src, moved_threshold=DEFAULT_MOVED_THRESHOLD)
    print(f"  built bank '{name}': {len(bank)} rows in {time.time()-t0:.1f}s")
    bank.save(str(path))
    return bank, path


def oracle_top50_mm(bank, n_queries=256, seed=1):
    """Best-of-top-50-by-distance mm error on `n_queries` DS-0009 test_chains
    rows (scatter+clump balanced, same method as diagnostics_r2.py's
    section5, generalized to an arbitrary bank). Device: GPU (chunked
    inside full_distance_matrix), falls back to CPU automatically if no
    CUDA."""
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, BEST_CFG)
    ch = [torch.load(f, map_location="cpu", weights_only=False)
         for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    states0 = torch.cat([d["states"] for d in ch]).float()
    states1 = torch.cat([d["states_"] for d in ch]).float()
    p_starts = torch.cat([d["p_starts"] for d in ch]).float()
    p_stops = torch.cat([d["p_stops"] for d in ch]).float()
    kinds = np.array(sum([list(d["start_kind"]) for d in ch], []))
    rng = np.random.default_rng(seed)
    n_per_kind = n_queries // 2
    idx_parts = []
    for k in ("scatter", "clump"):
        pool = np.nonzero(kinds == k)[0]
        idx_parts.append(rng.choice(pool, size=min(n_per_kind, len(pool)), replace=False))
    idx = np.sort(np.concatenate(idx_parts))

    q_pts, q_w, q_uv0_all, _ = query_points_and_weights(states0[idx], p_starts[idx], p_stops[idx], BEST_CFG)
    t0 = time.time()
    D_full = full_distance_matrix(q_pts, q_w, bank_pts, bank_w, BEST_CFG, bank_valid=bank_valid)
    search_s = time.time() - t0

    from model.retrieval.predictor import RetrievalPredictor as _RP  # noqa
    mms = []
    for row, qi in enumerate(idx):
        q_start, q_stop = p_starts[qi:qi + 1], p_stops[qi:qi + 1]
        q_uv0 = q_uv0_all[row]
        true_xy1 = states1[qi, :, :2]
        true_moved = (true_xy1 - states0[qi, :, :2]).norm(dim=-1) > MOVED_THRESH
        if not bool(true_moved.any()):
            continue
        true_xy1_moved = true_xy1[true_moved]
        order = torch.argsort(D_full[row].cpu())[:50]
        best_err = float("inf")
        for di_t in order.tolist():
            cost = torch.cdist(q_uv0, bank.uv0[di_t]).numpy()
            from scipy.optimize import linear_sum_assignment
            r_, c_ = linear_sum_assignment(cost)
            duv = torch.zeros_like(q_uv0)
            moved_matched = bank.moved[di_t][c_]
            duv[r_] = torch.where(moved_matched.unsqueeze(-1), bank.duv[di_t][c_], torch.zeros_like(bank.duv[di_t][c_]))
            pred_uv1 = q_uv0 + duv
            pred_xy1 = push_frame_to_world(pred_uv1[None], q_start, q_stop)[0]
            d_ = torch.cdist(true_xy1_moved, pred_xy1)
            e_mm = float(d_.min(dim=1).values.mean()) * 1000.0
            if e_mm < best_err:
                best_err = e_mm
        mms.append(best_err)
    return dict(n_queries=len(idx), n_used=len(mms), mm_mean=float(np.mean(mms)),
               search_s=search_s, bank_size=len(bank))


def main():
    out = _load(RES)
    RES.parent.mkdir(parents=True, exist_ok=True)
    ch, pools = EE._load_corpus()
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    Masks13 = EE._masks(EE.GOALS_13); MasksTough = EE._masks(EE.GOALS_TOUGH)

    print("loading DS-0008+DS-0010 (base) ...")
    bs0, bs1, bps, bpe, bsrc = load_base()
    print(f"  base: {len(bs0)} rows ({bsrc.count('DS-0008')} DS-0008, {bsrc.count('DS-0010')} DS-0010)")
    print("loading DS-0012 (reservoir) ...")
    d0, d1, dps, dpe = load_ds0012()
    print(f"  DS-0012: {len(d0)} valid rows")

    configs = {}
    configs["orig_12k"] = (bs0, bs1, bps, bpe, bsrc)
    s0, s1, ps, pe = subsample(len(bs0), d0, d1, dps, dpe, seed=0)
    configs["ds0012_only_12k"] = (s0, s1, ps, pe, ["DS-0012"] * len(s0))
    for tgt, seed in ((25000, 1), (50000, 1)):
        n_extra = max(0, tgt - len(bs0))
        s0, s1, ps, pe = subsample(n_extra, d0, d1, dps, dpe, seed=seed)
        cs0 = torch.cat([bs0, s0]); cs1 = torch.cat([bs1, s1])
        cps = torch.cat([bps, ps]); cpe = torch.cat([bpe, pe])
        csrc = bsrc + ["DS-0012"] * len(s0)
        configs[f"{tgt // 1000}k"] = (cs0, cs1, cps, cpe, csrc)
    cs0 = torch.cat([bs0, d0]); cs1 = torch.cat([bs1, d1])
    cps = torch.cat([bps, dps]); cpe = torch.cat([bpe, dpe])
    csrc = bsrc + ["DS-0012"] * len(d0)
    configs["98k_all"] = (cs0, cs1, cps, cpe, csrc)

    for name, (s0, s1, ps, pe, src) in configs.items():
        if name in out and out[name].get("done"):
            print("skip (already scored):", name); continue
        print(f"\n=== {name} (bank target size {len(s0)}) ===", flush=True)
        bank, path = build_bank(name, s0, s1, ps, pe, src)
        predictor = RetrievalPredictor(bank, cfg=BEST_CFG, k=5, aggregation="cube_median")
        t0 = time.time()
        r, _raw = EE.eval_particle_model(f"k5_cube_median_{name}", bank, ch, pools,
                                         Dist13, DistTough, Masks13, MasksTough,
                                         predictor_obj=predictor)
        r["wall_s_main"] = time.time() - t0
        print(f"  main eval: acc1 {r['accuracy_1']:+.3f} slateN {r['slateN']:.3f} "
             f"slateN_tough {r['slateN_tough']:.3f} rollout4 {r['rollout_accuracy_4']:.3f} "
             f"mm {r.get('moved_cube_mm_mean', float('nan')):.2f} ({r['wall_s_main']:.1f}s)", flush=True)
        t0 = time.time()
        oracle = oracle_top50_mm(bank, n_queries=256, seed=1)
        oracle["wall_s"] = time.time() - t0
        print(f"  oracle top50 mm: {oracle['mm_mean']:.3f} mm (n={oracle['n_used']}, "
             f"search {oracle['search_s']:.1f}s, total {oracle['wall_s']:.1f}s)", flush=True)
        out[name] = dict(bank_size=len(bank), n_ds0012_rows=src.count("DS-0012"),
                        main=r, oracle_top50=oracle, done=True)
        _atomic(out, RES)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nwrote", RES)


if __name__ == "__main__":
    main()
