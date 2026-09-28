"""model/retrieval_nfd/donors.py -- donor bank + chain-aware top-k / random
selection for EXP-0059 section 8 ("NFD with a retrieved reference").

Builds the SAME transition bank `model.retrieval.bank.TransitionBank` builds
(DS-0008 valid rows + all DS-0010 rows), but ALSO keeps a parallel per-row
"chain key" array so donor selection can exclude the query's own chain:

    DS-0008 row -> f"ds0008::{data_file}::{chain_env}"
    DS-0010 row -> f"ds0010::{source_file}"

`bank.py`'s own loader collapses this identity away (only keeps a coarse
"DS-0008"/"DS-0010" `source` tag), so this module reloads the two corpora
itself, mirroring `bank.py::_load_dir`'s glob pattern and valid-filter
EXACTLY (same sorted-glob order, same valid-mask-for-DS-0008-only rule) so
`TransitionBank.from_states(...)` built here has identical row order/content
to `TransitionBank.build()` -- checked by `build_bank_with_keys`'s own
assertion against a fresh `TransitionBank.build()` bank size.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import torch

from model.retrieval.bank import TransitionBank, DEFAULT_MOVED_THRESHOLD
from model.retrieval.distance import (DistanceConfig, query_points_and_weights,
                                       bank_points_and_weights, topk_search,
                                       corridor_weight_mask)

REPO = Path(__file__).resolve().parents[2]
DS0008_DIR = REPO / "Genesis/data/narrow_l20_n20/train"
DS0010_DIR = REPO / "Genesis/data/narrow_l20_n20/extra_18_22"


def _file_list_for_split(dir_path: Path, split: str, val_pct: int = 5, test_pct: int = 5) -> List[str]:
    """The exact file order `PileSweepData.__init__` iterates for ONE of its
    `paths` entries and ONE split -- same classmethods, same args, so
    `run_idx` (index into `ds.runs`) maps 1:1 to `result[run_idx]` when this
    is called for every `paths` entry in the SAME order `PileSweepData3Ch`
    was constructed with."""
    from Genesis.training.dataset import PileSweepData
    run_files = PileSweepData._collect_run_paths(None, dir_path)
    run_files = PileSweepData._filter_split(run_files, split, val_pct, test_pct)
    return [str(f) for f, _c in run_files]


def build_query_rows_from_pilesweepdata(split: str, val_pct: int = 5, test_pct: int = 5):
    """Builds the QUERY side (channels 0-2 + target, bit-exact with the
    narrow NFD's own training data) directly from `Baselines.NFD.nfd_lib.
    PileSweepData3Ch` -- NOT a from-scratch renderer -- per coordinator
    direction (2026-09-28): "channels 0-2 must be EXACTLY the narrow NFD's
    inputs". Also recovers each row's raw (states0, states1, p_start, p_stop)
    and chain key (same string format `build_bank_with_keys` uses) by
    replicating `PileSweepData`'s own file-order/split logic (`_file_list_
    for_split`), so `run_idx -> file path` is exact, not order-matched
    against a second, independent loader.

    -> dict: occ0/r_start/r_stop/target (N,H,W) float32, states0/states1
    (N,20,7), p_starts/p_stops (N,3), angles (N,), chain_keys (list[str]).
    """
    from Baselines.NFD.nfd_lib import PileSweepData3Ch
    ds = PileSweepData3Ch(paths=["narrow_l20_n20/train", "narrow_l20_n20/extra_18_22"],
                           split=split, val_pct=val_pct, test_pct=test_pct, resolution_scale=0.5)
    files_ds0008 = _file_list_for_split(DS0008_DIR, split, val_pct, test_pct)
    files_ds0010 = _file_list_for_split(DS0010_DIR, split, val_pct, test_pct)
    file_list = files_ds0008 + files_ds0010
    assert len(file_list) == len(ds.runs), (
        f"file_list ({len(file_list)}) != ds.runs ({len(ds.runs)}) for split={split!r} -- "
        "PileSweepData's own path/split iteration order drifted from this helper's replica.")

    N = len(ds)
    occ0 = torch.zeros(N, 64, 64); r_start = torch.zeros(N, 64, 64)
    r_stop = torch.zeros(N, 64, 64); target = torch.zeros(N, 64, 64)
    states0 = torch.zeros(N, 20, 7); states1 = torch.zeros(N, 20, 7)
    p_starts = torch.zeros(N, 3); p_stops = torch.zeros(N, 3); angles = torch.zeros(N)
    chain_keys = []
    for i in range(N):
        (inp, _phys), tgt = ds[i]
        occ0[i] = inp[0]; r_start[i] = inp[1]; r_stop[i] = inp[2]; target[i] = tgt
        run_idx = ds._run_lookup[i]
        sample_idx = i - ds._offsets[run_idx]
        run = ds.runs[run_idx]
        states0[i] = run["states"][sample_idx]
        states1[i] = run["states_"][sample_idx]
        p_starts[i] = run["p_starts"][sample_idx]
        p_stops[i] = run["p_stops"][sample_idx]
        angles[i] = run["angles"][sample_idx]
        f = file_list[run_idx]
        if "chain_env" in run:
            chain_keys.append(f"ds0008::{f}::{int(run['chain_env'][sample_idx])}")
        else:
            chain_keys.append(f"ds0010::{run['source_file']}")
    return dict(occ0=occ0, r_start=r_start, r_stop=r_stop, target=target,
                states0=states0, states1=states1, p_starts=p_starts, p_stops=p_stops,
                angles=angles, chain_keys=chain_keys)


def frozen_distance_config() -> DistanceConfig:
    """The EXP-0059 R1 sweep-chosen best retrieval key (see
    `experiments/EXP-0059-*/results/offline_eval_retrieval_bestconfigs.json`,
    shared by the shortlisted `k5_*` configs): capped+corridor-weighted
    chamfer, cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0. Used here
    with k=1/3 (not k=5) since section 8 wants top-1 (eval) / top-3 (train
    donor pool), not the k=5 hedged-occupancy aggregation those configs were
    tuned for -- only the DISTANCE itself is reused, not the aggregation."""
    return DistanceConfig(
        window_u=(-0.005, 0.045), window_v=(-0.05, 0.05),
        cap=0.02, mismatch_penalty=0.04,
        corridor_v_halfwidth=0.025, corridor_u_pad=0.005, corridor_weight=3.0,
        wall_weight=0.0, reduction="mean",
    )


def _load_ds0008_rows() -> Tuple[torch.Tensor, ...]:
    states0, states1, p_starts, p_stops, angles, chain_keys, bank_valid = [], [], [], [], [], [], []
    split_tag = []
    files = sorted(glob.glob(str(DS0008_DIR / "_*_data.pt")))
    from Genesis.training.dataset import PileSweepData
    run_files = PileSweepData._collect_run_paths(None, DS0008_DIR)
    split_map = {}
    for sp in ("train", "val", "test"):
        for f, _c in PileSweepData._filter_split(run_files, sp, 5, 5):
            split_map[str(f)] = sp
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = len(d["states"])
        v = d["valid"] if "valid" in d else torch.ones(n, dtype=torch.bool)
        chain_env = d["chain_env"]
        sp = split_map[f]
        states0.append(d["states"].float())
        states1.append(d["states_"].float())
        p_starts.append(d["p_starts"].float())
        p_stops.append(d["p_stops"].float())
        angles.append(d["angles"].float())
        bank_valid.append(v)
        split_tag.extend([sp] * n)
        chain_keys.extend(f"ds0008::{f}::{int(ce)}" for ce in chain_env.tolist())
    return (torch.cat(states0), torch.cat(states1), torch.cat(p_starts), torch.cat(p_stops),
            torch.cat(angles), torch.cat(bank_valid), chain_keys, split_tag,
            ["DS-0008"] * sum(len(s) for s in states0))


def _load_ds0010_rows() -> Tuple[torch.Tensor, ...]:
    states0, states1, p_starts, p_stops, angles, chain_keys = [], [], [], [], [], []
    split_tag = []
    files = sorted(glob.glob(str(DS0010_DIR / "*_data.pt")))
    from Genesis.training.dataset import PileSweepData
    run_files = PileSweepData._collect_run_paths(None, DS0010_DIR)
    split_map = {}
    for sp in ("train", "val", "test"):
        for f, _c in PileSweepData._filter_split(run_files, sp, 5, 5):
            split_map[str(f)] = sp
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        n = len(d["states"])
        src = d["source_file"]
        sp = split_map[f]
        states0.append(d["states"].float())
        states1.append(d["states_"].float())
        p_starts.append(d["p_starts"].float())
        p_stops.append(d["p_stops"].float())
        angles.append(d["angles"].float())
        split_tag.extend([sp] * n)
        chain_keys.extend([f"ds0010::{src}"] * n)
    n_total = sum(len(s) for s in states0)
    return (torch.cat(states0), torch.cat(states1), torch.cat(p_starts), torch.cat(p_stops),
            torch.cat(angles), torch.ones(n_total, dtype=torch.bool), chain_keys, split_tag,
            ["DS-0010"] * n_total)


def load_all_rows():
    """-> dict with states0/states1/p_starts/p_stops/angles (N,...),
    bank_valid (N,) bool, chain_keys (list[str] len N), split (list[str] len
    N, "train"/"val"/"test", the SAME file-level split `nfd_3ch_narrow_l20.
    yaml` uses: val_pct=5/test_pct=5 over these same two directories),
    source (list[str]). Every row of the actual narrow-NFD training corpus
    (unfiltered by `valid`), N ~= 6144 + 5777 = 11921."""
    a = _load_ds0008_rows()
    b = _load_ds0010_rows()
    keys = ("states0", "states1", "p_starts", "p_stops", "angles", "bank_valid",
            "chain_keys", "split", "source")
    out = {}
    for i, k in enumerate(keys):
        if k in ("chain_keys", "split", "source"):
            out[k] = a[i] + b[i]
        else:
            out[k] = torch.cat([a[i], b[i]])
    return out


def build_bank_with_keys(rows: dict) -> Tuple[TransitionBank, List[str]]:
    """Bank = valid rows only (matches `TransitionBank.build()` exactly --
    asserted below); `bank_chain_keys` is the parallel chain-key array,
    aligned 1:1 with the bank's own row order."""
    valid = rows["bank_valid"]
    idx = torch.nonzero(valid, as_tuple=True)[0]
    source = [rows["source"][int(i)] for i in idx]
    bank = TransitionBank.from_states(
        rows["states0"][idx], rows["states1"][idx], rows["p_starts"][idx], rows["p_stops"][idx],
        source=source, moved_threshold=DEFAULT_MOVED_THRESHOLD,
    )
    bank_chain_keys = [rows["chain_keys"][int(i)] for i in idx]
    ref = TransitionBank.build()
    assert len(bank) == len(ref), (
        f"donors.py bank ({len(bank)}) does not match TransitionBank.build() "
        f"({len(ref)}) -- loader order/filter drifted from bank.py."
    )
    return bank, bank_chain_keys


def _chain_keys_to_ids(*key_lists: List[str]) -> List[torch.Tensor]:
    """Map one or more parallel lists of chain-key strings to a SHARED
    integer id space (equal strings across lists get equal ids), so chain
    exclusion becomes a vectorised integer-tensor comparison instead of a
    per-row Python string-equality loop."""
    vocab: dict = {}
    out = []
    for keys in key_lists:
        ids = torch.empty(len(keys), dtype=torch.long)
        for i, k in enumerate(keys):
            j = vocab.setdefault(k, len(vocab))
            ids[i] = j
        out.append(ids)
    return out


def topk_donors_excluding_chain(states0_xy: torch.Tensor, p_starts: torch.Tensor, p_stops: torch.Tensor,
                                 query_chain_keys: List[str], bank: TransitionBank,
                                 bank_chain_keys: List[str], cfg: DistanceConfig,
                                 k: int = 3, search_k: int = 64, chunk: int = 1000,
                                 bank_chunk: int = 2000, device: str = None) -> torch.Tensor:
    """states0_xy: (N, n, 2) world metres (query cube xy, any n). -> (N, k)
    long bank indices, the top-k by the frozen chamfer distance EXCLUDING any
    bank row sharing the query's own chain key. Pads with -1 if fewer than k
    survive (rare; window is wide relative to per-chain group sizes).

    FULLY VECTORISED chain-exclusion (no per-row Python loop / `.tolist()`
    device-sync calls -- the previous version's dominant cost, not the
    chamfer search itself): chain keys are mapped to integer ids once, then
    for each query CHUNK a single `(chunk, search_k)` boolean mask + a
    stable `argsort` push excluded candidates to the end while preserving
    the search's own ascending-distance order among the survivors."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, cfg)
    query_ids, bank_ids = _chain_keys_to_ids(query_chain_keys, bank_chain_keys)
    bank_ids_dev = bank_ids.to(device)
    N = states0_xy.shape[0]
    out = torch.full((N, k), -1, dtype=torch.long)
    search_k = min(search_k, len(bank))
    import time as _time
    t0 = _time.time()
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        q_pts, q_w, _uv, _valid = query_points_and_weights(
            states0_xy[lo:hi], p_starts[lo:hi], p_stops[lo:hi], cfg)
        idx, _dist = topk_search(q_pts, q_w, bank_pts, bank_w, cfg, k=search_k,
                                  chunk=bank_chunk, device=device, bank_valid=bank_valid)
        idx = idx.to(device)                                    # (chunk, search_k)
        qid = query_ids[lo:hi].to(device).unsqueeze(1)           # (chunk, 1)
        cand_id = bank_ids_dev[idx]                              # (chunk, search_k)
        excluded = cand_id == qid                                # True -> same chain, drop
        # stable sort: valid (0) candidates first, in their original (ascending-distance) order
        order = torch.argsort(excluded.to(torch.uint8), dim=1, stable=True)
        idx_sorted = torch.gather(idx, 1, order)
        excl_sorted = torch.gather(excluded, 1, order)
        picked = idx_sorted[:, :k].clone()
        picked[excl_sorted[:, :k]] = -1     # still-excluded slot (fewer than k survivors) -> -1
        out[lo:hi] = picked.cpu()
        print(f"[donors] topk {hi}/{N} ({_time.time() - t0:.1f}s)", flush=True)
    return out


def random_donor_excluding_chain(query_chain_keys: List[str], bank: TransitionBank,
                                  bank_chain_keys: List[str], corridor_bucket: torch.Tensor,
                                  query_bucket: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """(N,) long bank indices: a uniform-random bank row, excluding rows that
    share the query's chain key, PREFERRING (not requiring) a row with the
    SAME corridor-cube-count bucket as the query -- a cheap stand-in for the
    design doc's "random donor ... with the same number of corridor cubes"
    control (section 3), so the control isolates "extra input capacity" from
    "right kind of information" rather than from "how much stuff is nearby"."""
    g = torch.Generator().manual_seed(seed)
    N = query_bucket.shape[0]
    T = len(bank)
    out = torch.zeros(N, dtype=torch.long)
    # bucket -> list of bank indices
    buckets = {}
    for i in range(T):
        buckets.setdefault(int(corridor_bucket[i]), []).append(i)
    n_fallback = 0
    for i in range(N):
        qk = query_chain_keys[i]
        qb = int(query_bucket[i])
        cand = [c for c in buckets.get(qb, []) if bank_chain_keys[c] != qk]
        if not cand:
            n_fallback += 1
            # rejection sampling over the WHOLE bank (rare: empty bucket
            # match), bounded tries before giving up the chain-exclusion
            # guarantee rather than an O(T) list scan per query.
            for _try in range(50):
                c = int(torch.randint(T, (1,), generator=g))
                if bank_chain_keys[c] != qk:
                    cand = [c]
                    break
            else:
                cand = [int(torch.randint(T, (1,), generator=g))]
        pick = cand[int(torch.randint(len(cand), (1,), generator=g))]
        out[i] = pick
    if n_fallback:
        print(f"[donors] random-donor bucket-empty fallback for {n_fallback}/{N} rows")
    return out


def corridor_cube_count(uv: torch.Tensor, push_len: torch.Tensor, cfg: DistanceConfig) -> torch.Tensor:
    """(N, n, 2), (N,) -> (N,) int: number of cubes inside the corridor mask
    (weight > 1) -- used to bucket-match the random-donor control."""
    w = corridor_weight_mask(uv, push_len, cfg)
    return (w > 1.0).sum(dim=-1)
