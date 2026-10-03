"""Benchmark compute time: retrieval transition model vs NFD, and retrieval
search scaling with bank size (EXP-0059 coordinator request, 2026-09-28).

Timing methodology follows `Baselines/common/benchmark_time.py`: warm-up
iterations discarded, `torch.cuda.synchronize()` immediately before/after
each timed region, median + IQR over repeats (`_median_iqr`, reused here).

Sections
--------
1. End-to-end latency: retrieval (nn1, cube_median k=5) vs NFD (single seed,
   3-seed ensemble) at B in {1, 64, 512}, GPU (+CPU for B=1). Per-stage
   breakdown for retrieval at B=64.
2. Scaling with bank size (~10k..200k) using the Sean corpus (n20 cube
   files only -- see NOTE below), search-only and end-to-end at B=64.
3. Sub-linear search prototype: sklearn BallTree over a flattened push-frame
   descriptor, top-M candidates + exact chamfer re-rank, query time and
   recall@5 vs N.

NOTE on Sean corpus / n!=20: `TransitionBank`'s per-object fields are fixed
at n=20 (`bank.py`'s own docstring). Sean's corpus mixes n20/n50/n100 cube
counts; loading n50/n100 files into this fixed-n bank layout would need
padding/truncation logic this bank doesn't have. Since this section is
TIMING-ONLY (content is irrelevant, only bank SIZE matters for search cost),
we use only the n20-cube Sean files (150 files, 512 rows each = 76,800 raw
transitions) and, to reach target sizes beyond what's on disk (100k, 200k),
tile (repeat) the loaded tensors in memory rather than loading more files.
This is stated explicitly, not hidden: it means the 100k/200k rows contain
duplicated content, which does not affect search/Hungarian cost (both are
pure functions of tensor shape, not content).
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

REPO = "/home/alon/Code/pile_manipulation"
sys.path.insert(0, REPO)
os.chdir(REPO)

from Baselines.common.benchmark_time import _median_iqr, check_gpu_contention  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402
from model.retrieval.distance import DistanceConfig, bank_points_and_weights  # noqa: E402
from model.retrieval.interaction import interaction_set  # noqa: E402
from simple_mpc.adapters import make_occ_adapter, occ_from_particles  # noqa: E402

CURATED_BANK = f"{REPO}/experiments/EXP-0059-retrieval-transition-model/artifacts/bank_train_v2_curated.pt"
TEST_POOLS = f"{REPO}/Genesis/data/narrow_l20_n20/test_pools_v2/pools_0.pt"
SEAN_N20_GLOB = f"{REPO}/Genesis/data/Sean/**/n20/**/*_data.pt"

WARMUP = 5
REPEATS = 20


def timed(fn, warmup=WARMUP, repeats=REPEATS, device="cuda"):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    ms = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000.0)
    return _median_iqr(ms)


def load_test_batch(n):
    d = torch.load(TEST_POOLS, map_location="cpu", weights_only=False)
    return d["states"][:n], d["p_starts"][:n], d["p_stops"][:n]


def make_action(p_start, p_stop):
    return torch.cat([p_start[:, :2], p_stop[:, :2]], dim=-1)


def retrieval_end_to_end(pred, states0, p_start, p_stop, device):
    def run():
        out = pred.predict_particles(states0, p_start, p_stop)
        occ = occ_from_particles(out.to(device))
        torch.cuda.synchronize() if device == "cuda" else None
        return occ
    return run


def nfd_end_to_end(adapter, states0, p_start, p_stop, device):
    occ0 = occ_from_particles(states0.to(device))
    act = make_action(p_start, p_stop).to(device)

    def run():
        with torch.no_grad():
            return adapter.predict_step(occ0, act)
    return run


def nfd_ensemble_end_to_end(adapters, states0, p_start, p_stop, device):
    occ0 = occ_from_particles(states0.to(device))
    act = make_action(p_start, p_stop).to(device)

    def run():
        with torch.no_grad():
            outs = [a.predict_step(occ0, act) for a in adapters]
            return torch.stack(outs).mean(0)
    return run


def section1(results, gpu_busy):
    print("=== Section 1: end-to-end latency ===")
    bank = TransitionBank.load_curated(CURATED_BANK)
    print(f"curated bank rows: {len(bank)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred_nn1 = RetrievalPredictor(bank, k=1, aggregation="nn1", device=device)
    pred_k5 = RetrievalPredictor(bank, k=5, aggregation="cube_median", device=device)

    nfd_seeds = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1",
                 "nfd_3ch_narrow_l20_v2_seed2"]
    nfd_adapters = [make_occ_adapter(mid, device=device) for mid in nfd_seeds]

    out = {}
    for B in (1, 64, 512):
        states0, p_start, p_stop = load_test_batch(B)
        row = {}
        for tag, pred in (("retrieval_nn1", pred_nn1), ("retrieval_k5_cube_median", pred_k5)):
            r = timed(retrieval_end_to_end(pred, states0, p_start, p_stop, device),
                      device="cpu")  # retrieval predict_particles runs on CPU internally (see below)
            row[tag] = {"total_ms": r, "per_candidate_us":
                        {k: v * 1000.0 / B for k, v in r.items() if k != "n"}}
        r = timed(nfd_end_to_end(nfd_adapters[0], states0, p_start, p_stop, device), device=device)
        row["nfd_single"] = {"total_ms": r, "per_candidate_us":
                             {k: v * 1000.0 / B for k, v in r.items() if k != "n"}}
        r = timed(nfd_ensemble_end_to_end(nfd_adapters, states0, p_start, p_stop, device), device=device)
        row["nfd_ensemble3"] = {"total_ms": r, "per_candidate_us":
                               {k: v * 1000.0 / B for k, v in r.items() if k != "n"}}
        if B == 1:
            # cheap CPU check for retrieval (already CPU-bound) and NFD-CPU
            nfd_cpu_adapter = make_occ_adapter(nfd_seeds[0], device="cpu")
            r = timed(nfd_end_to_end(nfd_cpu_adapter, states0, p_start, p_stop, "cpu"), device="cpu")
            row["nfd_single_cpu"] = {"total_ms": r}
        out[f"B{B}"] = row
        print(f"B={B}: " + ", ".join(f"{k}={v['total_ms']['median']:.2f}ms" for k, v in row.items()))

    # per-stage breakdown for retrieval nn1 at B=64
    states0, p_start, p_stop = load_test_batch(64)
    states0c, p_start_c, p_stop_c = states0.cpu(), p_start.cpu().float(), p_stop.cpu().float()
    stage_search = timed(lambda: pred_nn1._search(states0c, p_start_c, p_stop_c), device="cpu")

    idx, dist, uv0, q_valid, q_in_set = pred_nn1._search(states0c, p_start_c, p_stop_c)
    from model.retrieval.frame import yaw_from_quat
    yaw0_w = yaw_from_quat(states0c[..., 3:7])

    def transfer_loop():
        for b in range(64):
            if not bool(q_valid[b]):
                continue
            pred_nn1._transfer_one(uv0[b], int(idx[b][0]), q_in_set[b])
    stage_transfer = timed(transfer_loop, device="cpu")

    def apply_loop():
        for b in range(64):
            duv, dyaw = pred_nn1._transfer_one(uv0[b], int(idx[b][0]), q_in_set[b])
            pred_nn1._apply_delta(states0c[b], uv0[b], p_start_c[b:b + 1], p_stop_c[b:b + 1],
                                  duv, dyaw, yaw0_w[b])
    stage_apply = timed(apply_loop, device="cpu")

    stage_raster = timed(lambda: occ_from_particles(states0c.to(device)), device=device)

    out["stage_breakdown_B64_nn1_ms"] = {
        "search": stage_search, "hungarian_transfer_loop": stage_transfer,
        "transfer_plus_apply_delta": stage_apply, "rasterise": stage_raster,
    }
    print("stage breakdown (B=64, nn1): search={:.2f}ms transfer_loop={:.2f}ms "
         "transfer+apply={:.2f}ms rasterise={:.2f}ms".format(
             stage_search["median"], stage_transfer["median"],
             stage_apply["median"], stage_raster["median"]))
    results["section1_end_to_end"] = out
    results["gpu_contention_at_start"] = gpu_busy


def load_sean_n20(max_files=150):
    files = sorted(glob.glob(SEAN_N20_GLOB, recursive=True))[:max_files]
    assert files, "no Sean n20 files found"
    t0 = time.perf_counter()
    states0, states1, p_starts, p_stops = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        states0.append(d["states"].float())
        states1.append(d["states_"].float())
        p_starts.append(d["p_starts"].float())
        p_stops.append(d["p_stops"].float())
    load_s = time.perf_counter() - t0
    states0, states1 = torch.cat(states0), torch.cat(states1)
    p_starts, p_stops = torch.cat(p_starts), torch.cat(p_stops)
    return states0, states1, p_starts, p_stops, len(files), load_s


def tile_to(t, n):
    reps = math.ceil(n / t.shape[0])
    return t.repeat(reps, *([1] * (t.dim() - 1)))[:n]


def section2(results):
    print("=== Section 2: scaling with bank size (Sean n20 corpus) ===")
    states0, states1, p_starts, p_stops, n_files, load_s = load_sean_n20(150)
    print(f"loaded {n_files} Sean n20 files -> {len(states0)} raw rows in {load_s:.2f}s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_states0, test_p_start, test_p_stop = load_test_batch(64)  # fixed B=64 query batch

    sizes = [10_000, 25_000, 50_000, 100_000, 200_000]
    out = {"n_sean_files_loaded": n_files, "raw_rows_loaded": len(states0),
          "disk_load_seconds": load_s, "note": "n20-only Sean files; 100k/200k rows are "
          "in-memory tiled repeats of the 76,800 loaded rows (timing-only, content duplicated)"}
    per_size = {}
    for N in sizes:
        s0 = tile_to(states0, N)
        s1 = tile_to(states1, N)
        ps = tile_to(p_starts, N)
        pe = tile_to(p_stops, N)
        t0 = time.perf_counter()
        bank = TransitionBank.from_states(s0, s1, ps, pe, source=["sean"] * N)
        build_s = time.perf_counter() - t0
        try:
            pred = RetrievalPredictor(bank, k=1, aggregation="nn1", device=device)
        except RuntimeError as e:
            per_size[N] = {"error": str(e), "build_seconds": build_s}
            print(f"N={N}: OOM/error building predictor: {e}")
            continue
        states0c = test_states0.cpu()
        p_start_c, p_stop_c = test_p_start.cpu().float(), test_p_stop.cpu().float()
        search_t = timed(lambda: pred._search(states0c, p_start_c, p_stop_c), device="cpu")
        e2e_t = timed(retrieval_end_to_end(pred, test_states0, test_p_start, test_p_stop, device),
                     device="cpu")
        per_size[N] = {"build_seconds": build_s, "search_ms": search_t, "end_to_end_ms": e2e_t}
        print(f"N={N}: build={build_s:.2f}s search_median={search_t['median']:.2f}ms "
             f"e2e_median={e2e_t['median']:.2f}ms")
        del bank, pred
        if device == "cuda":
            torch.cuda.empty_cache()
    out["by_size"] = per_size

    ns = [n for n in sizes if "error" not in per_size[n]]
    if len(ns) >= 2:
        logN = np.log10(ns)
        log_e2e = np.log10([per_size[n]["end_to_end_ms"]["median"] for n in ns])
        log_srch = np.log10([per_size[n]["search_ms"]["median"] for n in ns])
        out["loglog_slope_end_to_end"] = float(np.polyfit(logN, log_e2e, 1)[0])
        out["loglog_slope_search"] = float(np.polyfit(logN, log_srch, 1)[0])
        print(f"log-log slope: search={out['loglog_slope_search']:.2f} "
             f"end_to_end={out['loglog_slope_end_to_end']:.2f}  (1.0 == linear O(N))")
    results["section2_bank_scaling"] = out
    return states0, states1, p_starts, p_stops  # reuse for section 3


def section3(results, raw_sean, results_s2):
    print("=== Section 3: sub-linear search prototype (BallTree) ===")
    from sklearn.neighbors import BallTree
    states0, states1, p_starts, p_stops = raw_sean
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_states0, test_p_start, test_p_stop = load_test_batch(64)
    cfg = DistanceConfig()

    out = {}
    for N in [10_000, 25_000, 50_000, 100_000, 200_000]:
        s0 = tile_to(states0, N); s1 = tile_to(states1, N)
        ps = tile_to(p_starts, N); pe = tile_to(p_stops, N)
        bank = TransitionBank.from_states(s0, s1, ps, pe, source=["sean"] * N)
        bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, cfg)  # (T,n,2),(T,n)
        descriptor = bank_pts.reshape(N, -1).numpy()  # flattened push-frame uv, fixed n=20 -> 40 dims

        t0 = time.perf_counter()
        tree = BallTree(descriptor)
        tree_build_s = time.perf_counter() - t0

        pred = RetrievalPredictor(bank, k=5, aggregation="cube_median", device=device)
        from model.retrieval.distance import query_points_and_weights, chamfer_distance_chunk
        q_pts, q_w, q_uv, q_valid = query_points_and_weights(test_states0.float(), test_p_start.float(),
                                                             test_p_stop.float(), cfg)
        q_desc = q_pts.reshape(64, -1).numpy()

        M = 200  # candidates retrieved by the approximate index before exact re-rank

        def approx_query():
            _, cand_idx = tree.query(q_desc, k=M)
            return cand_idx
        approx_t = timed(approx_query, device="cpu")

        def approx_plus_rerank():
            _, cand_idx = tree.query(q_desc, k=M)
            cand_idx_t = torch.from_numpy(cand_idx)  # (64, M)
            best = torch.full((64,), -1, dtype=torch.long)
            for b in range(64):
                ci = cand_idx_t[b]
                d = chamfer_distance_chunk(q_pts[b:b + 1], q_w[b:b + 1],
                                           bank_pts[ci], bank_w[ci], cfg)  # (1,M)
                best[b] = ci[d[0].argmin()]
            return best
        full_t = timed(approx_plus_rerank, device="cpu", repeats=5)

        # recall@5: exact top-5 (exhaustive, reuse section2's predictor search) vs
        # whether the approx-candidate-set's top-1 rerank matches the exact top-1,
        # and whether the exact top-5 indices are all inside the approx top-M set.
        exact_idx, exact_dist = pred._search(test_states0.cpu(), test_p_start.cpu().float(),
                                             test_p_stop.cpu().float())[0:2]
        _, cand_idx = tree.query(q_desc, k=M)
        recall_hits = 0
        total = 0
        for b in range(64):
            exact5 = set(exact_idx[b].tolist())
            cand_set = set(cand_idx[b].tolist())
            recall_hits += len(exact5 & cand_set)
            total += len(exact5)
        recall_at_5 = recall_hits / total

        out[N] = {"tree_build_seconds": tree_build_s, "approx_query_ms": approx_t,
                 "approx_plus_exact_rerank_ms": full_t, "recall_at_5": recall_at_5,
                 "M_candidates": M}
        print(f"N={N}: tree_build={tree_build_s:.2f}s approx_query_median={approx_t['median']:.3f}ms "
             f"approx+rerank_median={full_t['median']:.2f}ms recall@5={recall_at_5:.3f}")
        del bank, pred
        if device == "cuda":
            torch.cuda.empty_cache()
    results["section3_sublinear_prototype"] = out


def main():
    busy, info = check_gpu_contention()
    print(f"GPU contention check: busy={busy} info={info}")
    results = {"gpu_check": info}
    section1(results, busy)
    raw_sean = section2(results)
    section3(results, raw_sean, results["section2_bank_scaling"])

    out_path = f"{REPO}/experiments/EXP-0059-retrieval-transition-model/results/benchmark_timing.json"
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
