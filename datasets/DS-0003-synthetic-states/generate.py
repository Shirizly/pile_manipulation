"""Generate DS-0003 -- synthetic particle states in the same schema as real
corpus states: (n_objects, 7) = xyz (m) + wxyz quaternion, z fixed at the
single-layer resting height, yaw-only rotation, bounds DEFAULT_BOUNDS = +-0.064 m.

2026-09-17 REWRITE: replaces the old scattered+clump generator
(`goal_configs.py::sample_synthetic_state`) with
`Baselines/common/pile_compaction.py::sample_compacted_state` (grid seed +
per-object relaxation toward the centroid), legality checked with the EXACT
separating-axis test (`Baselines/common/cube_overlap.py::state_is_legal`,
tol=0.0) instead of the old axis-aligned circumscribed-box bound
(`assert_no_penetration`), which rejects legal diagonal contact by a factor
of sqrt(2) and rejects 91-99% of REAL simulated DS-0002 states outright --
i.e. it could never express "contact", so nothing could ever be compacted
against it. See DATASET.md for the full rationale and measured numbers.

Per state:
  - `compaction` ~ Uniform(0.05, 1.0) -- the dispersed<->compact diversity
    axis, REPLACING the old clump_prob/clump_size machinery entirely.
  - `yaw_kappa` ~ Uniform(0, 6) -- von Mises concentration of yaws around a
    random common heading (real cubes partially align and pack tighter than
    uniform-random yaws allow). Do not exceed 6: at higher kappa the seed
    lattice starts too tight and legality drops (measured 4/8 at kappa=40).
  - `sample_compacted_state` returns a CENTRED pile; this driver places it
    at a random offset within DEFAULT_BOUNDS, rejecting offsets that would
    push any cube outside the workspace -- this is what gives centre-of-mass
    spread (the compaction routine itself has no notion of the workspace).

Every generated state is asserted legal with `state_is_legal(..., tol=0.0)`
(the exact test) before being kept; failures are counted and reported, not
silently dropped.

Runs with multiprocessing (one worker per (n_objects) chunk range) since
compaction=1.0 states cost ~200-400 ms each at n=100.

Exact regeneration command (recorded per repo convention, `data/` is
gitignored):

    OMP_NUM_THREADS=4 /home/alon/anaconda3/envs/pme/bin/python -u \
        datasets/DS-0003-synthetic-states/generate.py \
        --n-per-count 10000 --counts 20 50 100 --workers 4 --seed 0

To extend an existing run to more states per count, rerun with a larger
--n-per-count and a NEW --seed (states are not resumable/appendable; the RNG
stream is reseeded per (n_objects, seed) pair via SeedSequence.spawn, so a
different seed gives independent, non-overlapping states).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import multiprocessing as mp
import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from Baselines.common.pile_synth import synthesize_state
from Baselines.common.cube_overlap import state_is_legal
from Baselines.common.goal_configs import DEFAULT_BOUNDS, CUBE_SIZE, REST_Z

OUT_DIR = "/home/alon/Code/pile_manipulation/datasets/DS-0003-synthetic-states/data"

COMPACTION_RANGE = (0.05, 1.0)
YAW_KAPPA_RANGE = (0.0, 6.0)


def _place_in_bounds(xy_centred, bounds, rng, max_tries=200):
    """Offset a centred pile to a random location inside `bounds`, rejecting
    offsets that push any cube outside the workspace. Returns None if no
    legal offset was found in `max_tries` (only possible for a pile whose
    own extent already exceeds the workspace)."""
    lo_x = xy_centred[:, 0].min(); hi_x = xy_centred[:, 0].max()
    lo_y = xy_centred[:, 1].min(); hi_y = xy_centred[:, 1].max()
    half = CUBE_SIZE * np.sqrt(2.0) / 2.0  # conservative any-yaw footprint pad
    cx_lo = bounds["x_min"] - lo_x + half
    cx_hi = bounds["x_max"] - hi_x - half
    cy_lo = bounds["y_min"] - lo_y + half
    cy_hi = bounds["y_max"] - hi_y - half
    if cx_lo > cx_hi or cy_lo > cy_hi:
        return None
    cx = rng.uniform(cx_lo, cx_hi)
    cy = rng.uniform(cy_lo, cy_hi)
    return xy_centred + np.array([cx, cy])


def _worker(args):
    n, count, seed_seq, worker_idx = args
    rng = np.random.default_rng(seed_seq)
    xyz_list, quat_list = [], []
    compaction_list, yaw_kappa_list = [], []
    n_illegal = 0
    n_placement_fail = 0
    t0 = time.time()
    made = 0
    while made < count:
        # synthesize_state mixes SCATTERED singletons with COMPACT clumps and
        # places them itself, inside a sampled region of the workspace -- so no
        # separate _place_in_bounds step. region_frac/clump_frac are drawn
        # inside, calibrated against real POST-SWEEP statistics.
        placed, yaw = synthesize_state(n, rng, size=CUBE_SIZE, bounds=DEFAULT_BOUNDS)
        if placed is None:
            n_placement_fail += 1
            continue
        compaction = float("nan")   # superseded by region_frac/clump_frac
        yaw_kappa = float("nan")
        legal = state_is_legal(placed, yaw, CUBE_SIZE, tol=0.0)
        if not legal:
            n_illegal += 1
            continue
        qw = np.cos(yaw / 2.0)
        qz = np.sin(yaw / 2.0)
        xyz = np.zeros((n, 3), dtype=np.float32)
        xyz[:, 0] = placed[:, 0]
        xyz[:, 1] = placed[:, 1]
        xyz[:, 2] = REST_Z
        quat = np.zeros((n, 4), dtype=np.float32)
        quat[:, 0] = qw
        quat[:, 3] = qz
        xyz_list.append(xyz)
        quat_list.append(quat)
        compaction_list.append(compaction)
        yaw_kappa_list.append(yaw_kappa)
        made += 1
    return {
        "n": n, "worker_idx": worker_idx,
        "xyz": xyz_list, "quat": quat_list,
        "compaction": compaction_list, "yaw_kappa": yaw_kappa_list,
        "n_illegal": n_illegal, "n_placement_fail": n_placement_fail,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-count", type=int, default=10000)
    ap.add_argument("--counts", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    max_n = max(args.counts)
    ss = np.random.SeedSequence(args.seed)

    jobs = []
    for n in args.counts:
        per_worker = [args.n_per_count // args.workers] * args.workers
        for i in range(args.n_per_count % args.workers):
            per_worker[i] += 1
        child_seqs = ss.spawn(args.workers)
        for wi, (count, cseq) in enumerate(zip(per_worker, child_seqs)):
            if count > 0:
                jobs.append((n, count, cseq, wi))

    t0 = time.time()
    print(f"[0s] launching {len(jobs)} jobs across {args.workers} workers "
          f"({args.n_per_count} states x {len(args.counts)} counts)")
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, jobs)
    print(f"[{time.time()-t0:.0f}s] all workers done")

    by_n = {n: [] for n in args.counts}
    for r in results:
        by_n[r["n"]].append(r)

    xyz_list, quat_list, n_obj_list = [], [], []
    compaction_list, yaw_kappa_list, group_list = [], [], []
    total_illegal = 0
    total_placement_fail = 0
    idx = 0
    for n in args.counts:
        rs = by_n[n]
        n_states_this = sum(len(r["xyz"]) for r in rs)
        n_illegal = sum(r["n_illegal"] for r in rs)
        n_pf = sum(r["n_placement_fail"] for r in rs)
        total_illegal += n_illegal
        total_placement_fail += n_pf
        elapsed = max((r["elapsed_s"] for r in rs), default=0.0)
        print(f"n_objects={n}: {n_states_this} states, illegal_rejected={n_illegal}, "
              f"placement_fail_rejected={n_pf}, worker_wall_s={elapsed:.0f}")
        for r in rs:
            for xy_i, quat_i, comp_i, kap_i in zip(r["xyz"], r["quat"], r["compaction"], r["yaw_kappa"]):
                xyz = np.zeros((max_n, 3), dtype=np.float32)
                quat = np.zeros((max_n, 4), dtype=np.float32)
                xyz[:n] = xy_i
                quat[:n] = quat_i
                xyz_list.append(xyz)
                quat_list.append(quat)
                n_obj_list.append(n)
                compaction_list.append(comp_i)
                yaw_kappa_list.append(kap_i)
                group_list.append(f"synthetic:n{n}:seed{args.seed}:{idx}")
                idx += 1

    N = len(xyz_list)
    payload = {
        "xyz": torch.from_numpy(np.stack(xyz_list)),
        "quat": torch.from_numpy(np.stack(quat_list)),
        "n_objects": torch.tensor(n_obj_list, dtype=torch.int64),
        "compaction": torch.tensor(compaction_list, dtype=torch.float32),
        "yaw_kappa": torch.tensor(yaw_kappa_list, dtype=torch.float32),
        "group": group_list,
        "generator_config": vars(args),
        "cube_size": CUBE_SIZE,
        "rest_z": REST_Z,
        "bounds": DEFAULT_BOUNDS,
    }
    out_path = os.path.join(OUT_DIR, "states.pt")
    torch.save(payload, out_path)
    print(f"[{time.time()-t0:.0f}s] saved {out_path}: {N} synthetic states "
          f"(illegal_rejected total={total_illegal}, placement_fail_rejected total={total_placement_fail})")

    with open(os.path.join(OUT_DIR, "generate_summary.json"), "w") as fh:
        json.dump({
            "n_states": N,
            "config": vars(args),
            "elapsed_s": time.time() - t0,
            "n_illegal_rejected": total_illegal,
            "n_placement_fail_rejected": total_placement_fail,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
