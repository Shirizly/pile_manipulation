"""Build per-transition analytic descriptors for overnight_randlen and cache them.

Reuses (unmodified):
  - transforms.functional.particles_to_occupancy  -- particle -> occupancy rasterisation
  - dmdc_baseline.occupancy_descriptors / descriptor_slices -- phi(occ)

Rasterisation choice: `particles_to_occupancy` with `footprint_radius` (not the
default point-splat), because this corpus is /cube/ (rigid cubes, not point
grains) -- same convention `occupancy_foresight.py`'s `--view mask
--cube-size` path uses for cube datasets. Grid 64x64, bounds
{-0.064..0.064}^2 m, matching this box's `vol: [0.128, 0.128, 0.045]`
(verified from a sample `_config.yaml`) and the repo's existing
`occupancy_foresight.BOUNDS`/`--grid 64` default.

Cache contract (cache/README.md documents this too):
  phi_t     [N,D] float32   -- descriptor at t   (dmdc_baseline layout)
  phi_t1    [N,D] float32   -- descriptor at t+1
  length_m  [N]   float32   -- ||p_stop_xy - p_start_xy||
  action    [N,5] float32   -- [sx, sy, ex, ey, angle] world metres/radians
  file_id   [N]   int64     -- index into `files` (this run's file list, saved too)
  spawn_mode[N]   int64     -- 0=mixed, 1=piled, 2=scattered
"""
from __future__ import annotations

import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from transforms.functional import particles_to_occupancy
from dmdc_baseline import occupancy_descriptors

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
N_FOURIER = 8
SPAWN_MODES = ["mixed", "piled", "scattered"]


def list_files(root="Genesis/data/overnight_randlen"):
    files = sorted(glob.glob(f"{root}/*/cube/**/_*_data.pt", recursive=True))
    return files


def spawn_of(path: str) -> int:
    for i, m in enumerate(SPAWN_MODES):
        if f"/{m}/cube/" in path:
            return i
    raise ValueError(path)


def build_all(files: list[str]):
    phi_t_all, phi_t1_all, len_all, act_all, fid_all, spawn_all = [], [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3]
        states_ = d["states_"][:, :, :3]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        phi0 = occupancy_descriptors(occ0, n_fourier=N_FOURIER)
        phi1 = occupancy_descriptors(occ1, n_fourier=N_FOURIER)

        p_start = d["p_starts"][:, :2]
        p_stop = d["p_stops"][:, :2]
        angle = d["angles"]
        length_m = (p_stop - p_start).norm(dim=-1)
        action = torch.cat([p_start, p_stop, angle.unsqueeze(1)], dim=1)

        n = phi0.shape[0]
        phi_t_all.append(phi0.numpy().astype(np.float32))
        phi_t1_all.append(phi1.numpy().astype(np.float32))
        len_all.append(length_m.numpy().astype(np.float32))
        act_all.append(action.numpy().astype(np.float32))
        fid_all.append(np.full(n, i, dtype=np.int64))
        spawn_all.append(np.full(n, spawn_of(f), dtype=np.int64))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")
    print(f"done: {len(files)} files, {time.time()-t0:.1f}s")
    return dict(
        phi_t=np.concatenate(phi_t_all),
        phi_t1=np.concatenate(phi_t1_all),
        length_m=np.concatenate(len_all),
        action=np.concatenate(act_all),
        file_id=np.concatenate(fid_all),
        spawn_mode=np.concatenate(spawn_all),
    )


def split_files(files: list[str], seed: int = 0, holdout_frac: float = 0.2):
    """Stratified by spawn mode, split by FILE."""
    rng = np.random.RandomState(seed)
    by_mode: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i, f in enumerate(files):
        by_mode[spawn_of(f)].append(i)
    train_idx, test_idx = [], []
    for m, idxs in by_mode.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * holdout_frac)))
        test_idx.extend(idxs[:n_test].tolist())
        train_idx.extend(idxs[n_test:].tolist())
    return sorted(train_idx), sorted(test_idx)


if __name__ == "__main__":
    import os

    outdir = "experiments/temp/dmdc-lenbins/cache"
    os.makedirs(outdir, exist_ok=True)
    files = list_files()
    print(f"{len(files)} files found")
    train_i, test_i = split_files(files, seed=0, holdout_frac=0.2)
    print(f"train files: {len(train_i)}, test files: {len(test_i)}")

    for split_name, idx_list in [("train", train_i), ("test", test_i)]:
        sub_files = [files[i] for i in idx_list]
        d = build_all(sub_files)
        # remap file_id to the GLOBAL file index (files list) not the sub-list
        # index, so file_id is stable/interpretable across caches.
        remap = np.array(idx_list, dtype=np.int64)
        d["file_id"] = remap[d["file_id"]]
        np.savez(f"{outdir}/desc_{split_name}.npz", **d, files=np.array(files, dtype=object))
        print(f"wrote {outdir}/desc_{split_name}.npz: {d['phi_t'].shape[0]} rows")
