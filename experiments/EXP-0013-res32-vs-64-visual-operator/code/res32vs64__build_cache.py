#!/usr/bin/env python -u
"""Rasterise Sean + overnight_randlen_{train,test} to occ0/occ1/actions and
cache to disk once. Fixes the >10 min/call `load_randlen_cell` bottleneck
that stalled the previous attempt (docs/CODEMAP.md's LOADER TRAP): glob
`*_data.pt` directly and rasterise with `particles_to_occupancy`, the fast
path measured in experiments/EXP-0004-*/code/dmdc-lenbins__descriptors.py
(full 213-file overnight_randlen corpus in ~40s).

Native raster grid fixed at 64x64 for ALL corpora (Sean, overnight train/test,
slates_binned control corpus) -- same convention as
scripts/probes/binned_pool_cache.py and EXP-0004. The res=32/64 ABLATION this
experiment is about is the CANONICAL push-frame resolution `canonicalise`
warps into, which is independent of this native raster grid.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from transforms.functional import particles_to_occupancy  # noqa: E402

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
PITCH = (BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID
RADIUS = 0.5 * CUBE_SIZE / PITCH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path(__file__).resolve().parent.parent / "cache"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def spawn_of_sean(path: str) -> str:
    for m in ("inbetween", "piled", "scattered"):
        if f"/{m}/cube/" in path:
            return m
    return "unknown"


def spawn_of_overnight(path: str) -> str:
    for m in ("mixed_n20", "piled_n20", "scattered_n20", "piled_n50", "scattered_n50"):
        if f"/{m}/" in path:
            return m
    return "unknown"


def load_files(files, spawn_fn, label):
    occ0_l, occ1_l, act_l, len_l, npart_l, spawn_l, fid_l = [], [], [], [], [], [], []
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3].to(DEVICE)
        states_ = d["states_"][:, :, :3].to(DEVICE)
        n_particles = states.shape[1]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS).cpu()
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS).cpu()
        p_start = d["p_starts"][:, :2].float()
        p_stop = d["p_stops"][:, :2].float()
        action = torch.cat([p_start, p_stop], dim=1)
        length_m = (p_stop - p_start).norm(dim=-1)
        n = occ0.shape[0]
        occ0_l.append(occ0.half())
        occ1_l.append(occ1.half())
        act_l.append(action)
        len_l.append(length_m)
        npart_l.append(torch.full((n,), n_particles, dtype=torch.int32))
        spawn_l.append([spawn_fn(f)] * n)
        fid_l.append(torch.full((n,), i, dtype=torch.int64))
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  [{label}] {i+1}/{len(files)} files, {time.time()-t0:.1f}s", flush=True)
    spawn_flat = [s for sub in spawn_l for s in sub]
    return dict(
        occ0=torch.cat(occ0_l).float(), occ1=torch.cat(occ1_l).float(),
        actions=torch.cat(act_l), length_m=torch.cat(len_l),
        n_particles=torch.cat(npart_l), spawn=spawn_flat,
        file_id=torch.cat(fid_l), files=files,
    )


def main():
    t0 = time.time()

    sean_files = sorted(glob.glob("Genesis/data/Sean/**/*_data.pt", recursive=True))
    sean_files = [f for f in sean_files if "_failed" not in f]
    print(f"Sean: {len(sean_files)} files")
    sean = load_files(sean_files, spawn_of_sean, "sean")
    torch.save(sean, OUT_DIR / "sean.pt")
    print(f"Sean cached: {sean['occ0'].shape[0]} rows, {time.time()-t0:.1f}s total")

    t1 = time.time()
    ov_train_files = sorted(glob.glob("Genesis/data/overnight_randlen_train/**/*_data.pt", recursive=True))
    ov_train_files = [f for f in ov_train_files if "_failed" not in f]
    print(f"overnight_randlen_train: {len(ov_train_files)} files")
    ov_train = load_files(ov_train_files, spawn_of_overnight, "ov_train")
    torch.save(ov_train, OUT_DIR / "overnight_train.pt")
    print(f"overnight_train cached: {ov_train['occ0'].shape[0]} rows, {time.time()-t1:.1f}s")

    t2 = time.time()
    ov_test_files = sorted(glob.glob("Genesis/data/overnight_randlen_test/**/*_data.pt", recursive=True))
    ov_test_files = [f for f in ov_test_files if "_failed" not in f]
    print(f"overnight_randlen_test: {len(ov_test_files)} files")
    ov_test = load_files(ov_test_files, spawn_of_overnight, "ov_test")
    torch.save(ov_test, OUT_DIR / "overnight_test.pt")
    print(f"overnight_test cached: {ov_test['occ0'].shape[0]} rows, {time.time()-t2:.1f}s")

    print(f"\nTOTAL wall-clock: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
