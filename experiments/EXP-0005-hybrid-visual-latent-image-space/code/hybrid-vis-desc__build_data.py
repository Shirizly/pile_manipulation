"""Stage-2 data builder: cache, per split, everything fit_hybrid.py needs to
fit/evaluate switched-linear operators on state = [32x32 canonical visual
occupancy || analytic descriptors].

Reuses (unmodified):
  - transforms.functional.particles_to_occupancy   -- same rasterisation stage 1 used
  - fit_linear_foresight.actions_to_pixels / canonicalise -- the SAME world->pixel
    convention and canonical-frame warp the LinearForesight baseline uses, so
    the "pure visual" cell here reproduces that baseline exactly at --res 32.
  - experiments/temp/dmdc-lenbins/descriptors_d.py::push_frame_full_descriptors --
    the local push-frame descriptor block (mass/com/moments2/ahead-behind/bands
    + DFT), unchanged.
  - experiments/temp/dmdc-lenbins/descriptors.py::list_files/split_files -- the
    EXACT same 80/20 file split (seed=0, stratified by spawn mode) stage 1's
    cache used, so stage 2 is comparable to stage 1.

One convention choice made explicit: `push_frame_full_descriptors` is generic
in its (start_px, end_px) argument -- it does not care which world->pixel
convention produced them, only that the SAME pair is used for occ0 and occ1
(that invariant is preserved here: computed once per file, reused for both).
Stage 1's descD cache used `descriptors_b.world_to_pushframe_px` (a slightly
different sub-pixel convention: (GRID-1) denominator, no -0.5 center offset)
while the visual block here must use `fit_linear_foresight.actions_to_pixels`
(the LinearForesight baseline's own convention) so the "pure visual" cell
reproduces that baseline. Rather than mixing two conventions inside one
hybrid state vector, THIS SCRIPT uses actions_to_pixels for BOTH the visual
canonicalisation and the descriptor push-frame warp -- internal consistency
of the hybrid state matters more here than bit-for-bit matching stage 1's
raw descriptor cache values (the file split, corpus, and descriptor formula
family are still identical; only the sub-pixel px convention differs, by a
fraction of a pixel).

Output: cache/{train,test}_cache.pt, each a dict of:
  canon0, canon1   float32 [N,32,32]  -- canonical-frame occupancy at t / t+1
  desc0,  desc1    float32 [N,94]     -- push_frame_full_descriptors layout
                                          (descriptors_d.slices_d()): global_mass(1),
                                          global_length(1), mass(1), com(2),
                                          moments2(3), mass_ahead(1), mass_behind(1),
                                          band_mass(4), dft_real(40), dft_imag(40)
  occ0,   occ1     float16 [N,64,64]  -- world-frame occupancy (for accuracy metric)
  action           float32 [N,4]      -- [sx,sy,ex,ey] world metres
  length_m         float32 [N]        -- push length, metres
  file_id          int64   [N]
  files            list[str]          -- full sorted file list (same index space
                                          as file_id, matches dmdc-lenbins cache)
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")

from transforms.functional import particles_to_occupancy
from descriptors import BOUNDS, GRID, RADIUS, list_files, split_files  # noqa: E402
from descriptors_d import push_frame_full_descriptors  # noqa: E402
from fit_linear_foresight import actions_to_pixels, canonicalise  # noqa: E402

RES = 32       # paper's canonical visual resolution (stage-2 spec)
CROP = 1.0     # full-image warp, matching LinearForesight's own default
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WS_MIN = torch.tensor([BOUNDS["x_min"], BOUNDS["y_min"]])
WS_MAX = torch.tensor([BOUNDS["x_max"], BOUNDS["y_max"]])


def process_file(f: str, file_idx: int):
    d = torch.load(f, map_location="cpu")
    states = d["states"][:, :, :3]
    states_ = d["states_"][:, :, :3]
    occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
    occ0, occ1 = occ0.to(DEVICE), occ1.to(DEVICE)

    p_start = d["p_starts"][:, :2].to(DEVICE)
    p_stop = d["p_stops"][:, :2].to(DEVICE)
    length_m = (p_stop - p_start).norm(dim=-1)
    action = torch.cat([p_start, p_stop], dim=1)

    start_px, end_px = actions_to_pixels(action, WS_MIN, WS_MAX, (GRID, GRID))

    canon0 = canonicalise(occ0, start_px, end_px, RES, CROP)
    canon1 = canonicalise(occ1, start_px, end_px, RES, CROP)

    H, W = occ0.shape[-2:]
    gm0 = occ0.sum(dim=(-2, -1)) / (H * W)
    gm1 = occ1.sum(dim=(-2, -1)) / (H * W)
    local0 = push_frame_full_descriptors(occ0, start_px, end_px)  # [N,92] incl DFT
    local1 = push_frame_full_descriptors(occ1, start_px, end_px)
    desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)  # [N,94]
    desc1 = torch.cat([gm1[:, None], length_m[:, None], local1], dim=1)

    n = occ0.shape[0]
    return dict(
        canon0=canon0.cpu().float(), canon1=canon1.cpu().float(),
        desc0=desc0.cpu().float(), desc1=desc1.cpu().float(),
        occ0=occ0.cpu().half(), occ1=occ1.cpu().half(),
        action=action.cpu().float(), length_m=length_m.cpu().float(),
        file_id=torch.full((n,), file_idx, dtype=torch.int64),
    )


def build_split(files, idx_list, split_name):
    keys = ["canon0", "canon1", "desc0", "desc1", "occ0", "occ1", "action",
            "length_m", "file_id"]
    acc = {k: [] for k in keys}
    t0 = time.time()
    for j, i in enumerate(idx_list):
        rec = process_file(files[i], i)
        for k in keys:
            acc[k].append(rec[k])
        if (j + 1) % 30 == 0 or (j + 1) == len(idx_list):
            print(f"  [{split_name}] {j+1}/{len(idx_list)} files, "
                  f"{time.time()-t0:.1f}s", flush=True)
    out = {k: torch.cat(acc[k], dim=0) for k in keys}
    out["files"] = files
    return out


def main():
    files = list_files()
    print(f"{len(files)} files found")
    train_i, test_i = split_files(files, seed=0, holdout_frac=0.2)
    print(f"train files: {len(train_i)}, test files: {len(test_i)}")

    outdir = "/home/alon/Code/pile_manipulation/experiments/temp/hybrid-vis-desc/cache"
    for split_name, idx_list in [("train", train_i), ("test", test_i)]:
        data = build_split(files, idx_list, split_name)
        path = f"{outdir}/{split_name}_cache.pt"
        torch.save(data, path)
        print(f"wrote {path}: {data['canon0'].shape[0]} rows")


if __name__ == "__main__":
    main()
