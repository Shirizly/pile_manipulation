"""Variant B: push-frame LOCAL descriptors, built to concatenate onto variant A's
global descriptor vector (dmdc_baseline.occupancy_descriptors, D=87).

Reuses (unmodified):
  - transforms.functional.particles_to_occupancy   -- same rasterisation as variant A
  - transforms.functional.to_push_frame / push_frame_transform / push_frame_validity_mask
    (Suh & Tedrake 2020 canonical push frame: origin at push midpoint, +x along
    push direction, occ warped by SE(2) affine_grid/grid_sample.)

Axis convention verified empirically (see chat/agent notes, not re-derived here):
occ from particles_to_occupancy has occ[b, ix, iy] (dim1=x-bin, dim2=y-bin,
axes=('x','y') per get_grid_axes). push_frame_transform/warp_affine_occ treat
their input as a plain (B,H,W) image and index it with grid_sample's (x,y)
normalized-coordinate convention, where grid-sample-x addresses the LAST
tensor dim (W) and grid-sample-y the second-to-last (H). Since our occ's last
dim is the y-bin and second-to-last is the x-bin, the (col,row) pixel pair
`push_frame_transform` wants is **(y_bin, x_bin)**, i.e. swapped relative to
naive (x_bin, y_bin). Verified with a synthetic single-voxel spike at a real
p_start: warping it into the push frame with this mapping puts it at
canonical row≈center (lateral offset ≈0, correct: start point lies ON the
push line) and canonical col offset from center ≈ -L_px/2 (correct: start is
half a push-length BEHIND the midpoint). Get this wrong and every push-frame
descriptor below is meaningless.

New descriptor block computed in the canonical push frame (all using the
SAME start_px/end_px derived from the action, applied identically to occ_t
and occ_t1 -- the frame is defined by the action, not the state):
  - mass_ahead, mass_behind : occupancy mass with canonical col > center vs
    < center (masked by push_frame_validity_mask -- corners lost to the
    round-trip warp are excluded, not counted as either).
  - com_push (2), moments2_push (3): COM / central 2nd moments computed in
    canonical pixel coordinates (mask-weighted), same layout convention as
    dmdc_baseline.occupancy_descriptors' mass/com/moments2 blocks.
  - band_mass (4): occupancy mass in 4 equal along-push distance bands on the
    ahead side, col-center in [0, GRID/2), i.e. out to the edge of the frame
    (a push cannot physically reach further than that in this canonical
    warp).

Total new block: 1+1+2+3+4 = 11 dims. Layout exposed via `slices_b()`.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
from transforms.functional import (
    particles_to_occupancy,
    to_push_frame,
    push_frame_validity_mask,
)
from descriptors import BOUNDS, GRID, RADIUS, list_files, spawn_of, split_files

N_BANDS = 4
MASK_THRESH = 0.5  # keep canonical pixels where the round-trip validity mask >= this


def slices_b() -> dict[str, slice]:
    s: dict[str, slice] = {}
    i = 0
    for name, n in [("mass_ahead", 1), ("mass_behind", 1), ("com_push", 2),
                    ("moments2_push", 3), ("band_mass", N_BANDS)]:
        s[name] = slice(i, i + n)
        i += n
    s["_total"] = slice(0, i)
    return s


def world_to_pushframe_px(p: torch.Tensor) -> torch.Tensor:
    """World (x,y) metres [B,2] -> push_frame_transform's (col,row) pixel pair.

    col = y-bin, row = x-bin (see module docstring for why the axes are
    swapped relative to naive expectation).
    """
    x, y = p[:, 0], p[:, 1]
    xb = (x - BOUNDS["x_min"]) / (BOUNDS["x_max"] - BOUNDS["x_min"]) * (GRID - 1)
    yb = (y - BOUNDS["y_min"]) / (BOUNDS["y_max"] - BOUNDS["y_min"]) * (GRID - 1)
    return torch.stack([yb, xb], dim=-1)


def push_frame_descriptors(occ: torch.Tensor, start_px: torch.Tensor,
                            end_px: torch.Tensor) -> torch.Tensor:
    """occ: [B,H,W] -> phi_B: [B,11] per slices_b() layout."""
    B, H, W = occ.shape
    canon = to_push_frame(occ, start_px, end_px, out_res=(H, W), scale=1.0)
    mask = push_frame_validity_mask(start_px, end_px, (H, W), scale=1.0)
    valid = (mask >= MASK_THRESH).float()
    occ_m = canon * valid
    eps = 1e-8

    center_col = (W - 1) / 2.0
    center_row = (H - 1) / 2.0
    cols = torch.arange(W, dtype=occ.dtype, device=occ.device)
    rows = torch.arange(H, dtype=occ.dtype, device=occ.device)
    grow, gcol = torch.meshgrid(rows, cols, indexing="ij")

    ahead = (gcol > center_col).float().unsqueeze(0)
    behind = 1.0 - ahead
    mass_ahead = (occ_m * ahead).sum(dim=(-2, -1)) / (H * W)
    mass_behind = (occ_m * behind).sum(dim=(-2, -1)) / (H * W)

    m = occ_m.sum(dim=(-2, -1)).clamp_min(eps)
    com_row = (occ_m * grow).sum(dim=(-2, -1)) / m / H
    com_col = (occ_m * gcol).sum(dim=(-2, -1)) / m / W

    dcol = (gcol.unsqueeze(0) - center_col) / W
    drow = (grow.unsqueeze(0) - center_row) / H
    mu_rr = (occ_m * drow * drow).sum(dim=(-2, -1)) / m
    mu_cc = (occ_m * dcol * dcol).sum(dim=(-2, -1)) / m
    mu_rc = (occ_m * drow * dcol).sum(dim=(-2, -1)) / m

    band_edges = torch.linspace(0.0, W / 2.0, N_BANDS + 1)
    offset = (gcol - center_col).clamp_min(0.0).unsqueeze(0)  # ahead-side distance, px
    bands = []
    for k in range(N_BANDS):
        band_mask = ((offset >= band_edges[k]) & (offset < band_edges[k + 1])).float()
        bands.append((occ_m * band_mask).sum(dim=(-2, -1)) / (H * W))
    band_mass = torch.stack(bands, dim=1)

    return torch.cat([
        mass_ahead.unsqueeze(1), mass_behind.unsqueeze(1),
        torch.stack([com_row, com_col], dim=1),
        torch.stack([mu_rr, mu_cc, mu_rc], dim=1),
        band_mass,
    ], dim=1)


def build_all_b(files: list[str]):
    phiB_t_all, phiB_t1_all = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3]
        states_ = d["states_"][:, :, :3]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)

        p_start = d["p_starts"][:, :2]
        p_stop = d["p_stops"][:, :2]
        start_px = world_to_pushframe_px(p_start)
        end_px = world_to_pushframe_px(p_stop)

        phiB0 = push_frame_descriptors(occ0, start_px, end_px)
        phiB1 = push_frame_descriptors(occ1, start_px, end_px)
        phiB_t_all.append(phiB0.numpy().astype(np.float32))
        phiB_t1_all.append(phiB1.numpy().astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")
    print(f"done: {len(files)} files, {time.time()-t0:.1f}s")
    return dict(
        phiB_t=np.concatenate(phiB_t_all),
        phiB_t1=np.concatenate(phiB_t1_all),
    )


if __name__ == "__main__":
    import os

    outdir = "experiments/temp/dmdc-lenbins/cache"
    os.makedirs(outdir, exist_ok=True)
    files = list_files()
    train_i, test_i = split_files(files, seed=0, holdout_frac=0.2)
    print(f"train files: {len(train_i)}, test files: {len(test_i)}")

    for split_name, idx_list in [("train", train_i), ("test", test_i)]:
        sub_files = [files[i] for i in idx_list]
        d = build_all_b(sub_files)
        np.savez(f"{outdir}/descB_{split_name}.npz", **d)
        # sanity: row count must match variant-A cache exactly (same file order/split)
        a = np.load(f"{outdir}/desc_{split_name}.npz", allow_pickle=True)
        assert a["phi_t"].shape[0] == d["phiB_t"].shape[0], (
            f"{split_name}: row count mismatch vs variant-A cache "
            f"({a['phi_t'].shape[0]} vs {d['phiB_t'].shape[0]})")
        print(f"wrote {outdir}/descB_{split_name}.npz: {d['phiB_t'].shape[0]} rows (matches A)")
