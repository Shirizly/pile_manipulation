"""Variant D: an ALL-LOCAL descriptor basis, built entirely in the canonical
push frame (Suh & Tedrake 2020), rather than bolting local features onto
variant A's global (world-frame) basis the way variant B did.

Reuses (unmodified):
  - transforms.functional.particles_to_occupancy   -- same rasterisation as A/B
  - transforms.functional.to_push_frame / push_frame_validity_mask -- same
    warp machinery variant B already uses (see descriptors_b.py docstring for
    the axis-swap gotcha; world_to_pushframe_px is imported from there rather
    than re-derived).

CORRECTNESS-CRITICAL: occ_t and occ_t1 are warped with the SAME
start_px/end_px (both derived from the action: p_start/p_stop of THIS row),
via the SAME `push_frame_transform`/`to_push_frame` call for each. The frame
is a property of the action, not of the state being warped, so a fixed
per-row transform is applied identically to occ0 and occ1 -- this is what
makes the fitted linear operator map push-frame-state -> push-frame-state
(not push-frame-state -> some-other-frame-state). Verified two ways:
  (1) by construction: `push_frame_descriptors_full` is called once per
      state with the same (start_px, end_px) tensors used for both occ0 and
      occ1 -- there is only one code path, so it can't drift between the two
      calls (see build_all_d below: `start_px`/`end_px` computed once per
      file, reused for both phi0 and phi1).
  (2) empirically: for a handful of rows, mass_ahead+mass_behind (push-frame,
      masked) tracks total occupancy mass (world-frame, unmasked) up to the
      validity-mask crop, for BOTH t and t+1 -- if the two states were warped
      with different transforms, that relationship would not hold consistently
      across the two occupancies of the same synthetic check used for B.

Descriptor family: the SAME family variant A used in world frame -- mass,
COM, central 2nd moments, low-frequency rfft2 block (n_fourier=8) -- computed
instead on the masked push-frame occupancy, plus B's ahead/behind mass split
and along-push band masses (both already local). Only two GLOBAL scalars are
kept, because the push frame genuinely cannot express them: total occupancy
mass in the WORLD frame (closure -- the push-frame mask crops corners, so its
own "mass" is not a true closure quantity) and the push length itself (the
canonical frame's rotation+translation encodes push identity but not scale,
so a 1cm and 10cm push produce the same transform structure; length is not
otherwise recoverable from within the frame without cropping information).

Layout (slices_d()): global_mass(1), global_length(1), mass(1), com(2),
moments2(3), mass_ahead(1), mass_behind(1), band_mass(4), dft_real(nf),
dft_imag(nf), nf = n_fourier*(n_fourier//2+1) = 40 for n_fourier=8.
D = 2+1+2+3+1+1+4+40+40 = 94.

`slices_d_no_dft()` is the SAME layout with dft_real/dft_imag dropped; since
they are laid out last, "no DFT" is simply the first `slices_d_no_dft()['_total'].stop`
columns of the full vector -- no separate build needed (cell 2 of fit_d.py
just slices columns off the cell-1 cache).
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
from descriptors_b import world_to_pushframe_px

N_FOURIER = 8
N_BANDS = 4
MASK_THRESH = 0.5


def slices_d(n_fourier: int = N_FOURIER) -> dict[str, slice]:
    nf = n_fourier * (n_fourier // 2 + 1)
    s: dict[str, slice] = {}
    i = 0

    def take(name: str, n: int) -> None:
        nonlocal i
        s[name] = slice(i, i + n)
        i += n

    take("global_mass", 1)
    take("global_length", 1)
    take("mass", 1)
    take("com", 2)
    take("moments2", 3)
    take("mass_ahead", 1)
    take("mass_behind", 1)
    take("band_mass", N_BANDS)
    take("dft_real", nf)
    take("dft_imag", nf)
    s["_total"] = slice(0, i)
    return s


def slices_d_no_dft(n_fourier: int = N_FOURIER) -> dict[str, slice]:
    """Same layout as slices_d, dft_real/dft_imag dropped. Since DFT is last
    in the layout, this is just a PREFIX of the full vector."""
    full = slices_d(n_fourier)
    cut = full["dft_real"].start
    out = {k: v for k, v in full.items() if k not in ("dft_real", "dft_imag", "_total")}
    out["_total"] = slice(0, cut)
    return out


def push_frame_full_descriptors(occ: torch.Tensor, start_px: torch.Tensor,
                                 end_px: torch.Tensor,
                                 n_fourier: int = N_FOURIER) -> torch.Tensor:
    """occ: [B,H,W] world-frame occupancy -> phi_local: [B, D_local] where
    D_local excludes the two global scalars (those are added by the caller,
    since they don't depend on the push-frame warp)."""
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

    mass = occ_m.sum(dim=(-2, -1)) / (H * W)
    m = occ_m.sum(dim=(-2, -1)).clamp_min(eps)

    com_row = (occ_m * grow).sum(dim=(-2, -1)) / m / H
    com_col = (occ_m * gcol).sum(dim=(-2, -1)) / m / W

    dcol = (gcol.unsqueeze(0) - center_col) / W
    drow = (grow.unsqueeze(0) - center_row) / H
    mu_rr = (occ_m * drow * drow).sum(dim=(-2, -1)) / m
    mu_cc = (occ_m * dcol * dcol).sum(dim=(-2, -1)) / m
    mu_rc = (occ_m * drow * dcol).sum(dim=(-2, -1)) / m

    ahead = (gcol > center_col).float().unsqueeze(0)
    behind = 1.0 - ahead
    mass_ahead = (occ_m * ahead).sum(dim=(-2, -1)) / (H * W)
    mass_behind = (occ_m * behind).sum(dim=(-2, -1)) / (H * W)

    band_edges = torch.linspace(0.0, W / 2.0, N_BANDS + 1)
    offset = (gcol - center_col).clamp_min(0.0).unsqueeze(0)
    bands = []
    for k in range(N_BANDS):
        band_mask = ((offset >= band_edges[k]) & (offset < band_edges[k + 1])).float()
        bands.append((occ_m * band_mask).sum(dim=(-2, -1)) / (H * W))
    band_mass = torch.stack(bands, dim=1)

    F = torch.fft.rfft2(occ_m, norm="forward")
    block = F[:, :n_fourier, : n_fourier // 2 + 1]

    return torch.cat([
        mass.unsqueeze(1),
        torch.stack([com_row, com_col], dim=1),
        torch.stack([mu_rr, mu_cc, mu_rc], dim=1),
        mass_ahead.unsqueeze(1), mass_behind.unsqueeze(1),
        band_mass,
        block.real.flatten(1),
        block.imag.flatten(1),
    ], dim=1)


def build_all_d(files: list[str]):
    phiD_t_all, phiD_t1_all = [], []
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3]
        states_ = d["states_"][:, :, :3]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)

        H, W = occ0.shape[-2:]
        global_mass0 = occ0.sum(dim=(-2, -1)) / (H * W)
        global_mass1 = occ1.sum(dim=(-2, -1)) / (H * W)

        p_start = d["p_starts"][:, :2]
        p_stop = d["p_stops"][:, :2]
        length_m = (p_stop - p_start).norm(dim=-1)
        # SAME action-derived transform (start_px, end_px) used for BOTH
        # occ0 (phi_t) and occ1 (phi_t1) below -- the frame is defined by the
        # action, not by which state is being warped.
        start_px = world_to_pushframe_px(p_start)
        end_px = world_to_pushframe_px(p_stop)

        local0 = push_frame_full_descriptors(occ0, start_px, end_px)
        local1 = push_frame_full_descriptors(occ1, start_px, end_px)

        phiD0 = torch.cat([global_mass0.unsqueeze(1), length_m.unsqueeze(1), local0], dim=1)
        phiD1 = torch.cat([global_mass1.unsqueeze(1), length_m.unsqueeze(1), local1], dim=1)

        phiD_t_all.append(phiD0.numpy().astype(np.float32))
        phiD_t1_all.append(phiD1.numpy().astype(np.float32))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")
    print(f"done: {len(files)} files, {time.time()-t0:.1f}s")
    return dict(
        phiD_t=np.concatenate(phiD_t_all),
        phiD_t1=np.concatenate(phiD_t1_all),
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
        d = build_all_d(sub_files)
        np.savez(f"{outdir}/descD_{split_name}.npz", **d)
        a = np.load(f"{outdir}/desc_{split_name}.npz", allow_pickle=True)
        assert a["phi_t"].shape[0] == d["phiD_t"].shape[0], (
            f"{split_name}: row count mismatch vs variant-A cache "
            f"({a['phi_t'].shape[0]} vs {d['phiD_t'].shape[0]})")
        print(f"wrote {outdir}/descD_{split_name}.npz: {d['phiD_t'].shape[0]} rows (matches A)")
