"""Variant C: richer spectral / higher-order descriptor bases for the
push-length-binned DMDc screen. Builds on variant A's rasterisation
(same BOUNDS/GRID/footprint_radius, same file list + split) but computes
SEVERAL descriptor variants per occupancy pair in one pass (occupancy is
shared; descriptors are cheap once rasterised).

Variants (each a separate cell, per experiment design):
  stock8    : dmdc_baseline.occupancy_descriptors equivalent, n_fourier=8 (D=87)
  nf16      : same layout, n_fourier=16 (D=295)
  nf24      : same layout, n_fourier=24 (D=631)
  highorder : stock8 + 3rd/4th central moments block (9 dims) (D=96)
  pooled4   : stock8 + 4x4 adaptive-avg-pooled occupancy, NOT push-frame-aware
              (honest control for variant B's push-frame locality) (D=103)
  pooled8   : stock8 + 8x8 pooled occupancy (D=151)

Does NOT edit / depend on descriptors.py's cache format; writes its own
cache under cache/ with a `_c` suffix to avoid clobbering variant A/B files.
Reuses (unmodified) descriptors.list_files / descriptors.split_files so the
file list + 80/20 stratified-by-spawn-mode split (seed 0) are IDENTICAL to
variant A.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/alon/Code/pile_manipulation")
sys.path.insert(0, "/home/alon/Code/pile_manipulation/experiments/temp/dmdc-lenbins")
from transforms.functional import particles_to_occupancy
import descriptors as descA  # variant A's list_files/split_files/BOUNDS/GRID (read-only import)

BOUNDS = descA.BOUNDS
GRID = descA.GRID
RADIUS = descA.RADIUS
SPAWN_MODES = descA.SPAWN_MODES

VARIANTS = ["stock8", "nf16", "nf24", "highorder", "pooled4", "pooled8"]


def compute_descriptors(occ: torch.Tensor, n_fourier: int = 8, highorder: bool = False,
                         pool: int | None = None):
    """occ: [B,H,W] -> (phi [B,D], slices dict). Same base layout/convention
    as dmdc_baseline.occupancy_descriptors (const/mass/com/moments2/dft), with
    optional highorder / pooled blocks appended after moments2 / at the end.
    """
    B, H, W = occ.shape
    eps = 1e-8
    mass = occ.sum(dim=(-2, -1))
    m = mass.clamp_min(eps)

    ys = torch.linspace(0.0, 1.0, H, device=occ.device)
    xs = torch.linspace(0.0, 1.0, W, device=occ.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    com_y = (occ * gy).sum(dim=(-2, -1)) / m
    com_x = (occ * gx).sum(dim=(-2, -1)) / m
    dy = gy.unsqueeze(0) - com_y[:, None, None]
    dx = gx.unsqueeze(0) - com_x[:, None, None]
    mu_yy = (occ * dy * dy).sum(dim=(-2, -1)) / m
    mu_xx = (occ * dx * dx).sum(dim=(-2, -1)) / m
    mu_xy = (occ * dy * dx).sum(dim=(-2, -1)) / m

    blocks = [torch.ones(B, 1, device=occ.device), (mass / (H * W)).unsqueeze(1),
              torch.stack([com_y, com_x], dim=1), torch.stack([mu_yy, mu_xx, mu_xy], dim=1)]
    layout = [("const", 1), ("mass", 1), ("com", 2), ("moments2", 3)]

    if highorder:
        mu30 = (occ * dx ** 3).sum(dim=(-2, -1)) / m
        mu03 = (occ * dy ** 3).sum(dim=(-2, -1)) / m
        mu21 = (occ * dx ** 2 * dy).sum(dim=(-2, -1)) / m
        mu12 = (occ * dx * dy ** 2).sum(dim=(-2, -1)) / m
        mu40 = (occ * dx ** 4).sum(dim=(-2, -1)) / m
        mu04 = (occ * dy ** 4).sum(dim=(-2, -1)) / m
        mu31 = (occ * dx ** 3 * dy).sum(dim=(-2, -1)) / m
        mu13 = (occ * dx * dy ** 3).sum(dim=(-2, -1)) / m
        mu22 = (occ * dx ** 2 * dy ** 2).sum(dim=(-2, -1)) / m
        blocks.append(torch.stack([mu30, mu03, mu21, mu12, mu40, mu04, mu31, mu13, mu22], dim=1))
        layout.append(("highorder", 9))

    Fr = torch.fft.rfft2(occ, norm="forward")
    block = Fr[:, :n_fourier, : n_fourier // 2 + 1]
    blocks.append(block.real.flatten(1))
    blocks.append(block.imag.flatten(1))
    nf = block.shape[1] * block.shape[2]
    layout.append(("dft_real", nf))
    layout.append(("dft_imag", nf))

    if pool:
        pooled = F.adaptive_avg_pool2d(occ.unsqueeze(1), (pool, pool)).flatten(1)
        blocks.append(pooled)
        layout.append((f"pooled{pool}", pool * pool))

    phi = torch.cat(blocks, dim=1)
    slices, i = {}, 0
    for name, n in layout:
        slices[name] = slice(i, i + n)
        i += n
    slices["_total"] = slice(0, i)
    return phi, slices


def variant_kwargs(name: str) -> dict:
    return {
        "stock8": dict(n_fourier=8),
        "nf16": dict(n_fourier=16),
        "nf24": dict(n_fourier=24),
        "highorder": dict(n_fourier=8, highorder=True),
        "pooled4": dict(n_fourier=8, pool=4),
        "pooled8": dict(n_fourier=8, pool=8),
    }[name]


def build_all(files: list[str]):
    """One raster pass per file; compute all VARIANTS' descriptors from the
    same occ0/occ1. Returns dict: shared arrays + per-variant phi_t/phi_t1."""
    shared = {"length_m": [], "action": [], "file_id": [], "spawn_mode": []}
    per_variant = {v: {"phi_t": [], "phi_t1": []} for v in VARIANTS}
    slices_by_variant = {}
    t0 = time.time()
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        states = d["states"][:, :, :3]
        states_ = d["states_"][:, :, :3]
        occ0 = particles_to_occupancy(states, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)
        occ1 = particles_to_occupancy(states_, BOUNDS, (GRID, GRID), footprint_radius=RADIUS)

        for v in VARIANTS:
            kw = variant_kwargs(v)
            phi0, sl = compute_descriptors(occ0, **kw)
            phi1, _ = compute_descriptors(occ1, **kw)
            slices_by_variant[v] = sl
            per_variant[v]["phi_t"].append(phi0.numpy().astype(np.float32))
            per_variant[v]["phi_t1"].append(phi1.numpy().astype(np.float32))

        p_start = d["p_starts"][:, :2]
        p_stop = d["p_stops"][:, :2]
        angle = d["angles"]
        length_m = (p_stop - p_start).norm(dim=-1)
        action = torch.cat([p_start, p_stop, angle.unsqueeze(1)], dim=1)
        n = phi0.shape[0]
        shared["length_m"].append(length_m.numpy().astype(np.float32))
        shared["action"].append(action.numpy().astype(np.float32))
        shared["file_id"].append(np.full(n, i, dtype=np.int64))
        shared["spawn_mode"].append(np.full(n, descA.spawn_of(f), dtype=np.int64))
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] {time.time()-t0:.1f}s")
    print(f"done: {len(files)} files, {time.time()-t0:.1f}s")

    out = {k: np.concatenate(v) for k, v in shared.items()}
    out["_variants"] = {}
    for v in VARIANTS:
        out["_variants"][v] = {
            "phi_t": np.concatenate(per_variant[v]["phi_t"]),
            "phi_t1": np.concatenate(per_variant[v]["phi_t1"]),
            "slices": {name: [sl.start, sl.stop] for name, sl in slices_by_variant[v].items()},
        }
    return out


if __name__ == "__main__":
    outdir = "experiments/temp/dmdc-lenbins/cache"
    os.makedirs(outdir, exist_ok=True)
    files = descA.list_files()
    print(f"{len(files)} files found")
    train_i, test_i = descA.split_files(files, seed=0, holdout_frac=0.2)
    print(f"train files: {len(train_i)}, test files: {len(test_i)}")

    for split_name, idx_list in [("train", train_i), ("test", test_i)]:
        sub_files = [files[i] for i in idx_list]
        d = build_all(sub_files)
        remap = np.array(idx_list, dtype=np.int64)
        d["file_id"] = remap[d["file_id"]]
        for v in VARIANTS:
            vd = d["_variants"][v]
            path = f"{outdir}/descC_{v}_{split_name}.npz"
            np.savez(path, phi_t=vd["phi_t"], phi_t1=vd["phi_t1"],
                     length_m=d["length_m"], action=d["action"],
                     file_id=d["file_id"], spawn_mode=d["spawn_mode"],
                     slices_names=np.array(list(vd["slices"].keys()), dtype=object),
                     slices_bounds=np.array(list(vd["slices"].values()), dtype=np.int64))
            print(f"wrote {path}: {vd['phi_t'].shape}")
