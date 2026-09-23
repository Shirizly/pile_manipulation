"""Fast, batch-vectorised disk-splat rasteriser + physically-meaningful view
augmentation for EXP-0019.

WHY NOT `transforms.functional.particles_to_occupancy`: its `footprint_radius`
branch loops `for b in range(B)` in Python and materialises a
`(N, H, W, 2)` diff tensor per sample.  That is fine for a one-pass encode
(EXP-0016 used it) but it is the inner loop of encoder TRAINING here, where the
corpus is rasterised twice per state per epoch.  This module is a drop-in
numerical equivalent for the 2-D `footprint_radius` case, fully vectorised over
the batch, and additionally supports a PER-SAMPLE radius (needed for the
radius-jitter view).  Equivalence to the project rasteriser is asserted in
`check_raster.py`.
"""
from __future__ import annotations
import torch

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
GRID = 64
CUBE_SIZE = 0.005
BASE_RADIUS = 0.5 * CUBE_SIZE / ((BOUNDS["x_max"] - BOUNDS["x_min"]) / GRID)  # voxels


def rasterise(pts_xy, radius, grid=GRID, bounds=BOUNDS, valid=None):
    """pts_xy: (B,N,2) world metres. radius: scalar or (B,) voxel radius.
    valid: optional (B,N) bool mask -- masked-out particles contribute nothing.
    Returns (B,grid,grid) float occupancy in {0,1}."""
    B, N, _ = pts_xy.shape
    dev = pts_xy.device
    lo = torch.tensor([bounds["x_min"], bounds["y_min"]], device=dev, dtype=pts_xy.dtype)
    hi = torch.tensor([bounds["x_max"], bounds["y_max"]], device=dev, dtype=pts_xy.dtype)
    p = (pts_xy - lo) / (hi - lo) * (grid - 1)                     # (B,N,2) voxel coords
    ax = torch.arange(grid, device=dev, dtype=pts_xy.dtype)
    dx2 = (p[..., 0:1] - ax[None, None, :]) ** 2                   # (B,N,G)
    dy2 = (p[..., 1:2] - ax[None, None, :]) ** 2                   # (B,N,G)
    d2 = dx2[:, :, :, None] + dy2[:, :, None, :]                   # (B,N,G,G)
    if torch.is_tensor(radius):
        r2 = (radius.to(dev) ** 2)[:, None, None, None]
    else:
        r2 = float(radius) ** 2
    hit = d2 <= r2
    if valid is not None:
        hit = hit & valid[:, :, None, None]
    return hit.any(dim=1).float()


def make_views(pts_xy, n_real, gen, keep_lo=0.80, keep_hi=1.0,
               rad_jit=0.15, occ_noise=0.02, grid=GRID):
    """Two PHYSICALLY-MEANINGFUL views of the same underlying pile state.

    Nuisance applied (design doc 6.2 -- no rotations, flips or crops, which
    would change the pile/tool frame relationship):
      1. particle dropout: keep a U(keep_lo, keep_hi) fraction of grains --
         models *which grains you happen to observe*, not a different pile.
      2. footprint-radius jitter: radius * U(1-rad_jit, 1+rad_jit) -- models
         rasterisation/segmentation footprint uncertainty.
      3. small additive occupancy noise, std `occ_noise`, clamped to [0,1].
    `n_real` is the true particle count per row (rows are padded by repetition
    to a common N; padding is excluded from the dropout mask).
    """
    B, N, _ = pts_xy.shape
    dev = pts_xy.device
    real = torch.arange(N, device=dev)[None, :] < n_real[:, None]      # (B,N)
    out = []
    for _ in range(2):
        keep_frac = torch.empty(B, 1, device=dev).uniform_(keep_lo, keep_hi, generator=gen)
        u = torch.rand(B, N, device=dev, generator=gen)
        keep = (u < keep_frac) & real
        # never drop every grain
        keep = torch.where(keep.any(1, keepdim=True), keep, real)
        r = BASE_RADIUS * torch.empty(B, device=dev).uniform_(1 - rad_jit, 1 + rad_jit, generator=gen)
        occ = rasterise(pts_xy, r, grid=grid, valid=keep)
        if occ_noise > 0:
            occ = (occ + occ_noise * torch.randn(occ.shape, device=dev, generator=gen)).clamp(0, 1)
        out.append(occ)
    return out[0], out[1]
