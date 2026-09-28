"""model/retrieval_nfd/render.py -- cube-box rasteriser for donor channels
(EXP-0059 section 8, "NFD with a retrieved reference").

Reimplements `Genesis.training.dataset.PileSweepData._draw_particle_grid`'s
box path (cv2.boxPoints + fillPoly, OpenCV (row,col) scratch buffer
transposed once at the end -- see that method's docstring for the axis-
convention history this guards against) as a FREE FUNCTION taking pixel-
space centres/yaws directly, so donor-channel rendering does not need a live
`PileSweepData` instance and can be reused identically at train-precompute
time (where channels 0-2/target come from the REAL `PileSweepData3Ch`, per
coordinator direction 2026-09-28: "channels 0-2 must be EXACTLY the narrow
NFD's inputs") and at eval time.

Measured cost: ~2ms per 20-cube grid (cv2 fillPoly is cheap) -- the earlier
(reverted) concern that this would be too slow at ~12k rows was unfounded;
a fully vectorised soft-renderer was tried and abandoned because it does not
bit-match `_draw_particle_grid`, which the mandatory unit test (rendering a
query's own cubes through the donor path must reproduce occ0) requires.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch

CUBE_SIZE_M = 0.005  # narrow-domain cube edge (Genesis/data/narrow_l20_n20 configs)


def cube_dim_px(to_pxl: float) -> tuple[float, float]:
    d = CUBE_SIZE_M * to_pxl
    return d, d


def render_cube_boxes_np(centers_px: np.ndarray, yaws: np.ndarray,
                          box_dim_px: tuple[float, float],
                          grid_hw: tuple[int, int]) -> np.ndarray:
    """centers_px: (n, 2) [x_px, y_px] (already ctr-offset, NOT row/col).
    yaws: (n,) radians. box_dim_px: (dim_x_px, dim_y_px). grid_hw: (H, W).
    -> (H, W) float32, 1.0 inside any cube's rotated rectangle, 0.0 else
    (fillPoly paints density=1; overlaps stay 1, matching the original).

    Axis convention: identical to `_draw_particle_grid` -- draws into a
    (row=y, col=x) OpenCV scratch buffer, then transposes once so the
    returned array is (dim0=x, dim1=y), matching occ0/the plate channels.
    Points far outside the grid (e.g. a padding/dummy centre) are silently
    clipped by fillPoly -- no special-casing needed.
    """
    H, W = grid_hw
    grid_np = np.zeros((H, W), dtype=np.float32)
    dim_x_px, dim_y_px = box_dim_px
    n = centers_px.shape[0]
    for i in range(n):
        cx, cy = float(centers_px[i, 0]), float(centers_px[i, 1])
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        angle_deg = float(yaws[i]) * 180.0 / math.pi
        rotated_rect = ((int(cx), int(cy)), (int(round(dim_x_px)), int(round(dim_y_px))), int(angle_deg))
        box = cv2.boxPoints(rotated_rect)
        box = np.int32(box)
        cv2.fillPoly(grid_np, [box], 1)
    # (row=y, col=x) -> (dim0=x, dim1=y)
    return grid_np.T.copy()


def render_cube_boxes_batch(centers_px: torch.Tensor, yaws: torch.Tensor,
                             box_dim_px: tuple[float, float],
                             grid_hw: tuple[int, int]) -> torch.Tensor:
    """centers_px: (B, n, 2), yaws: (B, n) -> (B, H, W) float32 torch tensor.
    Loops over the batch in Python (cv2 has no batched box-fill primitive);
    measured ~2ms/row (20 cubes), so ~12k rows costs single-digit seconds."""
    B = centers_px.shape[0]
    H, W = grid_hw
    out = torch.zeros(B, H, W, dtype=torch.float32)
    cp = centers_px.detach().cpu().numpy()
    yw = yaws.detach().cpu().numpy()
    for b in range(B):
        out[b] = torch.from_numpy(render_cube_boxes_np(cp[b], yw[b], box_dim_px, grid_hw))
    return out
