"""The `grid-convention` and `rasteriser-identity` invariants.

Written BEFORE the fix, per .claude/skills/experiment-log/SKILL.md's bug
procedure, so the failure was recorded in the suite rather than in prose. The
occupancy channel and the plate/action channel of a PileSweepData sample placed
world x on opposite grid axes; see docs/experiments/EXP-0001-occupancy-transpose.md.

**Fixed 2026-09-05** in `PileSweepData._draw_particle_grid`. The fourth test was
`xfail(strict=True)` until then; it now asserts normally. The first three pin
the individual conventions so the fix cannot silently regress.
"""
import glob

import cv2
import numpy as np
import pytest
import torch

from transforms.functional import draw_plate_soft, particles_to_occupancy

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
RES = (64, 64)


def test_cv2_takes_points_as_column_row():
    """The fact the dataset rasteriser trips over."""
    grid = np.zeros((16, 16), dtype=np.float32)
    cv2.circle(grid, (3, 11), 0, color=1, thickness=-1)   # (x=3, y=11)
    assert grid[11, 3] == 1.0, "cv2 writes (row=y, col=x)"
    assert grid[3, 11] == 0.0


def test_particles_to_occupancy_puts_world_x_on_dim0():
    pts = torch.tensor([[[0.048, 0.0, 0.0]]])             # +x, y centred
    occ = particles_to_occupancy(pts, BOUNDS, RES, sigma=0.0)
    row, col = (occ[0] > 0).nonzero()[0].tolist()
    assert row > RES[0] * 0.75, "world +x should land high on dim0"
    assert abs(col - RES[1] / 2) <= 1, "world y=0 should land mid dim1"


def test_draw_plate_soft_puts_world_x_on_dim0():
    """Its centre argument is (x, y) in pixels and x indexes dim0 — note this
    contradicts the function's own docstring, which is part of how the bug
    survived."""
    plate = draw_plate_soft(torch.tensor([[48.0, 32.0]]), torch.tensor([0.0]),
                            RES, 6.0, 2.0, intensity=1.0, sigma=0.5)[0]
    row, col = [float(v) for v in (plate > 0.5).nonzero().float().mean(0)]
    assert abs(row - 48.0) < 2.0, "centre[0] (world x) should index dim0"
    assert abs(col - 32.0) < 2.0


def test_dataset_occupancy_agrees_with_particles_to_occupancy():
    """Fixed 2026-09-05. Was xfail(strict) from 2026-09-03; see EXP-0001."""
    if not glob.glob("Genesis/data/foresight/L040/**/*_data.pt", recursive=True):
        pytest.skip("L040 dataset not present")
    from dmdc_baseline import load_transition_arrays
    from occupancy_foresight import load_transition_fields

    d = load_transition_arrays("configs/dataset/genesis_foresight_L040.yaml",
                               split="train", max_samples=64)
    mine, *_ = load_transition_fields("Genesis/data/foresight/L040/**/*_data.pt", 64,
                                0.0, "mean", 39.0, "cpu", view="mask",
                                min_grains=1.0, cube_size=0.007)
    a, b = (d.occ_t[:64] > 0.5).float(), (mine[:64] > 0.5).float()
    iou = float((a * b).sum() / (a + b).clamp(max=1).sum())
    iou_t = float((a * b.transpose(1, 2)).sum()
                  / (a + b.transpose(1, 2)).clamp(max=1).sum())
    assert iou > iou_t, (f"the dataset raster matches a TRANSPOSED "
                         f"particles_to_occupancy better ({iou_t:.3f}) than the "
                         f"untransposed one ({iou:.3f})")
