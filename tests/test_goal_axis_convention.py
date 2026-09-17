"""The `goal-mask-axis-convention-row-y-col-x` invariant.

Written BEFORE the fix, per `register-validator`'s "When a bug is found"
procedure, so the defect was recorded in the suite rather than in prose. All
eight assertions below were `xfail(strict=True)` and failed on the pre-fix
source; the fix landed the same day (2026-09-17) and the markers came off, so
they now assert normally and the tag is `fixed`.

`transforms/functional.py::particles_to_occupancy` puts **world x on the row
axis and world y on the column axis** (convention A -- pinned independently by
`tests/test_grid_convention.py`, and baked into every fitted checkpoint in the
repo). The goal-mask side (`Baselines/common/goals.py`,
`control_utility_test.lyapunov_weights`, `Baselines/common/goal_configs.py`)
built masks with row = world y, col = world x (convention B), so every
value function multiplied an A-occupancy by a B-mask. For a SYMMETRIC goal
(`corner`, `center`, quadrants 0 and 3) that is invisible -- those masks are
their own transpose -- which is why the bug survived. The tests below therefore
use deliberately ASYMMETRIC regions, the only ones that can detect it.

Each test states the same property: mass placed in a known WORLD region scores
as inside that region's own mask, and not inside its transpose.
"""
import os
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Baselines.common.goals import (LETTER_FONT, dist_field_from_mask,  # noqa: E402
                                    letter_mask, mass_in_region, quadrant_mask)
from Baselines.common.goal_configs import (DEFAULT_BOUNDS,             # noqa: E402
                                           mask_to_configuration)
from control_utility_test import lyapunov, lyapunov_weights            # noqa: E402
from transforms.functional import particles_to_occupancy               # noqa: E402

BOUNDS = DEFAULT_BOUNDS
GRID = 64
RES = (GRID, GRID)


def occ_from_world_xy(xs, ys):
    """(N,) world x, (N,) world y -> a single (H,W) occupancy, via the
    repo's own rasteriser. This is the convention-A side of every assertion;
    the tests never re-implement it."""
    pts = torch.zeros(1, len(xs), 3, dtype=torch.float32)
    pts[0, :, 0] = torch.as_tensor(np.asarray(xs), dtype=torch.float32)
    pts[0, :, 1] = torch.as_tensor(np.asarray(ys), dtype=torch.float32)
    return particles_to_occupancy(pts, BOUNDS, RES, sigma=0.0)[0]


def world_grid(x_range, y_range, n=16):
    """A dense grid of world points inside the given (lo, hi) x/y box."""
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return gx.ravel(), gy.ravel()


def frac_in(occ, mask):
    m = torch.as_tensor(np.asarray(mask), dtype=occ.dtype)
    total = float(occ.sum())
    assert total > 0
    return float(mass_in_region(occ.unsqueeze(0), m)[0]) / total


# --- quadrant_mask -----------------------------------------------------------
# Quadrants 0 and 3 are their own transpose and cannot detect the bug; 1 and 2
# are the two that can. Quadrant indices are {0,1,2,3} = world
# {(-x,-y), (+x,-y), (-x,+y), (+x,+y)}.
QUADRANT_WORLD = {
    1: ((0.016, 0.056), (-0.056, -0.016)),   # +x, -y
    2: ((-0.056, -0.016), (0.016, 0.056)),   # -x, +y
}


@pytest.mark.parametrize("q", [1, 2])
def test_quadrant_mask_agrees_with_occupancy(q):
    xs, ys = world_grid(*QUADRANT_WORLD[q])
    occ = occ_from_world_xy(xs, ys)
    mask = quadrant_mask(GRID, GRID, q)
    assert frac_in(occ, mask) > 0.99, (
        f"mass in the world region of quadrant {q} scores "
        f"{frac_in(occ, mask):.3f} inside quadrant {q}'s own mask "
        f"(and {frac_in(occ, mask.T):.3f} inside its transpose)")


def test_quadrants_0_and_3_are_transpose_invariant():
    """Not a bug test -- a blast-radius bound. These two must be unchanged by
    the fix, which is what protects every record scored on them."""
    for q in (0, 3):
        m = quadrant_mask(GRID, GRID, q)
        assert np.array_equal(m, m.T)


# --- letter_mask -------------------------------------------------------------
def letter_world_points(name):
    """World (x, y) of every glyph pixel, derived from the ASSET, not from
    `letter_mask` -- so this is an independent statement of where the letter
    is in the world, not a restatement of the function under test.

    The asset is stored in the orientation a human reads it: image row
    increases with world y, image column increases with world x.
    """
    goal = np.load(os.path.join(f"env/target_shapes/{LETTER_FONT}",
                                f"helvetica_{name}.npy"))
    vis = cv2.resize(goal, (GRID, GRID), interpolation=cv2.INTER_AREA) <= 0.5
    iy, ix = np.nonzero(vis)
    x = BOUNDS["x_min"] + (ix + 0.5) / GRID * (BOUNDS["x_max"] - BOUNDS["x_min"])
    y = BOUNDS["y_min"] + (iy + 0.5) / GRID * (BOUNDS["y_max"] - BOUNDS["y_min"])
    return x, y


@pytest.mark.parametrize("name", ["T"])
def test_letter_mask_agrees_with_occupancy(name):
    x, y = letter_world_points(name)
    occ = occ_from_world_xy(x, y)
    mask = letter_mask(name, GRID, GRID)
    inside, transposed = frac_in(occ, mask), frac_in(occ, mask.T)
    assert inside > 0.95, (f"letter '{name}': {inside:.3f} of the glyph's own "
                           f"mass scores inside its mask, {transposed:.3f} "
                           f"inside the mask's transpose")


# --- lyapunov_weights, asymmetric goal --------------------------------------
def test_lyapunov_weights_stripe_agrees_with_occupancy():
    """`stripe` is the one asymmetric key in `lyapunov_weights`: a central
    band in world y, free in world x. A pile built ON that band must have
    V ~ 0 (the field is 0 inside the target), and a pile on `occ.T` must not."""
    xs, ys = world_grid((-0.056, 0.056), (-0.006, 0.006), n=24)
    occ = occ_from_world_xy(xs, ys)
    d = lyapunov_weights(RES, "stripe", torch.device("cpu"))
    v_on = float(lyapunov(occ.unsqueeze(0), d)[0])
    v_off = float(lyapunov(occ.T.contiguous().unsqueeze(0), d)[0])
    assert v_on < 0.05 and v_off > 0.15, (
        f"V on the stripe = {v_on:.3f}, V on its transpose = {v_off:.3f}")


def test_dist_field_from_mask_matches_lyapunov_weights_on_stripe():
    """The two distance-field paths (`goals.dist_field_from_mask` on a
    `goals`-built mask, and `lyapunov_weights`'s own general branch) must
    describe the same world region."""
    stripe = np.zeros((GRID, GRID), dtype=bool)
    stripe[:, GRID // 2 - GRID // 8: GRID // 2 + GRID // 8] = True   # band in y
    a = dist_field_from_mask(stripe)
    b = lyapunov_weights(RES, "stripe", torch.device("cpu")).numpy()
    assert np.allclose(a, b, atol=1e-6)


# --- mask -> configuration -> rasterise round trip ---------------------------
def test_mask_to_configuration_round_trips_thick_mask():
    """The clean case: an asymmetric mask big enough that the generator needs
    no dilation, so essentially ALL of the configuration must land inside the
    mask it was generated from."""
    mask = quadrant_mask(GRID, GRID, 1)                  # world (+x, -y)
    res = mask_to_configuration(mask, n_objects=20, seed=0)
    assert res.method == "grid", res.method              # no dilation involved
    occ = occ_from_world_xy(res.poses[:, 0], res.poses[:, 1])
    inside, transposed = frac_in(occ, mask), frac_in(occ, mask.T)
    assert inside > 0.95, (f"the configuration places {inside:.3f} of its mass "
                           f"inside its own mask and {transposed:.3f} inside "
                           f"the mask's transpose")


@pytest.mark.parametrize("name", ["T", "L"])
def test_mask_to_configuration_round_trips(name):
    """`goal_configs.mask_to_configuration` emits world poses. Rasterising
    them with `particles_to_occupancy` must reproduce the mask they came
    from, not its transpose."""
    mask = letter_mask(name, GRID, GRID)
    res = mask_to_configuration(mask, n_objects=20, seed=0)
    xy = res.poses[:, :2]
    occ = occ_from_world_xy(xy[:, 0], xy[:, 1])
    inside, transposed = frac_in(occ, mask), frac_in(occ, mask.T)
    # Not 1.0: a thin glyph stroke cannot fit 20 non-penetrating cubes, so
    # `mask_to_configuration` dilates the PLACEMENT mask (documented in its
    # own docstring) and some cubes legitimately land just outside the
    # original mask. That is a packing property, not an axis property -- what
    # the axis convention controls is which of the mask and its transpose the
    # material lands on, so assert the ratio, not an absolute fraction. The
    # dilation-free case is covered above by
    # `test_mask_to_configuration_round_trips_thick_mask`.
    assert inside > 0.2 and inside > 3.0 * transposed, (
        f"letter '{name}': the configuration places {inside:.3f} of its mass "
        f"inside its own mask and {transposed:.3f} inside the mask's transpose")
