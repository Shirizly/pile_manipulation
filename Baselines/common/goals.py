"""Baselines/common/goals.py -- goal-region masks and value functions for
the multi-goal / multi-value-function MPC-effectiveness ("capture"/`slateN`)
report (2026-09-10 task: score GNN/NFD baselines across L20mm, L40mm, and
the overnight_randlen held-out test, for both image accuracy and control
utility). See `docs/experiments/METRICS.md` for the formal writeup this
module implements.

Three goal SHAPES, each a binary (H,W) numpy mask (True = inside the
target region), in **`transforms/functional.py::particles_to_occupancy`'s
convention: row = world x, col = world y** ("convention A"), which is also
what `control_utility_test.py::lyapunov_weights` now uses.

CONVENTION CHANGE, 2026-09-17 (invariant `goal-mask-axis-convention-row-y-col-x`,
which was `broken`, now `fixed`): until this date this module built masks with
row = world y, col = world x, the opposite of the rasteriser, and the docstring
here documented that divergence as if it were harmless. It is not: every value
function multiplies an occupancy by a mask, so for any ASYMMETRIC target the
two were TRANSPOSED relative to each other. Symmetric targets (`corner`,
`center`, quadrants 0 and 3) are their own transpose, which is why it survived.
Masks are now built natively in convention A -- do NOT "fix" a mask by
transposing it at the call site; see `tests/test_goal_axis_convention.py`:

  * random-quadrant -- one of the workspace's 4 quadrants, chosen by a
    SEEDED rng so the SAME quadrant is assigned to a given slate across
    every model being compared. Never re-randomise per model/per call --
    that would make cross-model comparison meaningless (model A being
    scored against a different goal than model B for "the same" slate).
  * ring ("O")   -- `utils.py::gen_goal_shape`'s own 'O' glyph asset
    (`env/target_shapes/{font}/helvetica_O.npy`), binarised with the SAME
    threshold that function uses internally (`goal <= 0.5`) so this mask
    is pixel-identical to what `gen_goal_shape('O', ...)` would binarise
    to, not a separately re-derived approximation. A genuine ring/annulus
    (the glyph's hollow centre is NOT part of the mask).
  * T-shape      -- same, for 'T'.

Three VALUE functions over a mask `m` and an occupancy grid `occ`:

  * lyapunov distance-to-goal -- the EXISTING `control_utility_test.lyapunov`
    (a COST: lower is better), fed a distance-transform weight field built
    the same way `lyapunov_weights`'s own general branch does
    (`dist_field_from_mask`, below) so it is consistent with the rest of
    this project's Lyapunov scoring, not a new formula.
  * mass_in_region -- sum(occ * m): raw pile mass inside the target,
    UNNORMALISED (deliberately not divided by total occupied mass, unlike
    `lyapunov`'s own normalisation) -- a literal "binary mask multiplied by
    the state". A VALUE (higher is better).
  * signed_mass_in_region -- sum(occ * (2*m - 1)): like the above, but
    pixels OUTSIDE the target subtract instead of contributing zero. A
    VALUE (higher is better).

FOLLOW-UP IDEA, not implemented here (recorded per 2026-09-10 direction):
some combination of `mass_in_region`/`signed_mass_in_region` (which respect
fine detail -- a single misplaced pixel changes them) and the Lyapunov
distance field (which is smooth and gradient-friendly for MPC gradient
descent) might give a metric with both properties. Not attempted; see
`docs/experiments/METRICS.md`.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

LETTER_FONT = "helvetica_thin"


def quadrant_mask(H: int, W: int, quadrant: int) -> np.ndarray:
    """`quadrant` in {0,1,2,3} indexes the four WORLD quadrants
    {(-x,-y), (+x,-y), (-x,+y), (+x,+y)} -- the same four world regions the
    old row=y/col=x version named {top-left, top-right, bottom-left,
    bottom-right}, but built in convention A (row = world x, col = world y).

    Quadrants 0 and 3 are their own transpose and are therefore BYTE-IDENTICAL
    to the pre-2026-09-17 masks; 1 and 2 swap. That is the whole blast radius
    of the convention fix on this function."""
    if quadrant not in (0, 1, 2, 3):
        raise ValueError(f"quadrant must be 0-3, got {quadrant}")
    mask = np.zeros((H, W), dtype=bool)
    # row = world x: low half for -x (quadrants 0, 2), high half for +x (1, 3).
    r0, r1 = (0, H // 2) if quadrant in (0, 2) else (H // 2, H)
    # col = world y: low half for -y (quadrants 0, 1), high half for +y (2, 3).
    c0, c1 = (0, W // 2) if quadrant in (0, 1) else (W // 2, W)
    mask[r0:r1, c0:c1] = True
    return mask


def random_quadrant_mask(H: int, W: int, seed) -> tuple[np.ndarray, int]:
    """Picks one of the 4 quadrants via a SEEDED rng (`seed` should be a
    stable per-slate identifier, e.g. the slate's flat index or file name
    hash) so repeated calls with the same `seed` -- across different
    models being compared on the same slate -- always agree. Returns
    (mask, quadrant_index)."""
    rng = np.random.default_rng(seed)
    q = int(rng.integers(4))
    return quadrant_mask(H, W, q), q


def letter_mask(name: str, H: int, W: int, font_name: str = LETTER_FONT) -> np.ndarray:
    """Binary mask for a font glyph, thresholded EXACTLY as
    `utils.py::gen_goal_shape` does internally (`goal <= 0.5`) on the SAME
    precomputed asset (`env/target_shapes/{font_name}/helvetica_{name}.npy`)
    -- not reimplemented from scratch, just the binarisation step isolated
    from that function's distance-transform/visualisation outputs, which
    this caller does not need.

    AXIS CONVENTION: the `.npy` asset is stored the way a human reads the
    glyph -- asset row increases with world y, asset column increases with
    world x. This module's masks are in convention A (row = world x, col =
    world y), so the asset's two axes are read out in the opposite order.
    This is an axis RELABEL of an external asset (there is no row/col
    construction here to swap), not a post-hoc correction: the returned mask
    is the glyph as it stands in the world, and rasterising material placed
    on the glyph with `particles_to_occupancy` reproduces exactly this array.
    `cv2.resize` takes its size argument as (width, height), i.e. (n_cols,
    n_rows) of the ASSET's own layout, so it is sized (H, W) here -- the
    asset's column axis becomes the mask's row axis."""
    root_dir = f"env/target_shapes/{font_name}"
    shape_path = os.path.join(root_dir, f"helvetica_{name}.npy")
    goal = np.load(shape_path)
    # asset (row=y, col=x) -> world (x, y): resize to (W_asset=H, H_asset=W)
    # so that after the axis swap the mask is (H, W) = (n_x, n_y).
    goal = cv2.resize(goal, (H, W), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray((goal <= 0.5).T)


def dist_field_from_mask(mask: np.ndarray) -> np.ndarray:
    """Normalised [0,1] distance-to-mask field, matching
    `control_utility_test.lyapunov_weights`'s own general (non-`ind-`,
    non-`distclip-`) branch exactly: `distance_transform_edt(~mask)`,
    (axis-convention-agnostic: the field inherits whatever convention the
    mask it is given is in, so feed it a convention-A mask),
    divided by its own max. 0 inside/at the mask, up to 1 at the point(s)
    farthest from it."""
    d = distance_transform_edt(~mask).astype(np.float32)
    return d / max(float(d.max()), 1e-6)


def mass_in_region(occ: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """occ: (N,H,W) or (N,H*W). mask: (H,W) or (H*W,), bool/float. Returns
    (N,) unnormalised mass-in-region VALUE (higher is better)."""
    n = occ.shape[0]
    flat = occ.reshape(n, -1)
    m = mask.reshape(1, -1).to(flat.dtype) if isinstance(mask, torch.Tensor) \
        else torch.as_tensor(mask, dtype=flat.dtype).reshape(1, -1)
    return (flat * m).sum(1)


def signed_mass_in_region(occ: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Like `mass_in_region`, but pixels OUTSIDE the mask subtract instead
    of contributing zero: sum(occ * (2*mask - 1))."""
    n = occ.shape[0]
    flat = occ.reshape(n, -1)
    m = mask.reshape(1, -1).to(flat.dtype) if isinstance(mask, torch.Tensor) \
        else torch.as_tensor(mask, dtype=flat.dtype).reshape(1, -1)
    signed = 2 * m - 1
    return (flat * signed).sum(1)


def slate_n_capture(value_pred: torch.Tensor, value_true: torch.Tensor,
                     higher_is_better: bool) -> float:
    """The full-pool ("slateN") capture metric over one slate (one
    same-state pool of candidate actions): the model picks the argmax (or
    argmin, if `higher_is_better=False`) of its OWN predicted value across
    every candidate in the pool; scoring then switches entirely to ground
    truth, asking how much of the gap between a random pick and the true
    best the model's choice actually captured:

        capture = (true[chosen] - mean(true)) / (true[best] - mean(true))

    0 = no better than a random pick, 1 = as good as an oracle that could
    see the true values directly, <0 = worse than random. Mathematically
    the K=n (full pool, no replacement) case of `docs/experiments/
    METRICS.md`'s `slateK_exact`/`slateN` metric -- reduces to this closed
    form exactly because an n-candidate pool has only one possible
    size-n subset (`slateK_exact`'s rank-weighting collapses to weight 1
    on the model's own top rank at K=n).
    """
    if not higher_is_better:
        value_pred = -value_pred
        value_true = -value_true
    best_idx = int(torch.argmax(value_pred))
    chosen = float(value_true[best_idx])
    true_best = float(value_true.max())
    mean_true = float(value_true.mean())
    denom = true_best - mean_true
    if abs(denom) < 1e-9:
        return float("nan")
    return (chosen - mean_true) / denom
