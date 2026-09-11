"""Baselines/common/goals.py -- goal-region masks and value functions for
the multi-goal / multi-value-function MPC-effectiveness ("capture"/`slateN`)
report (2026-09-10 task: score GNN/NFD baselines across L20mm, L40mm, and
the overnight_randlen held-out test, for both image accuracy and control
utility). See `docs/experiments/METRICS.md` for the formal writeup this
module implements.

Three goal SHAPES, each a binary (H,W) numpy mask (True = inside the
target region), in `control_utility_test.py::lyapunov_weights`'s own
row=y/col=x convention -- NOT `transforms/functional.py`'s row=x/col=y
convention (a different subsystem; mixing the two up would be exactly the
axis-swap bug this repo has a documented history of):

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
    """`quadrant` in {0,1,2,3} = {top-left, top-right, bottom-left,
    bottom-right} (row=y/col=x convention, matching `lyapunov_weights`)."""
    if quadrant not in (0, 1, 2, 3):
        raise ValueError(f"quadrant must be 0-3, got {quadrant}")
    mask = np.zeros((H, W), dtype=bool)
    r0, r1 = (0, H // 2) if quadrant in (0, 1) else (H // 2, H)
    c0, c1 = (0, W // 2) if quadrant in (0, 2) else (W // 2, W)
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
    this caller does not need."""
    root_dir = f"env/target_shapes/{font_name}"
    shape_path = os.path.join(root_dir, f"helvetica_{name}.npy")
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (W, H), interpolation=cv2.INTER_AREA)
    return goal <= 0.5


def dist_field_from_mask(mask: np.ndarray) -> np.ndarray:
    """Normalised [0,1] distance-to-mask field, matching
    `control_utility_test.lyapunov_weights`'s own general (non-`ind-`,
    non-`distclip-`) branch exactly: `distance_transform_edt(~mask)`,
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
