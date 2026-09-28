"""Cube-cube non-overlap projection (EXP-0059 R2 diagnostics: the "transport +
5 mm non-overlap projection" transfer rule).

A small, reusable, vectorised relaxation: any pair of cubes closer than
`min_dist` is pushed apart along their connecting vector, split evenly
between the two (symmetric), repeated for `n_iters` passes. This is the
documented limitation flagged in `predictor.py` ("clean" meant "one
well-defined state", not "verified non-penetrating") -- this module is the
fix, kept separate and opt-in rather than folded into every predictor, since
not every transfer rule needs it and it is a real (if cheap) extra cost.

Not a physically exact contact solver (no mass, no friction, a handful of
Jacobi-style iterations) -- good enough to remove gross overlap introduced
by transferring cube positions independently of each other, not a substitute
for re-simulating.
"""
from __future__ import annotations

import torch


def resolve_overlaps(xy: torch.Tensor, min_dist: float = 0.005, n_iters: int = 5) -> torch.Tensor:
    """xy: (..., n, 2) world/push-frame positions (any consistent frame).
    -> (..., n, 2) with every pairwise distance nudged toward >= min_dist.

    Vectorised over the batch dimension(s); the pairwise n x n distance
    computation is the only non-linear cost, negligible at this project's
    n=20 cubes/state.
    """
    xy = xy.clone()
    n = xy.shape[-2]
    eye = torch.eye(n, device=xy.device, dtype=torch.bool)
    for _ in range(n_iters):
        diff = xy.unsqueeze(-2) - xy.unsqueeze(-3)         # (..., n, n, 2): j -> i
        dist = diff.norm(dim=-1).clamp_min(1e-9)            # (..., n, n)
        deficit = (min_dist - dist).clamp_min(0.0)
        deficit = deficit.masked_fill(eye.expand_as(deficit), 0.0)
        if float(deficit.max()) <= 0.0:
            break
        direction = diff / dist.unsqueeze(-1)                # unit vector j -> i
        # each pair contributes half the deficit to EACH of its two cubes,
        # pushing them apart along `direction`; sum contributions from every
        # other cube onto cube i, then average so a cube in several
        # overlaps does not get pushed by the FULL sum of all of them at once
        push = (direction * (0.5 * deficit).unsqueeze(-1)).sum(dim=-2)   # (..., n, 2)
        n_active = (deficit > 0).sum(dim=-1).clamp_min(1).unsqueeze(-1)
        xy = xy + push / n_active
    return xy
