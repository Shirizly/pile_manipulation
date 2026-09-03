"""
transforms/sand_occupancy.py — projecting a granular CONTINUUM to model inputs.

Why this is not `particles_to_occupancy`
---------------------------------------
The rigid-cube path represents the scene as a binary silhouette: every occupied
cell is exactly 1.0, because `particles_to_occupancy` scatter-adds a count and
then clamps to [0, 1]. For 30-80 discrete cubes that is a defensible
representation of "is there an object here".

For sand it throws away the only thing that matters. MPM sand is a continuum
sampled by thousands of equal-mass particles, so the number of particles in a
column *is* the material's local depth. Clamping it makes a 1-particle-deep
smear and a 40-particle-deep dune identical — precisely the confusion that made
the cube datasets uninformative about depth
(docs/linear_foresight_findings.md §3.2).

So the sand projections here are **mass-preserving and unclamped**:

    sand_to_density     top-down column mass per cell. The direct analogue of
                        the paper's greyscale image, and the natural input for
                        an image-space transport operator: a push moves mass
                        between cells, and this is the quantity that is
                        conserved when it does.
    sand_to_heightmap   max particle height per cell. Complementary, not a
                        substitute: it says how TALL the material is, which
                        column mass cannot distinguish from how WIDE.

Both are pure torch with no `genesis` import, so they are unit tested without a
GPU (tests/test_sand_occupancy.py), following the same ownership rule as the
rest of `transforms/` (docs/UTILITIES.md).

A note on smoothing. The linear-foresight work found that the SE(2) warp its
operator needs costs more accuracy than one push changes when the field has
features at the pixel scale, and that a sigma ~ 1 px blur removes that cost
entirely. A sand density map is naturally smoother than a cube silhouette, but
`sigma` is exposed here so the same precondition can be met explicitly rather
than hoped for.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def _grid_index(particles: torch.Tensor, bounds: Dict[str, float],
                grid_res: Tuple[int, int]):
    """Map (B, N, 3) world xy to integer cell indices, with an in-bounds mask.

    Grid convention matches the cube path: ``dim 0`` spans x, ``dim 1`` spans y
    (verified empirically against the dataset's own rasterised plate channel —
    see the mapping self-check in `fit_linear_foresight.py`).
    """
    H, W = int(grid_res[0]), int(grid_res[1])
    x0, x1 = float(bounds["x_min"]), float(bounds["x_max"])
    y0, y1 = float(bounds["y_min"]), float(bounds["y_max"])

    fx = (particles[..., 0] - x0) / max(x1 - x0, 1e-12) * H
    fy = (particles[..., 1] - y0) / max(y1 - y0, 1e-12) * W
    ix = fx.floor().long()
    iy = fy.floor().long()
    inside = (ix >= 0) & (ix < H) & (iy >= 0) & (iy < W)
    return ix.clamp(0, H - 1), iy.clamp(0, W - 1), inside


def _gaussian_blur2d(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on (B, H, W), reflect-padded to conserve mass."""
    if sigma <= 0:
        return field
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=field.device, dtype=field.dtype) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma * sigma))
    g = g / g.sum()
    x = field.unsqueeze(1)
    pad = k // 2
    x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode="reflect")
    x = torch.nn.functional.conv2d(x, g.view(1, 1, -1, 1))
    return x.squeeze(1)


def sand_to_density(particles: torch.Tensor,
                    bounds: Dict[str, float],
                    grid_res: Tuple[int, int],
                    sigma: float = 0.0,
                    normalize: Optional[str] = "mean",
                    particle_mass: float = 1.0) -> torch.Tensor:
    """Top-down column mass per cell: (B, N, 3) particles -> (B, H, W).

    Every particle contributes ``particle_mass`` to the cell its xy falls in,
    regardless of height — so the value is the material's local depth, and the
    total is conserved by any push that does not throw sand out of bounds.
    **Deliberately unclamped**; see the module docstring.

    Parameters
    ----------
    normalize : how to scale the result.
        ``None``   raw mass (particle counts x ``particle_mass``). Use when
                  absolute mass matters, e.g. checking conservation.
        ``"mean"`` divide by the mean over *occupied* cells, so a typical
                  occupied cell reads ~1.0 and the map is comparable across
                  particle counts and grid resolutions. This is the default
                  because it makes sand maps directly comparable to the cube
                  silhouettes, whose occupied cells are 1.0 by construction.
        ``"max"``  divide by the per-frame maximum, giving [0, 1].
    sigma : Gaussian blur in cells, applied after binning. Mass-preserving
        (reflect padding). Use ~1.0 if the map feeds an SE(2)-warping model.

    Out-of-bounds particles are dropped, not clamped to the edge: clamping
    would pile escaped sand into a false ridge along the wall, which is exactly
    the artefact a transport model would then learn.
    """
    if particles.dim() != 3:
        raise ValueError(f"particles must be (B, N, 3), got {tuple(particles.shape)}")
    B = particles.shape[0]
    H, W = int(grid_res[0]), int(grid_res[1])
    ix, iy, inside = _grid_index(particles, bounds, grid_res)

    flat = (ix * W + iy).clamp(0, H * W - 1)
    weight = inside.to(particles.dtype) * float(particle_mass)
    out = torch.zeros((B, H * W), dtype=particles.dtype, device=particles.device)
    out.scatter_add_(1, flat, weight)
    out = out.view(B, H, W)

    out = _gaussian_blur2d(out, sigma)

    if normalize == "mean":
        occupied = (out > 0).flatten(1).sum(1).clamp_min(1)
        scale = out.flatten(1).sum(1) / occupied
        out = out / scale.clamp_min(1e-12).view(B, 1, 1)
    elif normalize == "max":
        peak = out.flatten(1).max(1).values.clamp_min(1e-12)
        out = out / peak.view(B, 1, 1)
    elif normalize is not None:
        raise ValueError(f"normalize must be None, 'mean' or 'max', got {normalize!r}")
    return out


def sand_to_heightmap(particles: torch.Tensor,
                      bounds: Dict[str, float],
                      grid_res: Tuple[int, int],
                      floor_z: float = 0.0,
                      sigma: float = 0.0) -> torch.Tensor:
    """Height of the topmost particle per cell: (B, N, 3) -> (B, H, W), metres.

    Empty cells read ``0.0`` (i.e. floor level), not ``-inf``, so the map is
    usable as a model input directly.

    Complementary to `sand_to_density`, not a replacement. Column mass cannot
    separate "a wide thin layer" from "a narrow tall dune" once the pile spreads
    past one cell; height can. Height in turn cannot see how much material sits
    under the surface. A model that needs both should take them as two channels.
    """
    if particles.dim() != 3:
        raise ValueError(f"particles must be (B, N, 3), got {tuple(particles.shape)}")
    B = particles.shape[0]
    H, W = int(grid_res[0]), int(grid_res[1])
    ix, iy, inside = _grid_index(particles, bounds, grid_res)

    flat = (ix * W + iy).clamp(0, H * W - 1)
    z = (particles[..., 2] - float(floor_z)).clamp_min(0.0)
    z = torch.where(inside, z, torch.zeros_like(z))

    out = torch.zeros((B, H * W), dtype=particles.dtype, device=particles.device)
    out.scatter_reduce_(1, flat, z, reduce="amax", include_self=True)
    return _gaussian_blur2d(out.view(B, H, W), sigma)


def sand_mass(particles: torch.Tensor, bounds: Dict[str, float]) -> torch.Tensor:
    """Fraction of particles inside the grid bounds, per frame: (B,).

    The conservation check for a sand dataset. A push should move sand, not
    delete it; a run where this drifts is losing material through a wall or out
    of the MPM domain, and its transitions are not describing the physics the
    model is meant to learn. The rigid-cube datasets conserved mass to 0.4%
    (docs/linear_foresight_findings.md §3.2) — sand should be checked to the
    same standard rather than assumed.
    """
    _, _, inside = _grid_index(particles, bounds, (1, 1))
    x0, x1 = float(bounds["x_min"]), float(bounds["x_max"])
    y0, y1 = float(bounds["y_min"]), float(bounds["y_max"])
    ok = ((particles[..., 0] >= x0) & (particles[..., 0] < x1)
          & (particles[..., 1] >= y0) & (particles[..., 1] < y1))
    return ok.to(particles.dtype).mean(dim=1)
