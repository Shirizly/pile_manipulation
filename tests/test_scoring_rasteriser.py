"""Ground-truth scoring rasteriser (EXP-0027, invariant `score-occupancy-subpixel-stable`).

The hard footprint rasteriser gives a particle 4-6 pixels depending on its
sub-pixel position, which made image-based dv swing by up to ~40% of a push's
effect under ~1 mm of motion (`occupancy-dv-subpixel-stable`, broken). The
scoring rasteriser must (1) give every particle the same mass wherever it
falls, (2) keep lyapunov stable under sub-pixel shifts, and (3) sit on the
same grid/axis convention as the model-input rasteriser. Genesis-free, CPU.
"""
import torch

from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.adapters import OCC_GRID, occ_for_scoring, occ_from_particles
from transforms.functional import splat_particles_mass


def _states(n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(1, n, 7)
    s[0, :, :2] = (torch.rand(n, 2, generator=g) - 0.5) * 0.10
    return s


def test_mass_is_constant_under_subpixel_shift():
    uv = torch.tensor([[[20.0, 30.0]]])
    masses = [float(splat_particles_mass(uv + torch.tensor([du, dv]), (64, 64)).sum())
              for du in torch.linspace(0, 1, 7) for dv in torch.linspace(0, 1, 7)]
    assert max(masses) - min(masses) < 1e-4


def test_lyapunov_stable_under_subpixel_shift_unlike_hard_footprint():
    dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
    s = _states()
    shifts = torch.linspace(0, 0.002, 9)                 # 0-2 mm = 0-1 px
    soft, hard = [], []
    for d in shifts:
        t = s.clone(); t[0, :, 0] += d
        soft.append(float(lyapunov(occ_for_scoring(t), dw)))
        hard.append(float(lyapunov(occ_from_particles(t), dw)))
    soft, hard = torch.tensor(soft), torch.tensor(hard)
    # a rigid shift moves V smoothly (almost linearly) under soft scoring
    resid_soft = (soft - torch.linspace(soft[0], soft[-1], len(soft))).abs().max()
    resid_hard = (hard - torch.linspace(hard[0], hard[-1], len(hard))).abs().max()
    assert resid_soft < 5e-4
    assert resid_hard > 3 * resid_soft                   # documents why the change was made


def test_same_grid_convention_as_model_input_rasteriser():
    """One particle at a time (so the hard rasteriser's own overlap/aliasing
    does not enter): the soft centroid sits at the particle's continuous grid
    coordinate, and the hard footprint's pixel centroid is within half a
    pixel of it, on the same axes (dim 0 = world x)."""
    from simple_mpc.adapters import OCC_BOUNDS
    lo = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]])
    hi = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]])
    g = torch.Generator().manual_seed(0)
    iu = torch.arange(OCC_GRID).float()
    for _ in range(20):
        s = torch.zeros(1, 1, 7)
        s[0, 0, :2] = (torch.rand(2, generator=g) - 0.5) * 0.10
        uv = (s[0, 0, :2] - lo) / (hi - lo) * (OCC_GRID - 1)
        a, b = occ_for_scoring(s)[0], occ_from_particles(s)[0]
        ca = torch.stack([(a.sum(1) * iu).sum() / a.sum(), (a.sum(0) * iu).sum() / a.sum()])
        cb = b.nonzero().float().mean(0)
        assert (ca - uv).abs().max() < 0.05
        assert (cb - uv).abs().max() < 0.5
