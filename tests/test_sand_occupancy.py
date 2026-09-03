"""Tests for transforms/sand_occupancy.py — continuum projections. Genesis-free."""

import pytest
import torch

from transforms.sand_occupancy import (
    sand_mass, sand_to_density, sand_to_heightmap,
)

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
RES = (64, 64)


def _column(n, x=0.0, y=0.0, z0=0.0, dz=0.001):
    """n particles stacked in one xy column."""
    p = torch.zeros(1, n, 3)
    p[0, :, 0] = x
    p[0, :, 1] = y
    p[0, :, 2] = z0 + dz * torch.arange(n).float()
    return p


def test_density_counts_depth_rather_than_presence():
    """The whole reason this is not particles_to_occupancy: a deep column and a
    shallow one must not read the same."""
    shallow = sand_to_density(_column(1), BOUNDS, RES, normalize=None)
    deep = sand_to_density(_column(40), BOUNDS, RES, normalize=None)
    assert float(shallow.max()) == pytest.approx(1.0)
    assert float(deep.max()) == pytest.approx(40.0)


def test_density_conserves_mass():
    g = torch.Generator().manual_seed(0)
    p = (torch.rand(2, 5000, 3, generator=g) - 0.5) * 0.1
    d = sand_to_density(p, BOUNDS, RES, normalize=None)
    assert torch.allclose(d.flatten(1).sum(1), torch.full((2,), 5000.0), rtol=1e-4)


def test_blur_conserves_mass():
    g = torch.Generator().manual_seed(1)
    p = (torch.rand(1, 3000, 3, generator=g) - 0.5) * 0.08
    raw = sand_to_density(p, BOUNDS, RES, sigma=0.0, normalize=None)
    smooth = sand_to_density(p, BOUNDS, RES, sigma=1.5, normalize=None)
    assert float(smooth.sum()) == pytest.approx(float(raw.sum()), rel=1e-3)


def test_out_of_bounds_is_dropped_not_clamped():
    """Clamping would pile escaped sand into a false ridge along the wall —
    exactly the artefact a transport model would then learn."""
    p = torch.zeros(1, 10, 3)
    p[0, :, 0] = 0.5          # far outside
    d = sand_to_density(p, BOUNDS, RES, normalize=None)
    assert float(d.sum()) == 0.0


def test_normalize_mean_makes_a_typical_cell_about_one():
    g = torch.Generator().manual_seed(2)
    for n in (2000, 20000):
        p = (torch.rand(1, n, 3, generator=g) - 0.5) * 0.06
        d = sand_to_density(p, BOUNDS, RES, normalize="mean")
        occ = d[d > 0]
        assert 0.5 < float(occ.mean()) < 2.0, n


def test_normalize_max_is_bounded():
    g = torch.Generator().manual_seed(3)
    p = (torch.rand(1, 4000, 3, generator=g) - 0.5) * 0.06
    d = sand_to_density(p, BOUNDS, RES, normalize="max")
    assert float(d.max()) == pytest.approx(1.0)
    assert float(d.min()) >= 0.0


def test_density_is_unclamped():
    """A regression guard: the cube path clamps to [0,1] and that is the bug
    this module exists to avoid."""
    d = sand_to_density(_column(100), BOUNDS, RES, normalize=None)
    assert float(d.max()) > 1.0


def test_heightmap_reports_the_topmost_particle():
    p = _column(20, z0=0.01, dz=0.002)          # top at 0.01 + 19*0.002
    h = sand_to_heightmap(p, BOUNDS, RES, floor_z=0.01)
    assert float(h.max()) == pytest.approx(19 * 0.002, abs=1e-6)


def test_heightmap_empty_cells_read_floor_level():
    h = sand_to_heightmap(_column(5, z0=0.01), BOUNDS, RES, floor_z=0.01)
    assert float(h.min()) == 0.0
    assert torch.isfinite(h).all()


def test_density_and_height_separate_wide_from_tall():
    """Neither map alone distinguishes these; that is why both exist."""
    tall = _column(64, z0=0.0, dz=0.001)
    wide = torch.zeros(1, 64, 3)
    wide[0, :, 0] = torch.linspace(-0.03, 0.03, 64)
    d_tall = sand_to_density(tall, BOUNDS, RES, normalize=None)
    d_wide = sand_to_density(wide, BOUNDS, RES, normalize=None)
    assert float(d_tall.max()) > float(d_wide.max()) * 10
    h_tall = sand_to_heightmap(tall, BOUNDS, RES)
    h_wide = sand_to_heightmap(wide, BOUNDS, RES)
    assert float(h_tall.max()) > float(h_wide.max())


def test_sand_mass_flags_material_leaving_the_tray():
    p = torch.zeros(1, 100, 3)
    p[0, :50, 0] = 0.0            # inside
    p[0, 50:, 0] = 0.5            # escaped
    assert float(sand_mass(p, BOUNDS)[0]) == pytest.approx(0.5)


def test_batch_independence():
    g = torch.Generator().manual_seed(4)
    p = (torch.rand(3, 1000, 3, generator=g) - 0.5) * 0.06
    d = sand_to_density(p, BOUNDS, RES, normalize=None)
    for i in range(3):
        one = sand_to_density(p[i:i + 1], BOUNDS, RES, normalize=None)
        assert torch.allclose(d[i], one[0])


def test_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(B, N, 3\)"):
        sand_to_density(torch.zeros(10, 3), BOUNDS, RES)
