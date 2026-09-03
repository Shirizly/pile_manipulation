"""Tests for Genesis/spawn_geometry.py — pyramid spawn layouts. Genesis-free."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from spawn_geometry import (  # noqa: E402
    heap_positions, pyramid_layer_plan, pyramid_positions, stagger_layers,
)

SIZE = 0.005


def test_every_cube_is_placed():
    """Parking the remainder was the first design and it wrecked the
    measurement — parked cubes either fall through the floor or register as a
    spurious bottom layer. So the plan must account for all n."""
    for n in (1, 5, 14, 30, 50, 80, 137):
        assert sum(pyramid_layer_plan(n)) == n
        pos, _ = pyramid_positions(n, SIZE)
        assert pos.shape == (n, 3)


def test_layer_plan_is_a_complete_pyramid_with_the_remainder_in_the_base():
    assert pyramid_layer_plan(14) == [9, 4, 1]
    assert pyramid_layer_plan(30) == [16, 9, 4, 1]
    assert pyramid_layer_plan(50) == [36, 9, 4, 1]     # 30 complete + 20 in base
    assert pyramid_layer_plan(55) == [25, 16, 9, 4, 1]


def test_fifty_cubes_gives_four_layers():
    """50 was asked about specifically: it comfortably supports real depth."""
    pos, n_layers = pyramid_positions(50, SIZE)
    assert n_layers == 4
    assert len(torch.unique(torch.round(pos[:, 2] / SIZE))) == 4


def test_layer_count_is_monotone_in_n():
    """Caught a real wart: choosing the base from the smallest full-pyramid sum
    that REACHES n and truncating gave 5 layers at n=55 and only 2 at n=56."""
    prev = 0
    for n in range(1, 200):
        _, k = pyramid_positions(n, SIZE)
        assert k >= prev, f"layer count fell from {prev} to {k} at n={n}"
        prev = k


def test_bottom_layer_rests_on_the_floor():
    floor = 0.01
    pos, _ = pyramid_positions(30, SIZE, floor_z=floor)
    lowest = float(pos[:, 2].min())
    assert lowest == pytest.approx(floor + SIZE / 2, abs=1e-4), (
        "bottom cube centres must sit half a cube above the floor, or they "
        "interpenetrate it and get ejected")


def test_no_two_cubes_overlap():
    pos, _ = pyramid_positions(50, SIZE)
    d = (pos[:, None, :] - pos[None, :, :]).abs()
    same = torch.eye(len(pos), dtype=torch.bool)
    # Non-overlapping means separated by >= one cube edge on some axis.
    clear = (d >= SIZE * 0.99).any(dim=-1)
    assert bool((clear | same).all()), "cubes are born interpenetrating"


def test_each_layer_is_centred():
    pos, _ = pyramid_positions(30, SIZE, centre=(0.02, -0.01))
    for z in torch.unique(pos[:, 2]):
        layer = pos[pos[:, 2] == z]
        assert torch.allclose(layer[:, :2].mean(0),
                              torch.tensor([0.02, -0.01]), atol=1e-6)


def test_is_taller_and_narrower_than_a_flat_layer():
    """The whole point: depth in a small footprint."""
    pos, k = pyramid_positions(50, SIZE)
    span_z = float(pos[:, 2].max() - pos[:, 2].min())
    span_xy = float(pos[:, :2].max(0).values.sub(pos[:, :2].min(0).values).max())
    assert span_z == pytest.approx((k - 1) * SIZE, rel=0.05)
    # 50 cubes in one layer need a ~7x7 footprint; the pyramid's base is 6x6.
    assert span_xy < 7 * SIZE


def test_empty_input():
    assert pyramid_layer_plan(0) == []


class TestStaggerLayers:
    """Brick-bonding alternate layers of a pyramid."""

    def test_zero_stagger_is_a_noop(self):
        pos, _ = pyramid_positions(30, 0.005, floor_z=0.01)
        out = stagger_layers(pos, 0.005, stagger=0.0, floor_z=0.01)
        assert torch.equal(out, pos)

    def test_does_not_mutate_input(self):
        pos, _ = pyramid_positions(30, 0.005, floor_z=0.01)
        before = pos.clone()
        stagger_layers(pos, 0.005, stagger=0.5, floor_z=0.01)
        assert torch.equal(pos, before)

    def test_offsets_odd_layers_only(self):
        size, gap, floor_z = 0.005, 1.15, 0.01
        pos, _ = pyramid_positions(30, size, gap=gap, floor_z=floor_z)
        out = stagger_layers(pos, size, gap=gap, stagger=0.5, floor_z=floor_z)
        layer = ((pos[:, 2] - floor_z - 0.5 * size) / size).round()
        shift = 0.5 * size * gap
        even, odd = layer % 2 == 0, layer % 2 == 1
        assert torch.allclose(out[even], pos[even])
        assert torch.allclose(out[odd, :2], pos[odd, :2] + shift)
        # z is untouched: staggering is lateral only.
        assert torch.allclose(out[:, 2], pos[:, 2])


class TestHeapPositions:
    """The irregular two-layer heap used for the cube-count spectrum."""

    def test_places_every_cube(self):
        for n in (1, 7, 20, 50, 80):
            pos, _ = heap_positions(n, 0.003, floor_z=0.01)
            assert pos.shape == (n, 3)

    def test_is_two_layers_with_the_requested_split(self):
        size, floor_z = 0.003, 0.01
        pos, n_layers = heap_positions(50, size, floor_z=floor_z, base_frac=0.6)
        z0 = floor_z + 0.5 * size * 1.001
        layer = ((pos[:, 2] - z0) / size).round().long()
        assert n_layers == 2
        assert int((layer == 0).sum()) == 30
        assert int((layer == 1).sum()) == 20

    def test_every_upper_cube_rests_on_an_occupied_site(self):
        # This is what keeps the heap STABLE: unsupported cubes would fall and
        # the pile would flatten into the monolayer a dropped spawn gives.
        size, floor_z = 0.003, 0.01
        pos, _ = heap_positions(80, size, floor_z=floor_z)
        z0 = floor_z + 0.5 * size * 1.001
        layer = ((pos[:, 2] - z0) / size).round().long()
        base = {(round(float(x), 6), round(float(y), 6))
                for x, y in pos[layer == 0][:, :2]}
        for x, y in pos[layer > 0][:, :2]:
            assert (round(float(x), 6), round(float(y), 6)) in base

    def test_sits_on_the_floor(self):
        size, floor_z = 0.003, 0.01
        pos, _ = heap_positions(20, size, floor_z=floor_z)
        assert float(pos[:, 2].min()) == pytest.approx(
            floor_z + 0.5 * size * 1.001, abs=1e-9)

    def test_draws_differ(self):
        # The whole reason heap exists: a pyramid is the same lattice every
        # episode, which caps state diversity.
        a, _ = heap_positions(50, 0.003, floor_z=0.01,
                              generator=torch.Generator().manual_seed(0))
        b, _ = heap_positions(50, 0.003, floor_z=0.01,
                              generator=torch.Generator().manual_seed(1))
        assert not torch.allclose(a[:, :2].flatten().sort().values,
                                  b[:, :2].flatten().sort().values)

    def test_reproducible_given_a_generator(self):
        a, _ = heap_positions(50, 0.003, floor_z=0.01,
                              generator=torch.Generator().manual_seed(7))
        b, _ = heap_positions(50, 0.003, floor_z=0.01,
                              generator=torch.Generator().manual_seed(7))
        assert torch.equal(a, b)

    def test_stays_compact(self):
        # A heap must remain a PILE; if it spread like a dropped spawn
        # (65-67 mm measured) it would defeat the purpose.
        pos, _ = heap_positions(80, 0.003, floor_z=0.01)
        span = float(pos[:, :2].max(0).values.sub(pos[:, :2].min(0).values).max())
        assert span < 0.040

    def test_empty(self):
        pos, n_layers = heap_positions(0, 0.003)
        assert pos.shape == (0, 3) and n_layers == 0

    def test_survives_a_non_cpu_default_device(self):
        # Genesis sets torch's DEFAULT device to cuda, which made a bare
        # torch.arange inside heap_positions land on the GPU while the
        # CPU-built index tensors did not: "Expected all tensors to be on the
        # same device" at the final cat. Every creation now names its device.
        if not torch.cuda.is_available():
            pytest.skip("no cuda device to set as default")
        try:
            torch.set_default_device("cuda")
            pos, n_layers = heap_positions(50, 0.003, floor_z=0.01,
                                           device="cuda")
            assert pos.shape == (50, 3) and n_layers == 2
        finally:
            torch.set_default_device("cpu")
