"""Tests for Genesis/action_sampling.py — batch-aware action shaping.

Genesis-free: the module is pure torch geometry.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from action_sampling import (  # noqa: E402
    duplicate_action_mask, equalize_travel_distance, shared_batch_distance,
)

LOW = torch.tensor([-0.05, -0.05])
HIGH = torch.tensor([0.05, 0.05])


def _bounds(shape):
    return LOW.expand(*shape, 2).clone(), HIGH.expand(*shape, 2).clone()


def test_equalized_pushes_all_travel_the_target_distance():
    starts = torch.tensor([[0.0, 0.0], [-0.01, 0.02], [0.03, -0.03]])
    stops = torch.tensor([[0.01, 0.0], [-0.01, 0.03], [0.01, -0.03]])
    low, high = _bounds((3,))
    target = torch.full((3, 1), 0.02)

    new_stops, clipped = equalize_travel_distance(starts, stops, low, high, target)

    travelled = (new_stops - starts).norm(dim=-1)
    assert torch.allclose(travelled, torch.full((3,), 0.02), atol=1e-6)
    assert not clipped.any()


def test_direction_is_preserved():
    starts = torch.tensor([[0.0, 0.0]])
    stops = torch.tensor([[0.006, 0.008]])          # direction (0.6, 0.8)
    low, high = _bounds((1,))

    new_stops, _ = equalize_travel_distance(
        starts, stops, low, high, torch.tensor([[0.02]]))

    unit_before = (stops - starts) / (stops - starts).norm()
    unit_after = (new_stops - starts) / (new_stops - starts).norm()
    assert torch.allclose(unit_before, unit_after, atol=1e-6)


def test_pushes_stay_inside_their_box():
    """A target longer than the box must truncate at the boundary, not escape."""
    starts = torch.tensor([[0.04, 0.0]])            # near the +x wall
    stops = torch.tensor([[0.045, 0.0]])            # heading further +x
    low, high = _bounds((1,))

    new_stops, clipped = equalize_travel_distance(
        starts, stops, low, high, torch.tensor([[0.5]]))

    assert clipped.all(), "an unreachable target must be reported as clipped"
    assert new_stops[0, 0] <= 0.05 + 1e-9
    assert torch.allclose(new_stops, torch.tensor([[0.05, 0.0]]), atol=1e-6)


def test_shrinking_is_always_possible():
    """A target shorter than the original never needs clipping."""
    g = torch.Generator().manual_seed(0)
    starts = (torch.rand(64, 2, generator=g) - 0.5) * 0.09
    stops = (torch.rand(64, 2, generator=g) - 0.5) * 0.09
    low, high = _bounds((64,))
    dist = (stops - starts).norm(dim=-1, keepdim=True)

    _, clipped = equalize_travel_distance(starts, stops, low, high, dist * 0.5)
    assert not clipped.any()


def test_shared_batch_distance_takes_one_envs_draw():
    """A summary statistic would collapse between-batch variation as envs grow."""
    dist = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]])  # (envs, samples, 1)
    shared = shared_batch_distance(dist)

    assert shared.shape == (1, 2, 1)
    assert torch.allclose(shared[0], dist[0]), "must be env 0's own draw"
    # broadcasting it gives every env the same per-sample distance
    assert torch.allclose(shared.expand_as(dist)[:, 0, 0],
                          torch.full((3,), 1.0))


def test_equalization_makes_the_batch_maximum_equal_the_target():
    """The point of the exercise: sweep_steps follows the batch maximum."""
    g = torch.Generator().manual_seed(1)
    starts = (torch.rand(16, 2, generator=g) - 0.5) * 0.05
    stops = (torch.rand(16, 2, generator=g) - 0.5) * 0.05
    low, high = _bounds((16,))
    dist = (stops - starts).norm(dim=-1, keepdim=True)

    before_spread = float(dist.max() - dist.min())
    target = torch.full_like(dist, float(dist.min()))
    new_stops, _ = equalize_travel_distance(starts, stops, low, high, target)
    after = (new_stops - starts).norm(dim=-1)

    assert before_spread > 1e-3, "fixture should have varied distances"
    assert float(after.max() - after.min()) < 1e-6


# ---------------------------------------------------------------------------
# Action-space restriction: perpendicular pushes and fixed push length
# ---------------------------------------------------------------------------

from action_sampling import (  # noqa: E402
    blade_normal, constrain_push, relative_blade_angle, sampling_box,
)

# The tray/blade the collection configs actually use (Genesis/configs/basic.yaml).
VOL = [0.27, 0.27, 0.1]
TOOL_L, TOOL_W, MARGIN = 0.04, 0.002, 0.02


def _box(angles):
    return sampling_box(angles, VOL, TOOL_L, TOOL_W, MARGIN)


def _random_angles(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (-torch.pi / 2) + torch.rand(n, generator=g) * torch.pi


def _random_starts(angles, seed=1):
    """Uniform in each entry's own yaw-dependent box, as the sampler draws."""
    low, high = _box(angles)
    g = torch.Generator().manual_seed(seed)
    return low + (high - low) * torch.rand(angles.shape + (2,), generator=g)


def test_sampling_box_matches_the_original_inline_formula():
    """The extracted helper must not move any existing dataset's bounds."""
    angles = _random_angles(32)
    low, high = _box(angles)

    space_x = VOL[0] / 2 - (torch.cos(angles) * TOOL_L / 2
                            + abs(torch.sin(angles)) * TOOL_W / 2 + MARGIN)
    space_y = VOL[1] / 2 - (abs(torch.sin(angles)) * TOOL_L / 2
                            + torch.cos(angles) * TOOL_W / 2 + MARGIN)
    assert torch.allclose(high, torch.stack([space_x, space_y], dim=1), atol=1e-9)
    assert torch.allclose(low, -high, atol=1e-9)


def test_blade_normal_is_perpendicular_to_the_blade_axis():
    angles = _random_angles(16)
    axis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    n_hat = blade_normal(angles)

    assert torch.allclose((axis * n_hat).sum(-1), torch.zeros(16), atol=1e-6)
    assert torch.allclose(n_hat.norm(dim=-1), torch.ones(16), atol=1e-6)


def test_perpendicular_pushes_are_perpendicular():
    angles = _random_angles(256)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=2)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True)

    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_perpendicular_preserves_the_push_length_distribution():
    """Only the direction is replaced — v1 datasets stay length-comparable."""
    angles = _random_angles(256)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=3)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True)

    before = (stops - starts).norm(dim=-1)
    after = (push.stops_xy - push.starts_xy).norm(dim=-1)
    # The start may be nudged to make the drawn length fit along the normal;
    # what must survive is the LENGTH, so v1 datasets stay comparable to the
    # unrestricted ones already collected.
    keep = ~push.truncated
    assert torch.allclose(before[keep], after[keep], atol=1e-6)
    assert keep.float().mean() > 0.9, (
        "nudging the start should rescue nearly every draw; only lengths "
        "exceeding the tray extent along the normal may truncate")


def test_perpendicular_pushes_go_both_ways():
    """Yaw is drawn from (-pi/2, pi/2), so cos(theta) > 0 always: without a
    random sign every push would travel into the +y half-plane."""
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=4)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, perpendicular=True,
                          generator=torch.Generator().manual_seed(7))

    dy = (push.stops_xy - push.starts_xy)[:, 1]
    assert (dy > 0).float().mean() > 0.35
    assert (dy < 0).float().mean() > 0.35


def test_fixed_length_pushes_all_travel_that_length():
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=5)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    travelled = (push.stops_xy - push.starts_xy).norm(dim=-1)
    assert not push.truncated.any(), "40 mm fits the 270 mm tray from anywhere"
    assert torch.allclose(travelled, torch.full((512,), 0.04), atol=1e-6)
    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_fixed_length_pushes_stay_inside_the_box():
    angles = _random_angles(512)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=6)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    for pts in (push.starts_xy, push.stops_xy):
        assert (pts >= low - 1e-6).all()
        assert (pts <= high + 1e-6).all()


def test_a_push_blocked_by_the_wall_flips_instead_of_truncating():
    """The +/- choice is free, so it is spent on reaching the target length."""
    angles = torch.tensor([0.0])                 # normal is +y
    low, high = _box(angles)
    starts = torch.stack([torch.zeros(1), high[:, 1] - 0.005], dim=-1)  # 5 mm from the +y wall
    stops = starts + torch.tensor([[0.0, 0.001]])

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    assert not push.truncated.any()
    assert not push.starts_moved.any(), "flipping should suffice; no nudge needed"
    assert torch.equal(push.starts_xy, starts), "start must be preserved"
    assert (push.stops_xy - starts)[0, 1] < 0, "should have flipped to -y"
    assert torch.allclose((push.stops_xy - starts).norm(dim=-1),
                          torch.tensor([0.04]), atol=1e-6)


def test_an_unreachable_length_is_reported_not_silently_shortened():
    """A truncated push is not in the requested length bin — it must be loud."""
    angles = torch.tensor([0.0])
    low, high = _box(angles)
    starts = torch.zeros(1, 2)
    stops = starts + torch.tensor([[0.0, 0.001]])

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=10.0)

    assert push.truncated.all()


def test_no_restriction_requested_is_a_no_op():
    angles = _random_angles(64)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=8)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high)

    assert torch.equal(push.stops_xy, stops)
    assert torch.equal(push.starts_xy, starts)
    assert not push.truncated.any()
    assert not push.starts_moved.any()


def test_fixed_length_without_perpendicular_keeps_the_drawn_direction():
    angles = _random_angles(64)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=9)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high, length=0.03)

    keep = ~push.truncated
    unit_before = (stops - starts) / (stops - starts).norm(dim=-1, keepdim=True)
    step = push.stops_xy - push.starts_xy
    unit_after = step / step.norm(dim=-1, keepdim=True)
    assert torch.allclose(unit_before[keep], unit_after[keep], atol=1e-5)
    assert torch.allclose(step.norm(dim=-1)[keep],
                          torch.full((int(keep.sum()),), 0.03), atol=1e-6)


def test_restriction_works_on_batched_env_sample_shapes():
    """The sampler applies this to (n_envs, n_samples, ...) tensors."""
    angles = _random_angles(24).reshape(4, 6)
    starts = _random_starts(angles)
    stops = _random_starts(angles, seed=10)
    low, high = _box(angles)

    push = constrain_push(starts, stops, angles, low, high,
                          perpendicular=True, length=0.04)

    assert push.stops_xy.shape == (4, 6, 2)
    assert push.truncated.shape == (4, 6)
    assert relative_blade_angle(push.starts_xy, push.stops_xy, angles).max() < 1e-6


def test_relative_blade_angle_spans_plow_to_shear():
    angles = torch.tensor([0.0, 0.0])
    starts = torch.zeros(2, 2)
    stops = torch.tensor([[0.0, 0.04],     # along the normal -> plow
                          [0.04, 0.0]])    # along the blade axis -> shear
    rel = relative_blade_angle(starts, stops, angles)

    assert abs(float(rel[0])) < 1e-6
    assert abs(float(rel[1]) - torch.pi / 2) < 1e-6


# ---------------------------------------------------------------------------
# Pile-aware action sampling
# ---------------------------------------------------------------------------

from action_sampling import pile_contact_starts  # noqa: E402


def _pile(n=30, extent=0.015, seed=0, batch=8):
    """A compact heap at the origin, as the piled spawn produces."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, n, 2, generator=g) - 0.5) * 2 * extent


def _headings(batch=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, generator=g) * 2 * torch.pi


def _along(pts, headings):
    u = torch.stack([torch.cos(headings), torch.sin(headings)], dim=-1)
    return (pts * u).sum(-1) if pts.dim() == headings.dim() + 1 else \
        (pts * u.unsqueeze(-2)).sum(-1)


def test_start_sits_one_clearance_behind_the_pile():
    """The whole point: no sweep distance is spent reaching the pile."""
    p, h = _pile(), _headings()
    starts, n_in, ok = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005, min_swath=3,
        generator=torch.Generator().manual_seed(2))

    gap = _along(p, h).min(dim=-1).values - _along(starts, h)
    # The nearest particle IN THE SWATH is exactly `clearance` ahead; the
    # nearest particle overall can only be nearer still, never further.
    assert (gap <= 0.005 + 1e-6).all()
    assert (gap > 0).all(), "the blade must start behind the pile, not inside it"
    assert ok.all()


def test_swath_actually_contains_material():
    p, h = _pile(), _headings()
    _, n_in, ok = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005, min_swath=5,
        generator=torch.Generator().manual_seed(3))
    assert ok.all()
    assert (n_in >= 5).all()


def test_a_wider_blade_sweeps_more_of_the_pile():
    p, h = _pile(), _headings()
    g = lambda: torch.Generator().manual_seed(4)
    _, narrow, _ = pile_contact_starts(p, h, blade_half_length=0.004,
                                       clearance=0.005, generator=g())
    _, wide, _ = pile_contact_starts(p, h, blade_half_length=0.04,
                                     clearance=0.005, generator=g())
    assert float(wide.float().mean()) > float(narrow.float().mean())


def test_unreachable_min_swath_is_reported_not_hidden():
    """A single isolated particle cannot fill a swath; the caller must know."""
    p = torch.zeros(4, 1, 2)
    _, n_in, ok = pile_contact_starts(p, _headings(batch=4), blade_half_length=0.02,
                                      clearance=0.005, min_swath=5)
    assert not ok.any()
    assert (n_in == 1).all()


def test_start_is_finite_even_when_min_swath_fails():
    p = torch.zeros(4, 1, 2)
    starts, _, ok = pile_contact_starts(p, _headings(batch=4), blade_half_length=0.02,
                                        clearance=0.005, min_swath=5)
    assert torch.isfinite(starts).all(), "must fall back, not return inf"
    assert not ok.any()


def test_push_direction_points_into_the_pile():
    """Every particle in the swath must lie AHEAD of the start, so the sweep
    travels through the pile rather than away from it."""
    p, h = _pile(), _headings()
    starts, _, _ = pile_contact_starts(
        p, h, blade_half_length=0.02, clearance=0.005,
        generator=torch.Generator().manual_seed(5))
    a_p, a_s = _along(p, h), _along(starts, h)
    assert (a_p.min(dim=-1).values > a_s).all()
    # and the pile's far side is further ahead still, i.e. there is depth to sweep
    assert (a_p.max(dim=-1).values > a_p.min(dim=-1).values).all()


def test_clearance_is_respected_exactly():
    p, h = _pile(batch=32), _headings(batch=32)
    for clr in (0.002, 0.005, 0.01):
        starts, n_in, ok = pile_contact_starts(
            p, h, blade_half_length=0.02, clearance=clr, min_swath=3,
            generator=torch.Generator().manual_seed(6))
        # Distance from start to the nearest swath particle == clearance.
        u = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
        nv = torch.stack([-torch.sin(h), torch.cos(h)], dim=-1)
        lat = (p * nv.unsqueeze(-2)).sum(-1)
        c = (starts * nv).sum(-1)
        in_sw = (lat - c.unsqueeze(-1)).abs() <= 0.02
        a = (p * u.unsqueeze(-2)).sum(-1)
        near = torch.where(in_sw, a, torch.full_like(a, 1e3)).min(dim=-1).values
        assert torch.allclose(near - (starts * u).sum(-1),
                              torch.full_like(near, clr), atol=1e-6)


def test_batched_env_sample_shape():
    p = _pile(batch=4).unsqueeze(1).expand(-1, 6, -1, -1)
    h = torch.rand(4, 6, generator=torch.Generator().manual_seed(7)) * 2 * torch.pi
    starts, n_in, ok = pile_contact_starts(p, h, blade_half_length=0.02,
                                           clearance=0.005)
    assert starts.shape == (4, 6, 2)
    assert n_in.shape == (4, 6) and ok.shape == (4, 6)


def test_pile_contact_start_can_be_far_outside_a_small_box():
    """Documents WHY SandboxManipulation clamps the start.

    `pile_contact_starts` places the blade behind the pile's near face; it knows
    nothing about the workspace box. For a pile whose radius approaches the box
    half-extent, that start is outside the box — measured on real data, 35.8% of
    the time, and a start outside the box has nowhere to travel (3.3% of pushes
    came out at ~0 mm). The clamp lives in the caller, so this test pins the
    property the caller has to defend against rather than asserting it away.
    """
    # A spread pile: radius ~35 mm, like a settled-and-pushed 30-cube heap.
    g = torch.Generator().manual_seed(11)
    ang = torch.rand(256, generator=g) * 2 * torch.pi
    rad = 0.030 + 0.005 * torch.rand(256, generator=g)
    p = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], dim=-1)
    p = p.unsqueeze(0).expand(64, -1, -1)
    h = _headings(batch=64, seed=12)

    starts, _, ok = pile_contact_starts(p, h, blade_half_length=0.02,
                                        clearance=0.005, min_swath=3)
    assert ok.all()
    # The tightest blade box in the 127 mm tray is ~23.5 mm half-extent.
    outside = (starts.abs() > 0.0235).any(dim=-1)
    assert outside.any(), (
        "fixture should reproduce the out-of-box condition the clamp exists for")


# ---------------------------------------------------------------------------
# duplicate_action_mask
# ---------------------------------------------------------------------------


def test_duplicate_action_mask_flags_a_close_pair():
    starts = torch.tensor([[0.0, 0.0], [0.0002, 0.0001], [0.03, 0.03]])
    headings = torch.tensor([0.1, 0.101, 0.1])
    dup = duplicate_action_mask(starts, headings, pos_tol=0.001, angle_tol=0.05)
    assert dup.tolist() == [True, True, False]


def test_duplicate_action_mask_no_false_positives_when_all_distinct():
    g = torch.Generator().manual_seed(3)
    starts = torch.rand(64, 2, generator=g) * 0.1
    headings = torch.rand(64, generator=g) * 2 * torch.pi
    dup = duplicate_action_mask(starts, headings, pos_tol=1e-6, angle_tol=1e-6)
    assert not dup.any()


def test_duplicate_action_mask_wraps_the_heading_circle():
    # +pi and -pi are the same direction.
    starts = torch.zeros(2, 2)
    headings = torch.tensor([torch.pi - 1e-4, -torch.pi + 1e-4])
    dup = duplicate_action_mask(starts, headings, pos_tol=1e-3, angle_tol=1e-3)
    assert dup.all()


# ---------------------------------------------------------------------------
# ISS-010 fix: touchdown legality (overlaps_rect_pairs_torch, quat_yaw,
# pile_aware_action_batch) -- Genesis-free
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Baselines.common.cube_overlap import overlaps_rect_pairs as _overlaps_np  # noqa: E402

from action_sampling import (  # noqa: E402
    overlaps_rect_pairs_torch, pile_aware_action_batch, quat_yaw,
)


def test_overlaps_rect_pairs_torch_matches_the_numpy_reference():
    g = torch.Generator().manual_seed(21)
    m = 500
    xy_a = (torch.rand(m, 2, generator=g) - 0.5) * 0.05
    xy_b = (torch.rand(m, 2, generator=g) - 0.5) * 0.05
    yaw_a = torch.rand(m, generator=g) * torch.pi
    yaw_b = torch.rand(m, generator=g) * torch.pi
    half_a = torch.tensor([0.02, 0.001])
    half_b = torch.tensor([0.0025, 0.0025])

    got = overlaps_rect_pairs_torch(xy_a, yaw_a, half_a, xy_b, yaw_b, half_b)
    want = _overlaps_np(xy_a.numpy(), yaw_a.numpy(), half_a.numpy(),
                        xy_b.numpy(), yaw_b.numpy(), half_b.numpy())
    assert (got.numpy() == want).all()


def test_overlaps_rect_pairs_torch_agrees_at_a_tolerance_margin():
    g = torch.Generator().manual_seed(22)
    m = 300
    xy_a = (torch.rand(m, 2, generator=g) - 0.5) * 0.05
    xy_b = (torch.rand(m, 2, generator=g) - 0.5) * 0.05
    yaw_a = torch.rand(m, generator=g) * torch.pi
    yaw_b = torch.rand(m, generator=g) * torch.pi
    half = torch.tensor([0.0025, 0.0025])

    for tol in (0.0, 0.001, -0.001):
        got = overlaps_rect_pairs_torch(xy_a, yaw_a, half, xy_b, yaw_b, half, tol=tol)
        want = _overlaps_np(xy_a.numpy(), yaw_a.numpy(), half.numpy(),
                            xy_b.numpy(), yaw_b.numpy(), half.numpy(), tol=tol)
        assert (got.numpy() == want).all(), f"mismatch at tol={tol}"


def test_quat_yaw_recovers_a_pure_z_rotation():
    yaw = torch.tensor([0.0, 0.3, -1.2, torch.pi / 2])
    quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw),
                        torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
    got = quat_yaw(quat)
    assert torch.allclose(got, yaw, atol=1e-6)


def _dense_disk_pile(n=300, rmax=0.04, seed=0, envs=1):
    """A disk-shaped pile dense enough that the sampling box interior is
    packed with cubes -- reproduces the real corpora's "pile spreads well
    past its spawn extent" condition (ISS-010), unlike the thin-shell
    `test_pile_contact_start_can_be_far_outside_a_small_box` fixture, which
    never actually overlaps the box boundary it clamps into."""
    g = torch.Generator().manual_seed(seed)
    r = rmax * torch.sqrt(torch.rand(n, generator=g))
    ang = torch.rand(n, generator=g) * 2 * torch.pi
    p = torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=-1)
    return p.unsqueeze(0).expand(envs, -1, -1).clone()


def test_pile_aware_action_batch_reproduces_illegal_touchdowns_without_redraw():
    """Sanity check on the FIXTURE, not the fix: with max_redraws=0 the old
    clamp-with-no-cube-check behaviour is recovered, and it must produce
    illegal touchdowns on this dense pile -- otherwise the fixture doesn't
    exercise the bug and the next test proves nothing."""
    E, S = 64, 4
    p = _dense_disk_pile(envs=E)
    yaw = torch.zeros(E, p.shape[1])
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.rand(E, S, generator=torch.Generator().manual_seed(1)) * 2 * torch.pi

    _, _, _, _, n_illegal, n_redraws, _ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=0.02, blade_half_width=0.001,
        granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
        max_redraws=0)
    assert n_redraws == 0
    assert n_illegal > 0, "dense-disk fixture should reproduce ISS-010's bug"


def test_pile_aware_action_batch_redraws_to_zero_illegal_touchdowns():
    """The actual fix: with redraws enabled, every returned touchdown clears
    every particle's footprint -- checked directly with the SAME exact SAT
    test the audit script uses, not re-derived."""
    E, S = 64, 4
    p = _dense_disk_pile(envs=E)
    yaw = torch.zeros(E, p.shape[1])
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.rand(E, S, generator=torch.Generator().manual_seed(1)) * 2 * torch.pi

    starts_xy, stops_xy, angles, ok, n_illegal, n_redraws, _ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=0.02, blade_half_width=0.001,
        granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
        max_redraws=40, generator=torch.Generator().manual_seed(2))

    assert n_illegal == 0, f"{n_illegal} touchdowns still illegal after redraws"

    # Independent re-check with the numpy reference the audit script itself
    # uses, so this test does not just trust the same function under test.
    N = p.shape[1]
    bxy = starts_xy.reshape(-1, 1, 2).expand(-1, N, -1).reshape(-1, 2).numpy()
    byaw = angles.reshape(-1, 1).expand(-1, N).reshape(-1).numpy()
    cxy = p.unsqueeze(1).expand(-1, S, -1, -1).reshape(-1, 2).numpy()
    cyaw = yaw.unsqueeze(1).expand(-1, S, -1).reshape(-1).numpy()
    ov = _overlaps_np(bxy, byaw, [0.02, 0.001], cxy, cyaw, [0.0025, 0.0025])
    assert not ov.any()


def test_pile_aware_action_batch_never_shortens_a_fixed_length_push():
    """The one hard rule the fix must not violate: a legal touchdown is found
    by redrawing the START (a fresh heading), never by shortening or
    lengthening the push once a start is accepted."""
    E, S = 32, 4
    p = _dense_disk_pile(envs=E, seed=5)
    yaw = torch.zeros(E, p.shape[1])
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.rand(E, S, generator=torch.Generator().manual_seed(3)) * 2 * torch.pi
    length = 0.02

    starts_xy, stops_xy, angles, ok, n_illegal, _, _ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=0.02, blade_half_width=0.001,
        granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
        push_length=length, max_redraws=40,
        generator=torch.Generator().manual_seed(4))

    travelled = (stops_xy - starts_xy).norm(dim=-1)
    # Every push that reached its target inside the box travels exactly
    # `length`; only a push whose clamped start has no room at all is capped
    # by t_max (a real tray-boundary limit, not a shortening introduced by
    # the redraw logic) -- assert the exact-length case is the overwhelming
    # majority, matching the old code's own behaviour under the same box.
    at_length = torch.isclose(travelled, torch.full_like(travelled, length), atol=1e-6)
    assert at_length.float().mean() > 0.9
    assert n_illegal == 0


# ---------------------------------------------------------------------------
# start_gap_range (2026-09-28 coordinator spec): sample the touchdown gap
# instead of a fixed clearance, applied at pile_contact_starts's own
# `clearance` argument. A single-particle-at-the-origin fixture makes the
# realized gap exactly recoverable (a_near = 0 identically, regardless of the
# lateral jitter `pile_contact_starts` draws), so this checks the actual
# sampled distribution rather than just "it runs".
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def test_start_gap_range_samples_uniformly_in_the_requested_window():
    E, S = 1, 20000
    L, lo, hi_margin = 0.02, 0.005, 0.005
    p = torch.zeros(E, 1, 2)          # one particle at the origin
    yaw = torch.zeros(E, 1)
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.zeros(E, S)      # push direction = +x -> a_near = 0 always

    starts_xy, stops_xy, angles, ok, n_illegal, _, _ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=0.02, blade_half_width=0.001,
        granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
        push_length=L, start_gap_range=(lo, hi_margin), max_redraws=0,
        generator=torch.Generator().manual_seed(15))

    # a_start = a_near(=0) - centre_clearance; centre_clearance is the
    # face-to-face gap PLUS blade_half_width PLUS cube_half (see
    # pile_aware_action_batch's start_gap_range docstring) -- recover the
    # face-to-face gap the coordinator's spec is actually about.
    centre_clearance = -starts_xy[..., 0]
    gap = centre_clearance - 0.001 - 0.0025
    assert n_illegal == 0
    assert float(gap.min()) >= lo - 1e-6
    assert float(gap.max()) <= (L - hi_margin) + 1e-6
    mid = (lo + (L - hi_margin)) / 2
    assert abs(float(gap.mean()) - mid) < 5e-4, "should be roughly uniform, not clustered"
    assert (gap < lo + 0.001).any() and (gap > (L - hi_margin) - 0.001).any(), \
        "both ends of the window should be populated at this sample count"

    # the push is never shortened/lengthened to make room for the sampled gap
    travelled = (stops_xy - starts_xy).norm(dim=-1)
    assert torch.allclose(travelled, torch.full_like(travelled, L), atol=1e-6)


def test_no_start_gap_range_reproduces_the_old_fixed_clearance():
    E, S = 1, 50
    p = torch.zeros(E, 1, 2)
    yaw = torch.zeros(E, 1)
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.zeros(E, S)

    starts_xy, *_ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=0.02, blade_half_width=0.001,
        granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
        push_length=0.02, start_gap_range=None, max_redraws=0)

    gap = -starts_xy[..., 0]
    assert torch.allclose(gap, torch.full_like(gap, 0.005), atol=1e-6)


def test_start_gap_range_requires_a_scalar_push_length():
    p = torch.zeros(1, 1, 2)
    yaw = torch.zeros(1, 1)
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.zeros(1, 3)
    with pytest.raises(NotImplementedError):
        pile_aware_action_batch(
            p, yaw, cube_half, headings,
            blade_half_length=0.02, blade_half_width=0.001,
            granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
            push_length=None, start_gap_range=(0.005, 0.005))


def test_a_clamped_draw_is_redrawn_not_silently_kept_with_a_corrupted_gap():
    """The coordinator's 2026-09-28 review fix: `pile_contact_starts` places
    the touchdown in (push, lateral) coordinates, but the box clamp moves it
    in WORLD (x, y) -- any clamp therefore invalidates the along-push-axis
    gap the draw was built with, whether or not the clamped point happens to
    still be legal. Before this fix, only the illegal case triggered a
    redraw, so a legal-but-clamped draw silently kept an arbitrary
    (occasionally far-outside-the-window) realized gap -- this is what a real
    smoke test caught (gap p95 ~34mm against a requested 5-15mm window).

    Uses REAL narrow-domain geometry (configs/basic.yaml-scale tray/blade) and
    a pile some of whose particles sit near/outside the tightest box extent,
    so clamping is exercised for a meaningful fraction of draws -- then checks
    the ACCEPTED touchdown's along-axis gap to its own first-contact cube
    (recomputed independently, the same way the audit script would) against
    the requested window."""
    granular_vol = [0.127, 0.127]
    safety_margin = 0.005
    tool_length, tool_width = 0.04, 0.002
    L, lo, hi_margin = 0.02, 0.005, 0.005
    cube_size = 0.005

    E, S = 48, 8
    p = _dense_disk_pile(n=40, rmax=0.032, envs=E, seed=17)   # spans past the tightest box half-extent
    yaw = torch.zeros(E, p.shape[1])
    cube_half = torch.tensor([cube_size / 2, cube_size / 2])
    headings = torch.rand(E, S, generator=torch.Generator().manual_seed(18)) * 2 * torch.pi

    starts_xy, stops_xy, angles, ok, n_illegal, _, _ = pile_aware_action_batch(
        p, yaw, cube_half, headings,
        blade_half_length=tool_length / 2, blade_half_width=tool_width / 2,
        granular_vol=granular_vol, safety_margin=safety_margin, clearance=0.005,
        push_length=L, start_gap_range=(lo, hi_margin), max_redraws=40,
        generator=torch.Generator().manual_seed(19))
    assert n_illegal == 0

    direction = stops_xy - starts_xy
    length = direction.norm(dim=-1, keepdim=True)
    u = direction / length
    nvec = torch.stack([-u[..., 1], u[..., 0]], dim=-1)
    rel = p.unsqueeze(1) - starts_xy.unsqueeze(2)             # (E,S,N,2)
    a = (rel * u.unsqueeze(2)).sum(-1)                        # along push axis
    lat = (rel * nvec.unsqueeze(2)).sum(-1)
    in_swath = lat.abs() <= (tool_length / 2)
    near_face_a = a - cube_size / 2
    ahead = near_face_a > (tool_width / 2)
    gap = near_face_a - tool_width / 2

    ok_count = 0
    total = 0
    for e in range(E):
        for s in range(S):
            cand = in_swath[e, s] & ahead[e, s]
            if not bool(cand.any()):
                continue
            total += 1
            g = float(gap[e, s][cand].min())
            if lo - 1e-3 <= g <= (L - hi_margin) + 1e-3:
                ok_count += 1
    assert total > 0
    assert ok_count / total > 0.9, (
        f"only {ok_count}/{total} accepted touchdowns kept a gap inside the "
        f"requested window -- the clamp-invalidates-the-gap case is not fixed")


def test_start_gap_range_rejects_a_window_that_does_not_fit_the_push_length():
    p = torch.zeros(1, 1, 2)
    yaw = torch.zeros(1, 1)
    cube_half = torch.tensor([0.0025, 0.0025])
    headings = torch.zeros(1, 3)
    with pytest.raises(ValueError):
        pile_aware_action_batch(
            p, yaw, cube_half, headings,
            blade_half_length=0.02, blade_half_width=0.001,
            granular_vol=VOL, safety_margin=MARGIN, clearance=0.005,
            push_length=0.02, start_gap_range=(0.015, 0.015))
