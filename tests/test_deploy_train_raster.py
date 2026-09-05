"""The `deploy-train-raster` invariant.

The occupancy and action channels a learned model sees at MPC time must be
built the same way as the ones it trained on. `UNetFiLMPushModel.forward`
converts from the EulerianWrapper convention into "the dataset convention"
before calling the network -- so when the dataset convention changes, this
conversion must change with it, or every deployed model is fed a transposed
scene it never saw in training.

That is exactly what happened on 2026-09-05: the grid-convention fix
(EXP-0001) moved the dataset's occupancy channel from dim0=world_y to
dim0=world_x, and this path still transposed into the old layout. These tests
pin both halves of the contract so the two cannot drift apart again.

Conventions, after the fix:
    dataset / training : grid[dim0, dim1] = grid[world_x_idx, world_y_idx]
    EulerianWrapper    : grid[dim0, dim1] = grid[cam_x, cam_y] = [world_x, -world_y]
so the conversion between them is a flip of dim1 and NOTHING ELSE.
"""
import math

import pytest
import torch

from transforms.functional import draw_plate_soft, particles_to_occupancy

BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
N = 64


def _wrapper(grid=(N, N)):
    from model.eulerian_wrapper import UNetFiLMPushModel

    class _Identity(torch.nn.Module):
        """Returns its own occupancy channel as a logit, so `forward`'s output
        conversion can be checked independently of any trained weights."""
        def forward(self, x, phys):
            # invert the sigmoid applied downstream, so the round trip is exact
            occ = x[:, 0:1].clamp(1e-4, 1 - 1e-4)
            return torch.log(occ / (1 - occ))

    return UNetFiLMPushModel(_Identity(), torch.zeros(3), grid,
                             plate_length_px=20.0, plate_width_px=2.0, sigma=1.0)


def _px(v, lo, hi, n):
    return (v - lo) / (hi - lo) * n - 0.5


def test_occupancy_channel_reaches_the_net_in_dataset_convention():
    """A particle at +world_x must land high on dim 0 of what the net sees."""
    m = _wrapper()
    pts = torch.tensor([[[0.048, 0.016, 0.0]]])
    occ_ds_expected = particles_to_occupancy(pts, BOUNDS, (N, N), sigma=0.0)

    # Same scene in EulerianWrapper convention: dim1 is cam_y = -world_y.
    occ_cam = occ_ds_expected.flip(dims=[-1])

    seen = {}
    orig = m.unet_film.forward
    m.unet_film.forward = lambda x, phys: (seen.setdefault("x", x), orig(x, phys))[1]
    act = torch.tensor([[32.0, 32.0]])
    m(occ_cam, act, act + 1.0)

    got = seen["x"][:, 0]
    assert torch.allclose(got, occ_ds_expected, atol=1e-5), (
        "the occupancy channel handed to the network is not in the dataset's "
        "convention -- a deployed model is seeing a scene it never trained on")


def test_action_channel_matches_the_datasets_own_plate_rasteriser():
    """The action channel must agree with `draw_plate_soft` called exactly as
    `PileSweepData._draw_plate` calls it: centre in (x_px, y_px), angle = the
    PHYSICAL plate yaw, which every MPC computes as atan2(dy, dx) + pi/2."""
    m = _wrapper()
    sx, sy, ex, ey = 0.010, -0.020, 0.040, 0.005
    to_x = lambda v: _px(v, BOUNDS["x_min"], BOUNDS["x_max"], N)
    to_y = lambda v: _px(v, BOUNDS["y_min"], BOUNDS["y_max"], N)
    s_ds = torch.tensor([[to_x(sx), to_y(sy)]])
    e_ds = torch.tensor([[to_x(ex), to_y(ey)]])
    yaw = math.atan2(ey - sy, ex - sx) + math.pi / 2

    expect = torch.maximum(
        draw_plate_soft(s_ds, torch.tensor([yaw]), (N, N), 20.0, 2.0, 0.5, 1.0),
        draw_plate_soft(e_ds, torch.tensor([yaw]), (N, N), 20.0, 2.0, 1.0, 1.0))

    # EulerianWrapper takes pixel coords with dim1 = cam_y = (N-1) - world_y_idx
    s_cam = torch.tensor([[to_x(sx), (N - 1) - to_y(sy)]])
    e_cam = torch.tensor([[to_x(ex), (N - 1) - to_y(ey)]])
    seen = {}
    orig = m.unet_film.forward
    m.unet_film.forward = lambda x, phys: (seen.setdefault("x", x), orig(x, phys))[1]
    m(torch.zeros(1, N, N), s_cam, e_cam)

    got = seen["x"][:, 1]
    com_g = (got > 0.4).float().nonzero().float().mean(0)
    com_e = (expect > 0.4).float().nonzero().float().mean(0)
    assert torch.allclose(com_g[1:], com_e[1:], atol=1.5), (
        f"action channel centroid {com_g[1:].tolist()} vs dataset's "
        f"{com_e[1:].tolist()} -- the plate is drawn in the wrong frame")
    # Compare the SOFT fields directly. A thresholded IoU is the wrong
    # instrument here: draw_plate_soft returns a sigmoid-edged field whose peak
    # is `intensity`, so even a perfect match scores sum(p^2)/sum(min(2p,1)),
    # which is below 0.5 by construction. (Caught by this test failing on a
    # correct implementation.)
    assert torch.allclose(got, expect, atol=1e-5), (
        f"action channel differs from the dataset's own plate raster "
        f"(max abs diff {float((got - expect).abs().max()):.4f})")


def test_output_conversion_is_the_inverse_of_the_input_conversion():
    """Whatever frame the net works in, what comes back out must be in the
    caller's frame -- otherwise a rollout transposes a little more each step."""
    m = _wrapper()
    pts = torch.tensor([[[0.048, 0.016, 0.0]]])
    occ_cam = particles_to_occupancy(pts, BOUNDS, (N, N), sigma=0.0).flip(dims=[-1])
    act = torch.tensor([[32.0, 32.0]])
    out = m(occ_cam, act, act + 1.0)
    assert out.shape == occ_cam.shape
    assert torch.allclose(out, occ_cam, atol=1e-3), (
        "input->output conversion is not a round trip; multi-step rollouts "
        "would accumulate a frame error")
