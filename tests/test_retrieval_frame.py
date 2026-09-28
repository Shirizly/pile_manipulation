"""Push-frame round-trip tests for model/retrieval/frame.py (EXP-0059).

Genesis-free, no data files needed.
"""
import math

import torch

from model.retrieval.frame import (
    push_frame_basis, world_to_push_frame, push_frame_to_world,
    push_angle, yaw_from_quat, yaw_to_quat, wrap_angle, tray_corners_push_frame,
)


def test_round_trip_random_points():
    torch.manual_seed(0)
    B, n = 8, 5
    p_start = torch.rand(B, 3) * 0.1 - 0.05
    p_stop = p_start + torch.randn(B, 3) * 0.02
    p_stop[:, 2] = p_start[:, 2]
    xy = torch.rand(B, n, 2) * 0.1 - 0.05

    uv = world_to_push_frame(xy, p_start, p_stop)
    xy_back = push_frame_to_world(uv, p_start, p_stop)
    assert torch.allclose(xy, xy_back, atol=1e-5)


def test_origin_and_stop_landmarks():
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])   # pure +x push, length 20mm
    pts = torch.stack([p_start[:, :2], p_stop[:, :2]], dim=1)  # (1, 2, 2): [start, stop]
    uv = world_to_push_frame(pts, p_start, p_stop)
    assert torch.allclose(uv[0, 0], torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(uv[0, 1], torch.tensor([0.02, 0.0]), atol=1e-6)


def test_lateral_axis_is_perpendicular_ccw():
    # push along +y: u_hat = (0, 1), so v_hat (90 deg CCW from u_hat) = (-1, 0)
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.0, 0.02, 0.0]])
    u_hat, v_hat, L = push_frame_basis(p_start, p_stop)
    assert torch.allclose(u_hat[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(v_hat[0], torch.tensor([-1.0, 0.0]), atol=1e-6)
    assert abs(float(L[0]) - 0.02) < 1e-6


def test_rotation_invariance_of_frame():
    """The SAME physical configuration, pushed in a rotated direction, must
    give the SAME push-frame (u, v) coordinates -- this is the entire point
    of canonicalisation: it is what lets two transitions with different
    world headings be compared directly."""
    torch.manual_seed(1)
    n = 6
    rel_xy = torch.rand(1, n, 2) * 0.05  # object positions relative to push start
    L = 0.02
    for phi in (0.0, math.pi / 3, math.pi, -math.pi / 4):
        p_start = torch.tensor([[0.01, -0.02, 0.0]])
        p_stop = p_start.clone()
        p_stop[0, 0] += L * math.cos(phi)
        p_stop[0, 1] += L * math.sin(phi)
        # rotate the same relative offsets into world coords for this heading
        c, s = math.cos(phi), math.sin(phi)
        R = torch.tensor([[c, -s], [s, c]])
        xy = p_start[:, :2].unsqueeze(1) + rel_xy @ R.T
        uv = world_to_push_frame(xy, p_start, p_stop)
        if phi == 0.0:
            uv_ref = uv
        else:
            assert torch.allclose(uv, uv_ref, atol=1e-5), phi


def test_wrap_angle_range_and_identity():
    a = torch.tensor([0.0, math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 0.1])
    w = wrap_angle(a)
    assert torch.all(w > -math.pi - 1e-6) and torch.all(w <= math.pi + 1e-6)
    # a value already in range is unchanged
    assert abs(float(wrap_angle(torch.tensor(0.1))) - 0.1) < 1e-6


def test_yaw_from_quat_and_back_pure_yaw():
    yaw = torch.tensor([0.0, math.pi / 6, -math.pi / 2, math.pi - 0.01])
    q = yaw_to_quat(yaw)
    yaw_back = yaw_from_quat(q)
    assert torch.allclose(wrap_angle(yaw), wrap_angle(yaw_back), atol=1e-5)


def test_push_angle_matches_atan2():
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[1.0, 1.0, 0.0]])
    assert abs(float(push_angle(p_start, p_stop)[0]) - math.pi / 4) < 1e-6


def test_tray_corners_push_frame_shape_and_round_trip():
    bounds = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
    p_start = torch.tensor([[0.0, 0.0, 0.0], [0.01, -0.01, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0], [0.01, 0.01, 0.0]])
    corners_pf = tray_corners_push_frame(p_start, p_stop, bounds)
    assert corners_pf.shape == (2, 4, 2)
    # transforming back to world must recover the 4 axis-aligned corners
    corners_world = push_frame_to_world(corners_pf, p_start, p_stop)
    expected = torch.tensor([[x, y] for x in (bounds["x_min"], bounds["x_max"])
                            for y in (bounds["y_min"], bounds["y_max"])])
    for b in range(2):
        assert torch.allclose(corners_world[b], expected, atol=1e-5)
