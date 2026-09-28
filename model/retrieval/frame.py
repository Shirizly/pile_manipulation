"""Push-frame canonicalisation of particle sets (EXP-0059).

This is the METRE-SPACE analogue of `transforms.functional.push_frame_transform`
/ `to_push_frame`: those warp OCCUPANCY GRIDS (a pixel affine transform fed to
`grid_sample`), which is the right tool once a state has already been
rasterised. A retrieval model instead needs to compare and transfer raw cube
POSITIONS and YAWS before any rasterisation happens (rasterising first would
alias distinct nearby configurations together and throw away exactly the
information retrieval needs to match on) -- hence a second, simpler transform
that works directly in (x, y) metres. Do not use `push_frame_transform` here:
it is image-centred (origin at the push MIDPOINT) and pixel-indexed; this
module is object-centred (origin at the push START, as the task spec and the
design doc's Section 1 "action-centric frame" both call for) and works in
continuous metres.

Convention: origin at the push START (`p_start`), +u axis along the push
direction (`p_stop - p_start`), +v axis 90 degrees CCW from +u (lateral).
A cube exactly at the push start has uv = (0, 0); one at the push stop
(assuming it moved bodily with the tool) has uv = (L, 0), L = push length.
"""
from __future__ import annotations

import math
import torch


def push_frame_basis(p_start: torch.Tensor, p_stop: torch.Tensor):
    """p_start, p_stop: (B, >=2) world metres. -> u_hat, v_hat: (B, 2), L: (B,).

    u_hat is the unit push direction, v_hat is u_hat rotated +90 degrees
    (CCW in the (x, y) plane, matching `swept_region`'s own
    `n = [-u_y, u_x]` lateral convention in
    `experiments/EXP-0053-*/code/eval_narrow.py`).
    """
    d = p_stop[..., :2] - p_start[..., :2]
    L = d.norm(dim=-1).clamp_min(1e-9)
    u_hat = d / L.unsqueeze(-1)
    v_hat = torch.stack([-u_hat[..., 1], u_hat[..., 0]], dim=-1)
    return u_hat, v_hat, L


def push_angle(p_start: torch.Tensor, p_stop: torch.Tensor) -> torch.Tensor:
    """(B, >=2) -> (B,) radians, the world heading of the push direction."""
    d = p_stop[..., :2] - p_start[..., :2]
    return torch.atan2(d[..., 1], d[..., 0])


def world_to_push_frame(xy: torch.Tensor, p_start: torch.Tensor,
                        p_stop: torch.Tensor) -> torch.Tensor:
    """xy: (B, n, 2) world metres (n objects per batch row, n=1 is fine).
    p_start, p_stop: (B, >=2). -> uv: (B, n, 2) in the push frame.
    """
    u_hat, v_hat, _ = push_frame_basis(p_start, p_stop)
    rel = xy - p_start[..., :2].unsqueeze(1)
    uu = (rel * u_hat.unsqueeze(1)).sum(-1)
    vv = (rel * v_hat.unsqueeze(1)).sum(-1)
    return torch.stack([uu, vv], dim=-1)


def push_frame_to_world(uv: torch.Tensor, p_start: torch.Tensor,
                        p_stop: torch.Tensor) -> torch.Tensor:
    """Inverse of `world_to_push_frame`. uv: (B, n, 2) -> xy: (B, n, 2)."""
    u_hat, v_hat, _ = push_frame_basis(p_start, p_stop)
    xy = (p_start[..., :2].unsqueeze(1)
          + uv[..., 0:1] * u_hat.unsqueeze(1)
          + uv[..., 1:2] * v_hat.unsqueeze(1))
    return xy


def tray_corners_push_frame(p_start: torch.Tensor, p_stop: torch.Tensor,
                            bounds: dict) -> torch.Tensor:
    """The 4 world tray corners (from an axis-aligned `bounds` dict with
    x_min/x_max/y_min/y_max, e.g. `simple_mpc.adapters.OCC_BOUNDS`),
    transformed into EACH transition's own push frame. -> (B, 4, 2).

    The tray is static in the world, but its walls matter near the pusher in
    an action-relative sense (a push toward a nearby wall behaves differently
    from the same push away from one) -- so this is the "wall/tray geometry
    in that frame" the task asks for, evaluated per-transition since every
    transition's own (p_start, p_stop) rotates the tray differently.
    """
    xs = [bounds["x_min"], bounds["x_max"]]
    ys = [bounds["y_min"], bounds["y_max"]]
    corners = torch.tensor([[x, y] for x in xs for y in ys],
                           dtype=p_start.dtype, device=p_start.device)  # (4,2)
    B = p_start.shape[0]
    corners = corners.unsqueeze(0).expand(B, -1, -1)
    return world_to_push_frame(corners, p_start, p_stop)


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """quat: (..., 4) in (w, x, y, z) order (this repo's convention, see
    `Genesis/state_library.py::_quat_mul`'s docstring and `_yaw_quat`) ->
    yaw (...,) radians via 2*atan2(qz, qw).

    Exact for a pure z-rotation quaternion (qx=qy=0); an approximation for
    the small roll/pitch real single-layer cubes carry (docs/CODEMAP.md:
    "real cubes TILT, mean 5-13 deg"). Harmless for THIS package's purposes:
    every scorer downstream (`occ_from_particles`, `occ_for_scoring`) rasters
    from cube CENTRES only and is orientation-agnostic, so yaw is carried
    for representation completeness (bank features, future R3/R4 models),
    not because current scoring reads it.
    """
    w, z = quat[..., 0], quat[..., 3]
    return 2.0 * torch.atan2(z, w)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Inverse of `yaw_from_quat` restricted to a pure z-rotation: (...,) ->
    (..., 4) in (w, x, y, z), with qx = qy = 0 (see that function's note on
    what this discards)."""
    half = yaw / 2.0
    w, z = torch.cos(half), torch.sin(half)
    zeros = torch.zeros_like(w)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap radians to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi
