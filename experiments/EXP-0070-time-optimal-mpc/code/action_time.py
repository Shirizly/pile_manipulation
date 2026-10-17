"""Physical execution time of one push, from the CURRENT plate pose (EXP-0070).

The simulator teleports the plate between pushes (execute_action: teleport to a 10 mm
clearance above p_start, simulate only the short descent, sweep, short ascent, teleport
away), so no record so far has counted the time a real gantry spends getting the tool to
the next push. This module is the analytic model of that time:

    t(a | p_prev) = T(h_up) + T(|p_start - p_prev|_xy) + T(h_down) + T(L) + t_dwell

  T(d)   trapezoid duration at the gantry's sweep limits (plate.speed 0.125 m/s,
         plate.acceleration 2 m/s^2): d/v + v/a if d >= v^2/a, else 2 sqrt(d/a)
         (Genesis/sandbox_manipulation_clean.py::_trapezoid_profile).
  h_up   lift from the previous push's end to the safe height (0 for the very first push,
         when the plate starts AT safe height above `home`).
  h_down descent from the safe height to the operating height.
  L      the sweep length |p_stop - p_start|.
  t_dwell per-push servo settling (sim: arrival_steps + sweep_settle_steps), a parameter;
         default 0 = the pure kinematic model. The sweep is NOT followed by a lift (that is
         charged to the next push as h_up), as specified by the user 2026-10-05.

Rotation to the new blade yaw runs concurrently with the transit and is assumed never
to be the limiting axis (user assumption). Everything is torch, differentiable a.e. in the
action, so it can sit inside a gradient planner's objective.
"""
from __future__ import annotations
import torch

V_MAX = 0.125      # m/s  Genesis/configs/basic.yaml plate.speed
ACC = 2.0          # m/s^2 plate.acceleration
H_SAFE = 0.010     # m    sim clearance height = max(8 mm, 2 x 5 mm cube) above operating height
HOME = (0.0, 0.0)  # plate parked at safe height above the tray centre at episode start


def trap_time(d: torch.Tensor, v: float = V_MAX, a: float = ACC) -> torch.Tensor:
    """Duration of a rest-to-rest trapezoidal move of length d (m)."""
    d = d.clamp_min(0.0)
    d_ramp = v * v / a
    tri = 2.0 * torch.sqrt(d / a + 1e-12)
    trap = d / v + v / a
    return torch.where(d >= d_ramp, trap, tri)


def action_time(act: torch.Tensor, p_prev: torch.Tensor, first: bool | torch.Tensor = False,
                h_safe: float = H_SAFE, t_dwell: float = 0.0, v: float = V_MAX, a: float = ACC,
                parts: bool = False):
    """act (B, 4) [sx, sy, ex, ey] metres; p_prev (2,) or (B, 2): where the plate is now (the
    previous push's end, at operating height; or HOME at safe height when `first`).
    Returns (B,) seconds, or a dict of the components if parts."""
    p_prev = p_prev.to(act).expand(act.shape[0], 2)
    first = torch.as_tensor(first, device=act.device).expand(act.shape[0])
    h = torch.full((act.shape[0],), h_safe, device=act.device, dtype=act.dtype)
    t_up = torch.where(first, torch.zeros_like(h), trap_time(h, v, a))
    t_tr = trap_time((act[:, 0:2] - p_prev).norm(dim=1), v, a)
    t_dn = trap_time(h, v, a)
    t_sw = trap_time((act[:, 2:4] - act[:, 0:2]).norm(dim=1), v, a)
    tot = t_up + t_tr + t_dn + t_sw + t_dwell
    if parts:
        return dict(total=tot, up=t_up, transit=t_tr, down=t_dn, sweep=t_sw)
    return tot


def episode_action_times(actions, h_safe: float = H_SAFE, t_dwell: float = 0.0, home=HOME):
    """Per-push times for a recorded episode (list of [sx, sy, ex, ey]); returns dict of
    lists (total + components)."""
    A = torch.as_tensor(actions, dtype=torch.float64)
    prev = torch.cat([torch.tensor([home], dtype=torch.float64), A[:-1, 2:4]], 0)
    first = torch.zeros(len(A), dtype=torch.bool); first[0] = True
    out = {k: [] for k in ("total", "up", "transit", "down", "sweep")}
    for i in range(len(A)):
        p = action_time(A[i:i + 1], prev[i], bool(first[i]), h_safe, t_dwell, parts=True)
        for k in out:
            out[k].append(float(p[k][0]))
    return out
