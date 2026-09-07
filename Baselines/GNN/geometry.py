"""Baselines/GNN/geometry.py -- action -> s_delta conversion, world-frame metres.

Implements Baselines/GNN/SPEC.md section 6 exactly (push direction/heading is
ALWAYS derived from p_stop - p_start, NEVER from `angles` -- see the
ORCHESTRATOR AMENDMENT at the bottom of SPEC.md and the confirmed hazard:
`angles` is the blade FACE orientation (heading + 90 deg mod 180 deg), not
the travel heading, and the sign is not recoverable from it).

All quantities here are plain world-frame metres; there is no PyFlex/OpenGL
camera-normalization step to port (SPEC.md hazard 5).
"""
from __future__ import annotations

import torch

# Sourced from the cell's own _0_config.yaml: plate.size = [0.04, 0.002, 0.01]
# -> lateral (swept-lane) full width 0.04 m -> half-width 0.02 m.
PUSHER_W = 0.02   # metres

# Mask softness -- copied verbatim from dataset_gnn_dyn.py's constant (there
# unitless in a pre-normalized space; here treated as a physical length, see
# SPEC.md hazard/§6 note). Sanity-checked in scripts/check_geometry.py before
# any training.
SOFTNESS = 0.01   # metres

# Contact radius for edge construction. NOT the paper's 0.08 (that value is
# tuned to PyFlex's camera-normalized coordinate scale and is meaningless
# here, see SPEC.md hazard 2). Chosen after the geometry sanity check
# (scripts/check_geometry.py) -- see Baselines/GNN/LOG.md for the actual
# edge-count distribution measured at a few candidate values. Cube edge is
# 0.005 m; box is 0.128 m across.
ADJ_THRESH = 0.012   # metres

PARTICLE_DENS = 1000.0   # fixed constant, not measured/sampled -- SPEC.md hazard 3


def compute_s_delta(s_cur_xyz: torch.Tensor, p_start: torch.Tensor,
                     p_stop: torch.Tensor, pusher_w: float = PUSHER_W,
                     softness: float = SOFTNESS) -> torch.Tensor:
    """s_cur_xyz: (B,20,3) world metres. p_start/p_stop: (B,3) world metres
    (only xy used -- push is planar, SPEC.md measured z-component <= 9.4e-7 m).
    Returns s_delta: (B,20,3), z-component always 0.
    """
    p0 = p_start[:, :2]
    p1 = p_stop[:, :2]
    d = p1 - p0
    push_l = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)   # (B,1)
    push_dir = d / push_l                                    # (B,2)
    push_ortho = torch.stack([-push_dir[:, 1], push_dir[:, 0]], dim=-1)  # (B,2)

    pos_xy = s_cur_xyz[..., :2]                               # (B,20,2)
    rel = pos_xy - p0[:, None, :]
    x_loc = (rel * push_dir[:, None, :]).sum(-1)              # (B,20)
    y_loc = (rel * push_ortho[:, None, :]).sum(-1)             # (B,20)

    l_mask = ((x_loc > 0) & (x_loc < push_l)).float()
    w_mask = torch.exp(-(y_loc.abs() - pusher_w).clamp(min=0) / softness)

    to_end = p1[:, None, :] - pos_xy                          # (B,20,2)
    dist_to_end = (to_end * push_dir[:, None, :]).sum(-1)      # (B,20)

    s_delta_xy = (dist_to_end[..., None] * push_dir[:, None, :]
                  * l_mask[..., None] * w_mask[..., None])
    s_delta = torch.cat([s_delta_xy, torch.zeros_like(s_delta_xy[..., :1])], dim=-1)
    return s_delta


def model_config(adj_thresh: float = ADJ_THRESH, nf_effect: int = 64) -> dict:
    return {"train": {"particle": {"adj_thresh": adj_thresh, "nf_effect": nf_effect,
                                    "add_delta": True}}}
