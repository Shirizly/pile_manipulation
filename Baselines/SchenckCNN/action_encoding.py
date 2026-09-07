"""Baselines/SchenckCNN/action_encoding.py -- build the "action map" input
channels for the Schenck single-net ablation, from world-metre
p_start/p_stop/angle.

**Reuse note (per task brief: reuse B2/NFD's action-rasterisation approach,
not a third one):** the world-to-pixel conversion (`raw.to_pxl`,
`raw.ctr_in_PXL`) and the `draw_plate_soft` rasteriser call are copied
verbatim from `Baselines/NFD/predictor.py::NFDPredictor.predict_occ`'s
3-channel branch (itself built on `Baselines/NFD/nfd_lib.py`'s training-time
construction) -- SAME primitive, SAME world->pixel formula, SAME
plate-geometry lookup (`raw.configs[0]["plate"]["size"]`, constant across
the pool per `nfd_lib.py::plate_geometry_px`'s own verification). We render
`p_start`/`p_stop` as two SEPARATE full-intensity channels (NFD's faithful
3-channel choice), not the repo's asymmetric 0.5/1.0 union -- this is more
informative and is a direct, not "third", reuse of NFD's own recommended
(non-ablation) encoding.

**What's added beyond NFD's encoding, and why:** Schenck's paper additionally
tiles each of its 3 action angles (start/end/roll) across 3 constant-valued
channels concatenated onto the position map (SPEC.md sec 2). We have a
single blade `angle` (no separate start/end/roll angles for a rigid plate
push), so we tile exactly ONE constant angle channel -- the one piece of
Schenck's action-map recipe NFD's own encoding doesn't need (NFD has no
angle-tiling; the orientation is implicit in the shape of the rendered
footprint). Total input = occ0 (1) + r_start (1) + r_stop (1) + angle (1)
= 4 channels.

**Heading hazard (C-018, per orchestration log):** `angle` here is used only
to orient the RENDERED PLATE FOOTPRINT (a real, physical, mod-180-recoverable
rotation of the rigid blade -- exactly how `draw_plate_soft`/
`PileSweepData._draw_plate` already use it to draw the ground-truth grids
byte-identically). It is never used to derive travel heading. Heading /
direction-of-motion enters ONLY through the two literal endpoint positions
`p_start`, `p_stop` (rendered as two separate channels, so the network sees
"where the plate was" and "where the plate ended up" as distinct fields) --
never through `angle - atan2(...)` or any other reconstruction from `angle`.
"""
from __future__ import annotations

import torch

from transforms.functional import draw_plate_soft


def plate_geometry_px(raw) -> tuple[float, float, float]:
    """(plate_dim_x_px, plate_dim_y_px, sigma) -- identical formula to
    Baselines/NFD/nfd_lib.py::plate_geometry_px / predictor.py's
    `_plate_geometry_px`; duplicated (not imported) so this module has no
    import-time dependency on Baselines/NFD."""
    plate_dim_x, plate_dim_y, _ = raw.configs[0]["plate"]["size"]
    plate_dim_x_px = plate_dim_x * raw.to_pxl
    plate_dim_y_px = plate_dim_y * raw.to_pxl
    sigma = max(0.5, 1.5 * raw.resolution_scale)
    return plate_dim_x_px, plate_dim_y_px, sigma


def build_action_channels(
    p_start: torch.Tensor,  # (B,3) world metres
    p_stop: torch.Tensor,   # (B,3) world metres
    angle: torch.Tensor,    # (B,)
    raw,                    # PileSweepData instance (to_pxl, ctr_in_PXL, configs, resolution_scale)
    H: int,
    W: int,
) -> torch.Tensor:
    """Returns (B, 3, H, W): [r_start, r_stop, angle_tiled]. Caller
    concatenates occ0 as channel 0 -> (B, 4, H, W) network input."""
    device = p_start.device
    ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(device)[:2]
    start_px = p_start[:, :2].to(device) * raw.to_pxl + ctr_xy
    stop_px = p_stop[:, :2].to(device) * raw.to_pxl + ctr_xy
    angle = angle.to(device)
    plate_x_px, plate_y_px, sigma = plate_geometry_px(raw)

    r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                               intensity=1.0, sigma=sigma)
    r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                              intensity=1.0, sigma=sigma)
    angle_ch = angle.view(-1, 1, 1).expand(-1, H, W).to(device)
    return torch.stack([r_start, r_stop, angle_ch], dim=1)


def build_input(occ0: torch.Tensor, p_start, p_stop, angle, raw, H: int, W: int) -> torch.Tensor:
    """(B,H,W) occ0 + the 3 action channels above -> (B,4,H,W) network input."""
    action = build_action_channels(p_start, p_stop, angle, raw, H, W)
    return torch.cat([occ0.unsqueeze(1).to(action.device), action], dim=1)
