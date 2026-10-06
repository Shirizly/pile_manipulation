"""model/zoom_nfd/window.py -- zoomed push-frame WINDOW rasters for a zoom NFD.

Instead of warping the 64x64 world raster into the push frame (`transforms.
functional.to_push_frame`, which resamples a 2 mm/px image), rasterise cube
footprints DIRECTLY into a small window that sits on the push, at the NFD's
standard 64x64 resolution. Only the window's cubes are drawn; nothing outside
it is ever rendered.

Window frame (metres): origin at the push start p0, u = unit push direction,
v = (u_y, -u_x) (the same handedness `push_frame_transform` uses: canonical
+col = push direction, canonical +row = rotate the push direction +90 deg in
(col,row) coordinates). Canonical raster layout matches `to_push_frame`'s:
dim1 (col) <-> u, dim0 (row) <-> v. The window is a SQUARE of side

    S = plate_len + 2 * margin_side

with two corners fixed by the plate's ends (+ margin_side beyond each end,
across the push) at the near edge (`margin_back` behind the plate's start), and
the far edge a distance S from the near edge, i.e. beyond the swept region.

Cube footprints: boxes of the particle's own (w, h) rotated by its quaternion
yaw, exactly as `PileSweepData._draw_particle_grid`, but with sub-pixel
polygon vertices (cv2 `shift`) -- the original truncates to int pixels, which
is fine at 2 mm/px and visibly lumpy at 0.8 mm/px.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from transforms.functional import draw_plate_soft


@dataclass(frozen=True)
class WindowSpec:
    plate_len: float = 0.04       # plate long axis (m), across the push
    plate_thick: float = 0.002    # plate thickness (m), along the push
    margin_side: float = 0.012    # beyond each plate end (m)
    margin_back: float = 0.001    # behind the plate's start position (m)
    res: int = 64                 # NFD standard resolution
    plate_sigma: float = 0.0015   # soft-plate edge sigma (m); = 1.5 mm, what
                                  # the 2 mm/px standard raster uses (0.75 px)

    @property
    def side(self) -> float:
        return self.plate_len + 2 * self.margin_side

    @property
    def px(self) -> float:        # metres per window pixel
        return self.side / self.res


def push_axes(p0: np.ndarray, p1: np.ndarray):
    u = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.linalg.norm(u)
    u = u / n
    return u, np.array([u[1], -u[0]]), n


def world_to_window_idx(xy: np.ndarray, p0, u, v, spec: WindowSpec) -> np.ndarray:
    """World (x, y) metres (..., 2) -> window (col, row) pixel-INDEX coords
    (pixel i's centre = index i; same convention as the dataset's ctr_in_PXL)."""
    d = np.asarray(xy, float) - np.asarray(p0, float)
    s = d @ u
    t = d @ v
    col = (s + spec.margin_back) / spec.px - 0.5
    row = (t + spec.side / 2) / spec.px - 0.5
    return np.stack([col, row], axis=-1)


def window_corners_world(p0, p1, spec: WindowSpec) -> np.ndarray:
    """(4,2) world corners of the window (for overlaying on the world raster)."""
    u, v, _ = push_axes(p0, p1)
    p0 = np.asarray(p0, float)
    out = []
    for s, t in [(-spec.margin_back, -spec.side / 2),
                 (-spec.margin_back, spec.side / 2),
                 (spec.side - spec.margin_back, spec.side / 2),
                 (spec.side - spec.margin_back, -spec.side / 2)]:
        out.append(p0 + s * u + t * v)
    return np.array(out)


def quat_yaw(q) -> float:
    w, x, y, z = [float(a) for a in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rasterise_window(states: torch.Tensor, sizes, p0, p1, spec: WindowSpec) -> torch.Tensor:
    """Cube footprints of `states` (n,7) [x,y,z,qw,qx,qy,qz] in the window.
    `sizes`: (n,3) or (n,2) box dims in metres (config particle_sizes).
    Returns (res, res) float32, dim0 = row (v), dim1 = col (u)."""
    u, v, _ = push_axes(p0, p1)
    res = spec.res
    img = np.zeros((res, res), np.float32)
    S = 4  # cv2 sub-pixel bits
    for st, dim in zip(states.numpy(), np.asarray(sizes, float)):
        yaw = quat_yaw(st[3:7])
        c, s = math.cos(yaw), math.sin(yaw)
        hw, hh = dim[0] / 2, dim[1] / 2
        corners = np.array([[sx * hw, sy * hh] for sx, sy in
                            [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
        world = st[:2] + corners @ np.array([[c, s], [-s, c]])   # R(yaw) on (x,y)
        idx = world_to_window_idx(world, p0, u, v, spec)
        if idx[:, 0].max() < -1 or idx[:, 0].min() > res or \
           idx[:, 1].max() < -1 or idx[:, 1].min() > res:
            continue
        cv2.fillPoly(img, [np.round(idx * (1 << S)).astype(np.int32)], 1.0, shift=S)
    return torch.from_numpy(img)


def plate_channels_window(p0, p1, spec: WindowSpec) -> torch.Tensor:
    """(2,res,res) [r(start), r(stop)] soft plates in the window (intensity 1,
    same soft rasteriser as the standard NFD channels)."""
    u, v, L = push_axes(p0, p1)
    px = spec.px
    row_c = (spec.side / 2) / px - 0.5
    cols = [(spec.margin_back) / px - 0.5, (spec.margin_back + L) / px - 0.5]
    centers = torch.tensor([[row_c, c] for c in cols], dtype=torch.float32)
    return draw_plate_soft(centers, torch.zeros(2), (spec.res, spec.res),
                           spec.plate_len / px, spec.plate_thick / px,
                           intensity=1.0, sigma=spec.plate_sigma / px)


# ---------------------------------------------------------------------------
# batch helpers, window <-> world, goal cut, window-frame scoring pieces
# ---------------------------------------------------------------------------
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID   # harness world grid (pixel i at lo + i*pitch)


def window_batch(states: torch.Tensor, sizes, p0s: np.ndarray, p1s: np.ndarray, spec: WindowSpec):
    """(B,n,7) states + (B,2) starts/stops (numpy, metres) -> (B,res,res)."""
    return torch.stack([rasterise_window(states[b], sizes, p0s[b], p1s[b], spec) for b in range(len(states))])


def plates_batch(p0s, p1s, spec):
    return torch.stack([plate_channels_window(p0s[b], p1s[b], spec) for b in range(len(p0s))])


def _window_pixel_world(p0, p1, spec, sub=(0.0,)):
    """World (x,y) of every window pixel centre -> (res,res,2)."""
    u, v, _ = push_axes(p0, p1)
    idx = (np.arange(spec.res) + 0.5) * spec.px
    s = idx[None, :] - spec.margin_back                    # col
    t = idx[:, None] - spec.side / 2                       # row
    return np.asarray(p0, float) + s[..., None] * u + t[..., None] * v


def sample_world_field_into_window(field: torch.Tensor, p0, p1, spec) -> torch.Tensor:
    """Cut a world-grid field (64,64; harness grid, dim0=x) into the window by bilinear
    sampling at the window's pixel centres (border-padded) -> (res,res)."""
    P = torch.from_numpy(_window_pixel_world(p0, p1, spec)).float()
    lo, hi = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]
    ij = (P - lo) / (hi - lo) * (OCC_GRID - 1)             # (row=x idx, col=y idx)
    g = torch.stack([ij[..., 1], ij[..., 0]], -1) / (OCC_GRID - 1) * 2 - 1   # grid_sample wants (col,row)
    out = torch.nn.functional.grid_sample(field[None, None].float(), g[None], mode="bilinear",
                                          padding_mode="border", align_corners=True)
    return out[0, 0]


def paste_window_into_world(pred_win: torch.Tensor, occ0_world: torch.Tensor, p0, p1, spec, sub=3) -> torch.Tensor:
    """World raster = window prediction where the window covers the world pixel, occ0 elsewhere.
    Each world pixel averages sub x sub sub-samples (the window is ~2x finer than the world grid)."""
    pitch = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / (OCC_GRID - 1)
    u, v, _ = push_axes(p0, p1)
    ax = (np.arange(OCC_GRID) * pitch + OCC_BOUNDS["x_min"])
    offs = (np.arange(sub) - (sub - 1) / 2) * pitch / sub
    acc = torch.zeros(OCC_GRID, OCC_GRID)
    for dx in offs:
        for dy in offs:
            X, Y = np.meshgrid(ax + dx, ax + dy, indexing="ij")
            d = np.stack([X - p0[0], Y - p0[1]], -1)
            col = ((d @ u) + spec.margin_back) / spec.px - 0.5
            row = ((d @ v) + spec.side / 2) / spec.px - 0.5
            inside = torch.from_numpy((col >= -0.5) & (col <= spec.res - 0.5) & (row >= -0.5) & (row <= spec.res - 0.5))
            g = torch.from_numpy(np.stack([col, row], -1)).float() / (spec.res - 1) * 2 - 1
            samp = torch.nn.functional.grid_sample(pred_win[None, None].float(), g[None], mode="bilinear",
                                                   padding_mode="border", align_corners=True)[0, 0]
            acc += torch.where(inside, samp, occ0_world.float())
    return acc / (sub * sub)


def swept_region_window(p0, p1, spec, lat=0.024, pad=0.020) -> torch.Tensor:
    """(res,res) bool: eval_narrow's swept region (lateral +-24 mm, 20 mm pre-pad, to push end) in window pixels."""
    u, v, L = push_axes(p0, p1)
    idx = (np.arange(spec.res) + 0.5) * spec.px
    s = (idx[None, :] - spec.margin_back) * np.ones((spec.res, 1))
    t = (idx[:, None] - spec.side / 2) * np.ones((1, spec.res))
    return torch.from_numpy((s >= -pad) & (s <= L) & (np.abs(t) <= lat))


def splat_window_mass(states: torch.Tensor, p0, p1, spec) -> torch.Tensor:
    """Mass-conserving bilinear splat of cube CENTRES that fall inside the window (one unit each),
    window-frame analogue of `occ_for_scoring` for true outcomes -> (res,res)."""
    u, v, _ = push_axes(p0, p1)
    ij = world_to_window_idx(states[:, :2].numpy(), p0, u, v, spec)       # (col,row) index coords
    img = torch.zeros(spec.res, spec.res)
    for c, r in ij:
        c0, r0 = math.floor(c), math.floor(r)
        for dc in (0, 1):
            for dr in (0, 1):
                cc, rr = c0 + dc, r0 + dr
                w = (1 - abs(c - cc)) * (1 - abs(r - rr))
                if 0 <= cc < spec.res and 0 <= rr < spec.res and w > 0:
                    img[rr, cc] += w
    return img
