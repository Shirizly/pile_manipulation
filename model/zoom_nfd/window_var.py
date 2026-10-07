"""model/zoom_nfd/window_var.py -- zoom window with a PER-SAMPLE side for variable push lengths (EXP-0074).

side(L) = max(64 mm, L + 44 mm)  (L = push length; at L = 20 mm this is exactly the narrow-pilot 64 mm window, 1 mm/px at 64 px).
Window frame is as in window.py: origin at the push start, +col = push direction, margin_back behind the start plate, lateral +-side/2,
res x res pixels, pixel = side/res (1.0-1.8 mm at 64 px for L up to 70 mm; 0.5-0.9 mm at 128 px).
All GPU functions are differentiable and take P0,P1 (B,2) metres and side (B,) metres.
"""
import math, numpy as np, torch
import torch.nn.functional as F
from model.zoom_nfd.window import WindowSpec, rasterise_window
from model.zoom_nfd.window_gpu import LO, SIDE, _axes

BASE, EXTRA, PLATE_LEN, PLATE_T, SIGMA, BACK = 0.064, 0.044, 0.04, 0.002, 0.0015, 0.001


import os
_BINS = [float(x) / 1000 for x in os.environ.get("ZSIDE_BINS", "").split(",") if x]     # e.g. ZSIDE_BINS=35,55 -> bins [0,35) [35,55) [55,70]
_UP = _BINS + [0.070]
_FIXED = float(os.environ["ZSIDE_FIXED"]) / 1000 if os.environ.get("ZSIDE_FIXED") else None   # constant window side (mm) for the zoom-factor ablation


def side_for(L):
    """push length(s) in metres (tensor or float) -> window side(s) in metres.
    Default: continuous max(64 mm, L+44 mm). With env ZSIDE_BINS (length-bin edges in mm) the window is CONSTANT within a length bin:
    side = (bin's maximum sweep length) + 44 mm (fallback A/B of EXP-0074)."""
    if _FIXED is not None:
        return torch.full_like(L, _FIXED) if torch.is_tensor(L) else _FIXED
    if _BINS:
        up = torch.tensor(_UP, dtype=torch.float32)
        if torch.is_tensor(L): return (up.to(L.device)[torch.bucketize(L, torch.tensor(_BINS, dtype=L.dtype, device=L.device), right=True)] + EXTRA).clamp_min(BASE)
        return max(BASE, _UP[int(np.searchsorted(_BINS, float(L), side="right"))] + EXTRA)
    if torch.is_tensor(L): return (L + EXTRA).clamp_min(BASE)
    return max(BASE, float(L) + EXTRA)


def spec_for(side, res):
    return WindowSpec(margin_side=(float(side) - PLATE_LEN) / 2, margin_back=BACK, res=res)


def window_batch_var(states, sizes, P0, P1, res):
    """CPU: direct pose-rendered (training-style, hard cv2 box) windows with per-sample side -> (B,res,res)."""
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float); out = []
    for b in range(len(states)):
        sp = spec_for(side_for(np.linalg.norm(P1[b] - P0[b])), res); out.append(rasterise_window(states[b], sizes, P0[b], P1[b], sp))
    return torch.stack(out)


def _soft(x, half, sigma): return torch.sigmoid((half - x.abs()) / sigma)


def plates_v(P0, P1, side, res):
    """(B,2,res,res) soft start/stop plates in the window (per-sample pixel size)."""
    B = len(P0); dev = P0.device; px = side / res; L = (P1 - P0).norm(dim=-1)
    ar = torch.arange(res, device=dev, dtype=torch.float32)
    row = (ar[None, :, None] - (res / 2 - 0.5)) * px[:, None, None]                    # lateral offset (m), (B,res,1)
    out = []
    for c in (BACK, BACK + L):
        col = (ar[None, None, :] + 0.5) * px[:, None, None] - BACK - (c - BACK)[..., None, None] if torch.is_tensor(c) else (ar[None, None, :] + 0.5) * px[:, None, None] - BACK - (c - BACK)
        out.append(_soft(row, PLATE_LEN / 2, SIGMA) * _soft(col, PLATE_T / 2, SIGMA))
    return torch.stack(out, 1)


def _coords(P0, P1, side, res, ss, dev):
    u, v = _axes(P0, P1); px = side / res
    off = (torch.arange(ss, device=dev) - (ss - 1) / 2) / ss
    ci = (torch.arange(res, device=dev)[:, None] + 0.5 + off[None, :]).reshape(-1)
    s = ci[None] * px[:, None] - BACK; t = ci[None] * px[:, None] - side[:, None] / 2
    return P0[:, None, None, :] + s[:, None, :, None] * u[:, None, None, :] + t[:, :, None, None] * v[:, None, None, :]


def extract_windows_v(canvas, P0, P1, side, res, ss=3):
    """canvas (B,C,C) -> (B,res,res) antialiased windows (rotation + per-sample zoom, one grid_sample)."""
    C = canvas.shape[-1]; pxc = SIDE / C
    idx = (_coords(P0, P1, side, res, ss, canvas.device) - LO) / pxc - 0.5
    g = torch.stack([idx[..., 1], idx[..., 0]], -1) / (C - 1) * 2 - 1
    return F.avg_pool2d(F.grid_sample(canvas[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True), ss).squeeze(1)


def _win_idx(pts, P0, P1, side, res):
    u, v = _axes(P0, P1); px = (side / res)[:, None, None]
    d = pts - P0[:, None, None, :]
    col = ((d * u[:, None, None, :]).sum(-1) + BACK) / px - 0.5
    row = ((d * v[:, None, None, :]).sum(-1) + side[:, None, None] / 2) / px - 0.5
    return torch.stack([col, row], -1) / (res - 1) * 2 - 1


def paste_canvas_v(delta, P0, P1, side, C):
    """window change (B,res,res) -> (B,C,C) on the canvas grid (zero outside the window)."""
    res = delta.shape[-1]; dev = delta.device; pxc = SIDE / C
    ax = LO + (torch.arange(C, device=dev) + 0.5) * pxc
    pts = torch.stack(torch.meshgrid(ax, ax, indexing="ij"), -1)[None].expand(len(delta), -1, -1, -1)
    return F.grid_sample(delta[:, None], _win_idx(pts, P0, P1, side, res), mode="bilinear", padding_mode="zeros", align_corners=True).squeeze(1)


def delta_to_world64_v(delta, P0, P1, side, grid=64, sub=3):
    """window change -> (B,grid,grid) on the corner-aligned 64-px world grid used by the pasted-frame scores."""
    res = delta.shape[-1]; dev = delta.device; pitch = SIDE / (grid - 1)
    ax = torch.arange(grid, device=dev) * pitch + LO
    off = (torch.arange(sub, device=dev) - (sub - 1) / 2) * pitch / sub; X = (ax[:, None] + off[None, :]).reshape(-1)
    pts = torch.stack(torch.meshgrid(X, X, indexing="ij"), -1)[None].expand(len(delta), -1, -1, -1)
    return F.avg_pool2d(F.grid_sample(delta[:, None], _win_idx(pts, P0, P1, side, res), mode="bilinear", padding_mode="zeros", align_corners=True), sub).squeeze(1)
