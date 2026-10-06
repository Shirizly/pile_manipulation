"""model/zoom_nfd/window_gpu.py -- the zoom-window pipeline starting from a VISUAL input:
a hi-res top-down occupancy raster of the whole tray (what a camera + segmentation gives), not particle poses.

  hr raster (1,1,H,H)  --extract_windows-->  K windows (K,64,64)   [rotation + zoom as ONE grid_sample, antialiased
  K actions            --plates_gpu------->  K x 2 plate channels     by an s x s sub-sample average]
  UNet forward                              K predictions
  paste_gpu                                  prediction CHANGE sampled back onto the 64x64 world grid

Conventions identical to window.py (pixel index i = centre; dim0 = world x, dim1 = world y; tray [-64, 64] mm).
"""
import math, cv2, numpy as np, torch
import torch.nn.functional as F
from model.zoom_nfd.window import WindowSpec, quat_yaw
from transforms.functional import draw_plate_soft

LO, SIDE = -0.064, 0.128


def raster_world(states: torch.Tensor, sizes, res: int, aa: int = 1) -> torch.Tensor:
    """Box raster of the whole tray at `res` x `res` (pixel = 128/res mm), stand-in for a camera image. aa > 1: rendered at
    aa*res and area-averaged (soft coverage; avoids the cv2 fill's boundary-inclusion inflation of cube area)."""
    if aa > 1:
        return F.avg_pool2d(raster_world(states, sizes, res * aa)[:, None], aa)[:, 0]
    px = SIDE / res
    out = np.zeros((len(states), res, res), np.float32)
    for b, st_all in enumerate(states.numpy()):
        for st, dim in zip(st_all, np.asarray(sizes, float)):
            yaw = quat_yaw(st[3:7]); c, s = math.cos(yaw), math.sin(yaw)
            hw, hh = dim[0] / 2, dim[1] / 2
            cr = np.array([[sx * hw, sy * hh] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
            w = st[:2] + cr @ np.array([[c, s], [-s, c]])
            idx = (w - LO) / px - 0.5
            pts = np.round(np.stack([idx[:, 1], idx[:, 0]], -1) * 16).astype(np.int32)
            cv2.fillPoly(out[b], [pts], 1.0, shift=4)
    return torch.from_numpy(out)


def _axes(P0, P1):
    d = P1 - P0; L = d.norm(dim=-1, keepdim=True); u = d / L
    return u, torch.stack([u[:, 1], -u[:, 0]], -1)


def extract_windows(hr: torch.Tensor, P0: torch.Tensor, P1: torch.Tensor, spec: WindowSpec, ss: int = 3) -> torch.Tensor:
    """hr (H,H) float on device; P0,P1 (K,2) metres on device -> (K,res,res) windows, area-averaged
    (ss x ss sub-samples per window pixel, bilinear in the hi-res raster). One grid_sample for all K."""
    H = hr.shape[-1]; K = len(P0); res = spec.res; pxh = SIDE / H
    u, v = _axes(P0, P1)
    off = (torch.arange(ss, device=hr.device) - (ss - 1) / 2) / ss
    ci = (torch.arange(res, device=hr.device)[:, None] + 0.5 + off[None, :]).reshape(-1)       # (res*ss,) col coords
    s = ci * spec.px - spec.margin_back; t = ci * spec.px - spec.side / 2                      # along / lateral offsets
    # world point for (row r, col c): P0 + s[c] u + t[r] v ; grid (K, res*ss rows, res*ss cols, 2)
    Pw = P0[:, None, None, :] + s[None, None, :, None] * u[:, None, None, :] + t[None, :, None, None] * v[:, None, None, :]
    idx = (Pw - LO) / pxh - 0.5                                                                  # (row=x, col=y) hr index
    g = torch.stack([idx[..., 1], idx[..., 0]], -1) / (H - 1) * 2 - 1                            # (K, R, R, 2)
    g = g.reshape(1, K * res * ss, res * ss, 2)
    out = F.grid_sample(hr[None, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.reshape(K, res * ss, res * ss)
    return F.avg_pool2d(out[:, None], ss).squeeze(1)


def plates_gpu(P0: torch.Tensor, P1: torch.Tensor, spec: WindowSpec) -> torch.Tensor:
    """(K,2,res,res) canonical soft plates (start, stop)."""
    K = len(P0); px = spec.px; L = (P1 - P0).norm(dim=-1)
    row = torch.full((K,), (spec.side / 2) / px - 0.5, device=P0.device)
    c0 = torch.full((K,), spec.margin_back / px - 0.5, device=P0.device); c1 = c0 + L / px
    cen = torch.cat([torch.stack([row, c0], -1), torch.stack([row, c1], -1)])
    pl = draw_plate_soft(cen, torch.zeros(2 * K, device=P0.device), (spec.res, spec.res), spec.plate_len / px,
                         spec.plate_thick / px, 1.0, spec.plate_sigma / px)
    return torch.stack([pl[:K], pl[K:]], 1)


def paste_gpu(delta: torch.Tensor, occ0_world: torch.Tensor, P0, P1, spec: WindowSpec, grid: int = 64, sub: int = 3):
    """delta (K,res,res) window change -> (K,grid,grid) = clamp(occ0_world + delta sampled at the world grid), zero outside."""
    K = len(P0); dev = delta.device; pitch = SIDE / (grid - 1)
    ax = torch.arange(grid, device=dev) * pitch + LO
    off = (torch.arange(sub, device=dev) - (sub - 1) / 2) * pitch / sub
    X = (ax[:, None] + off[None, :]).reshape(-1); Y = X
    pts = torch.stack(torch.meshgrid(X, Y, indexing="ij"), -1)                                   # (g*sub, g*sub, 2)
    u, v = _axes(P0, P1)
    d = pts[None] - P0[:, None, None, :]
    col = ((d * u[:, None, None, :]).sum(-1) + spec.margin_back) / spec.px - 0.5
    row = ((d * v[:, None, None, :]).sum(-1) + spec.side / 2) / spec.px - 0.5
    g = torch.stack([col, row], -1) / (spec.res - 1) * 2 - 1
    samp = F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True)
    samp = F.avg_pool2d(samp, sub).squeeze(1)
    return (occ0_world[None] + samp).clamp(0, 1)
