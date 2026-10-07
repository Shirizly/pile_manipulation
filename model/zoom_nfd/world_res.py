"""Plain world-frame rasters/plates at any resolution (vanilla NFD controls; res 64 -> 2 mm/px, 128 -> 1 mm/px). Same drawing style as
world128.py (sub-pixel boxes, soft plates), n-agnostic. dim0 = x, dim1 = y, pixel i covers world [-64 + i*128/res, ...)."""
import math, cv2, numpy as np, torch
from model.zoom_nfd.window import quat_yaw
LO, SIDE = -0.064, 0.128


def raster_res(states, sizes, res):
    px = SIDE / res; out = np.zeros((len(states), res, res), np.float32)
    for b, st_all in enumerate(states.numpy()):
        for st, dim in zip(st_all, np.asarray(sizes, float)):
            yaw = quat_yaw(st[3:7]); c, s = math.cos(yaw), math.sin(yaw); hw, hh = dim[0] / 2, dim[1] / 2
            cr = np.array([[sx * hw, sy * hh] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]); w = st[:2] + cr @ np.array([[c, s], [-s, c]])
            idx = (w - LO) / px - 0.5; cv2.fillPoly(out[b], [np.round(np.stack([idx[:, 1], idx[:, 0]], -1) * 16).astype(np.int32)], 1.0, shift=4)
    return torch.from_numpy(out)


def plates_res(P0, P1, res, plate_len=0.04, thick=0.002, sigma=0.0015):
    """GPU/torch: (B,2) metres -> (B,2,res,res) soft start/stop plates (plate long axis perpendicular to the push)."""
    px = SIDE / res; dev = P0.device; ar = (torch.arange(res, device=dev, dtype=torch.float32) + 0.5) * px + LO
    X, Y = torch.meshgrid(ar, ar, indexing="ij"); d = P1 - P0; u = d / d.norm(dim=-1, keepdim=True); out = []
    for P in (P0, P1):
        dx, dy = X[None] - P[:, 0, None, None], Y[None] - P[:, 1, None, None]
        along = dx * u[:, 0, None, None] + dy * u[:, 1, None, None]; lat = -dx * u[:, 1, None, None] + dy * u[:, 0, None, None]
        out.append(torch.sigmoid((plate_len / 2 - lat.abs()) / sigma) * torch.sigmoid((thick / 2 - along.abs()) / sigma))
    return torch.stack(out, 1)
