"""model/zoom_nfd/world128.py -- plain world-frame NFD inputs at 128x128 (1 mm/px over the 128 mm tray),
the no-crop control for the zoom window (same pixel pitch). Pixel i of dim0/dim1 covers world
[-64 + i, -64 + i + 1] mm (centre index i <-> x = -64 + (i+0.5) mm); dim0 = x, dim1 = y.
Cubes: rotated boxes with sub-pixel vertices (same style as the window raster); plates: soft."""
import math, cv2, numpy as np, torch
from transforms.functional import draw_plate_soft

RES, PX = 128, 0.001
LO = -0.064

def raster128(states: torch.Tensor, sizes) -> torch.Tensor:
    from model.zoom_nfd.window import quat_yaw
    out = np.zeros((len(states), RES, RES), np.float32)
    for b, st_all in enumerate(states.numpy()):
        for st, dim in zip(st_all, np.asarray(sizes, float)):
            yaw = quat_yaw(st[3:7]); c, s = math.cos(yaw), math.sin(yaw)
            hw, hh = dim[0] / 2, dim[1] / 2
            cr = np.array([[sx * hw, sy * hh] for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
            w = st[:2] + cr @ np.array([[c, s], [-s, c]])
            idx = (w - LO) / PX - 0.5                      # (row=x, col=y) index coords
            pts = np.round(np.stack([idx[:, 1], idx[:, 0]], -1) * 16).astype(np.int32)   # cv2 wants (col,row)
            cv2.fillPoly(out[b], [pts], 1.0, shift=4)
    return torch.from_numpy(out)

def plates128(P0, P1, plate_len=0.04, thick=0.002, sigma=0.0015) -> torch.Tensor:
    P0, P1 = np.asarray(P0, float), np.asarray(P1, float)
    ang = torch.from_numpy(np.arctan2(P1[:, 1] - P0[:, 1], P1[:, 0] - P0[:, 0]) - math.pi / 2).float()
    ch = []
    for P in (P0, P1):
        c = torch.from_numpy((P - LO) / PX - 0.5).float()
        ch.append(draw_plate_soft(c, ang, (RES, RES), plate_len / PX, thick / PX, 1.0, sigma / PX))
    return torch.stack(ch, 1)
