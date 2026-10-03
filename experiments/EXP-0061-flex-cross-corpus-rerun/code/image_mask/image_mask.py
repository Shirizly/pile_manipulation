"""Image -> top-down occupancy on the FleX model grid (EXP-0061 prototype).

NOTE (2026-10-01): the reusable owner is now ``FlexData/image_mask.py`` (same
algorithm, plane case; plus the cache builder). This file is kept unchanged as
the record of what figures/image_mask/ validated; new code imports FlexData.

Pipeline (see README in the figures dir / EXP-0061 report):

1. Segment: foreground = any RGB channel != 255. The FleX collector whitens
   every pixel whose rendered depth >= 0.599/0.8 * global_scale (= 17.97 units,
   i.e. the bare table) to pure white, so this rule reproduces the depth
   threshold of dyn-res-pile-manip/utils.py::_FG_DEPTH_THRESHOLD pixel-for-pixel
   (checked: 0 differing pixels on DS-0020 images). Works without depth, so it
   works on DS-0019.
2. Camera (cam_idx 0 of dyn-res-pile-manip/env/flex_env.py): pinhole at
   (0, 18, 0) looking straight down (pitch -90 deg, yaw 0), 720x720, vertical
   fov 45 deg (the PyFleX default; the "70" in that repo is the LIGHT fov and
   the fallback in experiment_analysis.py is wrong). Mapping, FleX world:
       u (image col) = cx + f * x / (cam_h - y)
       v (image row) = cy + f * z / (cam_h - y)
   f = 360 / tan(22.5 deg) = 869.12 px, cx = cy = 359.5. Confirmed by fitting
   f, cx, cy to particle projections (cam_params.json, `fit` block).
3. Warp: every image pixel is back-projected onto a horizontal plane at height
   `plane_y` (no depth on DS-0019) or, if a depth PNG is given, onto its own
   measured surface point; then binned into the model grid cells (row = table
   X = x_flex, col = table Y = -z_flex, half_extent 7.2, 64 px, pixel i's
   centre at world (i - 32) / 4.444, matching FlexData.dataset.world_to_px).
   Output = per-cell foreground AREA FRACTION in [0, 1] (each cell covers
   ~11x11 image pixels).
4. Binarise (optional, `threshold`): occ = frac > threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CAM_JSON = HERE / "cam_params.json"


@dataclass(frozen=True)
class GridSpec:
    half_extent: float = 7.2      # DS-0019 / DS-0020 config.yaml grid
    resolution: int = 64

    @property
    def to_pxl(self) -> float:
        return self.resolution / (2.0 * self.half_extent)

    @property
    def ctr(self) -> float:
        return float(round(self.resolution / 2))


@dataclass(frozen=True)
class Camera:
    f: float = 360.0 / np.tan(np.radians(22.5))
    cx: float = 359.5
    cy: float = 359.5
    cam_h: float = 18.0           # 6 * global_scale / 8
    width: int = 720
    height: int = 720
    plane_y: float = 0.0          # back-projection plane height when no depth

    @classmethod
    def load(cls, path: str | Path = CAM_JSON) -> "Camera":
        d = json.loads(Path(path).read_text())
        return cls(**{k: d[k] for k in ("f", "cx", "cy", "cam_h", "width", "height", "plane_y")})

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """(N, 3) FleX world (x, y, z) -> (N, 2) image (u=col, v=row)."""
        d = self.cam_h - xyz[:, 1]
        return np.stack([self.cx + self.f * xyz[:, 0] / d, self.cy + self.f * xyz[:, 2] / d], 1)


def segment(color: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) bool foreground (any channel != 255)."""
    return (color != 255).any(2)


_LUT_CACHE: dict = {}


def _pixel_cells(cam: Camera, grid: GridSpec, depth: np.ndarray | None):
    """Flat grid-cell index (or -1 outside) for every image pixel, plus the
    per-cell pixel count. Cached for the depth-free (plane) case."""
    key = (cam, grid)
    if depth is None and key in _LUT_CACHE:
        return _LUT_CACHE[key]
    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
    dist = (cam.cam_h - cam.plane_y) if depth is None else depth.astype(np.float32)
    x = (u - cam.cx) * dist / cam.f
    z = (v - cam.cy) * dist / cam.f
    R = grid.resolution
    gi = np.floor(x * grid.to_pxl + grid.ctr + 0.5).astype(np.int64)     # row = X = x
    gj = np.floor(-z * grid.to_pxl + grid.ctr + 0.5).astype(np.int64)    # col = Y = -z
    ok = (gi >= 0) & (gi < R) & (gj >= 0) & (gj < R)
    idx = np.where(ok, gi * R + gj, -1).ravel()
    out = idx
    if depth is None:
        cnt = np.bincount(idx[idx >= 0], minlength=R * R).astype(np.float32)
        out = (idx, cnt)
        _LUT_CACHE[key] = out
    return out


def mask_to_occupancy(mask: np.ndarray, cam: Camera, grid: GridSpec,
                      depth: np.ndarray | None = None) -> np.ndarray:
    """(720, 720) bool image mask -> (R, R) float32 area fraction on the grid.
    With `depth` (FleX units, camera-z distance), foreground pixels are placed
    at their measured surface point (removes parallax); background pixels still
    use the table plane for the per-cell denominator."""
    R = grid.resolution
    idx_plane, cnt = _pixel_cells(Camera(**{**cam.__dict__, "plane_y": 0.0}) if depth is not None else cam,
                                  grid, None)
    idx = idx_plane if depth is None else _pixel_cells(cam, grid, depth)
    m = mask.ravel()
    sel = m & (idx >= 0)
    fg = np.bincount(idx[sel], minlength=R * R).astype(np.float32)
    frac = np.divide(fg, cnt, out=np.zeros_like(fg), where=cnt > 0)
    return np.clip(frac, 0.0, 1.0).reshape(R, R)


def image_to_occupancy(color_png, grid_spec: GridSpec | None = None, cam: Camera | None = None,
                       depth_png=None, threshold: float | None = None) -> np.ndarray:
    """color_png: path or (720,720,3) uint8 BGR/RGB array (white = background).
    Returns (R, R) float32: area fraction, or binary {0,1} if `threshold` given.
    depth_png (optional, DS-0020 only): path or uint16 array (depth * 1000)."""
    grid_spec = grid_spec or GridSpec()
    cam = cam or Camera.load()
    color = cv2.imread(str(color_png), cv2.IMREAD_COLOR) if not isinstance(color_png, np.ndarray) else color_png
    depth = None
    if depth_png is not None:
        d = cv2.imread(str(depth_png), cv2.IMREAD_UNCHANGED) if not isinstance(depth_png, np.ndarray) else depth_png
        depth = d.astype(np.float32) / 1000.0
    occ = mask_to_occupancy(segment(color), cam, grid_spec, depth)
    if threshold is not None:
        occ = (occ > threshold).astype(np.float32)
    return occ
