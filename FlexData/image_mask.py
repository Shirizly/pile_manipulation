"""FlexData/image_mask.py -- top-down IMAGE-MASK occupancy for the FleX corpora
(the reusable owner; moved 2026-10-01 from the EXP-0061 prototype
``experiments/EXP-0061-flex-cross-corpus-rerun/code/image_mask/image_mask.py``,
which is kept unchanged as the record of what was validated there).

Pipeline (validated in ``experiments/EXP-0061-*/figures/image_mask/``):

1. Segment: foreground = any RGB channel != 255 (the FleX collector whitens
   every pixel at table depth, so this equals its depth threshold
   pixel-for-pixel, and works without depth -> works on DS-0019).
2. Camera (cam_idx 0): pinhole at (0, 18, 0) looking straight down, 720x720,
   vertical fov 45 deg -> f = 360 / tan(22.5 deg) = 869.12 px, cx = cy = 359.5
   (``FlexData/cam_params.json``, the DERIVED camera; the ``fit`` block there
   is only the confirmation):  u = cx + f x / (cam_h - y),  v = cy + f z / (cam_h - y).
3. Warp: every image pixel is back-projected onto the table plane y = 0 and
   binned into the model grid (row = table X = x_flex, col = table Y = -z_flex,
   pixel i centred at world (i - ctr) / to_pxl -- the ``FlexData.dataset``
   ``world_to_px`` convention). Output = per-cell foreground AREA FRACTION.
4. Binary mask (EXP-0061 user decision): occupied iff area fraction > 0.

Cache builder (chunked, atomic, manifest; resumable -- complete units skipped):

    python -u -m FlexData.image_mask ds0020 [--workers 14]
    python -u -m FlexData.image_mask ds0019 [--workers 14]
    python -u -m FlexData.image_mask ds0020_v1        # ARCHIVED v1 payload (old_data/_ported_v1/cache; already built)

DS-0020 v2 (2026-10-02) uses the same trajectories format: state k = 0 is the initial
image, k >= 1 the after-image of push k-1 (``cache/image_paths.json`` from
``python -u -m FlexData.build_cache paths``). Image paths resolve relative to the
dataset dir (= the dir holding ``config.yaml``; for v1 that is ``old_data/_ported_v1``,
whose index uses ``../<t>/...``).

writes per-unit parts under ``<dataset>/cache/image_masks_parts/`` and then the
consolidated ``<dataset>/cache/image_masks.npz`` in the format
``FlexData.dataset.ImageMaskSource`` reads, plus the area fraction as float16:

  DS-0020 (trajectories): traj_ids (T,), masks (T, 11, H, W) uint8 0/1, frac (T, 11, H, W) f16
  DS-0019 (slates):       state_ids (S,), init (S, H, W) u8, init_frac f16,
                          after_keys (M, 2) [state_idx, action_idx], after (M, H, W) u8, after_frac f16

and ``cache/image_masks.DONE`` (JSON: counts, grid, camera, sha256 of this file)
once the consolidated file is complete. Grid read from the instance config.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CAM_JSON = HERE / "cam_params.json"
DATASETS = {
    "ds0020": REPO / "datasets" / "DS-0020-training-data-flex-N864",          # v2 payload (2026-10-02)
    "ds0020_v1": REPO / "datasets" / "DS-0020-training-data-flex-N864" / "old_data" / "_ported_v1",  # ARCHIVED v1
    "ds0019": REPO / "datasets" / "DS-0019-slates-flex-pile-varN",
    "ds0021": REPO / "datasets" / "DS-0021-flex-carrots-countgroups-train",   # EXP-0064 (trajectories)
    "ds0022": REPO / "datasets" / "DS-0022-flex-carrots-countgroups-test-slates",  # EXP-0064 (slates)
}
THRESHOLD = 0.0          # occupied iff area fraction > THRESHOLD
CHUNK = 100              # DS-0020 trajectories per part file


@dataclass(frozen=True)
class GridSpec:
    half_extent: float = 7.2
    resolution: int = 64

    @property
    def to_pxl(self) -> float:
        return self.resolution / (2.0 * self.half_extent)

    @property
    def ctr(self) -> float:
        return float(round(self.resolution / 2))

    @classmethod
    def from_instance(cls, cfg: dict) -> "GridSpec":
        return cls(float(cfg["grid"]["half_extent"]), int(cfg["grid"]["resolution"]))


@dataclass(frozen=True)
class Camera:
    f: float = 360.0 / np.tan(np.radians(22.5))
    cx: float = 359.5
    cy: float = 359.5
    cam_h: float = 18.0
    width: int = 720
    height: int = 720
    plane_y: float = 0.0

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


_LUT: dict = {}


def _plane_lut(cam: Camera, grid: GridSpec):
    """Flat grid-cell index (-1 outside) of every image pixel back-projected onto
    the plane y = cam.plane_y, and the per-cell pixel count. Cached."""
    key = (cam, grid)
    if key not in _LUT:
        v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float32)
        dist = cam.cam_h - cam.plane_y
        x = (u - cam.cx) * dist / cam.f
        z = (v - cam.cy) * dist / cam.f
        R = grid.resolution
        gi = np.floor(x * grid.to_pxl + grid.ctr + 0.5).astype(np.int64)    # row = X = x
        gj = np.floor(-z * grid.to_pxl + grid.ctr + 0.5).astype(np.int64)   # col = Y = -z
        ok = (gi >= 0) & (gi < R) & (gj >= 0) & (gj < R)
        idx = np.where(ok, gi * R + gj, -1).ravel()
        cnt = np.bincount(idx[idx >= 0], minlength=R * R).astype(np.float32)
        _LUT[key] = (idx, cnt)
    return _LUT[key]


def mask_to_frac(mask: np.ndarray, cam: Camera, grid: GridSpec) -> np.ndarray:
    """(720, 720) bool image mask -> (R, R) float32 foreground area fraction."""
    idx, cnt = _plane_lut(cam, grid)
    R = grid.resolution
    sel = mask.ravel() & (idx >= 0)
    fg = np.bincount(idx[sel], minlength=R * R).astype(np.float32)
    frac = np.divide(fg, cnt, out=np.zeros_like(fg), where=cnt > 0)
    return np.clip(frac, 0.0, 1.0).reshape(R, R)


def image_to_frac(color_png, grid: GridSpec | None = None, cam: Camera | None = None) -> np.ndarray:
    """color_png: path or (720, 720, 3) uint8 array (white = background) ->
    (R, R) float32 area fraction on the grid."""
    grid = grid or GridSpec()
    cam = cam or Camera.load()
    color = color_png if isinstance(color_png, np.ndarray) else cv2.imread(str(color_png), cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError(color_png)
    return mask_to_frac(segment(color), cam, grid)


def image_to_occupancy(color_png, grid: GridSpec | None = None, cam: Camera | None = None,
                       threshold: float = THRESHOLD) -> np.ndarray:
    """Binary (R, R) float32 occupancy: area fraction > threshold (default 0)."""
    return (image_to_frac(color_png, grid, cam) > threshold).astype(np.float32)


# ----------------------------------------------------------------- cache builder
def _atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _code_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _frac_many(args):
    ds_dir, rels, grid, cam = args
    return np.stack([image_to_frac(Path(ds_dir) / r, grid, cam) for r in rels]).astype(np.float32)


def _instance(ds_dir: Path):
    return yaml.safe_load((ds_dir / "config.yaml").read_text())


def build(which: str, workers: int) -> None:
    ds_dir = DATASETS[which]
    cfg = _instance(ds_dir)
    grid, cam = GridSpec.from_instance(cfg), Camera.load()
    cache = ds_dir / "cache"
    parts = cache / "image_masks_parts"
    parts.mkdir(parents=True, exist_ok=True)
    idx = json.loads((cache / "image_paths.json").read_text())
    common = dict(grid=dict(half_extent=grid.half_extent, resolution=grid.resolution,
                            row="X = x_flex", col="Y = -z_flex"),
                  camera=cam.__dict__, threshold=THRESHOLD, code="FlexData/image_mask.py",
                  code_sha256=_code_sha(), source="colour PNG, table-plane back-projection (no depth)")
    man_path = cache / "image_masks_manifest.json"
    t0 = time.time()

    if cfg["kind"] == "trajectories":
        tids = sorted(int(t) for t in idx["trajectories"])
        units = [tids[i:i + CHUNK] for i in range(0, len(tids), CHUNK)]
        jobs = {}
        with ProcessPoolExecutor(workers) as ex:
            for u, ids in enumerate(units):
                if (parts / f"chunk_{u:03d}.npz").exists():
                    continue
                for t in ids:
                    rels = [idx["trajectories"][str(t)][k][0] for k in range(11)]
                    jobs[(u, t)] = ex.submit(_frac_many, (str(ds_dir), rels, grid, cam))
            done = []
            for u, ids in enumerate(units):
                f = parts / f"chunk_{u:03d}.npz"
                if not f.exists():
                    fr = np.stack([jobs[(u, t)].result() for t in ids])          # (T, 11, R, R)
                    _atomic_npz(f, traj_ids=np.asarray(ids, np.int32),
                                masks=(fr > THRESHOLD).astype(np.uint8), frac=fr.astype(np.float16))
                done.append(u)
                _atomic_json(man_path, dict(common, kind="trajectories", n_units=len(units),
                                            units_done=done, elapsed_s=time.time() - t0))
                print(f"[{which}] chunk {u + 1}/{len(units)}  {time.time() - t0:.0f}s", flush=True)
        zs = [np.load(parts / f"chunk_{u:03d}.npz") for u in range(len(units))]
        out = dict(traj_ids=np.concatenate([z["traj_ids"] for z in zs]),
                   masks=np.concatenate([z["masks"] for z in zs]),
                   frac=np.concatenate([z["frac"] for z in zs]))
        counts = dict(n_traj=int(len(out["traj_ids"])), n_states=int(np.prod(out["masks"].shape[:2])),
                      mean_occupied_frac=float(out["masks"].mean()))
    else:
        sids = sorted(int(s) for s in idx["states"])
        jobs = {}
        with ProcessPoolExecutor(workers) as ex:
            for s in sids:
                if (parts / f"state_{s:03d}.npz").exists():
                    continue
                st = idx["states"][str(s)]
                aids = sorted(int(a) for a in st["actions"])
                rels = [st["initial_color"]] + [st["actions"][str(a)] for a in aids]
                jobs[s] = (aids, ex.submit(_frac_many, (str(ds_dir), rels, grid, cam)))
            done = []
            for n, s in enumerate(sids):
                f = parts / f"state_{s:03d}.npz"
                if not f.exists():
                    aids, fut = jobs[s]
                    fr = fut.result()
                    _atomic_npz(f, state_idx=np.int32(s), action_idx=np.asarray(aids, np.int32),
                                init=(fr[0] > THRESHOLD).astype(np.uint8), init_frac=fr[0].astype(np.float16),
                                after=(fr[1:] > THRESHOLD).astype(np.uint8), after_frac=fr[1:].astype(np.float16))
                done.append(s)
                _atomic_json(man_path, dict(common, kind="slates", n_units=len(sids),
                                            units_done=done, elapsed_s=time.time() - t0))
                if n % 10 == 0 or n == len(sids) - 1:
                    print(f"[{which}] state {n + 1}/{len(sids)}  {time.time() - t0:.0f}s", flush=True)
        zs = [np.load(parts / f"state_{s:03d}.npz") for s in sids]
        keys = np.concatenate([np.stack([np.full(len(z["action_idx"]), int(z["state_idx"])), z["action_idx"]], 1)
                               for z in zs]).astype(np.int32)
        out = dict(state_ids=np.asarray(sids, np.int32),
                   init=np.stack([z["init"] for z in zs]), init_frac=np.stack([z["init_frac"] for z in zs]),
                   after_keys=keys, after=np.concatenate([z["after"] for z in zs]),
                   after_frac=np.concatenate([z["after_frac"] for z in zs]))
        counts = dict(n_states=len(sids), n_after=int(len(keys)),
                      mean_occupied_frac_init=float(out["init"].mean()),
                      mean_occupied_frac_after=float(out["after"].mean()))
    _atomic_npz(cache / "image_masks.npz", **out)
    done_rec = dict(common, kind=cfg["kind"], dataset=cfg["id"], file="cache/image_masks.npz",
                    keys=sorted(out), counts=counts, build_s=round(time.time() - t0, 1),
                    date=time.strftime("%Y-%m-%d %H:%M"))
    _atomic_json(man_path, dict(done_rec, units_done="all"))
    _atomic_json(cache / "image_masks.DONE", done_rec)
    print(f"[{which}] DONE {counts} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(DATASETS))
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    build(a.which, a.workers)


if __name__ == "__main__":
    main()
