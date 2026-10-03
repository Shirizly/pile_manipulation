"""FlexData/dataset.py -- FleX carrot-pile corpora (DS-0019, DS-0020) through
the project's standard Eulerian batch contract, in NATIVE FleX units.

Genesis-free: numpy/torch/yaml + ``transforms.functional.draw_plate_soft``.

FRAME (measured, EXP-0061 phase 1 -- ``experiments/EXP-0061-*/code/
plate_width_check.py``): the particle files store FleX ``(x, y_up, z)``; the
stored actions ``[a0, a1, a2, a3]`` are in a frame whose 2nd/4th components
are **-z**, not z (with actions taken as ``[x, z, x, z]`` literally, only ~7 %
of particles in the plate's path move; with ``a1 = -z`` ~100 % do). The
loader therefore works in the TABLE FRAME

    X = x_flex,   Y = -z_flex          (``flex_xz_to_table``)

which is exactly the actions' own frame, and is right-handed with FleX's y up
(X x Y = up), i.e. the same handedness as Genesis's (x, y, z-up). Actions are
used AS STORED; particles are converted. Grid convention is the repo's:
**row (dim 0) = X, col (dim 1) = Y**, pixel coordinate ``c = world * to_pxl +
ctr_in_PXL`` -- the same mapping ``PileSweepData`` uses, so every consumer that
reads ``raw.to_pxl`` / ``raw.ctr_in_PXL`` / ``raw.configs[0]['plate']['size']``
/ ``raw.configs[0]['box']['vol']`` (NFD predictors, GNN perception, the
eval harness) works unchanged.

Rasteriser: hard disk footprint, pixel ``i`` (centre at pixel coordinate i,
the SAME convention ``draw_plate_soft`` uses for the plate channels) is
occupied iff some particle centre lies within ``footprint_radius_px`` of it.
Genesis's ``_draw_particle_grid`` instead int-truncates (centre at i+0.5),
which is why ``eval_report.truth_for_scoring`` has a measured -1.0 px offset
there; this dataset exposes ``score_uv_offset_px = 0.0`` for its own frame.

Instance configs: ``datasets/DS-0019-*/config.yaml``, ``datasets/DS-0020-*/
config.yaml`` (DS-0020 **v2**, ``cache_format: traj_manifest_v2``, template
``configs/dataset/flex_pile_instance_v2.yaml``), and the ARCHIVED DS-0020 v1
``datasets/DS-0020-*/old_data/_ported_v1/config.yaml`` (``chunks_v1``, the data
EXP-0061 / MODEL-0004..0007 used; template ``configs/dataset/flex_pile_instance.yaml``).
Trajectory cache readers: ``_TrajSource`` (v1 chunks, fixed P) and
``_TrajManifestSource`` (v2, variable P per trajectory), selected by ``cache_format``.
Registered dataset type: ``flex`` (``registry/dataset_registry.py``).
Eval-cell loader: ``load_flex_cell`` (the ``load_randlen_cell`` interface).
"""
from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from transforms.functional import draw_plate_soft

REPO = Path(__file__).resolve().parent.parent


def flex_xz_to_table(xz: np.ndarray) -> np.ndarray:
    """(..., 2) FleX (x, z) -> table frame (X, Y) = (x, -z). See module docstring."""
    out = np.array(xz, dtype=np.float32, copy=True)
    out[..., 1] *= -1.0
    return out


def load_instance_config(cfg: str | Path | dict) -> dict:
    if isinstance(cfg, dict):
        return cfg
    p = Path(cfg)
    if not p.is_absolute():
        p = REPO / p
    return yaml.safe_load(p.read_text())


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


# ---------------------------------------------------------------- rasteriser
def rasterize_disk(px: np.ndarray, H: int, W: int, radius_px: float) -> np.ndarray:
    """(N, 2) continuous pixel coords (dim0, dim1), pixel i's centre at i ->
    (H, W) float32 binary occupancy: a pixel is 1 iff some point lies within
    ``radius_px`` of its centre. Vectorised over points (no per-point loop);
    non-finite points are ignored, points off the grid simply cover nothing."""
    occ = np.zeros((H, W), dtype=np.float32)
    px = px[np.isfinite(px).all(1)]
    if px.size == 0:
        return occ
    base = np.floor(px).astype(np.int64)
    k = int(math.ceil(radius_px)) + 1
    r2 = radius_px * radius_px
    for di in range(-k + 1, k + 1):
        for dj in range(-k + 1, k + 1):
            i = base[:, 0] + di
            j = base[:, 1] + dj
            ok = ((i - px[:, 0]) ** 2 + (j - px[:, 1]) ** 2 <= r2) & (i >= 0) & (i < H) & (j >= 0) & (j < W)
            occ[i[ok], j[ok]] = 1.0
    return occ


# ---------------------------------------------------------------- caches
class _TrajSource:
    """DS-0020: 2000 trajectories x 10 pushes; row = (trajectory, push k),
    before = state k, after = state k + 1."""

    kind = "trajectories"

    def __init__(self, cfg: dict, traj_ids: list[int] | None):
        files = sorted(glob.glob(str(_abs(cfg["cache_dir"]) / "chunk_*.npz")))
        assert files, f"no cache chunks in {cfg['cache_dir']} -- run python -u -m FlexData.build_cache ds0020"
        want = None if traj_ids is None else set(int(t) for t in traj_ids)
        xs, acts, tids, md = [], [], [], []
        for f in files:
            z = np.load(f)
            ids = z["traj_ids"]
            keep = np.ones(len(ids), bool) if want is None else np.isin(ids, list(want))
            if not keep.any():
                continue
            xs.append(z["xz"][keep]); acts.append(z["actions"][keep])
            tids.append(ids[keep]); md.append(z["max_disp"][keep])
        self.xz = np.concatenate(xs)                    # (T, 11, P, 2) float16, RAW flex (x, z)
        self.actions = np.concatenate(acts)             # (T, 10, 4) table frame (as stored)
        self.traj_ids = np.concatenate(tids)
        self.max_disp = np.concatenate(md)              # (T, 10)
        T, K = self.actions.shape[:2]
        self.rows = np.stack(np.meshgrid(np.arange(T), np.arange(K), indexing="ij"), -1).reshape(-1, 2)

    def before(self, r):
        t, k = self.rows[r]
        return flex_xz_to_table(self.xz[t, k])

    def after(self, r):
        t, k = self.rows[r]
        return flex_xz_to_table(self.xz[t, k + 1])

    def action(self, r):
        t, k = self.rows[r]
        return self.actions[t, k]

    def group(self, r):          # trajectory id
        return int(self.traj_ids[self.rows[r, 0]])

    def step(self, r):           # push index within the trajectory
        return int(self.rows[r, 1])

    def row_max_disp(self):
        return self.max_disp[self.rows[:, 0], self.rows[:, 1]]

    def frac_out(self, half_extent: float, escape_abs: float):
        """per row: (fraction of before/after particles outside the grid (max of
        the two), any NaN, any particle beyond escape_abs)."""
        T = self.xz.shape[0]
        out_s = np.zeros(self.xz.shape[:2]); nan_s = np.zeros(self.xz.shape[:2], bool)
        esc_s = np.zeros(self.xz.shape[:2], bool)
        for b in range(0, T, 50):                                   # blocked: no 3 GB float32 copy
            a = np.abs(self.xz[b:b + 50])                           # float16, (t, 11, P, 2)
            nan = ~np.isfinite(a).all(-1)
            out_s[b:b + 50] = ((a > half_extent).any(-1) | nan).mean(-1)
            nan_s[b:b + 50] = nan.any(-1)
            esc_s[b:b + 50] = (np.nan_to_num(a, nan=0.0) > escape_abs).any(-1).any(-1)
        t, k = self.rows[:, 0], self.rows[:, 1]
        return (np.maximum(out_s[t, k], out_s[t, k + 1]), nan_s[t, k] | nan_s[t, k + 1],
                esc_s[t, k] | esc_s[t, k + 1])


class _TrajManifestSource(_TrajSource):
    """DS-0020 v2 (``cache_format: traj_manifest_v2``): trajectories x 10 pushes with a
    VARIABLE particle count per trajectory (``FlexData/build_cache.py ds0020`` ->
    ``v2_chunk_*.npz``; particles concatenated along axis 1, trajectory j owns
    ``xz[:, p_off[j]:p_off[j+1]]``). Row = (trajectory, push k), before = state k
    (0 = initial), after = state k + 1 (= the after-state of push k; the chaining
    k-after == (k+1)-before is verified in DATASET.md). Same row/group/step/flag
    interface as the v1 ``_TrajSource`` so ``ImageMaskSource`` (masks (T, 11, H, W)
    keyed by ``traj_ids``) and every consumer work unchanged. Particles are for
    audit / flags / particle-truth scoring only; with ``occupancy.source:
    image_mask`` the model input never reads them."""

    def __init__(self, cfg: dict, traj_ids: list[int] | None):
        files = sorted(glob.glob(str(_abs(cfg["cache_dir"]) / "v2_chunk_*.npz")))
        assert files, f"no v2 cache chunks in {cfg['cache_dir']} -- run python -u -m FlexData.build_cache ds0020"
        want = None if traj_ids is None else set(int(t) for t in traj_ids)
        xs, offs, acts, tids, md, npart, nrig, ipos, plen = [], [], [], [], [], [], [], [], []
        base = 0
        for f in files:
            with np.load(f) as z:
                ids = z["traj_ids"]
                keep = np.ones(len(ids), bool) if want is None else np.isin(ids, list(want))
                if not keep.any():
                    continue
                po, xz = z["p_off"], z["xz"]
                for j in np.nonzero(keep)[0]:
                    blk = xz[:, po[j]:po[j + 1]]
                    xs.append(blk); offs.append(base); base += blk.shape[1]
                acts.append(z["actions"][keep]); tids.append(ids[keep]); md.append(z["max_disp"][keep])
                npart.append(z["n_particles"][keep]); nrig.append(z["n_rigids"][keep])
                ipos.append(z["init_pos"][keep]); plen.append(z["push_length"][keep])
        self.xz_flat = np.concatenate(xs, axis=1)       # (11, sumP, 2) float16, RAW flex (x, z)
        self.p_off = np.asarray(offs + [base], np.int64)
        self.actions = np.concatenate(acts)             # (T, 10, 4) table frame (as stored)
        self.traj_ids = np.concatenate(tids)
        self.max_disp = np.concatenate(md)              # (T, 10)
        self.n_particles = np.concatenate(npart)
        self.n_rigids = np.concatenate(nrig)
        self.init_pos = np.concatenate(ipos)            # 0 rand_blob, 1 rand_spread
        self.push_length = np.concatenate(plen)
        T, K = self.actions.shape[:2]
        self.rows = np.stack(np.meshgrid(np.arange(T), np.arange(K), indexing="ij"), -1).reshape(-1, 2)

    def state(self, t: int, k: int) -> np.ndarray:
        """(P_t, 2) RAW flex (x, z) of trajectory position t, state k."""
        return self.xz_flat[k, self.p_off[t]:self.p_off[t + 1]]

    def before(self, r):
        t, k = self.rows[r]
        return flex_xz_to_table(self.state(t, k))

    def after(self, r):
        t, k = self.rows[r]
        return flex_xz_to_table(self.state(t, k + 1))

    def frac_out(self, half_extent: float, escape_abs: float):
        T, S = len(self.traj_ids), self.xz_flat.shape[0]
        out_s = np.zeros((T, S)); nan_s = np.zeros((T, S), bool); esc_s = np.zeros((T, S), bool)
        for t in range(T):
            a = np.abs(self.xz_flat[:, self.p_off[t]:self.p_off[t + 1]])     # (11, P, 2) float16
            nan = ~np.isfinite(a).all(-1)
            out_s[t] = ((a > half_extent).any(-1) | nan).mean(-1)
            nan_s[t] = nan.any(-1)
            esc_s[t] = (np.nan_to_num(a, nan=0.0) > escape_abs).any(-1).any(-1)
        t, k = self.rows[:, 0], self.rows[:, 1]
        return (np.maximum(out_s[t, k], out_s[t, k + 1]), nan_s[t, k] | nan_s[t, k + 1],
                esc_s[t, k] | esc_s[t, k + 1])


class _SlateSource:
    """DS-0019: 100 same-state slates; row = one valid (state, action)."""

    kind = "slates"

    def __init__(self, cfg: dict, state_ids: list[int] | None):
        files = sorted(glob.glob(str(_abs(cfg["cache_dir"]) / "state_*.npz")))
        assert files, f"no cache in {cfg['cache_dir']} -- run python -u -m FlexData.build_cache ds0019"
        self.init, self.after_xz, rows, acts, aidx, md, plen, bins, nrig = [], [], [], [], [], [], [], [], []
        self.state_ids = []
        for f in files:
            s = int(Path(f).stem.split("_")[1])
            if state_ids is not None and s not in state_ids:
                continue
            z = np.load(f)
            j = len(self.state_ids)
            self.state_ids.append(s)
            self.init.append(z["init_xz"]); self.after_xz.append(z["after_xz"])
            n = len(z["action_idx"])
            rows += [(j, a) for a in range(n)]
            acts.append(z["actions"]); aidx.append(z["action_idx"]); md.append(z["max_disp"])
            plen.append(z["push_length"]); bins.append(z["bin"]); nrig.append(int(z["n_rigids"]))
        self.rows = np.asarray(rows, dtype=np.int64)
        self.actions_flat = np.concatenate(acts)
        self.action_idx = np.concatenate(aidx)
        self.max_disp_flat = np.concatenate(md)
        self.push_length = np.concatenate(plen)
        self.bin = np.concatenate(bins)
        self.n_rigids = nrig

    def before(self, r):
        return flex_xz_to_table(self.init[self.rows[r, 0]])

    def after(self, r):
        j, a = self.rows[r]
        return flex_xz_to_table(self.after_xz[j][a])

    def action(self, r):
        return self.actions_flat[r]

    def group(self, r):          # state (slate) id
        return int(self.state_ids[self.rows[r, 0]])

    def step(self, r):           # every slate row is a step-0 candidate
        return 0

    def row_max_disp(self):
        return self.max_disp_flat

    def frac_out(self, half_extent: float, escape_abs: float):
        fo, nn, es = [], [], []
        for j in range(len(self.state_ids)):
            a0 = np.abs(self.init[j].astype(np.float32))
            A = np.abs(self.after_xz[j].astype(np.float32))
            o0 = (a0 > half_extent).any(-1).mean()
            fo.append(np.maximum(o0, (A > half_extent).any(-1).mean(-1)))
            nn.append(~np.isfinite(A).all(-1).all(-1) | (not np.isfinite(a0).all()))
            es.append((np.nan_to_num(A, nan=0) > escape_abs).any(-1).any(-1) | bool((a0 > escape_abs).any()))
        return np.concatenate(fo), np.concatenate(nn), np.concatenate(es)


# ---------------------------------------------------------------- image-mask hook
class ImageMaskSource:
    """HOOK (EXP-0061, 2026-10-01 design change): occupancy from PRECOMPUTED
    top-down image masks (segmented colour/depth renders mapped onto THIS grid)
    instead of the particle raster. The segmentation is ``FlexData/image_mask.py``
    (builder: ``python -u -m FlexData.image_mask ds0020|ds0019``, which also
    stores the float16 area fraction as ``frac`` / ``init_frac`` /
    ``after_frac``); this class only loads the binary masks. Per-state PNG paths are indexed in
    ``<cache_dir>/image_paths.json`` (``FlexData/build_cache.py paths``).

    Expected file (instance config ``occupancy.mask_file``, default
    ``<cache_dir>/image_masks.npz``), on the instance's own grid (same
    ``half_extent``/``resolution``, row = X = x_flex, col = Y = -z_flex):
      kind=trajectories: ``traj_ids (T,) int``, ``masks (T, 11, H, W)``
      kind=slates:       ``state_ids (S,) int``, ``init (S, H, W)``,
                         ``after_keys (M, 2) int [state_idx, action_idx]``, ``after (M, H, W)``
    values in [0, 1] (uint8 0/1 or float). Missing rows raise -- never a silent
    fallback to the particle raster."""

    def __init__(self, cfg: dict, occ_cfg: dict, H: int, W: int):
        path = occ_cfg.get("mask_file") or str(Path(cfg["cache_dir"]) / "image_masks.npz")
        path = _abs(path)
        if not path.exists():
            raise FileNotFoundError(
                f"occ_source=image_mask but no mask file at {path}; produce it with the "
                f"EXP-0061 image_mask pipeline (format: FlexData.dataset.ImageMaskSource)")
        # materialise once: indexing an NpzFile re-reads the whole array per access
        with np.load(path) as z:
            self.z = {k: z[k] for k in z.files}
        self.kind = cfg["kind"]
        if self.kind == "trajectories":
            self.pos = {int(t): i for i, t in enumerate(self.z["traj_ids"])}
            shape = self.z["masks"].shape[-2:]
        else:
            self.pos = {int(s): i for i, s in enumerate(self.z["state_ids"])}
            self.after_pos = {(int(a), int(b)): i for i, (a, b) in enumerate(self.z["after_keys"])}
            shape = self.z["init"].shape[-2:]
        assert tuple(shape) == (H, W), f"mask grid {shape} != instance grid {(H, W)}"

    def pair(self, src, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
        if self.kind == "trajectories":
            t, k = src.rows[r]
            m = self.z["masks"][self.pos[int(src.traj_ids[t])]]
            return f(m[k]), f(m[k + 1])
        j, a = src.rows[r]
        s = int(src.state_ids[j])
        return f(self.z["init"][self.pos[s]]), f(self.z["after"][self.after_pos[(s, int(src.action_idx[r]))]])


# ---------------------------------------------------------------- dataset
class FlexPileData(Dataset):
    """Raw FleX dataset, ``PileSweepData``-compatible.

    ``__getitem__`` -> ``((input (C,H,W), physics (3,) zeros), target (H,W))``,
    C = 3 (``[occ0, r(p_start), r(p_stop)]``, the ``nfd-genesis-3ch`` layout)
    or 2 (``[occ0, union]``, the plain ``genesis`` layout).

    split: ``train`` / ``val`` (DS-0020's trajectory-level ``splits.json``),
    ``test`` (rows listed under ``test`` in the split file, if any) or ``all``.
    """

    def __init__(self, instance: str | Path | dict, split: str = "all", channels: int = 3,
                 exclude_flagged: bool = True, verbose: bool = True,
                 occ_source: str | None = None):
        cfg = load_instance_config(instance)
        self.cfg = cfg
        self.dataset_id = cfg["id"]
        assert channels in (2, 3)
        self.channels = channels
        g = cfg["grid"]
        self.half_extent = float(g["half_extent"])
        self.H = self.W = int(g["resolution"])
        self.to_pxl = self.H / (2.0 * self.half_extent)          # px per FleX unit
        self.footprint_radius_px = float(g["footprint_radius_px"])
        self.ctr_in_PXL = torch.tensor((round(self.H / 2), round(self.W / 2), 0))
        p = cfg["plate"]
        self.plate_size = [float(p["width"]), float(p["thickness"]), float(p.get("height") or 0.0)]
        self.plate_sigma_px = float(p["sigma_px"])
        # consumers compute sigma as max(0.5, 1.5 * resolution_scale) (PileSweepData
        # convention); this makes that expression evaluate to plate_sigma_px.
        self.resolution_scale = self.plate_sigma_px / 1.5
        self.score_uv_offset_px = 0.0
        self.configs = [{
            "plate": {"size": self.plate_size},
            "box": {"vol": [2 * self.half_extent, 2 * self.half_extent, 0.0]},
            "units": "flex", "dataset_id": self.dataset_id,
        }]
        self._physics = torch.zeros(3, dtype=torch.float32)

        # --- which source rows
        split_ids = None
        if split != "all":
            sp = json.loads(_abs(cfg["split_file"]).read_text())
            if split not in sp["splits"]:
                raise ValueError(f"{self.dataset_id}: split {split!r} not in {sorted(sp['splits'])}")
            split_ids = [int(x) for x in sp["splits"][split]]
            if not split_ids:
                raise ValueError(f"{self.dataset_id}: split {split!r} is empty")
        if cfg["kind"] == "trajectories":
            fmt = cfg.get("cache_format", "chunks_v1")
            if fmt == "chunks_v1":
                self.src = _TrajSource(cfg, split_ids)
            elif fmt == "traj_manifest_v2":
                self.src = _TrajManifestSource(cfg, split_ids)
            else:
                raise ValueError(f"{self.dataset_id}: unknown cache_format {fmt!r}")
        elif cfg["kind"] == "slates":
            self.src = _SlateSource(cfg, None if split_ids is None else set(split_ids))
        else:
            raise ValueError(cfg["kind"])

        # --- occupancy source (pluggable; particles is the default)
        occ_cfg = cfg.get("occupancy") or {}
        self.occ_source = occ_source or occ_cfg.get("source", "particles")
        self._masks = None
        if self.occ_source == "image_mask":
            self._masks = ImageMaskSource(cfg, occ_cfg, self.H, self.W)
        elif self.occ_source != "particles":
            raise ValueError(f"occ_source must be 'particles' or 'image_mask', got {self.occ_source!r}")

        # --- flags (computed from the cache; thresholds from the instance config)
        fl = cfg["flags"]
        frac_out, has_nan, escaped = self.src.frac_out(self.half_extent, float(fl["escape_abs"]))
        md = self.src.row_max_disp()
        null = ~(np.nan_to_num(md, nan=np.inf) >= float(fl["null_disp"]))
        self.flags = dict(nan=has_nan, escaped=escaped & ~has_nan,
                          out_of_grid=(frac_out > float(fl["max_out_of_grid_frac"])) & ~has_nan & ~escaped,
                          null=null & ~has_nan)
        bad = has_nan | escaped | self.flags["out_of_grid"] | self.flags["null"]
        self.flags["any"] = bad
        n = len(bad)
        self._index_map = np.nonzero(~bad)[0] if exclude_flagged else np.arange(n)
        if verbose:
            print(f"FlexPileData({self.dataset_id}, split={split}, exclude_flagged={exclude_flagged}): "
                  f"{n} rows; flagged nan={int(has_nan.sum())} escaped={int(self.flags['escaped'].sum())} "
                  f"out_of_grid={int(self.flags['out_of_grid'].sum())} null={int(self.flags['null'].sum())} "
                  f"-> kept {len(self._index_map)}", flush=True)
        # PileSweepData-style bookkeeping (one "run" per group: trajectory / slate)
        self._input_grid = torch.zeros((channels, self.H, self.W))
        self._output_grid = torch.zeros((self.H, self.W))

    # -------- PileSweepData-compatible accessors
    def __len__(self):
        return len(self._index_map)

    def _resolve_idx(self, idx: int) -> int:
        return int(self._index_map[idx])

    @property
    def workspace_bounds(self):
        h = self.half_extent
        return (-h, -h), (h, h)

    @property
    def plate_width_px(self) -> float:
        return self.plate_size[0] * self.to_pxl

    def get_run_index(self, idx: int) -> int:
        """Group id: trajectory id (DS-0020) / state (slate) id (DS-0019)."""
        return self.src.group(self._resolve_idx(idx))

    def get_step_index(self, idx: int) -> int:
        return self.src.step(self._resolve_idx(idx))

    def get_raw_action(self, idx: int) -> torch.Tensor:
        """``[sx, sy, ex, ey]`` in the TABLE frame, FleX units (= as stored)."""
        return torch.as_tensor(self.src.action(self._resolve_idx(idx)), dtype=torch.float32)

    def particles_before(self, idx: int) -> torch.Tensor:
        """(P, 2) table-frame (X, Y) particle centres before the push."""
        return torch.from_numpy(self.src.before(self._resolve_idx(idx)))

    def particles_after(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.src.after(self._resolve_idx(idx)))

    def world_to_px(self, xy: np.ndarray) -> np.ndarray:
        return xy * self.to_pxl + self.ctr_in_PXL[:2].numpy().astype(np.float32)

    def rasterize(self, xy: np.ndarray) -> np.ndarray:
        return rasterize_disk(self.world_to_px(np.asarray(xy, np.float32)), self.H, self.W,
                              self.footprint_radius_px)

    def plate_channels(self, action) -> tuple[torch.Tensor, torch.Tensor]:
        """(r_start, r_stop), each (H, W): ``draw_plate_soft`` at the push
        start/end, plate yaw = travel direction + pi/2 (``action_to_pose``
        convention), plate long side (width) perpendicular to the push."""
        a = torch.as_tensor(action, dtype=torch.float32)
        ang = torch.atan2(a[3] - a[1], a[2] - a[0]) + math.pi / 2
        ctr = self.ctr_in_PXL[:2].to(torch.float32)
        s = (a[:2] * self.to_pxl + ctr)[None]
        e = (a[2:] * self.to_pxl + ctr)[None]
        L, T = self.plate_size[0] * self.to_pxl, self.plate_size[1] * self.to_pxl
        ang = ang[None]
        rs = draw_plate_soft(s, ang, (self.H, self.W), L, T, intensity=1.0, sigma=self.plate_sigma_px)[0]
        re = draw_plate_soft(e, ang, (self.H, self.W), L, T, intensity=1.0, sigma=self.plate_sigma_px)[0]
        return rs, re

    def occupancy_pair(self, r: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(occ0, occ1) for SOURCE row r from the configured occupancy source:
        ``particles`` (default) = disk raster of the particle centres;
        ``image_mask`` = precomputed top-down masks (``ImageMaskSource``).
        Scoring truth (``truth_for_scoring_flex``) is ALWAYS particle-based."""
        if self._masks is not None:
            return self._masks.pair(self.src, r)
        return (torch.from_numpy(self.rasterize(self.src.before(r))),
                torch.from_numpy(self.rasterize(self.src.after(r))))

    def __getitem__(self, idx: int):
        r = self._resolve_idx(idx)
        occ0, occ1 = self.occupancy_pair(r)
        rs, re = self.plate_channels(self.src.action(r))
        if self.channels == 3:
            x = torch.stack([occ0, rs, re])
        else:  # PileSweepData._draw_plate's 0.5/1.0 union
            x = torch.stack([occ0, 1 - (1 - 0.5 * rs) * (1 - re)])
        return (x, self._physics.clone()), occ1


# ---------------------------------------------------------------- eval cell
@dataclass
class FlexCellData:
    """Same field names as ``Baselines.common.randlen_data.RandlenCellData``
    (and hence ``PredictorBatch``), so ``eval_report.py``'s predictors,
    ``_predict`` and ``_capture_report`` take it directly. No ``states`` field
    (particle counts vary per DS-0019 slate); particle-space consumers use
    ``raw.particles_before(i)`` / ``raw.particles_after(i)``."""
    tag: str
    occ0: torch.Tensor        # (N,H,W)
    occ1: torch.Tensor        # (N,H,W) ground truth (hard disk raster), eval-only
    actions: torch.Tensor     # (N,4) table frame, FleX units
    p_start: torch.Tensor     # (N,3) [X, Y, 0]
    p_stop: torch.Tensor      # (N,3)
    angle: torch.Tensor       # (N,) plate yaw = atan2(dY, dX) + pi/2
    run_idx: torch.Tensor     # (N,) group id (trajectory / slate)
    slate_idx: torch.Tensor   # (N,) slate id (DS-0019) / trajectory id (DS-0020)
    step_idx: torch.Tensor    # (N,) 0 for every DS-0019 row; push index for DS-0020
    push_length: torch.Tensor  # (N,)
    files: list
    workspace_min: tuple
    workspace_max: tuple
    H: int
    W: int
    raw: object               # the FlexPileData instance


def load_flex_cell(instance: str | Path | dict, split: str = "all", tag: str | None = None,
                   exclude_flagged: bool = True, occ_source: str | None = None) -> FlexCellData:
    """The ``load_randlen_cell`` interface for a FleX instance config.
    For DS-0019 every row is a step-0 candidate of its slate (``slate_idx`` =
    DS-0019 ``state_idx``), so ``_capture_report``'s ``step_idx == 0`` filter
    keeps every valid row and groups them into the 100 explicit slates."""
    raw = FlexPileData(instance, split=split, channels=3, exclude_flagged=exclude_flagged,
                       occ_source=occ_source)
    n = len(raw)
    occ0 = torch.empty((n, raw.H, raw.W)); occ1 = torch.empty((n, raw.H, raw.W))
    for i in range(n):
        r = raw._resolve_idx(i)
        occ0[i], occ1[i] = raw.occupancy_pair(r)
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    z = torch.zeros(n, 1)
    p_start = torch.cat([actions[:, :2], z], 1)
    p_stop = torch.cat([actions[:, 2:], z], 1)
    angle = torch.atan2(actions[:, 3] - actions[:, 1], actions[:, 2] - actions[:, 0]) + math.pi / 2
    grp = torch.tensor([raw.get_run_index(i) for i in range(n)], dtype=torch.long)
    step = torch.tensor([raw.get_step_index(i) for i in range(n)], dtype=torch.long)
    ws_min, ws_max = raw.workspace_bounds
    return FlexCellData(tag=tag or raw.dataset_id, occ0=occ0, occ1=occ1, actions=actions,
                        p_start=p_start, p_stop=p_stop, angle=angle, run_idx=grp,
                        slate_idx=grp.clone(), step_idx=step,
                        push_length=(actions[:, 2:] - actions[:, :2]).norm(dim=1),
                        files=None, workspace_min=ws_min, workspace_max=ws_max,
                        H=raw.H, W=raw.W, raw=raw)


def truth_for_scoring_flex(cell, rows: torch.Tensor, device: str | None = None) -> torch.Tensor:
    """Soft, mass-conserving post-push truth for FleX rows (the FleX branch of
    ``eval_report.truth_for_scoring``): ``splat_particles_mass`` at
    ``uv = pos * to_pxl + ctr + score_uv_offset_px`` (offset 0.0 here: the
    disk raster's pixel centres sit at integer coordinates, so no Genesis
    cv2-truncation correction applies). Each PARTICLE carries mass 1 -- a
    DS-0019 slate has 1.6k-20k particles, and value functions are
    mass-normalised or scored via scale-free slateN, so this is harmless
    within a slate; do not compare raw masses across slates. Escaped / NaN
    particles are dropped (they would carry ~0 weight off-grid anyway).
    Runs on CUDA when available (~0.25 ms/row vs ~40-90 ms/row on CPU for
    10k+ particles; same function, float32), result returned on CPU."""
    from transforms.functional import splat_particles_mass
    raw = cell.raw
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ctr = raw.ctr_in_PXL[:2].to(torch.float32)
    out = torch.zeros(len(rows), raw.H, raw.W)
    for k, i in enumerate(rows.tolist()):
        pos = raw.particles_after(int(i))
        pos = pos[torch.isfinite(pos).all(1)]
        uv = pos * float(raw.to_pxl) + ctr + float(raw.score_uv_offset_px)
        out[k] = splat_particles_mass(uv[None].to(dev), (raw.H, raw.W))[0].cpu()
    return out


def is_flex_dataset_cfg(cfg: str | Path | dict) -> bool:
    """True iff a configs/dataset/*.yaml (path or dict) is a ``type: flex`` config."""
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(_abs(str(cfg)).read_text())
    return cfg.get("type") == "flex"


def load_flex_cell_from_cfg(cfg: str | Path | dict, split: str, tag: str | None = None) -> FlexCellData:
    """``load_flex_cell`` for a ``type: flex`` dataset config (the same file the
    Trainer reads), honouring its ``split_instances`` override -- e.g. split
    ``test`` of ``configs/dataset/flex_ds0020_train_ds0019_test.yaml`` is
    DS-0019 ``all``."""
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(_abs(str(cfg)).read_text())
    assert cfg.get("type") == "flex", cfg.get("type")
    inst, sub = cfg["instance"], split
    ov = (cfg.get("split_instances") or {}).get(split)
    if ov is not None:
        inst, sub = ov["instance"], ov.get("split", "all")
    return load_flex_cell(inst, sub, tag=tag, exclude_flagged=bool(cfg.get("exclude_flagged", True)),
                          occ_source=cfg.get("occ_source"))
