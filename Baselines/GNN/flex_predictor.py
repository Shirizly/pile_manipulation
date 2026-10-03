"""Baselines/GNN/flex_predictor.py -- the ORIGINAL dyn-res-pile-manip GNN
(`PropNetDiffDenModel`, Wang et al. RSS 2023) run on the FleX carrot-pile
corpora (DS-0019 / DS-0020, EXP-0061) through the offline
`predict_occ(batch) -> (B, H, W)` contract `Baselines/common/eval_report.py`
scores.

Why a separate predictor: `Baselines/GNN/predictor.py` + `perception.py` +
`geometry.py` hard-code Genesis geometry (5 mm cubes, metres, the Genesis
action frame) and build nodes from the occupancy raster. This file instead
reproduces the SOURCE repo's camera pipeline in FleX units, so the released
checkpoint `Baselines/GNN/data/gnn_dyn_model/2023-01-28-10-42-05-114323/
net_epoch_0_iter_1000.pth` sees inputs distributed as in its own training.

Source pipeline reproduced (paths in /home/alon/Code/dyn-res-pile-manip):
  * model: `model/gnn_dyn.py::PropNetDiffDenModel` -- byte-identical copy
    already in this repo at `model/gnn_dyn.py` (diffed), imported from there.
    Config used by the checkpoint: `config/train/gnn_dyn.yaml` (nf_effect 64,
    adj_thresh 0.08, add_delta False) -- the only keys the model reads.
  * perception, test-time (`env/flex_env.py::obs2ptcl_fixed_num_batch`):
    depth (FleX units) / global_scale -> foreground = depth < 0.599/0.8
    (`utils.py::depth2fgpcd`) -> camera-frame point cloud (OpenCV frame,
    / global_scale) -> voxel downsample 0.01 (`utils.py::downsample_pcd`,
    open3d voxel mean) -> farthest-point sampling of `particle_num` points
    (`utils.py::fps`, dgl FPS from a random start) -> `recenter` with
    r = min(0.02, 0.5 * particle_r) -> particle_den = 1 / particle_r**2,
    particle_r = FPS covering radius.
  * action encoding, TRAINING convention (`dataset/dataset_gnn_dyn.py`
    l.150-215): s_3d = [a0, 0, -a1] FleX world, `opengl2cam` -> camera frame;
    per-node s_delta = (distance to the push end along the push) * push_dir *
    hard length gate (0 < proj < L) * soft width gate exp(-excess / 0.01)
    with pusher half-width 0.8 / global_scale. (The source's MPC planner
    `planners.py::gen_s_delta` uses 0.048 instead; the weights were trained
    with 0.8/24, so that is what is used here.)
  * attrs a_cur = 0 for every node; one predict_one_step per push (the
    training target is the post-push, post-settle state k+1).

Camera (verified, `FlexData/cam_params.json`): pinhole at (0, 18, 0) looking
straight down, f = 869.12, cx = cy = 359.5:  u = cx + f x / (18 - y),
v = cy + f z / (18 - y). OpenCV camera frame = (x, z, 18 - y); / 24.
(The source computes cx = cy = 360 from the window size; the 0.5 px
difference is 0.01 FleX units at the table and is ignored so that this file
and the image masks share one camera.) Table frame: X = x = 24 * cam_x,
Y = -z = -24 * cam_y.

Depth. DS-0019 ships NO depth PNGs. The foreground mask needs none (the
collector whitens every table-depth pixel, so "any RGB != 255" equals the
depth threshold pixel-for-pixel, `FlexData/image_mask.py`); depth only gives
each foreground pixel its height. `depth_mode`:
  * "true"  -- the DS-0020 depth PNG (depth * 1000, FleX units).
  * "plane" -- every foreground pixel at the constant height `plane_y`
    (FleX units above the table). See EXP-0061 RUNS_gnn_lf.md for the
    measured cost on DS-0020 and the chosen constant.

Rendering back to the 64x64 binary mask (the source never does this -- its
planner scores particles directly against the goal image,
`env/flex_rewards.py::config_reward_ptcl`): every foreground PIXEL of the
input image is carried by the predicted displacement of its nearest node
(`carry_k` = 1; k > 1 = inverse-distance blend over the k nearest nodes),
re-projected onto the table plane exactly as the image masks are built
(`FlexData/image_mask.py`: plane position = world * 18 / (18 - h)) and
binned; a cell is occupied iff any carried pixel lands in it. With zero
predicted motion this reproduces the input mask exactly (asserted in the
EXP-0061 check), so the rendering adds no error of its own to persistence.

Batch contract: needs `batch.raw` (a `FlexData.dataset.FlexPileData`),
`batch.run_idx` (trajectory id / slate id), `batch.step_idx` (push index /
0) and `batch.actions` (table frame). The pre-push IMAGE for each row is
found through `<cache_dir>/image_paths.json`; `batch.occ0` is NOT read.
Node sampling is seeded by the STATE (not the row's batch position), so all
candidates of one slate share one node set and chunking cannot change a
prediction (cf. the `gnn-node-sampling-consistent-within-state` trap of
`Baselines/GNN/predictor.py`).
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from model.gnn_dyn import PropNetDiffDenModel

REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT = "Baselines/GNN/data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth"

GLOBAL_SCALE = 24.0
CAM_H = 18.0                       # 6 * global_scale / 8
F = 360.0 / np.tan(np.radians(22.5))
CX = CY = 359.5
FG_DEPTH_SCALED = 0.599 / 0.8      # source foreground threshold, depth / global_scale
PUSHER_W = 0.8 / GLOBAL_SCALE      # training-time half-width (dataset_gnn_dyn.py)
VOXEL = 0.01                       # obs2ptcl_fixed_num_batch downsample (scaled units)
DEN_TRAIN_MAX = 6500.0             # top of the source training particle_den range (15-6500)
# config/train/gnn_dyn.yaml -- the keys PropNetDiffDenModel reads
MODEL_CFG = {"train": {"particle": {"nf_effect": 64, "add_delta": False, "adj_thresh": 0.08}}}


# ------------------------------------------------------------------ perception
def _voxel_down(p: np.ndarray, v: float) -> np.ndarray:
    """open3d voxel_down_sample equivalent: mean of the points in each voxel."""
    key = np.floor((p - p.min(0)) / v).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, p)
    return out / np.bincount(inv, minlength=n)[:, None]


def _fps(p: np.ndarray, n: int, start: int) -> tuple[np.ndarray, float]:
    """utils.py::fps (dgl farthest_point_sampler from `start`) + covering radius."""
    idx = [start]
    d = np.linalg.norm(p - p[start], axis=1)
    for _ in range(n - 1):
        j = int(d.argmax())
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(p - p[j], axis=1))
    return p[idx], float(d.max())


def _recenter(p: np.ndarray, s: np.ndarray, r: float) -> np.ndarray:
    """utils.py::recenter."""
    out = np.zeros_like(s)
    for i in range(len(s)):
        m = np.linalg.norm(p - s[i], axis=1) < r
        out[i] = p[m].mean(0) if m.any() else s[i]
    return out


def image_pointcloud(color: np.ndarray, depth: np.ndarray | None, plane_y: float):
    """(720,720,3) colour (+ optional (720,720) depth in FleX units) ->
    (fg pixel rows, cols, camera-frame scaled points (n,3), per-pixel height (n,))."""
    fg = (color != 255).any(2)
    if depth is not None:
        fg &= depth > 0
    v, u = np.nonzero(fg)
    d = (depth[v, u].astype(np.float64) if depth is not None
         else np.full(len(v), CAM_H - plane_y))
    pts = np.stack([(u - CX) * d / F, (v - CY) * d / F, d], 1) / GLOBAL_SCALE
    return v, u, pts, CAM_H - d


def perceive(color, depth, plane_y: float, particle_num: int, seed: int):
    """Source test-time perception -> (nodes (N,3) scaled cam frame, particle_den,
    dense fg pixel cam points (n,3), dense heights (n,), dense rows/cols)."""
    v, u, pts, h = image_pointcloud(color, depth, plane_y)
    ds = _voxel_down(pts, VOXEL)
    if len(ds) <= particle_num:
        # SMALL-PILE FALLBACK (EXP-0061, 2026-10-01): the source pipeline has no
        # path for a state with <= particle_num voxels (FPS takes every voxel, its
        # covering radius over the voxel set is 0 -> particle_den = inf). 43/100
        # DS-0019 states (compact rand_blob piles, 73-200 voxels) hit this at
        # N = 200. Then: every voxel is a node (no sampling -> seed-independent),
        # the covering radius is measured against the DENSE fg pixel cloud, and
        # particle_den is clamped to the training max (DEN_TRAIN_MAX). States with
        # more voxels than particle_num are unchanged (bit-identical).
        nodes = ds.copy()
        from scipy.spatial import cKDTree
        r = max(float(cKDTree(nodes).query(pts)[0].max()), 1e-6)
        return dict(nodes=nodes.astype(np.float32), den=min(1.0 / (r * r), DEN_TRAIN_MAX),
                    pts=pts.astype(np.float32), h=h.astype(np.float32), v=v, u=u, fallback=True)
    rng = np.random.default_rng(seed)
    nodes, r = _fps(ds, min(particle_num, len(ds)), int(rng.integers(len(ds))))
    nodes = _recenter(ds, nodes, min(0.02, 0.5 * r))
    return dict(nodes=nodes.astype(np.float32), den=1.0 / (r * r), pts=pts.astype(np.float32),
                h=h.astype(np.float32), v=v, u=u, fallback=False)


def s_delta_train(s_cur: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """dataset_gnn_dyn.py l.150-215, batched: (B,N,3) cam-frame nodes x (B,4)
    table-frame actions -> (B,N,3)."""
    s = actions[:, :2]
    e = actions[:, 2:]
    z = torch.full_like(s[:, :1], CAM_H / GLOBAL_SCALE)
    # world [a0, 0, -a1] -> opencv cam (x, z, 18 - y) / 24 = (a0, -a1, 18) / 24
    s3 = torch.cat([s[:, :1] / GLOBAL_SCALE, -s[:, 1:] / GLOBAL_SCALE, z], 1)
    e3 = torch.cat([e[:, :1] / GLOBAL_SCALE, -e[:, 1:] / GLOBAL_SCALE, z], 1)
    d = e3 - s3
    L = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
    d = d / L
    ortho = torch.cat([-d[:, 1:2], d[:, 0:1], torch.zeros_like(d[:, :1])], 1)
    diff = s_cur - s3[:, None]
    proj = (diff * d[:, None]).sum(-1)
    proj_o = (diff * ortho[:, None]).sum(-1)
    lmask = ((proj < L) & (proj > 0)).float()
    wmask = torch.exp(-torch.maximum((-PUSHER_W - proj_o).clamp_min(0), (proj_o - PUSHER_W).clamp_min(0)) / 0.01)
    to_end = ((e3[:, None] - s_cur) * d[:, None]).sum(-1)
    return to_end[..., None] * d[:, None] * lmask[..., None] * wmask[..., None]


def predict_one_step_vec(model: PropNetDiffDenModel, a_cur, s_cur, s_delta, particle_dens):
    """`PropNetDiffDenModel.predict_one_step` (model/gnn_dyn.py, byte-identical to the source) with its
    per-SAMPLE Python loop (`rels_idx = [torch.arange(n_rels[i]) for i in range(B)]`, one host sync per
    row) replaced by an equivalent vectorised index (EXP-0062 RUN-0003). Same adjacency (radius
    adj_thresh AND 10 nearest, on s_cur + s_delta), same Rr/Rs, same forward -> bit-identical output
    (asserted in Baselines/GNN/flex_train.py `check-vec`). No particle_nums path (constant N)."""
    B, N = a_cur.size()
    x = s_cur + s_delta
    dis = torch.sum((x[:, None, :, :] - x[:, :, None, :]) ** 2, -1)       # == (s_sender - s_receiv)**2
    thr = model.adj_thresh * model.adj_thresh
    topk_idx = torch.topk(dis, k=min(10, N), dim=2, largest=False).indices
    topk = torch.zeros_like(dis).scatter_(2, topk_idx, 1)
    adj = ((dis - thr) < 0).float() * topk
    rels = adj.nonzero()                                                   # sorted by (b, r, s)
    n_rels = torch.bincount(rels[:, 0], minlength=B)
    n_rel = int(n_rels.max().item()) if rels.shape[0] else 0
    starts = torch.cumsum(n_rels, 0) - n_rels
    rels_idx = torch.arange(rels.shape[0], device=s_cur.device) - starts[rels[:, 0]]
    Rr = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
    Rs = torch.zeros((B, n_rel, N), device=s_cur.device, dtype=s_cur.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return model.model.forward(a_cur, s_cur, s_delta, Rr, Rs, particle_dens)


# ------------------------------------------------------------------ predictor
class FlexGNNPredictor:
    name = "gnn_flex_drp"

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, particle_num: int = 50,
                 depth_mode: str = "plane", plane_y: float = 0.0, carry_k: int = 1,
                 device: str | None = None, fast_graph: bool = False, vector_render: bool = False,
                 cache_states: bool = True):
        """fast_graph / vector_render (EXP-0062): vectorised graph construction / mask rendering,
        bit-identical predictions; default False = the EXP-0061 code path. cache_states=False
        re-perceives the PNG on every call (for timing perception)."""
        assert depth_mode in ("true", "plane")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PropNetDiffDenModel(MODEL_CFG, use_gpu=False).to(self.device)
        sd = torch.load(str(REPO / ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path,
                        map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd)          # strict: the checkpoint is exactly this module
        self.model.eval()
        self.particle_num, self.depth_mode, self.plane_y, self.carry_k = particle_num, depth_mode, plane_y, carry_k
        self.n_particles = particle_num
        self.fast_graph, self.vector_render, self.cache_states = fast_graph, vector_render, cache_states
        self._cache: dict = {}
        self._paths: dict = {}
        self.name = f"gnn_flex_drp_n{particle_num}_{depth_mode}" + (f"{plane_y:g}" if depth_mode == "plane" else "")
        print(f"[FlexGNNPredictor] {ckpt_path}: particle_num={particle_num} depth={depth_mode} "
              f"plane_y={plane_y} carry_k={carry_k} device={self.device}", flush=True)

    # ---- image lookup
    def _image_paths(self, raw):
        key = raw.dataset_id
        if key not in self._paths:
            cache_dir = Path(raw.cfg["cache_dir"])
            cache_dir = cache_dir if cache_dir.is_absolute() else REPO / cache_dir
            self._paths[key] = (cache_dir.parent, json.loads((cache_dir / "image_paths.json").read_text()))
        return self._paths[key]

    def _state(self, raw, group: int, step: int):
        key = (raw.dataset_id, group, step)
        if self.cache_states and key in self._cache:
            return self._cache[key]
        root, idx = self._image_paths(raw)
        if raw.src.kind == "trajectories":
            cpath, dpath = idx["trajectories"][str(group)][step]
        else:
            cpath, dpath = idx["states"][str(group)]["initial_color"], None
        color = cv2.imread(str(root / cpath), cv2.IMREAD_COLOR)
        depth = None
        if self.depth_mode == "true":
            if dpath is None:
                raise ValueError(f"{raw.dataset_id} has no depth PNGs; use depth_mode='plane'")
            depth = cv2.imread(str(root / dpath), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        seed = zlib.crc32(f"{raw.dataset_id}/{group}/{step}".encode())
        st = perceive(color, depth, self.plane_y, self.particle_num, seed)
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[key] = st
        return st

    # ---- forward
    @torch.no_grad()
    def predict_nodes(self, states: list[dict], actions: torch.Tensor):
        """list of perceived states (one per row) + (B,4) actions -> (s_cur, s_pred) (B,N,3)."""
        dev = self.device
        s_cur = torch.from_numpy(np.stack([s["nodes"] for s in states])).to(dev)
        dens = torch.tensor([s["den"] for s in states], dtype=torch.float32, device=dev)
        a = actions.to(dev, torch.float32)
        sd = s_delta_train(s_cur, a)
        a_cur = torch.zeros(s_cur.shape[:2], device=dev)
        if self.fast_graph:
            return s_cur, predict_one_step_vec(self.model, a_cur, s_cur, sd, dens)
        return s_cur, self.model.predict_one_step(a_cur, s_cur, sd, dens)

    def render(self, st: dict, disp_world: torch.Tensor, nodes: torch.Tensor, H: int, W: int,
               to_pxl: float, ctr: float) -> torch.Tensor:
        """Carry every foreground pixel by its nearest node's (B rows of) world-XY displacement
        and bin onto the grid as the image masks are built. disp_world (B,N,2) table frame."""
        dev = self.device
        pts = torch.from_numpy(st["pts"]).to(dev)
        h = torch.from_numpy(st["h"]).to(dev)
        # table-plane position of each pixel: exactly the image-mask back-projection
        u = torch.from_numpy(st["u"]).to(dev, torch.float32)
        v = torch.from_numpy(st["v"]).to(dev, torch.float32)
        px = (u - CX) * CAM_H / F                       # x on the table plane
        pz = (v - CY) * CAM_H / F
        base = torch.stack([px, -pz], 1)                # table (X, Y)
        dist = torch.cdist(pts[:, :2], nodes[:, :2])    # (n, N) cam-xy
        k = min(self.carry_k, nodes.shape[0])
        dk, ik = dist.topk(k, dim=1, largest=False)
        wk = 1.0 / dk.clamp_min(1e-6)
        wk = wk / wk.sum(1, keepdim=True)               # (n,k)
        scale = CAM_H / (CAM_H - h)                     # world -> table-plane displacement
        out = torch.zeros(disp_world.shape[0], H * W, device=dev)
        if self.vector_render:                          # same cells as the loop below, one scatter
            Bn = disp_world.shape[0]
            dsp = (disp_world[:, ik] * wk[None, ..., None]).sum(2) * scale[None, :, None]   # (B,n,2)
            q = base[None] + dsp
            gi = torch.floor(q[..., 0] * to_pxl + ctr + 0.5).long()
            gj = torch.floor(q[..., 1] * to_pxl + ctr + 0.5).long()
            ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
            bi = torch.arange(Bn, device=dev)[:, None].expand_as(gi)
            out[bi[ok], gi[ok] * W + gj[ok]] = 1.0
            return out.reshape(-1, H, W)
        for b in range(disp_world.shape[0]):
            dsp = (disp_world[b][ik] * wk[..., None]).sum(1) * scale[:, None]
            q = base + dsp
            gi = torch.floor(q[:, 0] * to_pxl + ctr + 0.5).long()
            gj = torch.floor(q[:, 1] * to_pxl + ctr + 0.5).long()
            ok = (gi >= 0) & (gi < H) & (gj >= 0) & (gj < W)
            out[b, gi[ok] * W + gj[ok]] = 1.0
        return out.reshape(-1, H, W)

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        raw = batch.raw
        H, W = int(batch.H), int(batch.W)
        to_pxl, ctr = float(raw.to_pxl), float(raw.ctr_in_PXL[0])
        groups = batch.run_idx.tolist()
        steps = batch.step_idx.tolist()
        out = torch.zeros(len(groups), H, W)
        keys = list(zip(groups, steps))
        order: dict = {}
        for i, kk in enumerate(keys):
            order.setdefault(kk, []).append(i)
        for (g, s), rows in order.items():          # one perception + one batched forward per state
            st = self._state(raw, int(g), int(s))
            for c in range(0, len(rows), 256):
                rr = rows[c:c + 256]
                s_cur, s_pred = self.predict_nodes([st] * len(rr), batch.actions[rr])
                d = (s_pred - s_cur)[..., :2] * GLOBAL_SCALE
                disp_world = torch.stack([d[..., 0], -d[..., 1]], -1)   # cam (x, y) -> table (X, Y)
                out[rr] = self.render(st, disp_world, s_cur[0], H, W, to_pxl, ctr).cpu()
        return out


def build_predictor(**kw) -> FlexGNNPredictor:
    """eval_report factory; checkpoint from GNN_FLEX_CKPT (set by the harness from
    spec['ckpt']); kwargs (particle_num, depth_mode, plane_y, carry_k) from spec['kwargs']."""
    return FlexGNNPredictor(ckpt_path=os.environ.get("GNN_FLEX_CKPT", DEFAULT_CKPT), **kw)
