"""Baselines/GNN/dataset/dataset_genesis_gnn.py -- our own Dataset for the
dynamic-resolution GNN (`model/gnn_dyn.py::PropNetDiffDenModel`), built from
the harness's own occupancy raster, NOT from privileged ground-truth cube
centroids. Does NOT touch or import `Baselines/GNN/dataset/dataset_gnn_dyn.py`
(the vendored PyFlex-format dataset) -- see SPEC.md section 7 for why a
from-scratch dataset is the right call.

CORRECTED 2026-09-10 (see SPEC.md's "CORRECTION" section and LOG.md): an
earlier version of this file read `states[...,:3]`/`states_[...,:3]`
directly out of the raw `_*_data.pt` files and used those exact simulator
cube centroids as graph node positions -- privileged 3D state a real
top-down-camera deployment would never have. This version instead:

  1. rasterises the pre-push state to a top-down occupancy grid (`occ0`,
     the SAME routine that produces every ground-truth occ0/occ1 in this
     repo -- reused via the registry, not reinvented);
  2. reads that grid's occupied pixels as a point cloud (it is already
     pure foreground -- no background/clutter rendered into it, see
     `perception.py`'s module docstring) and Farthest-Point-Samples it
     down to a FIXED `n_particles` node positions
     (`Baselines/GNN/perception.py::sample_nodes_xy`) -- these ARE the
     node input positions, pixel-quantized and raster-derived, nothing
     privileged. `n_particles` is now just a chosen hyperparameter,
     decoupled from any cell's TRUE particle count (see point 4 below);
  3. for the TRAINING TARGET ONLY, tracks each sampled point to its
     nearest real cube (KDTree on privileged simulator state) and applies
     THAT cube's true displacement to the raster-derived point
     (`perception.py::track_displacement`) -- the standard sim-training
     pattern (privileged label, realistic input), mirroring the paper's
     own `dataset_gnn_dyn.py` KDTree-tracking step exactly;
  4. reads each row's own per-file `states`/`states_` directly off the
     registry's raw dataset (`registry.dataset_registry.build_dataset`),
     NOT through `Baselines.common.data.load_cell`/`CellData` --
     `CellData` preallocates a fixed `(n, 20, 7)` tensor and so cannot
     represent a pool that mixes different true particle counts (e.g.
     `Genesis/data/overnight_randlen_train`'s n20 AND n50 groups). Since
     step 2 already resamples every cell down to the SAME fixed
     `n_particles` regardless of its true count, pooling heterogeneous-N
     cells is safe once occupancy + per-row states are read this way.

Precomputation note: steps 1-3 run ONCE per row at construction time (in
`__init__`, not lazily in `__getitem__`) and are cached as plain tensors.
`sample_nodes_xy` is seeded deterministically per row (`seed=<flat row
index>`), so recomputing it every epoch would be pure waste -- for the
pooled overnight_randlen corpus (tens of thousands of rows) that waste is
also what caused an early smoke-training run to be killed (the identical,
epoch-repeated cost of foreground-extraction + FPS + KDTree per sample, in
a single DataLoader worker).
"""
from __future__ import annotations

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from Baselines.common.data import _resolve_sample
from Baselines.GNN.perception import N_PARTICLES, Z_CONST, sample_nodes_xy, track_displacement
from registry.dataset_registry import build_dataset


def _load_rows(cfg_path: str, split: str = "train"):
    """Yield `(occ0, gt_cur_xy, gt_next_xy, p_start, p_stop)` for every row
    of `cfg_path`, reading occupancy the exact way
    `Baselines.common.data.load_cell` does (`raw[i][0][0][0]`, the SAME
    rasteriser behind every ground-truth occ0/occ1 in this repo) but
    resolving each row's `states`/`states_`/`p_starts`/`p_stops` straight
    off `raw.runs` (reusing `Baselines.common.data._resolve_sample`'s own
    flat-index -> (run, row) arithmetic) instead of through `CellData` --
    so a row's true particle count is whatever THAT file has, never
    hardcoded. Also yields `raw.to_pxl`/`raw.ctr_in_PXL` once (constant for
    the whole config: same physical workspace across every pooled cell,
    only true particle count differs).
    """
    cfg = yaml.safe_load(open(cfg_path).read())
    wrapper = build_dataset(cfg, split)
    raw = wrapper.raw_dataset
    n = len(wrapper)
    to_pxl = raw.to_pxl
    ctr_in_pxl = raw.ctr_in_PXL
    for i in range(n):
        occ0 = raw[i][0][0][0].numpy()
        r, s = _resolve_sample(raw, i)
        run = raw.runs[r]
        gt_cur_xy = run["states"][s, :, :2].numpy()
        gt_next_xy = run["states_"][s, :, :2].numpy()
        p_start = run["p_starts"][s]
        p_stop = run["p_stops"][s]
        yield occ0, gt_cur_xy, gt_next_xy, p_start, p_stop, to_pxl, ctr_in_pxl


class GenesisGNNDataset(Dataset):
    """Flattened single-step (s_cur, p_start, p_stop, s_next) examples,
    node positions constructed per this module's docstring above.

    `cfg_paths` is one or more `configs/dataset/genesis_*.yaml` paths (any
    `type: genesis` / `paths:` config the registry can build) -- e.g. the
    pooled `n20_L20L40_train.yaml`, a list of per-cell `_eval.yaml` configs,
    or `genesis_overnight_randlen_train_all.yaml` (mixed n20+n50 groups).
    `n_particles` is a free hyperparameter (the true particle count of the
    underlying cell(s) no longer has to match it).
    """

    def __init__(self, cfg_paths: list[str] | str, n_particles: int = N_PARTICLES):
        if isinstance(cfg_paths, str):
            cfg_paths = [cfg_paths]
        self.n_particles = n_particles

        s_cur_xy_all: list[np.ndarray] = []
        s_next_xy_all: list[np.ndarray] = []
        p_start_all: list[torch.Tensor] = []
        p_stop_all: list[torch.Tensor] = []

        row = 0
        for cfg_path in cfg_paths:
            for occ0, gt_cur_xy, gt_next_xy, p_start, p_stop, to_pxl, ctr_in_pxl in _load_rows(cfg_path):
                s_cur_xy = sample_nodes_xy(occ0, to_pxl, ctr_in_pxl,
                                            n_particles=n_particles, seed=row)
                delta_xy = track_displacement(s_cur_xy, gt_cur_xy, gt_next_xy)
                s_cur_xy_all.append(s_cur_xy)
                s_next_xy_all.append(s_cur_xy + delta_xy)
                p_start_all.append(p_start)
                p_stop_all.append(p_stop)
                row += 1

        z = torch.full((row, n_particles, 1), Z_CONST)
        s_cur_xy_t = torch.from_numpy(np.stack(s_cur_xy_all)).float()
        s_next_xy_t = torch.from_numpy(np.stack(s_next_xy_all)).float()
        self.s_cur = torch.cat([s_cur_xy_t, z], dim=-1)
        self.s_next = torch.cat([s_next_xy_t, z], dim=-1)
        self.p_start = torch.stack(p_start_all).float()
        self.p_stop = torch.stack(p_stop_all).float()

    def __len__(self) -> int:
        return self.s_cur.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "s_cur": self.s_cur[idx],
            "s_next": self.s_next[idx],
            "p_start": self.p_start[idx],
            "p_stop": self.p_stop[idx],
        }


def collate(batch: list[dict]) -> dict:
    """Plain stack -- `n_particles` is fixed for every sample, no padding needed."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
