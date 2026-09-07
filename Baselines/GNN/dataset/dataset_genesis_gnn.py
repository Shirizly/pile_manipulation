"""Baselines/GNN/dataset/dataset_genesis_gnn.py -- our own Dataset for the
dynamic-resolution GNN (`model/gnn_dyn.py::PropNetDiffDenModel`), built
directly from the raw `_*_data.pt` files. Does NOT touch or import
`Baselines/GNN/dataset/dataset_gnn_dyn.py` (the vendored PyFlex-format
dataset) -- see SPEC.md section 7 for why a from-scratch dataset is the
right call (different I/O shape entirely: no images, no FPS/KDTree, no
camera transforms, fixed N=20).

Each `_<i>_data.pt` file holds 128 single-step transitions that all share
ONE starting state ("slate", see SPEC.md section 4) at step 0, or 128
independently-diverged continuations at steps 1/2 (ORCHESTRATOR AMENDMENT).
Either way every row is an independent, valid (s_cur, action, s_next)
single-step training example -- this dataset flattens across files and
rows with no further bookkeeping (no leakage risk: train/eval directories
are already the slate-level split from
`scripts/probes/prepare_slate_multistep_split.py`, reused as-is).
"""
from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset


class GenesisGNNDataset(Dataset):
    """Flattened single-step (s_cur, p_start, p_stop, s_next) examples.

    `roots` is one or more directories each holding `_*_data.pt` files
    (e.g. `Genesis/data/slates_multistep/n20_L20mm_train`). Pooling
    multiple cells (L20mm + L40mm) is just passing multiple roots --
    matches `configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml`'s
    own pooling, but read directly off disk rather than through the
    registry (this is a particle-space model; it never needs the occupancy
    view for training, only `Baselines.common.data`'s occupancy view is
    used for eval/rasterisation via `predictor.py`).
    """

    def __init__(self, roots: list[str] | str):
        if isinstance(roots, str):
            roots = [roots]
        self.files: list[str] = []
        for root in roots:
            fs = sorted(glob.glob(os.path.join(root, "_*_data.pt")))
            assert fs, f"no _*_data.pt files found under {root!r}"
            self.files.extend(fs)

        # Cache (file_idx, row_idx) -> flat index, and lazily load file
        # contents (128 rows each -> a few hundred files is fine to keep
        # fully in memory: 180 files * 128 * 20 * 7 floats * 4 tensors
        # ~= tens of MB, well within budget).
        self._data = []
        self._index = []
        for fi, path in enumerate(self.files):
            d = torch.load(path, weights_only=False)
            self._data.append(d)
            n = d["states"].shape[0]
            self._index.extend((fi, r) for r in range(n))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fi, r = self._index[idx]
        d = self._data[fi]
        return {
            "s_cur": d["states"][r, :, :3],     # (20,3) world metres
            "s_next": d["states_"][r, :, :3],   # (20,3) world metres, ground truth
            "p_start": d["p_starts"][r],        # (3,)
            "p_stop": d["p_stops"][r],          # (3,)
        }


def collate(batch: list[dict]) -> dict:
    """Plain stack -- N=20 is fixed for every sample, no padding needed."""
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
