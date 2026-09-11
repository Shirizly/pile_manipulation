"""Baselines/GNN/predictor.py -- BaselinePredictor for
Baselines/common/eval_baseline.py. Wraps the trained
`PropNetDiffDenModel` (model/gnn_dyn.py) and rasterises its particle-space
prediction to occupancy.

CORRECTED 2026-09-10 (SPEC.md's "CORRECTION" section, LOG.md): `predict_occ`
used to read `batch.states` -- privileged ground-truth 3D cube pose -- and
feed those exact centroids to the graph as node positions. That is not
available in a real camera-only deployment. Node positions are now built
from `batch.occ0` (the top-down occupancy raster) alone, via
`Baselines/GNN/perception.py`'s foreground-extraction + FPS pipeline --
the same pipeline `Baselines/GNN/dataset/dataset_genesis_gnn.py` uses for
training (minus the KDTree-to-ground-truth tracking step, which only
exists to build a training label and has no privileged state to run
against at inference time).

Rasterisation, second correction (also 2026-09-10): predicted nodes are now
rasterised as fixed-size, axis-aligned cubes (`perception.py::
rasterize_nodes_as_cubes_batch`), not the harness's own per-cube,
per-orientation box renderer. That renderer (formerly
`rasterize_particles_batch`/`_draw_particle_grid_fast` in this file, with
an extensive cv2-batching/threading history -- see
`Baselines/GNN/LOG.md`'s "E1-gnn-improve" summary for that work) assumed
one node per REAL cube, reading a per-cell config's own `n_particles`/
`particle_sizes` list by index. That assumption is gone now that node
count is a free hyperparameter decoupled from any cell's true particle
count (SPEC.md's "CORRECTION" section) -- scoring, say, a 30-node model on
an n50 cell would have silently truncated to 30 boxes or indexed past the
end of the prediction tensor, depending on which count was larger, with
the old code. `rasterize_nodes_as_cubes_batch` draws exactly as many boxes
as the model predicted, all the same real cube size, all axis-aligned (no
orientation head, none observable from a top-down raster either) -- see
`perception.py` for the full rationale. `Baselines/GNN/scripts/
verify_rasterizer.py` (the equivalence proof for the retired per-cube
renderer) was retired along with it: the property it proved (byte-identical
to the harness's exact per-cube-config renderer) no longer applies by
design, and the cv2 multi-contour fill-rule trap it also guarded against
cannot occur with `rasterize_nodes_as_cubes`'s plain numpy slice
assignment (correctly ORs overlapping boxes with no winding-rule surprise).

`build_predictor()` takes no arguments (harness contract) and loads
`Baselines/GNN/runs/ckpt_best.pth` by default (override via the
GNN_CKPT env var if scoring a different checkpoint).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from Baselines.GNN.geometry import PARTICLE_DENS, compute_s_delta
from Baselines.GNN.perception import (
    N_PARTICLES, Z_CONST, rasterize_nodes_as_cubes_batch, sample_nodes_xy,
)
from model.gnn_dyn import PropNetDiffDenModel

DEFAULT_CKPT = "Baselines/GNN/runs/ckpt_best.pth"


class GNNPredictor:
    name = "gnn"

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = PropNetDiffDenModel(ckpt["cfg"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.ckpt_epoch = ckpt.get("epoch")
        self.ckpt_val_mse = ckpt.get("val_mse")
        # Node count is a runtime FPS choice, not something baked into the
        # model's weights (PropNetDiffDenModel has no N-dependent
        # parameters) -- read it from the checkpoint so a model trained
        # with e.g. --n-particles 30 is scored with 30 nodes, not whatever
        # this module's default happens to be.
        self.n_particles = ckpt.get("n_particles", N_PARTICLES)
        print(f"[GNNPredictor] loaded {ckpt_path} (epoch={self.ckpt_epoch}, "
              f"val_mse={self.ckpt_val_mse}, n_particles={self.n_particles})")

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        # CORRECTED 2026-09-10 (SPEC.md's "CORRECTION" section): node
        # positions come ONLY from `batch.occ0`, the top-down occupancy
        # raster -- `batch.states` (privileged ground-truth cube pose) is
        # never read here, matching what a real camera-only deployment
        # would have. `occ0` is already pure foreground (no background to
        # segment out, `perception.py`'s module docstring) -- reading its
        # occupied pixels + FPS mirrors exactly what
        # `Baselines/GNN/dataset/dataset_genesis_gnn.py` does for training,
        # minus the KDTree-to-ground-truth tracking step (that step exists
        # only to build a training LABEL; there is no privileged state to
        # track against at inference time).
        occ0 = batch.occ0  # (B,H,W), cpu
        raw = batch.raw
        B, H, W = occ0.shape
        n_particles = self.n_particles

        s_cur_xy = np.stack([
            sample_nodes_xy(occ0[i].numpy(), raw.to_pxl, raw.ctr_in_PXL,
                             n_particles=n_particles, seed=i)
            for i in range(B)
        ])  # (B,n_particles,2)
        z = np.full((B, n_particles, 1), Z_CONST, dtype=np.float32)
        s_cur = torch.from_numpy(
            np.concatenate([s_cur_xy, z], axis=-1).astype(np.float32)
        ).to(self.device)  # (B,n_particles,3)

        p_start = batch.p_start.to(self.device)
        p_stop = batch.p_stop.to(self.device)

        s_delta = compute_s_delta(s_cur, p_start, p_stop)
        a_cur = torch.zeros(B, n_particles, device=self.device)
        dens = torch.full((B,), PARTICLE_DENS, device=self.device)

        s_pred = self.model.predict_one_step(a_cur, s_cur, s_delta, dens)  # (B,n_particles,3)
        s_pred_xy = s_pred[..., :2].cpu().numpy()

        occ_pred = rasterize_nodes_as_cubes_batch(s_pred_xy, raw.to_pxl, raw.ctr_in_PXL, H, W)
        return torch.from_numpy(occ_pred)


def build_predictor():
    ckpt_path = os.environ.get("GNN_CKPT", DEFAULT_CKPT)
    return GNNPredictor(ckpt_path=ckpt_path)
