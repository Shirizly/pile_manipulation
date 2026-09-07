"""Baselines/GNN/predictor.py -- BaselinePredictor for
Baselines/common/eval_baseline.py. Wraps the trained
`PropNetDiffDenModel` (model/gnn_dyn.py) and rasterises its particle-space
prediction to occupancy via `Baselines.common.data.rasterize_particles`
(the harness's own routine -- NOT a new rasteriser, per that module's
docstring and the task brief).

`build_predictor()` takes no arguments (harness contract) and loads
`Baselines/GNN/runs/ckpt_best.pth` by default (override via the
GNN_CKPT env var if scoring a different checkpoint).
"""
from __future__ import annotations

import os

import torch

from Baselines.common.data import rasterize_particles
from Baselines.GNN.geometry import PARTICLE_DENS, compute_s_delta
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
        print(f"[GNNPredictor] loaded {ckpt_path} (epoch={self.ckpt_epoch}, "
              f"val_mse={self.ckpt_val_mse})")

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        states = batch.states.to(self.device)          # (B,20,7)
        s_cur = states[..., :3]
        p_start = batch.p_start.to(self.device)
        p_stop = batch.p_stop.to(self.device)
        B, N, _ = s_cur.shape

        s_delta = compute_s_delta(s_cur, p_start, p_stop)
        a_cur = torch.zeros(B, N, device=self.device)
        dens = torch.full((B,), PARTICLE_DENS, device=self.device)

        s_pred = self.model.predict_one_step(a_cur, s_cur, s_delta, dens)  # (B,20,3)

        # Model has no orientation head (SPEC.md hazard/§6) -- reattach the
        # INPUT frame's quaternion unchanged rather than inventing one.
        quat = states[..., 3:7]
        particles_pred = torch.cat([s_pred, quat], dim=-1).cpu()  # (B,20,7)

        occ = torch.empty_like(batch.occ0)
        run_idx = batch.run_idx.tolist()
        for i in range(B):
            occ[i] = rasterize_particles(batch.raw, run_idx[i], particles_pred[i])
        return occ


def build_predictor():
    ckpt_path = os.environ.get("GNN_CKPT", DEFAULT_CKPT)
    return GNNPredictor(ckpt_path=ckpt_path)
