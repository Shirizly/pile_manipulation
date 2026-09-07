"""Baselines/SchenckCNN/predictor.py -- BaselinePredictor for
Baselines/common/eval_baseline.py.

Builds the same 4-channel [occ0, r_start, r_stop, angle] input at eval time
that `train_schenck.py` builds at train time (via
`Baselines.SchenckCNN.action_encoding.build_input`, straight from
`PredictorBatch`'s world-metre `p_start`/`p_stop`/`angle` -- never from
`batch.raw`'s own internal grids, same reasoning as
`Baselines/NFD/predictor.py`'s docstring: `PredictorBatch.raw` is built
through the plain 2-channel `genesis` dataset type by
`Baselines.common.data.load_cell`, so its own `_input_grid` is not this
baseline's 4-channel layout -- we rasterise the action channels ourselves).

`SchenckCNN.forward` already returns a plain linear (non-logit) regression
value with the residual add baked in (see model.py) -- no sigmoid needed
here, unlike the UNet-family predictors in this repo.
"""
from __future__ import annotations

import torch

from Baselines.SchenckCNN.action_encoding import build_input
from Baselines.SchenckCNN.model import SchenckCNN

CKPT = "Baselines/SchenckCNN/runs/schenck.pth"


class SchenckPredictor:
    name = "schenck_singlenet"

    def __init__(self, ckpt_path: str = CKPT, in_channels: int = 4, width: int = 32,
                 n_layers: int = 16):
        self.model = SchenckCNN(in_channels=in_channels, width=width, n_layers=n_layers)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        device = batch.occ0.device
        self.model.to(device)
        x = build_input(batch.occ0, batch.p_start, batch.p_stop, batch.angle,
                         batch.raw, batch.H, batch.W)
        pred = self.model(x.to(device))
        return pred.cpu()


def build_predictor() -> SchenckPredictor:
    """--predictor Baselines.SchenckCNN.predictor:build_predictor"""
    return SchenckPredictor()
