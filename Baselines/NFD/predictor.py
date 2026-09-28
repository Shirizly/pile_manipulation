"""Baselines/NFD/predictor.py -- BaselinePredictor for Baselines/common/eval_baseline.py.

Builds the SAME 3-channel (or, for the ablation, 2-channel) input tensor at
eval time that ``nfd_lib.py`` builds at train time, straight from
``PredictorBatch``'s world-metre ``p_start``/``p_stop``/``angle`` (never from
``batch.raw``'s own per-sample grids -- ``PredictorBatch`` carries the eval
split's OWN ``raw`` PileSweepData instance, built through the plain 2-channel
``genesis`` dataset type by ``Baselines.common.data.load_cell``, so its
``_input_grid`` is not 3-channel; we rasterise the action channels here
ourselves via the same ``draw_plate_soft`` primitive instead of touching that
instance's grids).

Checkpoint note: ``Trainer.from_config`` always builds train/val/test
datasets (unlike ``Baselines.common.data.load_cell``, which only ever
requests "train"), so a val_pct=0/test_pct=0 dataset config (the shared
pooled ``configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml``)
makes ``PileSweepData`` raise ``ValueError("No configs found for
dataset.")`` for the val/test splits -- hit directly on a 2-epoch smoke test.
Fixed in this baseline's own configs (``configs/nfd_train_3ch.yaml`` /
``nfd_train_2ch_ablation.yaml``) with a small ``val_pct: 5, test_pct: 5``
instead, so ``unet_best.pth`` (tracked against a real, if small, validation
slice) is meaningful and is what both factories below load by default.

Model output is a raw logit (repo-wide convention -- ``EulerianCombinedLoss``
always trains against ``sigmoid(logit)``, see ``training/losses.py``), so
``predict_occ`` applies ``torch.sigmoid`` before returning, exactly as
``Baselines/common/LOG.md`` documents other UNet-family predictors must.
"""
from __future__ import annotations

import os

import torch

from model.UNetModels_modular import UNet
from transforms.functional import draw_plate_soft

_STRUCTURE_3CH = dict(
    features=[4, 8, 16], in_channels=3, out_channels=1, kernel_size=3,
    final_kernel_size=1, activation="relu", residual=True,
    bottleneck_type="None", bottleneck_kwargs={},
)
_STRUCTURE_2CH = dict(_STRUCTURE_3CH, in_channels=2)

CKPT_3CH = "Baselines/NFD/runs/nfd_3ch/unet_best.pth"
CKPT_2CH = "Baselines/NFD/runs/nfd_2ch_ablation/unet_best.pth"


def _plate_geometry_px(raw) -> tuple[float, float, float]:
    """See Baselines/NFD/nfd_lib.py::plate_geometry_px -- duplicated here
    (rather than imported) to keep this predictor importable without pulling
    in Genesis.training.dataset / the dataset registry at all, since
    eval_baseline.py already constructed `raw` via Baselines.common.data;
    this predictor only ever needs the two-line geometry computation, not
    the registrations."""
    plate_dim_x, plate_dim_y, _ = raw.configs[0]["plate"]["size"]
    return plate_dim_x * raw.to_pxl, plate_dim_y * raw.to_pxl, max(0.5, 1.5 * raw.resolution_scale)


class NFDPredictor:
    """channels=3 -> faithful NFD (two separate full-intensity action
    renders). channels=2 -> ablation, replicating the repo's own
    0.5/1.0-intensity soft-union encoding exactly as
    Genesis/training/dataset.py::PileSweepData._draw_plate builds it."""

    def __init__(self, ckpt_path: str, channels: int = 3, name: str | None = None,
                 features: list | None = None):
        assert channels in (2, 3)
        self.channels = channels
        self.name = name or f"nfd_unet{channels}ch"
        structure = _STRUCTURE_3CH if channels == 3 else _STRUCTURE_2CH
        if features is not None:          # wider/deeper variants (e.g. [16, 32, 64], overnight 2026-09-25)
            structure = dict(structure, features=list(features))
        self.model = UNet(structure)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        device = batch.occ0.device
        self.model.to(device)
        H, W = batch.H, batch.W
        raw = batch.raw

        ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(device)[:2]
        start_px = batch.p_start[:, :2].to(device) * raw.to_pxl + ctr_xy
        stop_px = batch.p_stop[:, :2].to(device) * raw.to_pxl + ctr_xy
        angle = batch.angle.to(device)
        plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)

        occ0 = batch.occ0.to(device)
        if self.channels == 3:
            r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                       intensity=1.0, sigma=sigma)
            r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                      intensity=1.0, sigma=sigma)
            x = torch.stack([occ0, r_start, r_stop], dim=1)
        else:
            r1 = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=0.5, sigma=sigma)
            r2 = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
            action = 1 - (1 - r1) * (1 - r2)
            x = torch.stack([occ0, action], dim=1)

        logits = self.model(x)
        return torch.sigmoid(logits).squeeze(1).cpu()


def build_predictor() -> NFDPredictor:
    """Primary, faithful 3-channel NFD. --predictor Baselines.NFD.predictor:build_predictor

    Checkpoint path overridable via the NFD_CKPT env var (same pattern as
    Baselines/GNN/predictor.py's GNN_CKPT) -- e.g. for EXP-0030's
    overnight_randlen-trained checkpoint, same architecture/recipe, scored
    against the same eval slates for a generalisation comparison. `.name`
    stays "nfd_unet3ch" regardless (matches GNN's own convention: the
    predictor's identity is the architecture, not the training data; runs
    are told apart by --tag/--out-prefix, not by predictor name)."""
    import os
    ckpt_path = os.environ.get("NFD_CKPT", CKPT_3CH)
    return NFDPredictor(ckpt_path, channels=3, name="nfd_unet3ch")


def build_predictor_2ch_ablation() -> NFDPredictor:
    """2-channel ablation (repo's own union encoding). --predictor Baselines.NFD.predictor:build_predictor_2ch_ablation"""
    return NFDPredictor(CKPT_2CH, channels=2, name="nfd_unet2ch_ablation")


# Warped NFD (canonical-push-frame prediction) now lives in
# `model/warped_nfd/predictor.py` -- `build_canonical_stack`, `WarpedNFDPredictor`,
# `build_predictor_warped[_walls]`, `WARPED_DEFAULT_PLATE_MODE` moved there so
# that new-model-family code sits under `model/`, not `Baselines/`. This
# module keeps only the plain (unwarped) NFD predictor above. See
# `model/warped_nfd/predictor.py`'s module docstring for the "stays
# importable without Genesis.training.dataset" rationale that motivated
# keeping this split rather than merging the two.
