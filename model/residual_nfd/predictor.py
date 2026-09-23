"""model/residual_nfd/predictor.py -- offline eval-time predictors for
EXP-0022 R1 (unwarped residual) / R2 (warped residual), for use with
`Baselines/common/eval_report.py`'s `predict_occ(batch) -> (B, H, W)`
contract.

Does NOT edit `model/warped_nfd/predictor.py` -- imports `build_canonical_stack`
from it (and `_STRUCTURE_3CH`/`_plate_geometry_px` from the plain
`Baselines.NFD.predictor`), mirroring the existing `WarpedNFDPredictor`
pattern, and duplicates the tanh-residual math from `model/residual_nfd/
lib.py`'s training-time wrappers so eval and train agree (both call the SAME
`build_canonical_stack`; only the residual-add/clamp glue is duplicated here,
deliberately, since it is a few lines and pulling in the training wrapper
class would require importing the Genesis-backed dataset registry this
module is meant to stay free of -- same design note
`Baselines.NFD.predictor::NFDPredictor` already makes).
"""
from __future__ import annotations

import os

import torch

from Baselines.NFD.predictor import _STRUCTURE_3CH, _plate_geometry_px
from model.UNetModels_modular import UNet
from model.warped_nfd.predictor import build_canonical_stack
from transforms.functional import draw_plate_soft, from_push_frame, to_push_frame

CKPT_RESIDUAL_UNWARPED = (
    "Baselines/NFD/runs/nfd_residual_unwarped_L20mm_pilot_2/unet_best.pth"
)
CKPT_RESIDUAL_WARPED = (
    "Baselines/NFD/runs/nfd_residual_warped_L20mm_pilot_2/unet_best.pth"
)


class ResidualNFDPredictor:
    """R1: world-frame NFD UNet (`residual_mode=False`), explicit
    `clamp(occ0 + tanh(raw), 0, 1)` reconstruction. Matches
    `model/residual_nfd/lib.py::ResidualUnwarpedWrapper`'s forward
    exactly, minus the prob->logit trick (this returns a probability
    directly, matching `NFDPredictor.predict_occ`'s contract)."""

    def __init__(self, ckpt_path: str, name: str | None = None):
        self.name = name or "nfd_residual_unwarped"
        structure = dict(_STRUCTURE_3CH, residual=False)
        self.model = UNet(structure)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        device = batch.occ0.device
        self.model.to(device)
        H, W = batch.H, batch.W
        raw_ds = batch.raw

        ctr_xy = raw_ds.ctr_in_PXL.to(torch.float32).to(device)[:2]
        start_px = batch.p_start[:, :2].to(device) * raw_ds.to_pxl + ctr_xy
        stop_px = batch.p_stop[:, :2].to(device) * raw_ds.to_pxl + ctr_xy
        angle = batch.angle.to(device)
        plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw_ds)

        occ0 = batch.occ0.to(device)
        r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        x = torch.stack([occ0, r_start, r_stop], dim=1)

        raw = self.model(x)
        if raw.dim() == 4:
            raw = raw.squeeze(1)
        delta = torch.tanh(raw)
        occ_pred = torch.clamp(occ0 + delta, 0.0, 1.0)
        return occ_pred.cpu()


class ResidualWarpedNFDPredictor:
    """R2: warped residual -- warp occ0 in, predict residual in canonical
    frame, unwarp the RESIDUAL, add to pristine world occ0, clamp. No
    blend. Matches `model/residual_nfd/lib.py::ResidualWarpedWrapper`'s forward
    exactly, minus the prob->logit trick."""

    def __init__(self, ckpt_path: str, plate_mode: str = "canonical",
                 wall_channel: bool = False, canon_res: int | None = None,
                 scale: float = 1.0, name: str | None = None):
        self.plate_mode = plate_mode
        self.wall_channel = wall_channel
        self.canon_res = canon_res
        self.scale = scale
        in_channels = 4 if wall_channel else 3
        structure = dict(_STRUCTURE_3CH, in_channels=in_channels, residual=False)
        self.model = UNet(structure)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.name = name or "nfd_residual_warped"

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        device = batch.occ0.device
        self.model.to(device)
        H, W = batch.H, batch.W
        raw_ds = batch.raw

        ctr_xy = raw_ds.ctr_in_PXL.to(torch.float32).to(device)[:2]
        start_native = batch.p_start[:, :2].to(device) * raw_ds.to_pxl + ctr_xy
        stop_native = batch.p_stop[:, :2].to(device) * raw_ds.to_pxl + ctr_xy
        start_px = start_native[:, [1, 0]]
        end_px = stop_native[:, [1, 0]]
        angle = batch.angle.to(device)
        plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw_ds)

        occ0 = batch.occ0.to(device)
        r_start = draw_plate_soft(start_native, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_native, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        action_world = torch.stack([r_start, r_stop], dim=1)
        canon_res = self.canon_res or H

        canon_occ0 = to_push_frame(occ0, start_px, end_px, (canon_res, canon_res), self.scale)
        stack = build_canonical_stack(
            canon_occ0, start_px, end_px, action_world, occ0,
            self.plate_mode, self.wall_channel, canon_res, self.scale,
            plate_x_px, plate_y_px, sigma, H,
        )
        raw = self.model(stack)
        if raw.dim() == 4:
            raw = raw.squeeze(1)
        delta_canon = torch.tanh(raw)
        delta_world = from_push_frame(delta_canon, start_px, end_px, (H, H), self.scale)
        occ_pred = torch.clamp(occ0 + delta_world, 0.0, 1.0)
        return occ_pred.cpu()


def build_predictor_residual_unwarped() -> ResidualNFDPredictor:
    """--predictor model.residual_nfd.predictor:build_predictor_residual_unwarped"""
    ckpt_path = os.environ.get("NFD_RESIDUAL_CKPT", CKPT_RESIDUAL_UNWARPED)
    return ResidualNFDPredictor(ckpt_path)


def build_predictor_residual_warped() -> ResidualWarpedNFDPredictor:
    """--predictor model.residual_nfd.predictor:build_predictor_residual_warped"""
    ckpt_path = os.environ.get("NFD_RESIDUAL_WARPED_CKPT", CKPT_RESIDUAL_WARPED)
    return ResidualWarpedNFDPredictor(ckpt_path, plate_mode="canonical", wall_channel=False,
                                       canon_res=None, scale=1.0)
