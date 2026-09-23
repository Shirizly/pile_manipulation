"""model/flow_nfd/predictor.py -- offline eval-time predictor for EXP-0025
(flow/advection-warp NFD pilot), for use with
`Baselines/common/eval_report.py`'s `predict_occ(batch) -> (B, H, W)`
contract.

Duplicates the flow/warp math from `model/flow_nfd/lib.py::
FlowWarpWrapper.forward` (world-frame [occ0, r_start, r_stop] input,
tanh-bounded backward displacement field via `grid_sample`, optional bounded
source/sink term) so eval and train agree, following the same
duplicate-rather-than-import-the-training-wrapper pattern
`residual_predictor.py` already uses (keeps this module free of the
Genesis-backed dataset registry `nfd_lib`'s training wrapper would pull in).
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from Baselines.NFD.predictor import _plate_geometry_px
from model.UNetModels_modular import UNet
from transforms.functional import draw_plate_soft


def _base_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).expand(B, -1, -1, -1)


class FlowWarpPredictor:
    """Mirrors `model.flow_nfd.lib.FlowWarpWrapper.forward` exactly,
    minus the prob->logit trick (returns a probability directly, matching
    `NFDPredictor.predict_occ`'s contract)."""

    def __init__(self, ckpt_path: str, flow_res: int | None, max_disp_px: float,
                 source_sink: bool, source_scale: float = 0.3, name: str | None = None):
        self.name = name or "nfd_flow_warp"
        self.flow_res = flow_res
        self.max_disp_px = float(max_disp_px)
        self.source_sink = bool(source_sink)
        self.source_scale = float(source_scale)
        out_channels = 3 if source_sink else 2
        structure = dict(
            features=[4, 8, 16], in_channels=3, out_channels=out_channels,
            kernel_size=3, final_kernel_size=1, activation="relu",
            residual=False, bottleneck_type="None", bottleneck_kwargs={},
        )
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
        r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        x = torch.stack([occ0, r_start, r_stop], dim=1)

        raw_out = self.model(x)
        B = occ0.shape[0]
        flow_raw = raw_out[:, 0:2]
        if self.flow_res is not None and self.flow_res < H:
            k = H // self.flow_res
            flow_raw = F.avg_pool2d(flow_raw, kernel_size=k)
            flow_raw = F.interpolate(flow_raw, size=(H, W), mode="bilinear", align_corners=True)

        disp_px = torch.tanh(flow_raw) * self.max_disp_px
        disp_x_norm = disp_px[:, 0] * (2.0 / max(W - 1, 1))
        disp_y_norm = disp_px[:, 1] * (2.0 / max(H - 1, 1))
        disp_norm = torch.stack([disp_x_norm, disp_y_norm], dim=-1)

        base_grid = _base_grid(B, H, W, occ0.device, occ0.dtype)
        sampling_grid = base_grid + disp_norm

        occ_pred = F.grid_sample(
            occ0.unsqueeze(1), sampling_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze(1)

        if self.source_sink:
            src = torch.tanh(raw_out[:, 2]) * self.source_scale
            occ_pred = occ_pred + src

        occ_pred = occ_pred.clamp(0.0, 1.0)
        return occ_pred.cpu()


def _build(ckpt_env: str, default_ckpt: str, flow_res, max_disp_px, source_sink,
           source_scale=0.3, name=None):
    ckpt_path = os.environ.get(ckpt_env, default_ckpt)
    return FlowWarpPredictor(ckpt_path, flow_res=flow_res, max_disp_px=max_disp_px,
                              source_sink=source_sink, source_scale=source_scale, name=name)


# One factory per sweep cell (EXP-0025 RUN ids); each honours an env-var
# override of its checkpoint path (FLOW_<CELL>_CKPT), same convention as
# NFD_CKPT / NFD_RESIDUAL_CKPT elsewhere in this package.

def build_predictor_baseline():
    return _build("FLOW_BASELINE_CKPT",
                   "Baselines/NFD/runs/flow_baseline_L20mm_pilot/unet_best.pth",
                   flow_res=None, max_disp_px=12.0, source_sink=False,
                   name="flow_baseline")


def build_predictor_coarse():
    return _build("FLOW_COARSE_CKPT",
                   "Baselines/NFD/runs/flow_coarse16_L20mm_pilot/unet_best.pth",
                   flow_res=16, max_disp_px=12.0, source_sink=False,
                   name="flow_coarse16")


def build_predictor_small_disp():
    return _build("FLOW_SMALLDISP_CKPT",
                   "Baselines/NFD/runs/flow_smalldisp4_L20mm_pilot/unet_best.pth",
                   flow_res=None, max_disp_px=4.0, source_sink=False,
                   name="flow_smalldisp4")


def build_predictor_large_disp():
    return _build("FLOW_LARGEDISP_CKPT",
                   "Baselines/NFD/runs/flow_largedisp24_L20mm_pilot/unet_best.pth",
                   flow_res=None, max_disp_px=24.0, source_sink=False,
                   name="flow_largedisp24")


def build_predictor_source_sink():
    return _build("FLOW_SRCSINK_CKPT",
                   "Baselines/NFD/runs/flow_srcsink_L20mm_pilot/unet_best.pth",
                   flow_res=None, max_disp_px=12.0, source_sink=True,
                   name="flow_srcsink")


# EXP-0025 extension (results/supervised_flow.md): the flow head is now
# supervised DIRECTLY from particle-correspondence targets
# (`model/flow_nfd/supervised.py`), on top of the same photometric
# loss, instead of photometric-only. Same architecture/max_disp_px=24
# parameterisation as `flow_largedisp24` above (which was catastrophic under
# photometric-only training) -- reusing this predictor class is exactly the
# point: only the checkpoint (and the loss the checkpoint was trained
# under) differ.

def build_predictor_supervised():
    return _build("FLOW_SUPERVISED_CKPT",
                   "experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
                   "RUN-0008-supervised-flow/unet_best.pth",
                   flow_res=None, max_disp_px=24.0, source_sink=False,
                   name="flow_supervised")


def build_predictor_supervised_masked():
    return _build("FLOW_SUPERVISED_MASKED_CKPT",
                   "experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
                   "RUN-0009-supervised-flow-masked-penalty/unet_best.pth",
                   flow_res=None, max_disp_px=24.0, source_sink=False,
                   name="flow_supervised_masked")


# Coordinator-requested re-run (results/supervised_flow.md's "confound" note):
# same two cells, this time with the x8 flip/rotation augmentation restored
# (RUN-0012/RUN-0013), matching EXP-0025's photometric cells' effective
# training-set size. Same architecture/checkpoint format, so this predictor
# class is reused unchanged.

def build_predictor_supervised_augmented():
    return _build("FLOW_SUPERVISED_AUG_CKPT",
                   "experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
                   "RUN-0012-supervised-flow-augmented/unet_best.pth",
                   flow_res=None, max_disp_px=24.0, source_sink=False,
                   name="flow_supervised_augmented")


def build_predictor_supervised_masked_augmented():
    return _build("FLOW_SUPERVISED_MASKED_AUG_CKPT",
                   "experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
                   "RUN-0013-supervised-flow-masked-penalty-augmented/unet_best.pth",
                   flow_res=None, max_disp_px=24.0, source_sink=False,
                   name="flow_supervised_masked_augmented")
