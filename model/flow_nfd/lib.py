"""model/flow_nfd/lib.py -- EXP-0025 pilot: flow/advection field
prediction for the world-frame NFD
family, registered alongside `Baselines.NFD.nfd_lib`'s `nfd-unet3ch` /
`nfd-genesis-3ch`.

Motivation (see EXPERIMENT.md): every image-space model in this repo predicts
absolute next occupancy directly. This module instead predicts a per-pixel
BACKWARD displacement field (dx, dy) and warps the current occupancy through
it via `grid_sample` -- mass-conserving by construction (material can only
move, never appear/vanish, modulo an optional bounded source/sink term) and
zero-flow-is-identity by construction (residual by construction, unlike a
network that must re-derive "mostly copy the input" from scratch).

Interpretation of the field (backward warping, per `grid_sample`'s contract):
for each OUTPUT pixel, the field says "where did the material that ends up
here come from" -- NOT "where does the material at this input pixel go".
This is the well-behaved, differentiable direction; forward-splatting a
displacement (the physically more natural "push this pixel's mass to X")
would be non-injective (collisions, holes) and is not implemented here.

Reuses `Baselines.NFD.nfd_lib.PileSweepData3Ch` / `nfd-genesis-3ch` dataset
UNCHANGED (this module registers a MODEL type only, no new dataset type) --
same 3-channel [occ0, r_start, r_stop] world-frame input as the direct NFD
control, so the sweep varies only the model head.

Registered model type: ``nfd-flow-warp``. Config keys (all under `model:`):
    in_channels     (default 3)  -- must match dataset (3ch NFD)
    features        (default [4, 8, 16], same UNet backbone as nfd-unet3ch)
    kernel_size, final_kernel_size, activation, bottleneck_type/kwargs
    flow_res        (default None -> full resolution H). If < H, the raw
                     backbone flow output is average-pooled to (flow_res,
                     flow_res) THEN bilinearly upsampled back to (H,H)
                     before the tanh bound is applied -- a coarse,
                     low-dimensional field forced smooth by construction,
                     without changing the backbone architecture (only the
                     flow head's effective resolution is ablated; the
                     backbone still sees/produces full-res feature maps).
    max_disp_px     (default 12.0) -- tanh-scaled max displacement magnitude
                     in PIXELS at the (unwarped) world-frame resolution H.
    source_sink     (default False) -- if True, the backbone emits one extra
                     channel, bounded by tanh*source_scale, ADDED to the
                     warped occupancy after advection (a source/sink term:
                     material entering/leaving the 2D view, e.g. stacking).
                     If False, prediction is PURE advection: occ_pred is a
                     grid_sample warp of occ0 and nothing else, exactly
                     mass-conserving in the 2D view.
    source_scale    (default 0.3) -- bound on the source/sink term.

No blend/validity-mask step (unlike the warped-push-frame NFD family): the
whole operation stays in the WORLD frame (same frame as occ0, r_start,
r_stop already are), so there is no push-frame corner-loss to patch --
`grid_sample`'s zero-padding at the world-frame borders already means
"nothing sampled from off-canvas", which is the physically correct
fallback for a world-frame occupancy grid whose borders are workspace
walls/edges.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from Baselines.NFD.nfd_lib import PileSweepData3Ch  # noqa: F401 (re-export convenience)
from registry.model_registry import ModelTrainingWrapper, register_model

_EPS = 1e-6


def _prob_to_logit(prob: torch.Tensor) -> torch.Tensor:
    p = prob.clamp(_EPS, 1.0 - _EPS)
    return torch.log(p / (1.0 - p))


def _base_grid(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    """Standard normalised (-1..1) sampling grid, (B,H,W,2) in grid_sample's
    (x,y) = (col,row) order, align_corners=True convention."""
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)
    return grid.unsqueeze(0).expand(B, -1, -1, -1)


class FlowWarpWrapper(ModelTrainingWrapper):
    """World-frame backward-advection NFD variant. Backbone predicts a raw
    (B, C_out, H, W) tensor; channels 0:2 are the flow field, channel 2 (if
    `source_sink`) is the bounded source/sink term."""

    def __init__(self, model: torch.nn.Module, flow_res: int | None,
                 max_disp_px: float, source_sink: bool, source_scale: float,
                 uses_physics: bool = False):
        super().__init__(model, uses_physics=uses_physics)
        self.flow_res = flow_res
        self.max_disp_px = float(max_disp_px)
        self.source_sink = bool(source_sink)
        self.source_scale = float(source_scale)

    def forward(self, batch) -> torch.Tensor:
        x = batch["input"]              # (B, 3, H, W): [occ0, r_start, r_stop]
        occ0 = x[:, 0]                  # (B, H, W)
        B, H, W = occ0.shape
        raw = self.model(x)             # (B, C_out, H, W)

        flow_raw = raw[:, 0:2]          # (B, 2, H, W)
        if self.flow_res is not None and self.flow_res < H:
            k = H // self.flow_res
            flow_raw = F.avg_pool2d(flow_raw, kernel_size=k)
            flow_raw = F.interpolate(flow_raw, size=(H, W), mode="bilinear",
                                      align_corners=True)

        # tanh-bounded displacement in PIXELS, then to grid_sample's
        # normalised units (align_corners=True: dx_px pixels <-> dx_px*2/(N-1)
        # normalised units along that axis).
        disp_px = torch.tanh(flow_raw) * self.max_disp_px      # (B,2,H,W)
        disp_x_norm = disp_px[:, 0] * (2.0 / max(W - 1, 1))    # (B,H,W)
        disp_y_norm = disp_px[:, 1] * (2.0 / max(H - 1, 1))    # (B,H,W)
        disp_norm = torch.stack([disp_x_norm, disp_y_norm], dim=-1)  # (B,H,W,2)

        base_grid = _base_grid(B, H, W, occ0.device, occ0.dtype)
        sampling_grid = base_grid + disp_norm

        occ_pred = F.grid_sample(
            occ0.unsqueeze(1), sampling_grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        ).squeeze(1)  # (B, H, W)

        if self.source_sink:
            assert raw.shape[1] >= 3, (
                "source_sink=True requires the backbone to emit >=3 output "
                "channels (2 flow + 1 source/sink); got "
                f"{raw.shape[1]}"
            )
            src = torch.tanh(raw[:, 2]) * self.source_scale     # (B, H, W)
            occ_pred = occ_pred + src

        occ_pred = occ_pred.clamp(0.0, 1.0)
        logit = _prob_to_logit(occ_pred)
        return logit.unsqueeze(1)


@register_model("nfd-flow-warp")
def _build_nfd_flow_warp(cfg: dict) -> FlowWarpWrapper:
    from model.UNetModels_modular import UNet

    source_sink = bool(cfg.get("source_sink", False))
    out_channels = 3 if source_sink else 2
    flow_res = cfg.get("flow_res")
    flow_res = int(flow_res) if flow_res is not None else None

    structure_parameters = {
        "features": cfg.get("features", [4, 8, 16]),
        "in_channels": int(cfg.get("in_channels", 3)),
        "out_channels": out_channels,
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "final_kernel_size": int(cfg.get("final_kernel_size", 1)),
        "activation": cfg.get("activation", "relu"),
        "residual": False,   # flow head has its own, different, identity path (zero flow)
        "bottleneck_type": cfg.get("bottleneck_type", "None"),
        "bottleneck_kwargs": cfg.get("bottleneck_kwargs", {}),
    }
    model = UNet(structure_parameters)
    return FlowWarpWrapper(
        model,
        flow_res=flow_res,
        max_disp_px=float(cfg.get("max_disp_px", 12.0)),
        source_sink=source_sink,
        source_scale=float(cfg.get("source_scale", 0.3)),
        uses_physics=bool(cfg.get("uses_physics", False)),
    )
