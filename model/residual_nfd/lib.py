"""model/residual_nfd/lib.py -- EXP-0022 R1/R2: image-space RESIDUAL
prediction for NFD, both unwarped (R1, control) and warped (R2).

Context: `experiments/EXP-0022-*/results/residual_vs_direct_survey.md` found
that NO image-space model in this repo trains against a residual TARGET --
several (this NFD family included) add `occ0` back into the pre-sigmoid
LOGIT as an architectural skip (`UNetModels_modular.UNet(residual=True)`),
but the LOSS always compares `sigmoid(logit)` against the absolute `occ1`,
never against `occ1-occ0`. This module changes the PARAMETERISATION, not the
objective: the loss stays world-frame MSE against absolute `occ1`
(`eulerian_combined`, mse=1.0, unchanged, reused via the same
prob->safe-inverse-sigmoid->logit trick `WarpedNFDWrapper` already uses), but
the network internally predicts a signed residual through a `tanh` head and
the wrapper reconstructs `occ_pred = clamp(occ0 + delta, 0, 1)` before that
trick is applied.

Two registrations, mirroring the existing subclassing pattern
(`nfd_lib.py`/`warped_nfd_lib.py`) -- neither of those files, nor
`model/UNetModels_modular.py`, is edited:

  - model type ``nfd-unet3ch-residual`` (R1, unwarped control): reuses the
    EXISTING ``nfd-genesis-3ch`` dataset unchanged (target stays absolute
    `occ1` -- no residual-target dataset variant is created, per task scope).
    Config MUST set ``residual: false`` so ``UNetModels_modular.UNet``'s own
    occ0-into-logit skip is OFF -- this wrapper's explicit `occ0 + tanh(delta)`
    is the only identity path. Verified by inspection (`UNet.forward`,
    `model/UNetModels_modular.py:394-406`): `residual_mode=False` returns
    `raw` unchanged, so `raw` here is genuinely just the conv head's output,
    with no double-count against this wrapper's own residual add.

  - model type ``nfd-unet-warped-residual`` (R2, warped): reuses the
    EXISTING ``nfd-genesis-3ch-warped`` dataset (registered by
    `warped_nfd_lib.py`, imported not reimplemented) for the `push_px` key,
    and `model/warped_nfd/predictor.py::build_canonical_stack` for the
    canonical-frame input stack, exactly as `WarpedNFDWrapper` does. The
    mechanism that is the actual point of R2:

      1. warp the PRISTINE world-frame `occ0` into the canonical frame
         (`to_push_frame`) and predict the residual THERE;
      2. unwarp the RESIDUAL FIELD itself (not a reconstructed occupancy)
         back to the world frame (`from_push_frame`);
      3. add that unwarped residual to the SAME pristine, never-resampled
         world-frame `occ0` used in step 1 (not to any warped-and-back
         occupancy), then clamp to [0, 1].

    No `blend_push_prediction`/validity-mask blend is applied here -- see
    the module docstring's "on the blend" note below for why one is not
    needed in this formulation, verified (not just argued) by the dead-
    gradient/corner check in this experiment's RUN.md.

Import this module (registers all three types) alongside `nfd_lib` and
`warped_nfd_lib` before `Trainer.from_config` looks them up.

## On the blend (validity-mask corner handling)

Every prediction the un-residual warped model (`WarpedNFDWrapper`) makes
passes through `push_frame_roundtrip(..., blend=True)`: outside
`push_frame_validity_mask`'s ~1 interior, the corners a rotated square loses
are filled by COPYING `occ0` in explicitly, because a `grid_sample` zero-pad
there would read as "no material" -- wrong for an absolute occupancy field.

For a RESIDUAL field the same zero-padding reads as "no CHANGE" -- exactly
the correct fallback -- so no explicit copy step is needed: `from_push_frame`
already returns 0 for any world pixel whose canonical-frame source position
falls outside the canonical grid (grid_sample's `padding_mode='zeros'`), and
`occ0 + 0 = occ0` bit-exact (mod float rounding) in that region. This is the
one clean argument in the residual formulation's favour named in the task
brief ("does the change preserve the ~98% of the image that never changes
bit-exact") -- it is verified empirically in this experiment's RUN.md by
checking `occ_pred == occ0` outside the validity mask on held-out batches,
not merely argued from the grid_sample contract.
"""
from __future__ import annotations

import torch

from Baselines.NFD.nfd_lib import PileSweepData3Ch  # noqa: F401 (re-export for eval script convenience)
from model.warped_nfd.predictor import build_canonical_stack
from registry.model_registry import ModelTrainingWrapper, register_model
from transforms.functional import from_push_frame, to_push_frame

_EPS = 1e-6


def _prob_to_logit(prob: torch.Tensor) -> torch.Tensor:
    """Safe inverse-sigmoid: same trick `WarpedNFDWrapper.forward` uses so a
    reconstructed PROBABILITY can still be scored by the unmodified
    `eulerian_combined` loss (which always applies `sigmoid(logit)`)."""
    p = prob.clamp(_EPS, 1.0 - _EPS)
    return torch.log(p / (1.0 - p))


class ResidualUnwarpedWrapper(ModelTrainingWrapper):
    """R1 (control): world-frame NFD UNet, `residual_mode` OFF (config must
    set `residual: false`), with an EXPLICIT external residual: the conv
    head's raw output is passed through `tanh` (bounded, signed, unlike
    `sigmoid` which cannot express a negative delta) to get a residual in
    [-1, 1], added to `occ0`, then clamped to [0, 1] and converted back to a
    logit for the unmodified loss.

    Expected to gain little over the plain `nfd-unet3ch` control, BECAUSE
    `residual: true` there already adds `occ0` into the logit -- an identity
    path already exists in that baseline. This arm's point is to be the
    control that makes R2 (the warped residual) interpretable, not to beat
    the existing baseline.
    """

    def __init__(self, model: torch.nn.Module, uses_physics: bool = False):
        super().__init__(model, uses_physics=uses_physics)
        assert not getattr(model, "residual_mode", False), (
            "ResidualUnwarpedWrapper requires the wrapped UNet's OWN "
            "residual_mode=False (config residual: false) -- otherwise its "
            "occ0-into-logit skip double-counts with this wrapper's "
            "explicit residual add."
        )

    def forward(self, batch) -> torch.Tensor:
        x = batch["input"]              # (B, 3, H, W): [occ0, r_start, r_stop]
        occ0 = x[:, 0]
        raw = self.model(x)             # (B, 1, H, W); residual_mode=False -> no skip
        if raw.dim() == 4:
            raw = raw.squeeze(1)
        delta = torch.tanh(raw)         # signed residual, [-1, 1]
        occ_pred = torch.clamp(occ0 + delta, 0.0, 1.0)
        logit = _prob_to_logit(occ_pred)
        return logit.unsqueeze(1)


@register_model("nfd-unet3ch-residual")
def _build_nfd_unet3ch_residual(cfg: dict) -> ResidualUnwarpedWrapper:
    """Config keys: identical to `nfd-unet3ch` (features, kernel_size,
    final_kernel_size, activation, bottleneck_type/kwargs, uses_physics),
    EXCEPT `residual` MUST be `false` (asserted, not silently coerced) --
    see module docstring."""
    from model.UNetModels_modular import UNet

    assert not bool(cfg.get("residual", False)), (
        "nfd-unet3ch-residual: config `residual` must be false -- this "
        "model applies its OWN explicit tanh residual add; the UNet's "
        "internal occ0-into-logit skip must be off to avoid double-counting."
    )
    structure_parameters = {
        "features": cfg.get("features", [4, 8, 16]),
        "in_channels": int(cfg.get("in_channels", 3)),
        "out_channels": int(cfg.get("out_channels", 1)),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "final_kernel_size": int(cfg.get("final_kernel_size", 1)),
        "activation": cfg.get("activation", "relu"),
        "residual": False,
        "bottleneck_type": cfg.get("bottleneck_type", "None"),
        "bottleneck_kwargs": cfg.get("bottleneck_kwargs", {}),
    }
    model = UNet(structure_parameters)
    return ResidualUnwarpedWrapper(model, uses_physics=bool(cfg.get("uses_physics", False)))


class ResidualWarpedWrapper(ModelTrainingWrapper):
    """R2: the warped residual arm -- see module docstring for the exact
    3-step mechanism (warp occ0 in, predict residual there, unwarp the
    RESIDUAL, add to pristine world occ0). No blend: zero-padding outside
    the canonical grid's coverage is already the neutral value for a
    residual field (see module docstring)."""

    def __init__(self, model: torch.nn.Module, plate_mode: str, wall_channel: bool,
                 canon_res: int | None, scale: float, plate_dim_x_px: float,
                 plate_dim_y_px: float, plate_sigma_px: float,
                 uses_physics: bool = False):
        super().__init__(model, uses_physics=uses_physics)
        assert not getattr(model, "residual_mode", False), (
            "ResidualWarpedWrapper requires the wrapped UNet's OWN "
            "residual_mode=False (config residual: false) -- see "
            "ResidualUnwarpedWrapper's identical assertion."
        )
        assert plate_mode in ("canonical", "warp"), plate_mode
        self.plate_mode = plate_mode
        self.wall_channel = wall_channel
        self.canon_res = canon_res
        self.scale = scale
        self.plate_dim_x_px = plate_dim_x_px
        self.plate_dim_y_px = plate_dim_y_px
        self.plate_sigma_px = plate_sigma_px

    def forward(self, batch) -> torch.Tensor:
        x = batch["input"]                 # (B, 3, H, W): [occ0, r_start, r_stop]
        occ0 = x[:, 0]                     # PRISTINE world-frame occ0 -- never resampled
        action_world = x[:, 1:3]
        push_px = batch["push_px"]         # (B, 4) = [start_col, start_row, end_col, end_row]
        start_px, end_px = push_px[:, :2], push_px[:, 2:]
        H = occ0.shape[-1]
        canon_res = self.canon_res or H

        # 1. warp occ0 into the canonical frame and predict the residual there.
        canon_occ0 = to_push_frame(occ0, start_px, end_px, (canon_res, canon_res), self.scale)
        stack = build_canonical_stack(
            canon_occ0, start_px, end_px, action_world, occ0,
            self.plate_mode, self.wall_channel, canon_res, self.scale,
            self.plate_dim_x_px, self.plate_dim_y_px, self.plate_sigma_px, H,
        )
        raw = self.model(stack)            # (B, 1, canon_res, canon_res); residual_mode=False
        if raw.dim() == 4:
            raw = raw.squeeze(1)
        delta_canon = torch.tanh(raw)      # signed canonical-frame residual, [-1, 1]

        # 2. unwarp the RESIDUAL (not a reconstructed occupancy) back to world.
        delta_world = from_push_frame(delta_canon, start_px, end_px, (H, H), self.scale)

        # 3. add to the pristine world-frame occ0, then clamp. No blend: see
        # module docstring -- zero-padding outside the canonical grid's
        # coverage already means "no change" for a residual field.
        occ_pred = torch.clamp(occ0 + delta_world, 0.0, 1.0)
        logit = _prob_to_logit(occ_pred)
        return logit.unsqueeze(1)


@register_model("nfd-unet-warped-residual")
def _build_nfd_unet_warped_residual(cfg: dict) -> ResidualWarpedWrapper:
    """Config keys: identical to `nfd-unet-warped` (features, kernel_size,
    final_kernel_size, activation, bottleneck_type/kwargs, uses_physics,
    plate_mode, wall_channel, canon_res, scale, resolution_scale,
    plate_size_m), EXCEPT `residual` MUST be `false` -- see
    `ResidualUnwarpedWrapper`'s identical requirement and the module
    docstring's double-counting warning."""
    from model.UNetModels_modular import UNet

    assert not bool(cfg.get("residual", False)), (
        "nfd-unet-warped-residual: config `residual` must be false -- see "
        "module docstring."
    )
    wall_channel = bool(cfg.get("wall_channel", False))
    expected_in = 4 if wall_channel else 3
    in_channels = int(cfg.get("in_channels", expected_in))
    assert in_channels == expected_in, (
        f"nfd-unet-warped-residual: in_channels must be {expected_in} when "
        f"wall_channel={wall_channel}, got {in_channels}"
    )

    structure_parameters = {
        "features": cfg.get("features", [4, 8, 16]),
        "in_channels": in_channels,
        "out_channels": int(cfg.get("out_channels", 1)),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "final_kernel_size": int(cfg.get("final_kernel_size", 1)),
        "activation": cfg.get("activation", "relu"),
        "residual": False,
        "bottleneck_type": cfg.get("bottleneck_type", "None"),
        "bottleneck_kwargs": cfg.get("bottleneck_kwargs", {}),
    }
    model = UNet(structure_parameters)

    resolution_scale = float(cfg.get("resolution_scale", 0.5))
    plate_size_m = cfg.get("plate_size_m", [0.04, 0.002, 0.01])
    to_pxl = 1e3 * resolution_scale
    plate_dim_x_px = float(plate_size_m[0]) * to_pxl
    plate_dim_y_px = float(plate_size_m[1]) * to_pxl
    plate_sigma_px = max(0.5, 1.5 * resolution_scale)

    canon_res = cfg.get("canon_res")
    canon_res = int(canon_res) if canon_res is not None else None

    return ResidualWarpedWrapper(
        model,
        plate_mode=cfg.get("plate_mode", "canonical"),
        wall_channel=wall_channel,
        canon_res=canon_res,
        scale=float(cfg.get("scale", 1.0)),
        plate_dim_x_px=plate_dim_x_px,
        plate_dim_y_px=plate_dim_y_px,
        plate_sigma_px=plate_sigma_px,
        uses_physics=bool(cfg.get("uses_physics", False)),
    )
