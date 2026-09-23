"""model/warped_nfd/lib.py -- "warped NFD": the NFD UNet baseline,
predicting in the canonical PUSH FRAME (Suh & Tedrake 2020's SE(2) warp,
`transforms.functional.push_frame_roundtrip`) instead of the world frame.

Registers:
  - dataset type ``nfd-genesis-3ch-warped`` (``PileSweepData3ChWarped`` +
    a local ``EulerianDatasetWrapper`` subclass that additionally emits
    ``push_px``);
  - model type ``nfd-unet-warped`` (``WarpedNFDWrapper``).

Does NOT edit ``Baselines/NFD/nfd_lib.py`` -- subclasses/imports from it, per
task scope. The canonical-frame input-stack construction
(``build_canonical_stack``) and the warp/predict/unwarp/blend composition
(``push_frame_roundtrip``) are NOT reimplemented here: they live in
``model/warped_nfd/predictor.py`` and ``transforms/functional.py``
respectively, and this module's ``WarpedNFDWrapper.forward`` is a thin
caller of both, so the training path and ``WarpedNFDPredictor`` (the
eval-time twin in ``predictor.py``) cannot drift apart.

Coordinate-convention trap (see model/warped_nfd/WARPED_NFD_NOTES.md): a
dataset sample's plate-render pixel centers (``PileSweepData._extract_sample_
in_pxl``'s ``plate_pos``) are in (row=world_x_pixel, col=world_y_pixel)
order -- the SAME order ``draw_plate_soft``'s own ``center`` argument wants,
which is why ``nfd_lib.py``'s ``_draw_plate`` feeds them straight through.
``transforms.functional``'s push-frame module wants the TRANSPOSE of that,
(col, row) order. ``PileSweepData3ChWarped.push_px_pixels`` below computes
the native (row, col) pixel pair (identical formula to
``Baselines.NFD.predictor::NFDPredictor.predict_occ``) and then swaps it
into push_px's documented ``[start_col, start_row, end_col, end_row]``
layout -- do not "simplify" this by passing the native order directly to
push_frame code.

Import this module (registers both types) before ``Trainer.from_config``
looks them up -- ``Baselines/NFD/train_nfd.py`` imports it alongside
``nfd_lib``.
"""
from __future__ import annotations

import torch

from Baselines.NFD.nfd_lib import PileSweepData3Ch
from model.warped_nfd.predictor import build_canonical_stack
from physics.normalization import PhysicsBounds
from registry.dataset_registry import EulerianDatasetWrapper, register_dataset
from registry.model_registry import ModelTrainingWrapper, register_model
from transforms.functional import push_frame_roundtrip


class PileSweepData3ChWarped(PileSweepData3Ch):
    """Identical grids/channels to ``PileSweepData3Ch`` -- adds
    ``push_px_pixels(idx)``, the push endpoints in ``transforms.functional``'s
    push-frame pixel convention, derived the SAME way
    ``predictor.py::NFDPredictor.predict_occ`` derives them at eval time
    (``p_start[:2] * to_pxl + ctr_in_PXL[:2]``) so train and eval agree
    exactly, then swapped from (row, col) to (col, row) -- see the module
    docstring above.
    """

    def push_px_pixels(self, idx: int) -> torch.Tensor:
        action = self.get_raw_action(idx)  # [sx, sy, ex, ey] world metres
        ctr = self.ctr_in_PXL[:2].to(torch.float32)
        start_native = action[0:2] * self.to_pxl + ctr   # (x_px, y_px) = (row, col)
        end_native = action[2:4] * self.to_pxl + ctr
        # push_frame wants (col, row): swap.
        return torch.stack([
            start_native[1], start_native[0],
            end_native[1], end_native[0],
        ])


class _EulerianDatasetWrapperWithPushPx(EulerianDatasetWrapper):
    """``EulerianDatasetWrapper`` plus a ``push_px`` key (B,4 after
    collation) = ``[start_col, start_row, end_col, end_row]`` pixels, read
    off the raw dataset's ``push_px_pixels``. Everything else (input/target/
    physics, the ``EnsureRepresentation``/``EulerianOccupancyAliases``
    transforms) is unchanged from the base class."""

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        sample["push_px"] = self.raw_dataset.push_px_pixels(idx)
        return sample


@register_dataset("nfd-genesis-3ch-warped")
def _build_nfd_genesis_3ch_warped(cfg: dict, split: str) -> _EulerianDatasetWrapperWithPushPx:
    """Same config contract as ``nfd-genesis-3ch`` (nfd_lib.py) -- paths,
    val_pct (5), test_pct (5), resolution_scale (1.0), include_physics
    (false), min_push_length_m (None) -- backed by ``PileSweepData3ChWarped``
    and wrapped so the batch also carries ``push_px``."""
    bounds = (
        PhysicsBounds.from_config(cfg["physics"])
        if "physics" in cfg
        else PhysicsBounds.default()
    )
    min_push_length_m = cfg.get("min_push_length_m")
    raw = PileSweepData3ChWarped(
        paths=cfg["paths"],
        split=split,
        val_pct=int(cfg.get("val_pct", 5)),
        test_pct=int(cfg.get("test_pct", 5)),
        resolution_scale=float(cfg.get("resolution_scale", 1.0)),
        physics_bounds=bounds,
        min_push_length_m=(
            float(min_push_length_m) if min_push_length_m is not None else None
        ),
    )
    return _EulerianDatasetWrapperWithPushPx(
        raw,
        include_physics=bool(cfg.get("include_physics", False)),
        transforms_cfg=cfg.get("transforms"),
    )


class WarpedNFDWrapper(ModelTrainingWrapper):
    """NFD UNet predicting in the canonical push frame.

    ``forward(batch)`` reads ``batch["input"][:, 0]`` (occ0) and
    ``batch["push_px"]`` (B,4), warps occ0 into the canonical frame via
    ``push_frame_roundtrip``, builds the canonical-frame input stack with
    ``predictor.py::build_canonical_stack`` (occ0 + 2 plate channels + an
    optional 4th wall channel), runs the wrapped UNet, sigmoids it (the
    blend in `push_frame_roundtrip` must happen in PROBABILITY space, since
    it mixes in occ0 whose logit would be +/-inf), then converts the
    blended world-frame PROBABILITY back to a LOGIT (repo convention: a
    model returns a raw logit, the loss/metrics apply sigmoid) with a
    numerically safe inverse sigmoid.

    Returns ``Tensor[B, 1, H, W]`` -- matches ``EulerianTrainingWrapper``'s
    output shape today (``self.model(x)`` with ``out_channels=1``), since
    the loss/metrics depend on that shape.
    """

    def __init__(self, model: torch.nn.Module, plate_mode: str, wall_channel: bool,
                 canon_res: int | None, scale: float, plate_dim_x_px: float,
                 plate_dim_y_px: float, plate_sigma_px: float,
                 uses_physics: bool = False):
        super().__init__(model, uses_physics=uses_physics)
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
        occ0 = x[:, 0]
        action_world = x[:, 1:3]
        push_px = batch["push_px"]         # (B, 4) = [start_col, start_row, end_col, end_row]
        start_px, end_px = push_px[:, :2], push_px[:, 2:]
        H = occ0.shape[-1]
        canon_res = self.canon_res or H

        def fn(canon_occ0: torch.Tensor) -> torch.Tensor:
            stack = build_canonical_stack(
                canon_occ0, start_px, end_px, action_world, occ0,
                self.plate_mode, self.wall_channel, canon_res, self.scale,
                self.plate_dim_x_px, self.plate_dim_y_px, self.plate_sigma_px, H,
            )
            logits = self.model(stack)
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            return torch.sigmoid(logits)

        prob = push_frame_roundtrip(fn, occ0, start_px, end_px, canon_res,
                                     scale=self.scale, blend=True)
        eps = 1e-6
        prob = prob.clamp(eps, 1.0 - eps)
        logit = torch.log(prob / (1.0 - prob))
        return logit.unsqueeze(1)


@register_model("nfd-unet-warped")
def _build_warped_nfd_unet(cfg: dict) -> WarpedNFDWrapper:
    """Config keys: everything ``nfd-unet3ch`` takes (features, kernel_size,
    final_kernel_size, activation, residual, bottleneck_type/kwargs,
    uses_physics), plus:

      plate_mode   : "canonical" (canonical_plate_channels, analytic render)
                     or "warp" (to_push_frame the world-frame plate renders).
      wall_channel : bool -- 4th input channel, the warped workspace-extent
                     indicator. ``in_channels`` MUST be 3 (False) or 4
                     (True) -- asserted, not silently coerced.
      canon_res    : int, default = the batch's own grid resolution (H).
      scale        : float, default 1.0 -- forwarded to
                     `push_frame_transform`/`to_push_frame`.
      resolution_scale, plate_size_m : used ONLY to compute
                     ``plate_dim_x_px``/``plate_dim_y_px``/``sigma`` for the
                     "canonical" plate_mode (`canonical_plate_channels` needs
                     plate size in WORLD pixels but this wrapper has no
                     access to the raw `PileSweepData` instance the way
                     `nfd_lib.plate_geometry_px(raw)` does) -- MUST match the
                     paired dataset config's own `resolution_scale`. Defaults
                     (0.5, [0.04, 0.002, 0.01]) match every existing NFD
                     pilot config's dataset/plate geometry (constant across
                     the pool, per `nfd_lib.py`'s own `plate_geometry_px`
                     docstring).
    """
    from model.UNetModels_modular import UNet

    wall_channel = bool(cfg.get("wall_channel", False))
    expected_in = 4 if wall_channel else 3
    in_channels = int(cfg.get("in_channels", expected_in))
    assert in_channels == expected_in, (
        f"nfd-unet-warped: in_channels must be {expected_in} when "
        f"wall_channel={wall_channel}, got {in_channels}"
    )

    structure_parameters = {
        "features": cfg.get("features", [4, 8, 16]),
        "in_channels": in_channels,
        "out_channels": int(cfg.get("out_channels", 1)),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "final_kernel_size": int(cfg.get("final_kernel_size", 1)),
        "activation": cfg.get("activation", "relu"),
        "residual": bool(cfg.get("residual", True)),
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

    return WarpedNFDWrapper(
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
