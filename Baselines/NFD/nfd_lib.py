"""Baselines/NFD/nfd_lib.py -- registrations for the non-FiLM NFD baseline.

Implements the single biggest fidelity delta identified in ``SPEC.md``: the
paper renders the pusher pose as TWO SEPARATE one-channel fields
``a_t := [r(x_t), r(x_t+1)]`` (in_channels=3 total with the state), while
this repo's existing ``Genesis/training/dataset.py::PileSweepData._draw_plate``
unions the two ``draw_plate_soft`` renders into ONE channel via a 0.5/1.0
intensity trick (in_channels=2). Nothing here edits that file (read-only per
task scope) -- ``PileSweepData3Ch`` below is a subclass that overrides only
the two methods responsible for the union (``_create_grids``, ``_draw_plate``),
inheriting ``__getitem__``/``_extract_sample_in_pxl``/``_draw_particle_grid``
unmodified so every other piece of ground-truth construction (occupancy
rasterisation, axis convention, physics normalisation) stays byte-identical
to the register's own numbers.

Also registers ``nfd-unet3ch``, a model factory that builds
``model.UNetModels_modular.UNet`` directly (not through the existing
``unet-modular`` factory in ``registry/model_registry.py``, which does not
forward ``final_kernel_size`` from its config dict at all -- confirmed by
reading that factory; the paper's Fig.7 needs a 1x1 final conv, so a new
factory that does forward it is required). This factory is reused for BOTH
the primary 3-channel run (dataset ``nfd-genesis-3ch``, in_channels=3) and
the in_channels=2 ablation (dataset ``genesis``, the repo's own union
encoding, unmodified) -- see ``configs/nfd_train_2ch_ablation.yaml``.

Import this module (which runs both ``@register_dataset``/``@register_model``
decorators at import time) before touching ``registry.dataset_registry`` /
``registry.model_registry``'s ``_REGISTRY`` dicts, e.g. via
``train_nfd.py``'s ``import Baselines.NFD.nfd_lib  # noqa: F401`` at the top.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from Genesis.training.dataset import PileSweepData
from physics.normalization import PhysicsBounds
from registry.dataset_registry import EulerianDatasetWrapper, register_dataset
from registry.model_registry import EulerianTrainingWrapper, register_model
from transforms.functional import draw_plate_soft


class PileSweepData3Ch(PileSweepData):
    """PileSweepData variant with a 3-channel input ``[occ0, r(p_start),
    r(p_stop)]``, both renders at full intensity 1.0 (channel identity, not
    an asymmetric 0.5/1.0 union, tells the network start from stop) -- see
    SPEC.md sec 1.3/2. Overrides only the two methods that build the action
    channel(s); ``__getitem__`` (inherited) still does exactly:

        self._draw_particle_grid(particles, self._input_grid[0], config)
        self._draw_particle_grid(particles_, self._output_grid, config)
        self._draw_plate(plate_pos, plate_pos_, angle, config)

    which now writes into 2 channels (1, 2) instead of unioning into 1.
    """

    def _create_grids(self, config) -> None:
        x_dim, y_dim, _ = config["box"]["vol"]
        x_pxl = max(1, int(round(x_dim * self.to_pxl)))
        y_pxl = max(1, int(round(y_dim * self.to_pxl)))
        self.ctr_in_PXL = torch.tensor((round(x_pxl / 2), round(y_pxl / 2), 0))

        self._input_grid = torch.zeros((3, x_pxl, y_pxl), dtype=torch.float32)
        self._output_grid = torch.zeros((x_pxl, y_pxl), dtype=torch.float32)

        x_dim_plt, y_dim_plt, _ = self._get_plate_dims(config)
        self._plt_dim_in_m = (x_dim_plt, y_dim_plt)

    def _draw_plate(self, start_pos, end_pos, angle, config) -> None:
        plate_dim_x, plate_dim_y, _ = self._get_plate_dims(config)
        plate_dim_x *= self.to_pxl
        plate_dim_y *= self.to_pxl
        sigma = max(0.5, 1.5 * self.resolution_scale)
        angle_tensor = torch.as_tensor([angle], dtype=torch.float32)
        start_center = start_pos[:2].to(torch.float32).unsqueeze(0)
        end_center = end_pos[:2].to(torch.float32).unsqueeze(0)
        grid_size = (self._input_grid.shape[1], self._input_grid.shape[2])

        r_start = draw_plate_soft(
            start_center, angle_tensor, grid_size, plate_dim_x, plate_dim_y,
            intensity=1.0, sigma=sigma,
        )[0]
        r_stop = draw_plate_soft(
            end_center, angle_tensor, grid_size, plate_dim_x, plate_dim_y,
            intensity=1.0, sigma=sigma,
        )[0]
        # NOT unioned (contrast base class's `1 - (1-occ1)*(1-occ2)`): two
        # separate channels, matching the paper's a_t := [r(x_t), r(x_t+1)].
        self._input_grid[1] = r_start
        self._input_grid[2] = r_stop


@register_dataset("nfd-genesis-3ch")
def _build_nfd_genesis_3ch(cfg: dict, split: str) -> EulerianDatasetWrapper:
    """Same config contract as the existing ``genesis`` factory
    (registry/dataset_registry.py's ``_build_genesis_dataset``), just backed
    by ``PileSweepData3Ch`` instead of ``PileSweepData``.

    Config keys: paths, val_pct (5), test_pct (5), resolution_scale (1.0),
    include_physics (false -- nfd-unet3ch consumes no physics),
    min_push_length_m (None), physics.normalization (bounds, unused unless
    include_physics=true).
    """
    bounds = (
        PhysicsBounds.from_config(cfg["physics"])
        if "physics" in cfg
        else PhysicsBounds.default()
    )
    min_push_length_m = cfg.get("min_push_length_m")
    raw = PileSweepData3Ch(
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
    return EulerianDatasetWrapper(
        raw,
        include_physics=bool(cfg.get("include_physics", False)),
        transforms_cfg=cfg.get("transforms"),
    )


@register_model("nfd-unet3ch")
def _build_nfd_unet(cfg: dict) -> EulerianTrainingWrapper:
    """Builds ``UNetModels_modular.UNet`` directly (not via the existing
    ``unet-modular`` factory, which never forwards ``final_kernel_size`` from
    its config dict -- confirmed by reading ``registry/model_registry.py``'s
    ``_build_unet_modular``, whose ``structure_parameters`` dict has no
    ``final_kernel_size`` key at all, so it is unreachable through that path).
    Paper Fig.7 needs the final conv to be 1x1 (distinct from the 3x3 convs
    elsewhere), so this factory exists to forward it.

    Reused for both the 3-channel primary run (in_channels=3) and the
    2-channel ablation (in_channels=2, paired with the ``genesis`` dataset
    type) -- only ``in_channels`` (and the paired dataset ``type``) differ.
    """
    from model.UNetModels_modular import UNet

    structure_parameters = {
        "features": cfg.get("features", [4, 8, 16]),
        "in_channels": int(cfg.get("in_channels", 3)),
        "out_channels": int(cfg.get("out_channels", 1)),
        "kernel_size": int(cfg.get("kernel_size", 3)),
        "final_kernel_size": int(cfg.get("final_kernel_size", 1)),
        "activation": cfg.get("activation", "relu"),
        "residual": bool(cfg.get("residual", True)),
        "bottleneck_type": cfg.get("bottleneck_type", "None"),
        "bottleneck_kwargs": cfg.get("bottleneck_kwargs", {}),
    }
    model = UNet(structure_parameters)
    return EulerianTrainingWrapper(model, uses_physics=bool(cfg.get("uses_physics", False)))


def plate_geometry_px(raw) -> tuple[float, float, float]:
    """(plate_dim_x_px, plate_dim_y_px, sigma) for dataset instance `raw`,
    exactly as ``PileSweepData3Ch._draw_plate``/the base class's ``_draw_plate``
    compute them. Plate/box geometry is constant across the pool (checked
    directly against a couple of run configs per cell in both n20_L20mm_train
    and n20_L40mm_train -- both give plate size [0.04, 0.002, 0.01] m, box vol
    [0.128, 0.128, 0.04] m), so using ``raw.configs[0]`` for every row is safe
    and matches the same constant-plate-size assumption
    ``Baselines/common/eval_baseline.py`` already makes
    (``plate_px = 0.04 / 0.128 * W``, one number for the whole eval cell).
    """
    plate_dim_x, plate_dim_y, _ = raw.configs[0]["plate"]["size"]
    plate_dim_x_px = plate_dim_x * raw.to_pxl
    plate_dim_y_px = plate_dim_y * raw.to_pxl
    sigma = max(0.5, 1.5 * raw.resolution_scale)
    return plate_dim_x_px, plate_dim_y_px, sigma
