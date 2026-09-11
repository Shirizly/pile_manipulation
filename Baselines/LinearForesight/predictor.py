"""Baselines/LinearForesight/predictor.py -- BaselinePredictor for
Baselines/common/eval_baseline.py / eval_randlen_indist.py / eval_report.py.

Wraps the operator bundle `fit_switched.py` produces
(`Baselines/LinearForesight/runs/operators_res<R>.pt`) so both the switched
(per-length-bin) model AND the single global reference operator it was fit
alongside can be scored by the same image-accuracy / control-utility harness
every other baseline (GNN, NFD, SchenckCNN) uses, following the same
`.name`/`.predict_occ(batch)` contract those baselines' `predictor.py`
implement.

Two predictor classes, ONE bundle file: `fit_switched.py` fits
`single_operator` (the plain global operator, matching this repo's
established "linear" reference recipe) and the per-bin `operators` in the
same run, on the identical loaded train data -- so `linear-single` here is
never a stale/previously-saved checkpoint, it is retrained fresh every time
`fit_switched.py` is (re-)run, exactly like the switched operators it is
compared against. `SingleLinearForesightPredictor` reads `single_operator`
out of the very same file `SwitchedLinearForesightPredictor` reads
`operators` from -- there is deliberately no separate "fit the single
operator" script/checkpoint to go stale.

`build_predictor()` (switched) / `build_predictor_single()` (single) take no
arguments (harness contract) and load
`Baselines/LinearForesight/runs/operators_res64.pt` by default (override via
the LINEARFORESIGHT_CKPT env var, same pattern as GNN_CKPT/NFD_CKPT -- point
it at `operators_res32.pt` to score the 32x32 resolution ablation instead).
"""
from __future__ import annotations

import os

import torch

from fit_linear_foresight import actions_to_pixels, predict_world

from Baselines.LinearForesight.model import predict_switched, push_length_m

DEFAULT_OPERATORS = "Baselines/LinearForesight/runs/operators_res64.pt"


class SwitchedLinearForesightPredictor:
    name = "linear_foresight_switched"

    def __init__(self, operators_path: str = DEFAULT_OPERATORS):
        ckpt = torch.load(operators_path, map_location="cpu", weights_only=False)
        self.operators = ckpt["operators"]
        self.bin_edges = ckpt["bin_edges"]
        self.res = ckpt["res"]
        self.crop = ckpt["crop"]
        self.name = f"linear_foresight_switched_res{self.res}"
        print(f"[SwitchedLinearForesightPredictor] loaded {operators_path} "
              f"({ckpt['n_bins']} bins, res={self.res}, "
              f"constraint={ckpt['constraint']}, counts={ckpt['counts']})")

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        H, W = batch.H, batch.W
        s_px, e_px = actions_to_pixels(batch.actions, batch.workspace_min,
                                        batch.workspace_max, (H, W))
        lengths_m = push_length_m(batch.actions)
        return predict_switched(self.bin_edges, self.operators, batch.occ0,
                                 s_px, e_px, lengths_m, self.res, (H, W), self.crop)


class SingleLinearForesightPredictor:
    """The un-switched reference operator, from the SAME fit run/bundle as
    the switched one (see module docstring) -- so a switched-vs-single
    comparison run through this predictor is guaranteed to compare two
    operators fit on identical data, not one fresh and one stale."""
    name = "linear_foresight_single"

    def __init__(self, operators_path: str = DEFAULT_OPERATORS):
        ckpt = torch.load(operators_path, map_location="cpu", weights_only=False)
        self.operator = ckpt["single_operator"]
        self.res = ckpt["res"]
        self.crop = ckpt["crop"]
        self.name = f"linear_foresight_single_res{self.res}"
        print(f"[SingleLinearForesightPredictor] loaded {operators_path} "
              f"(res={self.res}, fit on {ckpt['train_cfg']})")

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        H, W = batch.H, batch.W
        s_px, e_px = actions_to_pixels(batch.actions, batch.workspace_min,
                                        batch.workspace_max, (H, W))
        return predict_world(self.operator, batch.occ0, s_px, e_px,
                              self.res, (H, W), self.crop)


def build_predictor() -> SwitchedLinearForesightPredictor:
    path = os.environ.get("LINEARFORESIGHT_CKPT", DEFAULT_OPERATORS)
    return SwitchedLinearForesightPredictor(path)


def build_predictor_single() -> SingleLinearForesightPredictor:
    path = os.environ.get("LINEARFORESIGHT_CKPT", DEFAULT_OPERATORS)
    return SingleLinearForesightPredictor(path)
