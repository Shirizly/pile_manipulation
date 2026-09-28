"""Baselines/common/eval_report.py -- cross-corpus, multi-metric report.

Scores every model in `MODELS` against every corpus in `CORPORA`:

  * accuracy (`fit_linear_foresight.metrics`'s `accuracy` key, swept-region
    masked) -- for a GNN model, BOTH sides of the comparison (prediction
    AND ground truth) are passed through the SAME node-count bottleneck
    (`Baselines/GNN/perception.py::resample_occupancy_through_nodes`) so
    the score isolates dynamics-prediction error from the representational
    loss of using only `n_particles` nodes to describe an arbitrary pile
    -- not because it's unfair to the GNN otherwise, but because comparing
    a 20/30-node prediction against a full-fidelity many-cube ground truth
    would conflate "wrong dynamics" with "fewer nodes than cubes", two
    different things. NFD (a full-resolution grid model, no node-count
    bottleneck) is compared directly against raw ground truth.

  * `slateN`/"capture" (docs/experiments/METRICS.md) on step-0 same-state
    pools, for 3 goal shapes (`Baselines/common/goals.py`: a per-slate
    random quadrant, a centred 'O' ring, a centred 'T') x 3 value
    functions (Lyapunov distance-to-goal, mass-in-region, signed
    mass-in-region), reported per (goal, value function) and averaged
    over goals per value function.

Usage:
    PYTHONPATH=. python Baselines/common/eval_report.py \\
        --out-prefix Baselines/common/runs/cross_corpus_report
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time

import numpy as np
import torch

from control_utility_test import lyapunov
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask

from Baselines.common.data import load_cell
from Baselines.common.eval_baseline import _predictor_batch
from Baselines.common.goals import (
    dist_field_from_mask, letter_mask, mass_in_region, quadrant_mask, random_quadrant_mask,
    signed_mass_in_region, slate_n_capture, two_squares_mask,
)
from Baselines.common.randlen_data import load_randlen_cell
from Baselines.GNN.perception import resample_occupancy_through_nodes
from transforms.functional import to_push_frame

MODELS = {
    "gnn_l20l40": dict(
        module="Baselines.GNN.predictor", factory="build_predictor",
        ckpt_env="GNN_CKPT", ckpt="Baselines/GNN/runs/ckpt_best.pth", is_gnn=True,
    ),
    "gnn_randlen_n30": dict(
        module="Baselines.GNN.predictor", factory="build_predictor",
        ckpt_env="GNN_CKPT", ckpt="Baselines/GNN/runs/randlen_train_all_n30/ckpt_best.pth",
        is_gnn=True,
    ),
    "nfd_randlen": dict(
        module="Baselines.NFD.predictor", factory="build_predictor",
        ckpt_env="NFD_CKPT", ckpt="Baselines/NFD/runs/nfd_3ch_randlen/unet_best.pth",
        is_gnn=False,
    ),
    # Baselines/LinearForesight (EXP-0003): switched (per-push-length-bin)
    # vs. single global pixel operator, at the repo's own res=64 convention
    # and at the paper's own res=32, all four fit on the SAME
    # overnight_randlen train_all corpus in the same `fit_switched.py` run
    # per resolution (see that module's docstring for why single/switched
    # are bundled together rather than fit separately).
    "linear_switched_res64": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res64.pt",
        is_gnn=False,
    ),
    "linear_single_res64": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor_single",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res64.pt",
        is_gnn=False,
    ),
    "linear_switched_res32": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res32.pt",
        is_gnn=False,
    ),
    "linear_single_res32": dict(
        module="Baselines.LinearForesight.predictor", factory="build_predictor_single",
        ckpt_env="LINEARFORESIGHT_CKPT", ckpt="Baselines/LinearForesight/runs/operators_res32.pt",
        is_gnn=False,
    ),
    # EXP-0022 RUN-0004: three L20mm-only pilot arms, identical recipe,
    # differing only in whether/how the NFD UNet sees the LinearForesight
    # canonical push-frame warp. `canon_res=None` (batch's own grid
    # resolution, 64 at resolution_scale=0.5), `plate_mode="canonical"`,
    # `scale=1.0` for both warped arms -- these are ALSO the checkpoints'
    # own training-time defaults (see each arm's `model_card.yaml`), and
    # `predictor.py::_assert_matches_model_card` checks that on every load,
    # so a drifted default here would raise rather than silently score a
    # mismatched setting.
    "nfd_unwarped_L20mm_pilot": dict(
        module="Baselines.NFD.predictor", factory="build_predictor",
        ckpt_env="NFD_CKPT", ckpt="Baselines/NFD/runs/nfd_3ch_L20mm_pilot/unet_best.pth",
        is_gnn=False,
    ),
    "nfd_warped_L20mm_pilot": dict(
        module="model.warped_nfd.predictor", factory="build_predictor_warped",
        ckpt_env="NFD_WARPED_CKPT", ckpt="Baselines/NFD/runs/nfd_warped_L20mm_pilot/unet_best.pth",
        is_gnn=False,
        kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0),
    ),
    "nfd_warped_walls_L20mm_pilot": dict(
        module="model.warped_nfd.predictor", factory="build_predictor_warped_walls",
        ckpt_env="NFD_WARPED_WALLS_CKPT",
        ckpt="Baselines/NFD/runs/nfd_warped_walls_L20mm_pilot/unet_best.pth",
        is_gnn=False,
        kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0),
    ),
    # EXP-0022 RUN-0005: warped NFD (no wall channel), canon_res=None (->64
    # at resolution_scale=0.5, matching the LinearForesight warp exactly),
    # trained on overnight_randlen_train with the SAME split/recipe/gradient
    # step count as nfd_randlen (the world-frame baseline above). The two
    # are the main-run comparison this MODELS entry exists for.
    "nfd_warped_randlen": dict(
        module="model.warped_nfd.predictor", factory="build_predictor_warped",
        ckpt_env="NFD_WARPED_CKPT", ckpt="Baselines/NFD/runs/nfd_warped_randlen/unet_best.pth",
        is_gnn=False,
        kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0),
    ),
    # EXP-0022 RUN-0010: the FAIR re-run of the arm above -- flip-only
    # augmentation (the x2 rotation-free subgroup; the x8 aug RUN-0005 used
    # collapses to 2 distinct canonical inputs for a push-frame model), 120
    # epochs = 668,040 gradient steps, matched to nfd_randlen's 668,100.
    # This is what replaces RUN-0005's verdict -- see PLAN.md's "Phase A
    # interim state". Same architecture/knobs as nfd_warped_randlen, only
    # the checkpoint (and the recipe that produced it) differ.
    # EXP-0025 flow-warp sweep: predict a per-pixel DISPLACEMENT field and warp
    # the current occupancy through it, instead of emitting the next occupancy.
    # Mass-conserving and residual by construction. All six cells share one
    # short-epoch recipe on a subset of slates_multistep/n20_L20mm_train, so they
    # are comparable to EACH OTHER and to `flow_direct_control` below -- NOT to
    # the fully-trained arms elsewhere in this dict.
    "flow_baseline": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_baseline",
        ckpt_env="FLOW_BASELINE_CKPT",
        ckpt="Baselines/NFD/runs/flow_baseline_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    "flow_coarse16": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_coarse",
        ckpt_env="FLOW_COARSE_CKPT",
        ckpt="Baselines/NFD/runs/flow_coarse16_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    "flow_smalldisp4": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_small_disp",
        ckpt_env="FLOW_SMALLDISP_CKPT",
        ckpt="Baselines/NFD/runs/flow_smalldisp4_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    "flow_largedisp24": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_large_disp",
        ckpt_env="FLOW_LARGEDISP_CKPT",
        ckpt="Baselines/NFD/runs/flow_largedisp24_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    "flow_srcsink": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_source_sink",
        ckpt_env="FLOW_SRCSINK_CKPT",
        ckpt="Baselines/NFD/runs/flow_srcsink_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    # The direct-prediction control for that sweep: identical subset, recipe and
    # epoch count, ordinary 3-channel NFD. Without it the flow numbers have
    # nothing to be read against.
    # EXP-0025 extension (results/supervised_flow.md): flow head supervised
    # DIRECTLY from particle correspondence (not photometric-only). Same
    # subset/epoch count as the sweep above; augmentation dropped (see that
    # run's training script docstring) so this is directional vs the
    # photometric-only cells, not a strict re-run under identical recipe.
    "flow_supervised": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_supervised",
        ckpt_env="FLOW_SUPERVISED_CKPT",
        ckpt="experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
             "RUN-0008-supervised-flow/unet_best.pth", is_gnn=False,
    ),
    "flow_supervised_masked": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_supervised_masked",
        ckpt_env="FLOW_SUPERVISED_MASKED_CKPT",
        ckpt="experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
             "RUN-0009-supervised-flow-masked-penalty/unet_best.pth", is_gnn=False,
    ),
    "flow_supervised_augmented": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_supervised_augmented",
        ckpt_env="FLOW_SUPERVISED_AUG_CKPT",
        ckpt="experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
             "RUN-0012-supervised-flow-augmented/unet_best.pth", is_gnn=False,
    ),
    "flow_supervised_masked_augmented": dict(
        module="model.flow_nfd.predictor", factory="build_predictor_supervised_masked_augmented",
        ckpt_env="FLOW_SUPERVISED_MASKED_AUG_CKPT",
        ckpt="experiments/EXP-0025-flow-warp-nfd-pilot/runs/"
             "RUN-0013-supervised-flow-masked-penalty-augmented/unet_best.pth", is_gnn=False,
    ),
    "flow_direct_control": dict(
        module="Baselines.NFD.predictor", factory="build_predictor",
        ckpt_env="NFD_CKPT",
        ckpt="Baselines/NFD/runs/direct_control_L20mm_pilot/unet_best.pth", is_gnn=False,
    ),
    "nfd_warped_randlen_flipaug": dict(
        module="model.warped_nfd.predictor", factory="build_predictor_warped",
        ckpt_env="NFD_WARPED_CKPT",
        ckpt="Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_best.pth",
        is_gnn=False,
        kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0),
    ),
    # Same run, epoch-30 checkpoint: a quarter of the baseline's gradient
    # steps/wall-clock at flip-only augmentation -- "does the warp buy a
    # cheaper model" reading, alongside the "does it buy a better one"
    # reading above.
    "nfd_warped_randlen_flipaug_epoch30": dict(
        module="model.warped_nfd.predictor", factory="build_predictor_warped",
        ckpt_env="NFD_WARPED_CKPT",
        ckpt="Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_epoch_30.pth",
        is_gnn=False,
        kwargs=dict(canon_res=None, plate_mode="canonical", scale=1.0),
    ),
    # EXP-0022 R1/R2 (residual_vs_direct_survey.md): image-space RESIDUAL
    # prediction -- loss stays world-frame MSE against absolute occ1 (same
    # as every arm above); the PARAMETERISATION is `clamp(occ0 + tanh(delta),
    # 0, 1)`. R1 unwarped control, R2 warped (residual predicted/unwarped in
    # the canonical frame, added to pristine world occ0, no blend -- see
    # model/residual_nfd/lib.py). Same L20mm_train recipe as
    # RUN-0001/RUN-0002 above, directly comparable.
    # RUN-0022: WORLD-FRAME NFD + explicit tanh residual on overnight_randlen,
    # NO augmentation. **INTERRUPTED at epoch 43 of a planned 240** (the session
    # hosting it was restarted), so it reached ~120k of the intended 668,100
    # gradient steps -- roughly 18% of the budget the baseline and every other
    # randlen arm were trained on. These two entries are NOT step-matched to
    # anything; read them as an early-training snapshot, not as an arm.
    "nfd_residual_worldframe_noaug_ep43": dict(
        module="model.residual_nfd.predictor", factory="build_predictor_residual_unwarped",
        ckpt_env="NFD_RESIDUAL_CKPT",
        ckpt="Baselines/NFD/runs/nfd_residual_unwarped_noaug_randlen/unet_best.pth",
        is_gnn=False,
    ),
    "nfd_residual_worldframe_noaug_ep30": dict(
        module="model.residual_nfd.predictor", factory="build_predictor_residual_unwarped",
        ckpt_env="NFD_RESIDUAL_CKPT",
        ckpt="Baselines/NFD/runs/nfd_residual_unwarped_noaug_randlen/unet_epoch_30.pth",
        is_gnn=False,
    ),
    "nfd_residual_unwarped_L20mm_pilot": dict(
        module="model.residual_nfd.predictor", factory="build_predictor_residual_unwarped",
        ckpt_env="NFD_RESIDUAL_CKPT",
        ckpt="Baselines/NFD/runs/nfd_residual_unwarped_L20mm_pilot_2/unet_best.pth",
        is_gnn=False,
    ),
    # RUN-0019: warped + flip-only augmentation + explicit tanh residual, trained
    # on overnight_randlen with the SAME corpus, split, recipe and gradient-step
    # count as the world-frame baseline. Stacks the two changes that individually
    # helped in EXP-0022; the combination had never been run before this.
    "nfd_residual_warped_flipaug_randlen": dict(
        module="model.residual_nfd.predictor", factory="build_predictor_residual_warped",
        ckpt_env="NFD_RESIDUAL_WARPED_CKPT",
        ckpt="Baselines/NFD/runs/nfd_residual_warped_flipaug_randlen/unet_best.pth",
        is_gnn=False,
    ),
    "nfd_residual_warped_L20mm_pilot": dict(
        module="model.residual_nfd.predictor", factory="build_predictor_residual_warped",
        ckpt_env="NFD_RESIDUAL_WARPED_CKPT",
        ckpt="Baselines/NFD/runs/nfd_residual_warped_L20mm_pilot_2/unet_best.pth",
        is_gnn=False,
    ),
}

CORPORA = {
    "L20mm": dict(
        kind="slate",
        train_cfg="configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
    ),
    "L40mm": dict(
        kind="slate",
        train_cfg="configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json",
    ),
    "randlen_test": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_all.yaml",
    ),
    # Spawn-mode-stratified variants of the above (EXP-0002): the SAME
    # held-out files, split by initialisation type instead of pooled
    # together, so a model's performance can be compared across spawn
    # modes. "mixed" has no n50 group at all (never collected -- see
    # Genesis/data/overnight_randlen's own directory listing), so its
    # config is the same n20-only one used elsewhere; piled/scattered each
    # pool their n20+n50 groups, mirroring `randlen_test`'s own pooling.
    "randlen_piled": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_piled_all.yaml",
    ),
    "randlen_scattered": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_scattered_all.yaml",
    ),
    "randlen_mixed": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_n20_mixed.yaml",
    ),
}

VALUE_FNS = ("lyapunov", "mass_in_region", "signed_mass")
GOAL_NAMES = ("random_quadrant", "ring_O", "T")


def _load_predictor(spec: dict):
    os.environ[spec["ckpt_env"]] = spec["ckpt"]
    mod = importlib.import_module(spec["module"])
    factory = getattr(mod, spec["factory"])
    return factory(**spec.get("kwargs", {}))


def _load_cell(corpus_spec: dict, tag: str):
    if corpus_spec["kind"] == "slate":
        # train_cfg is unused here (only needed to fit the linear/mean-delta
        # reference operators, out of scope for this report) but load_cell
        # doesn't require it as a pair -- only the eval_cfg matters.
        return load_cell(corpus_spec["eval_cfg"], "train",
                          manifest_path=corpus_spec["manifest"], tag=tag)
    return load_randlen_cell(corpus_spec["cfg"], "train", tag=tag)


_TRUTH_FIELDS = ("occ1", "states_")


def _predict(predictor, cell, device: str = "cpu", chunk: int = 1024,
             move_inputs: bool = True) -> torch.Tensor:
    """predictor.predict_occ over the whole cell, on `device`, in row chunks,
    returned on CPU. On CPU this is exactly the historical single call. On
    cuda every per-row tensor field of the batch is sliced and moved (truth
    fields are sliced but never moved -- predictors never read them), which
    fixes `eval-baseline-scorer-batch-on-requested-device` for this harness
    (predictors take their compute device from `batch.occ0.device`)."""
    import dataclasses
    batch = _predictor_batch(cell) if hasattr(cell, "states") else cell
    if device == "cpu" and move_inputs:
        return predictor.predict_occ(batch).to(torch.float32)
    N = batch.occ0.shape[0]
    outs = []
    for i in range(0, N, chunk):
        upd = {}
        for f in dataclasses.fields(batch):
            v = getattr(batch, f.name)
            if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == N:
                v = v[i:i + chunk]
                if f.name not in _TRUTH_FIELDS and move_inputs:
                    v = v.to(device)
                upd[f.name] = v
        with torch.no_grad():
            outs.append(predictor.predict_occ(dataclasses.replace(batch, **upd)).to(torch.float32).cpu())
    return torch.cat(outs)


def _accuracy(model_spec: dict, predictor, cell, device: str = "cpu") -> float:
    H, W = cell.occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    # GNN predictors are ALWAYS run the historical way (one unchunked call, CPU
    # inputs): `GNNPredictor.predict_occ` seeds its node sampling by the row's
    # POSITION in the batch (`sample_nodes_xy(..., seed=i)`), so chunking changes
    # its predictions (found 2026-09-24 while adding --device; invariant
    # `gnn-node-sampling-consistent-within-state`). It picks its own device.
    pred = _predict(predictor, cell, "cpu") if model_spec["is_gnn"] else _predict(predictor, cell, device)

    if model_spec["is_gnn"]:
        n_particles = predictor.n_particles
        raw = cell.raw
        occ0_np, occ1_np = cell.occ0.numpy(), cell.occ1.numpy()
        n = occ0_np.shape[0]
        truth_rs = np.stack([
            resample_occupancy_through_nodes(occ1_np[i], raw.to_pxl, raw.ctr_in_PXL,
                                              n_particles=n_particles, seed=i)
            for i in range(n)
        ])
        prev_rs = np.stack([
            resample_occupancy_through_nodes(occ0_np[i], raw.to_pxl, raw.ctr_in_PXL,
                                              n_particles=n_particles, seed=10_000_000 + i)
            for i in range(n)
        ])
        truth = torch.from_numpy(truth_rs).to(torch.float32)
        prev = torch.from_numpy(prev_rs).to(torch.float32)
    else:
        truth, prev = cell.occ1.to(torch.float32), cell.occ0.to(torch.float32)

    return metrics(pred, truth, prev, region=region)["accuracy"], pred


def _accuracy_canonical(model_spec: dict, predictor, cell, pred_world: torch.Tensor):
    """EXP-0022 A1: swept-region `accuracy` scored in the CANONICAL push
    frame instead of the world frame, as an ADDITIONAL reporting frame (this
    function does not touch `_accuracy` above; world-frame numbers are
    unaffected).

    This is NOT a symmetric comparison and is not treated as one:
      - a model that natively predicts in the canonical frame
        (`WarpedNFDPredictor`, exposing `.predict_occ_canonical`) is scored
        on that native output -- ZERO extra resamplings beyond the one warp
        every canonical model pays to enter its own frame.
      - every other model (world-frame NFD, the fitted linear operators)
        only ever produces a WORLD-frame prediction, so its already-computed
        `pred_world` is warped ONCE into the canonical frame to be scored
        here -- one extra `grid_sample` it would not otherwise pay.
    So a native-canonical model is structurally favoured in THIS frame,
    exactly as a world-frame model is structurally favoured when scored in
    the world frame (`_accuracy` above pays the warped model two
    resamplings + a validity blend it doesn't need). Report both frames
    side by side; do not average them into one number.

    Truth/prev/region are built once, in the canonical frame, using
    `canon_res = H` (the batch's own grid resolution) and `scale = 1.0` --
    matching `nfd_warped_randlen`'s own training-time default
    (`canon_res=None` -> batch's own H) and the LinearForesight warp
    convention (`fit_linear_foresight.py` never passes `scale`, so its
    default 1.0 applies) -- so every model in a report is scored against the
    SAME canonical grid.

    GNN models are skipped by the caller (`is_gnn` is True there and the
    node-resampling bottleneck doesn't compose cleanly with a second,
    per-model canonical frame in the time available -- see A1 in the task
    brief).
    """
    H, W = cell.occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    canon_res = H
    scale = 1.0

    native = hasattr(predictor, "predict_occ_canonical")
    if native:
        batch = _predictor_batch(cell) if hasattr(cell, "states") else cell
        pred_canon, s_px_n, e_px_n, canon_res, scale = predictor.predict_occ_canonical(batch)
        # Sanity check once per call: the predictor's own pixel derivation
        # (from batch.p_start/p_stop via raw.to_pxl+ctr_in_PXL) must agree
        # with actions_to_pixels' (from cell.actions via workspace bounds) --
        # both are supposed to describe the same push, just derived two
        # different ways (see `fit_linear_foresight.py::verify_pixel_mapping`).
        max_px_diff = float((s_px_n - s_px).abs().max())
        assert max_px_diff < 1.0, (
            f"canonical-frame pixel convention mismatch: {max_px_diff:.3f} px "
            f"(native vs actions_to_pixels) -- would silently misalign the "
            f"canonical truth/region against a native-canonical prediction")
        s_px, e_px = s_px_n, e_px_n
    else:
        pred_canon = to_push_frame(pred_world.to(torch.float32), s_px, e_px,
                                    (canon_res, canon_res), scale)

    region_world = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
    region_canon = (to_push_frame(region_world.to(torch.float32), s_px, e_px,
                                   (canon_res, canon_res), scale) > 0.5).to(torch.float32)
    truth_canon = to_push_frame(cell.occ1.to(torch.float32), s_px, e_px,
                                 (canon_res, canon_res), scale)
    prev_canon = to_push_frame(cell.occ0.to(torch.float32), s_px, e_px,
                                (canon_res, canon_res), scale)

    acc = metrics(pred_canon.to(torch.float32), truth_canon, prev_canon, region=region_canon)["accuracy"]
    return acc, native


GOAL_SETS = ("default", "many", "many_plus")
# Opt-in goals that are NOT part of the 30-goal "many" set (other results
# depend on that set staying fixed); "many_plus" = "many" + these (EXP-0046).
EXTRA_GOALS = ("two_squares",)
MANY_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def goal_names(goal_set: str = "default") -> tuple[str, ...]:
    """Goal names, in report order, for a goal set.

    "default": the original three (a per-slate seeded random quadrant, the
    letters O and T). "many" (EXP-0027): the SAME family widened -- all 26
    `helvetica_thin` letters plus all 4 quadrants, fixed across slates. The
    default is unchanged so every existing row stays comparable."""
    if goal_set == "default":
        return GOAL_NAMES
    if goal_set == "many":
        return tuple(f"letter_{c}" for c in MANY_LETTERS) + tuple(f"quadrant_{q}" for q in range(4))
    if goal_set == "many_plus":
        return goal_names("many") + EXTRA_GOALS
    raise ValueError(f"unknown goal set {goal_set!r}; known: {GOAL_SETS}")


def _fixed_goal(name: str, H: int, W: int):
    """(mask, dist) for a slate-independent goal name."""
    if name == "ring_O":
        m = letter_mask("O", H, W)
    elif name.startswith("letter_"):
        m = letter_mask(name[len("letter_"):], H, W)
    elif name.startswith("quadrant_"):
        m = quadrant_mask(H, W, int(name[len("quadrant_"):]))
    elif name == "two_squares":
        m = two_squares_mask(H, W)
    else:
        m = letter_mask(name, H, W)
    return (torch.from_numpy(m.astype(np.float32)),
            torch.from_numpy(dist_field_from_mask(m > 0)).float())


def _goals_for_slate(goal_set: str, H: int, W: int, sid: int, cache: dict):
    """[(name, mask, dist)] for one slate; fixed goals are built once."""
    out = []
    for name in goal_names(goal_set):
        if name == "random_quadrant":
            q_np, _ = random_quadrant_mask(H, W, seed=int(sid))
            out.append((name, torch.from_numpy(q_np.astype(np.float32)),
                        torch.from_numpy(dist_field_from_mask(q_np)).float()))
        else:
            if name not in cache:
                cache[name] = _fixed_goal(name, H, W)
            out.append((name, *cache[name]))
    return out


TRUTH_SCORINGS = ("soft", "image")


def truth_for_scoring(cell, rows: torch.Tensor) -> torch.Tensor:
    """Ground-truth post-push images for `rows`, redrawn from each row's stored
    particle states with the mass-conserving scoring rasteriser
    (`transforms.functional.splat_particles_mass`), in the dataset's own pixel
    frame: u = pos * raw.to_pxl + raw.ctr_in_PXL - 1.0, dim 0 = world x
    (matching `_draw_particle_grid`'s final transpose). The -1.0 is MEASURED,
    not derived: the dataset draws `int()`-truncated centres and fills
    `cv2.boxPoints` polygons, and on L40mm / randlen_test step-0 rows a -0.5
    offset left the soft centroid +0.47/+0.30 and +0.57/+0.62 px above the
    hard image's (dim 0 / dim 1); -1.0 centres it (EXP-0027). One particle carries mass 1 -- value functions here are either
    mass-normalised (lyapunov) or scored through slateN, which is scale-free.

    Why: the dataset's own `occ1` is a hard rasterisation whose per-particle
    pixel count depends on sub-pixel position; it adds noise of ~12% of the
    between-action spread to every true dv (EXP-0027 RUN-0005). See
    experiments/METRICS.md, "Ground-truth scoring"."""
    from Baselines.common.data import _resolve_sample
    from transforms.functional import splat_particles_mass
    raw = cell.raw
    H, W = cell.occ0.shape[-2:]
    ctr = torch.as_tensor(raw.ctr_in_PXL[:2], dtype=torch.float32)
    out = torch.zeros(len(rows), H, W)
    for k, i in enumerate(rows.tolist()):
        r, smp = _resolve_sample(raw, int(i))
        pos = torch.as_tensor(raw.runs[r]["states_"][smp][:, :2], dtype=torch.float32)
        uv = pos * float(raw.to_pxl) + ctr - 1.0
        out[k] = splat_particles_mass(uv[None], (H, W))[0]
    return out


def _capture_report(cell, pred: torch.Tensor, goal_set: str = "default",
                    truth_s0: torch.Tensor | None = None) -> dict:
    """Per (goal, value_fn) capture, averaged over step-0 same-state pools,
    plus a per-value-fn average over goals. `truth_s0`: step-0 ground-truth
    images to score against (default: the dataset's hard `occ1`; pass
    `truth_for_scoring(...)` for the soft, mass-conserving truth)."""
    step0 = (cell.step_idx == 0)
    slate_ids = cell.slate_idx[step0]
    occ1_s0 = (cell.occ1[step0] if truth_s0 is None else truth_s0).to(torch.float32)
    pred_s0 = pred[step0].to(torch.float32)
    unique_slates = slate_ids.unique().tolist()
    H, W = cell.occ0.shape[-2:]
    names = goal_names(goal_set)
    cache = {}

    per_goal = {g: {v: [] for v in VALUE_FNS} for g in names}
    for sid in unique_slates:
        rows = (slate_ids == sid).nonzero(as_tuple=True)[0]
        true_pool = occ1_s0[rows]
        pred_pool = pred_s0[rows]

        for goal, mask, dist in _goals_for_slate(goal_set, H, W, sid, cache):
            v_true = lyapunov(true_pool, dist)
            v_pred = lyapunov(pred_pool, dist)
            per_goal[goal]["lyapunov"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=False))

            v_true = mass_in_region(true_pool, mask)
            v_pred = mass_in_region(pred_pool, mask)
            per_goal[goal]["mass_in_region"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=True))

            v_true = signed_mass_in_region(true_pool, mask)
            v_pred = signed_mass_in_region(pred_pool, mask)
            per_goal[goal]["signed_mass"].append(
                slate_n_capture(v_pred, v_true, higher_is_better=True))

    def _mean(xs):
        xs = [x for x in xs if x == x]  # drop NaN
        return float(np.mean(xs)) if xs else float("nan")

    # `per_slate` keeps the raw per-slate capture (same order as `slate_ids`),
    # so a paired, slate-resampled comparison between two models is possible
    # (EXP-0026) -- the summary means below are unchanged by it.
    out = {"n_slates": len(unique_slates), "goal_set": goal_set,
           "truth_scoring": "image" if truth_s0 is None else "soft",
           "per_goal": {}, "averaged_over_goals": {},
           "slate_ids": [int(s) for s in unique_slates],
           "per_slate": {g: {v: [float(x) for x in per_goal[g][v]] for v in VALUE_FNS}
                         for g in names}}
    for goal in names:
        out["per_goal"][goal] = {v: _mean(per_goal[goal][v]) for v in VALUE_FNS}
    for v in VALUE_FNS:
        out["averaged_over_goals"][v] = _mean(
            [out["per_goal"][g][v] for g in names])
    return out


def _random_floor_capture(cell, n_seeds: int = 20, goal_set: str = "default",
                          truth_s0: torch.Tensor | None = None) -> dict:
    """`random` ranking floor: score every step-0 pool with i.i.d. random
    noise images standing in for a "prediction" -- the induced `v_pred` is
    then unrelated to `v_true`, so `slate_n_capture`'s own argmax reduces to
    a uniform-random pick from the pool. `slate_n_capture`'s definition
    (`chosen - mean_true) / (best_true - mean_true)`) has EXPECTATION
    EXACTLY 0 under a uniform random pick, since `E[true[uniform idx]] =
    mean_true` over the pool -- so this is a Monte-Carlo check of an exact
    analytic fact, not an independent floor to be trusted on its own."""
    names = goal_names(goal_set)
    accs = {g: {v: [] for v in VALUE_FNS} for g in names}
    for seed in range(n_seeds):
        g = torch.Generator().manual_seed(seed)
        noise = torch.rand(cell.occ0.shape, generator=g)
        rep = _capture_report(cell, noise, goal_set, truth_s0)
        for goal in names:
            for v in VALUE_FNS:
                accs[goal][v].append(rep["per_goal"][goal][v])
    out = {"n_slates": None, "per_goal": {}, "averaged_over_goals": {}}
    for goal in names:
        out["per_goal"][goal] = {v: float(np.nanmean(accs[goal][v])) for v in VALUE_FNS}
    for v in VALUE_FNS:
        out["averaged_over_goals"][v] = float(np.mean(
            [out["per_goal"][g][v] for g in names]))
    return out


def _print_capture(tag: str, capture: dict) -> None:
    goals = list(capture["per_goal"])
    for goal in (goals if len(goals) <= len(GOAL_NAMES) else []):
        row = capture["per_goal"][goal]
        print(f"    goal={goal:16s} "
              f"lyapunov={row['lyapunov']:+.4f}  "
              f"mass_in_region={row['mass_in_region']:+.4f}  "
              f"signed_mass={row['signed_mass']:+.4f}")
    avg = capture["averaged_over_goals"]
    print(f"    {'averaged':16s} "
          f"lyapunov={avg['lyapunov']:+.4f}  "
          f"mass_in_region={avg['mass_in_region']:+.4f}  "
          f"signed_mass={avg['signed_mass']:+.4f}"
          + (f"  ({len(goals)} goals)" if len(goals) > len(GOAL_NAMES) else ""))


def _write_json_atomic(path: str, obj) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="Baselines/common/runs/cross_corpus_report")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--corpora", default=",".join(CORPORA))
    ap.add_argument("--goal-set", default="default", choices=GOAL_SETS,
                     help="slateN goal set: 'default' = the original 3 goals; 'many' = "
                          "26 letters + 4 quadrants (EXP-0027), the same family widened")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                     help="where predictors run (chunked). 'cpu' reproduces every earlier report "
                          "exactly; 'cuda' is much faster (G2c).")
    ap.add_argument("--truth-scoring", default="soft", choices=TRUTH_SCORINGS,
                     help="how TRUE post-push outcomes are rasterised for slateN: 'soft' (default "
                          "since 2026-09-24) = redrawn from particle states with the mass-conserving "
                          "splat; 'image' = the dataset's hard occ1 (every slateN before 2026-09-24). "
                          "Model predictions are scored as images either way.")
    ap.add_argument("--ckpt", action="append", default=[], metavar="MODEL=PATH",
                     help="score MODEL with checkpoint PATH instead of its MODELS default (repeatable). "
                          "Setting the model's env var (e.g. NFD_CKPT) does NOT work: the loader "
                          "writes spec['ckpt'] into that variable before building.")
    ap.add_argument("--no-reference", action="store_true",
                     help="skip the persistence/random reference rows (they are ON by default)")
    ap.add_argument("--canonical-frame", action="store_true",
                     help="EXP-0022 A1: also score swept-region accuracy in the "
                          "canonical push frame, as an ADDITIONAL reporting frame "
                          "alongside (not instead of) the world-frame accuracy above. "
                          "Skipped for GNN models (is_gnn=True). Off by default so the "
                          "existing world-frame-only report is unchanged.")
    args = ap.parse_args()
    for kv in args.ckpt:
        name, path = kv.split("=", 1)
        MODELS[name] = dict(MODELS[name], ckpt=path)
        print(f"[ckpt override] {name} -> {path}")

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    model_names = args.models.split(",")
    corpus_names = args.corpora.split(",")

    results = {}
    for corpus_name in corpus_names:
        corpus_spec = CORPORA[corpus_name]
        t0 = time.time()
        cell = _load_cell(corpus_spec, tag=corpus_name)
        print(f"[{corpus_name}] loaded {cell.occ0.shape[0]} transitions "
              f"({time.time() - t0:.1f}s)")
        truth_s0 = None
        if args.truth_scoring == "soft":
            truth_s0 = truth_for_scoring(cell, (cell.step_idx == 0).nonzero(as_tuple=True)[0])

        if not args.no_reference:
            # persistence: accuracy is 0 by construction (metrics()'s own
            # denominator); its slateN row is a DEGENERATE ranker (predicts
            # dv=0 for every candidate, so slate_n_capture's argmax just
            # picks whatever index torch.argmax ties to -- not a meaningful
            # ranking baseline, kept only to show that degeneracy).
            H, W = cell.occ0.shape[-2:]
            s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min,
                                            cell.workspace_max, (H, W))
            plate_px = 0.04 / 0.128 * W
            region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
            acc_persist = metrics(cell.occ0.to(torch.float32), cell.occ1.to(torch.float32),
                                   cell.occ0.to(torch.float32), region=region)["accuracy"]
            capture_persist = _capture_report(cell, cell.occ0, args.goal_set, truth_s0)
            print(f"[{corpus_name} / persistence] accuracy={acc_persist:.4f} "
                  f"(reference floor; slateN row below is DEGENERATE, see docstring)")
            _print_capture(corpus_name, capture_persist)

            capture_random = _random_floor_capture(cell, goal_set=args.goal_set, truth_s0=truth_s0)
            print(f"[{corpus_name} / random] accuracy=n/a (ranking-only floor)")
            _print_capture(corpus_name, capture_random)

            results.setdefault(corpus_name, {})["persistence"] = {
                "accuracy": acc_persist, "capture": capture_persist}
            results.setdefault(corpus_name, {})["random"] = {
                "accuracy": None, "capture": capture_random}

        for model_name in model_names:
            model_spec = MODELS[model_name]
            t0 = time.time()
            predictor = _load_predictor(model_spec)
            device = "cpu" if model_spec["is_gnn"] else args.device
            acc, pred = _accuracy(model_spec, predictor, cell, args.device)
            capture = _capture_report(cell, pred, args.goal_set, truth_s0)
            dt = time.time() - t0
            print(f"[{corpus_name} / {model_name}] accuracy={acc:.4f}  device={device}  "
                  f"({dt:.1f}s)")
            _print_capture(corpus_name, capture)

            row = {"accuracy": acc, "capture": capture, "device": str(device)}

            if args.canonical_frame:
                if model_spec.get("is_gnn"):
                    print(f"[{corpus_name} / {model_name}] canonical-frame accuracy: "
                          f"SKIPPED (GNN node-resampling path, see A1 scope note)")
                else:
                    acc_canon, native = _accuracy_canonical(model_spec, predictor, cell, pred)
                    frame_note = "native (no unwarp)" if native else "warped-once from world pred"
                    print(f"[{corpus_name} / {model_name}] accuracy_canonical={acc_canon:.4f}  "
                          f"({frame_note})")
                    row["accuracy_canonical"] = acc_canon
                    row["canonical_native"] = native

            results.setdefault(corpus_name, {})[model_name] = row
            # checkpoint after every model: a cut-off run keeps everything
            # finished so far (atomic replace, never a half-written file)
            _write_json_atomic(f"{args.out_prefix}.json", results)

    out_path = f"{args.out_prefix}.json"
    _write_json_atomic(out_path, results)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
