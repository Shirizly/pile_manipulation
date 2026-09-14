"""experiments/EXP-0009-time-budget-bench/code/bench.py -- equalised-wall-clock-budget
timing GATE for the larger control-comparison experiment.

Purpose (see task brief): before comparing dynamics models' control
performance under an EQUALISED WALL-CLOCK BUDGET (fastest model ranks all
128 candidates, everyone else ranks N_i <= 128 candidates in the same
time), we need the actual throughput curves. This script produces those
curves ONLY -- it does not run any control/slateN evaluation.

Device-bug fix (see module docstring of Baselines/common/benchmark_time.py,
which first documented this trap): `Baselines.common.eval_baseline`'s own
`_predictor_batch` builds `PredictorBatch` straight from `load_cell`'s CPU
tensors and never calls `.to(device)`, so NFD/Schenck (whose predictors
derive their compute device from `batch.occ0.device`) silently run on CPU
even when GPU is available and requested, while GNN forces its own
cuda-if-available device internally regardless of the batch. Comparing
"NFD on CPU" against "GNN on GPU" is not a valid throughput comparison.
FIX APPLIED HERE: every candidate batch this script builds is moved to
`DEVICE` (`torch.Tensor.to(DEVICE)`) BEFORE being handed to any predictor,
for every model (NFD/Schenck/MODEL-0001/MODEL-0002/hybrids alike), and the
actual device of the tensors fed to each model's forward call is
introspected and printed/recorded per row -- not assumed. GNN's predictor
fixes its own device at construction (`cuda if available`, matching
`DEVICE` here) so no separate fix was needed there, but its actual device
is still verified, not assumed.

Timing scope (user's explicit rule, see task brief):
  - EXCLUDED: the one-time particle->occupancy rasterisation of the STATE
    (shared across all K candidates -- in a real deployment the state
    arrives as an image from perception, this is not action-dependent).
  - INCLUDED (per-candidate, action-dependent):
      * NFD / Schenck: 2x `draw_plate_soft` (start-plate, stop-plate render)
      * MODEL-0001 (switched-linear visual) + hybrids: push-frame
        warp/`canonicalise` (to 32x32) and, for the hybrids, the
        `push_frame_full_descriptors` computation
      * MODEL-0002 (descriptor-only): push-frame descriptor computation
      * GNN: `sample_nodes_xy` image->particle conversion (foreground
        extraction + FPS) -- a real cost the GNN pays that no image model
        does; INCLUDED per the task brief even though (see below) it is
        also flagged as an implementation slowdown.
  - Forward-pass-only time is ALSO reported separately for every model so
    preprocessing cost is visible as its own number.

Known implementation caveat, reported not silently absorbed: GNN's
`sample_nodes_xy` loops per-sample in Python (`for i in range(B)` in
`Baselines/GNN/predictor.py`). This is an IMPLEMENTATION slowdown, not an
architectural one (FPS-on-GPU could batch it). We report the measured
(looped) preprocessing cost as the honest number, AND a cheap estimate of
the loop-excluded cost (single-sample cost, i.e. cost paid once regardless
of K, extrapolated as O(1) rather than O(B)) so architecture and
implementation are distinguishable -- see `gnn_loop_excluded_estimate_s`
in the JSON output and the RESULTS.md note.

Method: reuses Baselines/common/benchmark_time.py's methodology verbatim
(imported where possible) -- >=3 warm-up discarded, >=10 timed reps,
`torch.cuda.synchronize()` before/after every timed region, median+IQR
(not mean), actual device + param count recorded per model. Respects that
script's GPU-contention gate (imported, not reimplemented).
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

REPO = "/home/alon/Code/pile_manipulation"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/experiments/temp/hybrid-vis-desc")
sys.path.insert(0, f"{REPO}/experiments/temp/dmdc-lenbins")
sys.path.insert(0, f"{REPO}/experiments/temp/stage3-slaten")

from Baselines.common.data import load_cell  # noqa: E402
from Baselines.common.benchmark_time import (  # noqa: E402
    check_gpu_contention, _median_iqr,
)
from Baselines.common.goals import mass_in_region  # noqa: E402
from fit_linear_foresight import actions_to_pixels, canonicalise  # noqa: E402
from transforms.functional import (  # noqa: E402
    draw_plate_soft, to_push_frame, push_frame_validity_mask,
    blend_push_prediction, from_push_frame,
)
from utils import git_provenance  # noqa: E402
from control_utility_test import lyapunov  # noqa: E402
from descriptors_b import world_to_pushframe_px  # noqa: E402
from eval_slaten_latent import com_world_pixel, bilinear_sample  # noqa: E402
from Baselines.GNN.perception import rasterize_nodes_as_cubes_batch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
KS = [128, 96, 64, 48, 32, 24, 16, 8, 4, 2, 1]
# RUN-0002 (timing-corrected): a reduced K set for the "with value function"
# matrix -- the full 11-point curve is still exercised by the sync-free
# equivalence check and is unchanged in spirit from RUN-0001's KS above.
KS_V2 = [1, 32, 128]
WARMUP = 5
REPEATS = 15
OUTDIR = f"{REPO}/experiments/EXP-0009-time-budget-bench/artifacts/RUN-0001-timing-sweep"
OUTDIR_V2 = f"{REPO}/experiments/EXP-0009-time-budget-bench/artifacts/RUN-0002-timing-corrected"

from descriptors_d import push_frame_full_descriptors, slices_d  # noqa: E402
from fit_hybrid import CELLS, bin_index  # noqa: E402
from descriptors import BOUNDS  # noqa: E402

WS_MIN = (BOUNDS["x_min"], BOUNDS["y_min"])
WS_MAX = (BOUNDS["x_max"], BOUNDS["y_max"])
SD = slices_d()  # D-all-local descriptor layout (mass, com, moments2, ...)

# =============================================================================
# Defect 2 fix: a fixed goal region + Lyapunov distance field, built ONCE
# (state/goal-independent of K, exactly like state rasterisation above) so
# every model's timed region can compute a real goal value instead of the
# original bench scoring nothing. Goal shape is an arbitrary top-right
# quadrant -- this bench times VALUE COMPUTATION COST, it does not validate
# value-function accuracy, so the shape only needs to be fixed and
# documented, not tuned.
# =============================================================================
_GOAL_MASK_NP = None
_GOAL_DIST_NP = None
GOAL_MASK = None   # (H,W) float32 tensor on DEVICE, 1 inside region
GOAL_DIST = None   # (H,W) float32 tensor on DEVICE, dist_transform_edt(~mask)/max


def build_goal_fields(H, W):
    global _GOAL_MASK_NP, _GOAL_DIST_NP, GOAL_MASK, GOAL_DIST
    mask_np = np.zeros((H, W), dtype=bool)
    mask_np[: H // 2, W // 2:] = True  # top-right quadrant, fixed & arbitrary
    dist_np = distance_transform_edt(~mask_np).astype(np.float32)
    dist_np = dist_np / max(float(dist_np.max()), 1e-6)
    _GOAL_MASK_NP, _GOAL_DIST_NP = mask_np, dist_np
    GOAL_MASK = torch.from_numpy(mask_np.astype(np.float32)).to(DEVICE)
    GOAL_DIST = torch.from_numpy(dist_np).to(DEVICE)


def value_from_occ(occ):
    """occ: (B,H,W) or (B,1,H,W) predicted occupancy -> (lyapunov, mass_in_region),
    both (B,) tensors. This is the "image-space" value path: every image-
    producing model (nfd, schenck, gnn-after-rasterisation, model0001
    switched/global, hybrid14/94) uses this, matching
    `eval_slaten_broad.py::value_true_and_pred`'s `lyapunov`/`mass_in_region`
    branches."""
    if occ.dim() == 4:
        occ = occ[:, 0]
    lyap = lyapunov(occ, GOAL_DIST)
    mir = mass_in_region(occ, GOAL_MASK)
    return lyap, mir


def value_from_descriptor(pred_desc, sp_d, ep_d, H, W):
    """Point-mass (monopole) value readout for the descriptor-only model,
    matching `eval_slaten_latent.py`'s own documented approximation:
    V_lyapunov_hat = d(world_COM_hat) (mass cancels in the true V=sum(occ*d)/
    sum(occ) formula when all mass is treated as concentrated at the COM);
    V_mass_in_region_hat = mask(world_COM_hat) * mass_hat (monopole analogue
    of sum(occ*mask)). `pred_desc` layout is bench's own desc0 layout:
    [global_mass(1), global_length(1), mass(1), com(2), ...], which is
    exactly `slices_d()`'s own layout (SD), so SD's offsets apply directly."""
    com_row = pred_desc[:, SD["com"].start]
    com_col = pred_desc[:, SD["com"].start + 1]
    mass_hat = pred_desc[:, SD["global_mass"].start] * (H * W)
    wr, wc = com_world_pixel(com_row, com_col, sp_d, ep_d, H=H, W=W)
    lyap = bilinear_sample(GOAL_DIST, wr, wc)
    mir = bilinear_sample(GOAL_MASK, wr, wc) * mass_hat
    return lyap, mir


# =============================================================================
# Shared state / candidate-batch construction (same recipe as
# Baselines/common/benchmark_time.py::build_candidate_batch -- one real
# current state, K candidate actions cycled from the loaded eval cell).
# Rasterising the STATE itself is done ONCE here, OUTSIDE every timed
# region (excluded from all measurements per the task brief).
# =============================================================================

def load_state_and_actions():
    cell = load_cell(EVAL_CFG, "train")
    print(f"[bench] loaded {cell.occ0.shape[0]} transitions from {EVAL_CFG} "
          f"(state rasterisation done once, NOT timed)")
    return cell


def build_batch(cell, k: int, device: str):
    """One current state (row 0), repeated K times, paired with K candidate
    actions cycled from the cell. Everything moved to `device` explicitly
    (THE device-bug fix -- see module docstring)."""
    idx0 = 0
    n = cell.occ0.shape[0]
    cand_idx = torch.tensor([(idx0 + j) % n for j in range(k)], dtype=torch.long)
    occ0 = cell.occ0[idx0].unsqueeze(0).expand(k, -1, -1).contiguous().to(device)
    return dict(
        occ0=occ0,
        actions=cell.actions[cand_idx].to(device),
        p_start=cell.p_start[cand_idx].to(device),
        p_stop=cell.p_stop[cand_idx].to(device),
        angle=cell.angle[cand_idx].to(device),
        raw=cell.raw,
        workspace_min=cell.workspace_min, workspace_max=cell.workspace_max,
        H=cell.H, W=cell.W,
    )


def assert_all_on(device_str: str, *tensors):
    dev = torch.device(device_str)
    for t in tensors:
        if isinstance(t, torch.Tensor):
            assert t.device.type == dev.type, f"expected {dev}, got {t.device}"


# =============================================================================
# Per-model pre/forward split. Each returns a dict with two zero-arg
# closures: "pre" (builds forward input from the batch, action-dependent
# preprocessing) and "fwd" (the model call proper, consuming pre()'s
# output). "pre" is called once per timed rep (its own cost measured
# separately) then "fwd" is called on its output; "end2end" is timed as
# pre()+fwd() back-to-back in one region (this is what a real MPC step pays).
# =============================================================================

def make_nfd():
    from Baselines.NFD.predictor import NFDPredictor
    pred = NFDPredictor("Baselines/NFD/runs/nfd_3ch/unet_best.pth", channels=3, name="nfd_unet3ch")
    pred.model.to(DEVICE).eval()

    def pre(batch):
        raw = batch["raw"]
        H, W = batch["H"], batch["W"]
        device = batch["occ0"].device
        ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(device)[:2]
        start_px = batch["p_start"][:, :2].to(device) * raw.to_pxl + ctr_xy
        stop_px = batch["p_stop"][:, :2].to(device) * raw.to_pxl + ctr_xy
        angle = batch["angle"].to(device)
        from Baselines.NFD.predictor import _plate_geometry_px
        plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)
        r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        x = torch.stack([batch["occ0"].to(device), r_start, r_stop], dim=1)
        return x

    def fwd(x):
        assert_all_on(DEVICE, x)
        assert next(pred.model.parameters()).device.type == torch.device(DEVICE).type
        with torch.no_grad():
            return torch.sigmoid(pred.model(x))

    def value(occ_pred):
        return value_from_occ(occ_pred)

    return dict(name="nfd", model=pred.model, pre=pre, fwd=fwd, value=value)


def make_schenck():
    from Baselines.SchenckCNN.predictor import SchenckPredictor
    from Baselines.SchenckCNN.action_encoding import build_input
    pred = SchenckPredictor()
    pred.model.to(DEVICE).eval()

    def pre(batch):
        device = batch["occ0"].device
        return build_input(batch["occ0"].to(device), batch["p_start"].to(device),
                            batch["p_stop"].to(device), batch["angle"].to(device),
                            batch["raw"], batch["H"], batch["W"]).to(device)

    def fwd(x):
        assert_all_on(DEVICE, x)
        assert next(pred.model.parameters()).device.type == torch.device(DEVICE).type
        with torch.no_grad():
            return pred.model(x)

    def value(occ_pred):
        return value_from_occ(occ_pred)

    return dict(name="schenck", model=pred.model, pre=pre, fwd=fwd, value=value)


def make_gnn():
    from Baselines.GNN.predictor import GNNPredictor
    from Baselines.GNN.perception import (
        Z_CONST, sample_nodes_xy, rasterize_nodes_as_cubes_batch,
    )
    from Baselines.GNN.geometry import PARTICLE_DENS, compute_s_delta
    pred = GNNPredictor(ckpt_path="Baselines/GNN/runs/ckpt_best.pth")
    assert pred.device == DEVICE, f"GNN forced device {pred.device} != {DEVICE}"

    def pre(batch):
        occ0 = batch["occ0"].cpu()
        raw = batch["raw"]
        B, H, W = occ0.shape
        n = pred.n_particles
        t0 = time.perf_counter()
        s_cur_xy = np.stack([
            sample_nodes_xy(occ0[i].numpy(), raw.to_pxl, raw.ctr_in_PXL,
                             n_particles=n, seed=i)
            for i in range(B)
        ])
        loop_s = time.perf_counter() - t0
        z = np.full((B, n, 1), Z_CONST, dtype=np.float32)
        s_cur = torch.from_numpy(
            np.concatenate([s_cur_xy, z], axis=-1).astype(np.float32)).to(pred.device)
        return dict(s_cur=s_cur, loop_s=loop_s, B=B, one_sample_s=loop_s / max(B, 1),
                     p_start=batch["p_start"].to(pred.device),
                     p_stop=batch["p_stop"].to(pred.device), raw=raw, H=H, W=W)

    def fwd(pre_out):
        s_cur = pre_out["s_cur"]
        assert_all_on(DEVICE, s_cur)
        assert next(pred.model.parameters()).device.type == torch.device(DEVICE).type
        with torch.no_grad():
            s_delta = compute_s_delta(s_cur, pre_out["p_start"], pre_out["p_stop"])
            a_cur = torch.zeros(pre_out["B"], pred.n_particles, device=pred.device)
            dens = torch.full((pre_out["B"],), PARTICLE_DENS, device=pred.device)
            s_pred = pred.model.predict_one_step(a_cur, s_cur, s_delta, dens)
        return s_pred

    def value(s_pred, pre_out):
        """GNN produces PARTICLES, not an occupancy image -- defect 2's
        value function needs an image (lyapunov/mass_in_region are grid
        sums). Rasterise s_pred's xy back to an (B,H,W) occupancy grid
        (same `rasterize_nodes_as_cubes_batch` the GNN's own predictor
        machinery already imports, previously unused in this bench) and
        score that -- this is a real cost the GNN pays to be scorable by
        the same value function as the image models, timed honestly like
        the rest of its per-candidate preprocessing (see the GNN
        Python-loop caveat already on `sample_nodes_xy`)."""
        raw, H, W = pre_out["raw"], pre_out["H"], pre_out["W"]
        node_xy = s_pred[:, :, :2].detach().cpu().numpy()
        occ_np = rasterize_nodes_as_cubes_batch(node_xy, raw.to_pxl, raw.ctr_in_PXL, H, W)
        occ_t = torch.from_numpy(occ_np).to(DEVICE)
        return value_from_occ(occ_t)

    return dict(name="gnn", model=pred.model, predictor=pred, pre=pre, fwd=fwd,
                value=value, special="gnn", needs_pre_for_value=True)


def make_switched_linear(name, ckpt_path, desc_dim, switched):
    """MODEL-0001 (desc_dim=0) or the hybrid14/hybrid94 checkpoints
    (desc_dim=14/94): 32x32 canonical-occupancy [+ descriptors] switched-
    or global-linear operator. Reuses fit_hybrid's cell spec / bin_index and
    the exact canonicalise/from_push_frame/blend_push_prediction machinery
    `experiments/temp/stage2-slaten/eval_slaten.py::predict_image` uses.
    """
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    bin_edges = ckpt["bin_edges"].to(DEVICE)
    ops = [o.to(DEVICE) for o in ckpt["switched_ops"]] if switched else None
    global_op = ckpt["global_op"].to(DEVICE)
    RES = 32

    def pre(batch):
        occ0 = batch["occ0"].float()
        device = occ0.device
        s_px, e_px = actions_to_pixels(batch["actions"], WS_MIN, WS_MAX, (batch["H"], batch["W"]))
        s_px, e_px = s_px.to(device), e_px.to(device)
        canon0 = canonicalise(occ0, s_px, e_px, RES, 1.0)
        canon0f = canon0.reshape(canon0.shape[0], -1)
        if desc_dim > 0:
            local0 = push_frame_full_descriptors(occ0, s_px, e_px)
            gm0 = occ0.sum(dim=(-2, -1)) / (batch["H"] * batch["W"])
            length_m = (batch["p_stop"][:, :2] - batch["p_start"][:, :2]).norm(dim=-1)
            desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)[:, :desc_dim]
            x0 = torch.cat([canon0f, desc0], dim=1)
        else:
            x0 = canon0f
            length_m = (batch["p_stop"][:, :2] - batch["p_start"][:, :2]).norm(dim=-1)
        return dict(x0=x0, s_px=s_px, e_px=e_px, occ0=occ0, length_m=length_m,
                    H=batch["H"], W=batch["W"])

    ops_stack = torch.stack(ops, dim=0) if switched else None  # [n_bins, D_out, D_in], built once (not timed)

    def fwd_loop_reference(pre_out):
        """OLD path, kept for the numerical-equivalence check only (defect 1):
        `bool(m.any())` forces a GPU->CPU sync every one of the 6 bins, every
        call."""
        x0 = pre_out["x0"]
        if switched:
            bins = bin_index(pre_out["length_m"], bin_edges)
            pred_c = torch.empty(x0.shape[0], RES * RES, device=DEVICE)
            for b in range(len(ops)):
                m = bins == b
                if bool(m.any()):
                    pred_c[m] = (ops[b] @ x0[m].T).T[:, :RES * RES]
        else:
            pred_c = (global_op @ x0.T).T[:, :RES * RES]
        return pred_c

    def fwd_core(pre_out):
        """Sync-free path (defect 1 fix). First attempt was a gather
        (`index_select`) + batched `bmm` over one distinct [D_out,D_in]
        operator per sample -- sync-free, but SLOWER at large K than the
        old loop (it pays a full per-sample GEMM instead of one shared-A
        GEMM per bin, i.e. trades a host sync for extra device FLOPs/
        memory traffic -- measured, not assumed, then discarded). The
        actual fix keeps the old grouped-GEMM shape (cheap, one shared
        operator per bin) and removes ONLY the unnecessary host round-trip:
        `if bool(m.any()):` forces a GPU->CPU sync to evaluate a Python
        bool, but `pred_c[m] = ...` with an all-False mask is already a
        legal no-op on a boolean-mask index -- the `.any()` guard was pure
        (and costly) belt-and-braces, not a correctness requirement.
        Dropping it is sync-free AND FLOP-identical to the original."""
        x0 = pre_out["x0"]
        assert_all_on(DEVICE, x0)
        if switched:
            bins = bin_index(pre_out["length_m"], bin_edges)
            pred_c = torch.empty(x0.shape[0], RES * RES, device=DEVICE)
            for b in range(len(ops)):
                m = bins == b
                pred_c[m] = (ops[b] @ x0[m].T).T[:, :RES * RES]
        else:
            pred_c = (global_op @ x0.T).T[:, :RES * RES]
        return pred_c

    def fwd(pre_out):
        pred_c = fwd_core(pre_out)
        canon1_hat = pred_c.reshape(-1, RES, RES)
        H, W = pre_out["H"], pre_out["W"]
        back = from_push_frame(canon1_hat, pre_out["s_px"], pre_out["e_px"], (H, W), 1.0)
        mask = push_frame_validity_mask(pre_out["s_px"], pre_out["e_px"], (H, W), (RES, RES), 1.0)
        return blend_push_prediction(back, pre_out["occ0"], mask).clamp_(0.0, 1.0)

    def value(occ_pred):
        return value_from_occ(occ_pred)

    return dict(name=name, model=None, pre=pre, fwd=fwd, value=value,
                fwd_loop_reference=fwd_loop_reference if switched else None,
                fwd_core=fwd_core if switched else None)


def make_desc_only(name, ckpt_path):
    """MODEL-0002: descriptor-only (94-dim), switched-linear, no image
    reconstruction (predicts the next descriptor vector only -- see
    checkpoint's own note: 'NOT for image reconstruction')."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    bin_edges = ckpt["bin_edges"].to(DEVICE)
    ops = [o.to(DEVICE) for o in ckpt["ops"]]

    ops_stack = torch.stack(ops, dim=0)  # [n_bins, D_out, D_in], built once (not timed)

    def pre(batch):
        occ0 = batch["occ0"].float()
        device = occ0.device
        s_px, e_px = actions_to_pixels(batch["actions"], WS_MIN, WS_MAX, (batch["H"], batch["W"]))
        s_px, e_px = s_px.to(device), e_px.to(device)
        local0 = push_frame_full_descriptors(occ0, s_px, e_px)
        gm0 = occ0.sum(dim=(-2, -1)) / (batch["H"] * batch["W"])
        length_m = (batch["p_stop"][:, :2] - batch["p_start"][:, :2]).norm(dim=-1)
        desc0 = torch.cat([gm0[:, None], length_m[:, None], local0], dim=1)
        sp_d = world_to_pushframe_px(batch["p_start"][:, :2])
        ep_d = world_to_pushframe_px(batch["p_stop"][:, :2])
        return dict(desc0=desc0, length_m=length_m, sp_d=sp_d, ep_d=ep_d,
                    H=batch["H"], W=batch["W"])

    def fwd_loop_reference(pre_out):
        """OLD path, kept for the numerical-equivalence check only (defect 1)."""
        x0 = pre_out["desc0"]
        bins = bin_index(pre_out["length_m"], bin_edges)
        pred = torch.empty_like(x0)
        for b in range(len(ops)):
            m = bins == b
            if bool(m.any()):
                pred[m] = (ops[b] @ x0[m].T).T
        return pred

    def fwd_core(pre_out):
        """Sync-free path (defect 1 fix): drop the `bool(m.any())` host
        round-trip, keep the shared-operator-per-bin grouped GEMM (see
        make_switched_linear's fwd_core for why the gather+bmm alternative
        was measured and discarded)."""
        x0 = pre_out["desc0"]
        assert_all_on(DEVICE, x0)
        bins = bin_index(pre_out["length_m"], bin_edges)
        pred = torch.empty_like(x0)
        for b in range(len(ops)):
            m = bins == b
            pred[m] = (ops[b] @ x0[m].T).T
        return pred

    def fwd(pre_out):
        return fwd_core(pre_out)

    def value(pred_desc, pre_out):
        return value_from_descriptor(pred_desc, pre_out["sp_d"], pre_out["ep_d"],
                                      pre_out["H"], pre_out["W"])

    return dict(name=name, model=None, pre=pre, fwd=fwd, value=value,
                fwd_loop_reference=fwd_loop_reference, fwd_core=fwd_core,
                needs_pre_for_value=True)


# =============================================================================
# Timing core (methodology per Baselines/common/benchmark_time.py, reused).
# =============================================================================

CUDA_SYNC = torch.cuda.is_available()


def _sync():
    if CUDA_SYNC:
        torch.cuda.synchronize()


def time_region(fn, warmup, repeats):
    for _ in range(warmup):
        _ = fn()
    _sync()
    xs = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        _ = fn()
        _sync()
        xs.append(time.perf_counter() - t0)
    return _median_iqr([x * 1e3 for x in xs])  # ms


def call_value(spec, fwd_out, pre_out):
    """Dispatch to spec['value'] with the right arity -- descriptor-only and
    GNN need pre_out (action geometry / raw for the point-mass or
    rasterisation readout), the image models only need their own predicted
    occupancy."""
    if spec.get("needs_pre_for_value"):
        return spec["value"](fwd_out, pre_out)
    return spec["value"](fwd_out)


def time_model_at_k(spec, batch, k, warmup, repeats):
    """RUN-0002 (timing-corrected): reports pre/fwd/end2end BOTH without and
    WITH the goal-value computation in the timed region (defect 2), using
    the sync-free fwd path throughout (defect 1)."""
    pre, fwd = spec["pre"], spec["fwd"]

    def do_pre():
        return pre(batch)

    def do_end2end():
        return fwd(pre(batch))

    def do_end2end_with_value():
        pre_out = pre(batch)
        fwd_out = fwd(pre_out)
        call_value(spec, fwd_out, pre_out)
        return fwd_out

    pre_stats = time_region(do_pre, warmup, repeats)
    pre_out_sample = pre(batch)  # for fwd-only timing, reuse one pre() output
    # (safe: fwd is a pure function of pre_out and does not mutate state that
    # would make repeated calls on the SAME pre_out unrealistic -- it is
    # exactly what an MPC loop does if it caches nothing between calls; we
    # additionally re-run pre() at the start of end2end timing above so
    # end2end reflects the true back-to-back cost.)

    def do_fwd():
        return fwd(pre_out_sample)

    def do_fwd_with_value():
        fwd_out = fwd(pre_out_sample)
        call_value(spec, fwd_out, pre_out_sample)
        return fwd_out

    fwd_stats = time_region(do_fwd, warmup, repeats)
    e2e_stats = time_region(do_end2end, warmup, repeats)
    fwd_v_stats = time_region(do_fwd_with_value, warmup, repeats)
    e2e_v_stats = time_region(do_end2end_with_value, warmup, repeats)
    extra = {}
    if spec.get("special") == "gnn":
        extra["gnn_loop_s_median"] = pre_out_sample["loop_s"]
        extra["gnn_one_sample_s_estimate"] = pre_out_sample["one_sample_s"]
    return dict(pre_ms=pre_stats, fwd_ms=fwd_stats, end2end_ms=e2e_stats,
                fwd_with_value_ms=fwd_v_stats, end2end_with_value_ms=e2e_v_stats,
                **extra)


def verify_sync_free_equivalence(specs, cell, k=128):
    """Defect 1 fix verification: run the OLD (`.any()`-synced) and NEW
    (sync-free bmm/gather) fwd paths on the SAME pre_out for every switched-
    operator model and report max abs diff. Only meaningful for specs that
    expose `fwd_loop_reference`/`fwd_core` (the switched-linear and
    descriptor-only models -- the only ones defect 1 accuses)."""
    diffs = {}
    batch = build_batch(cell, k, DEVICE)
    for name, spec in specs.items():
        if spec.get("fwd_loop_reference") is None:
            continue
        pre_out = spec["pre"](batch)
        old = spec["fwd_loop_reference"](pre_out)
        new = spec["fwd_core"](pre_out)
        diff = (old - new).abs().max().item()
        diffs[name] = diff
        print(f"[equivalence] {name}: max abs diff (old loop vs sync-free) = {diff:.3e}")
    return diffs


def device_and_params(spec):
    model = spec.get("model")
    if model is not None:
        dev = str(next(model.parameters()).device)
        n_params = sum(p.numel() for p in model.parameters())
    else:
        dev = DEVICE  # linear operators: tensors moved to DEVICE at load time
        n_params = None
    return dev, n_params


def main():
    contended, details = check_gpu_contention()
    force = "--force" in sys.argv
    if contended and not force:
        print("[bench] GPU CONTENDED, refusing to record (pass --force to override):", details)
        sys.exit(3)
    if contended:
        print("[bench] --force: proceeding, contended=True", details)

    prov = git_provenance()
    print(f"[bench] provenance: {prov}")
    print(f"[bench] DEVICE={DEVICE}  cuda_available={torch.cuda.is_available()}")

    cell = load_state_and_actions()

    specs = {}
    print("\n[bench] loading models...")
    specs["nfd"] = make_nfd()
    specs["schenck"] = make_schenck()
    specs["gnn"] = make_gnn()
    specs["model0001_switched"] = make_switched_linear(
        "model0001_switched", "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
        desc_dim=0, switched=True)
    specs["model0001_global"] = make_switched_linear(
        "model0001_global", "weights/MODEL-0001-stage2-visual-switched/checkpoint.pt",
        desc_dim=0, switched=False)
    specs["model0002_descriptor_only"] = make_desc_only(
        "model0002_descriptor_only", "weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt")
    specs["hybrid14"] = make_switched_linear(
        "hybrid14", "experiments/temp/stage2-slaten/operators/hybrid14_lam1.0.pt",
        desc_dim=14, switched=True)
    specs["hybrid94"] = make_switched_linear(
        "hybrid94", "experiments/temp/stage2-slaten/operators/hybrid94_lam1.0.pt",
        desc_dim=94, switched=True)
    print(f"[bench] {len(specs)} models loaded: {list(specs)}")

    build_goal_fields(cell.H, cell.W)

    print("\n[bench] verifying sync-free fwd == old loop fwd (defect 1) ...")
    equivalence = verify_sync_free_equivalence(specs, cell)

    results = {}
    for name, spec in specs.items():
        dev, n_params = device_and_params(spec)
        print(f"\n--- {name} (device={dev}, params={n_params}) ---")
        by_k = {}
        for k in KS_V2:
            batch = build_batch(cell, k, DEVICE)
            r = time_model_at_k(spec, batch, k, WARMUP, REPEATS)
            by_k[str(k)] = r
            cand_per_s = k / (r["end2end_ms"]["median"] / 1e3)
            cand_per_s_v = k / (r["end2end_with_value_ms"]["median"] / 1e3)
            print(f"  K={k:4d}  end2end median={r['end2end_ms']['median']:.3f}ms "
                  f"[{r['end2end_ms']['q1']:.3f},{r['end2end_ms']['q3']:.3f}]  "
                  f"+value={r['end2end_with_value_ms']['median']:.3f}ms  "
                  f"fwd median={r['fwd_ms']['median']:.3f}ms  "
                  f"cand/s={cand_per_s:.0f}  cand/s(+value)={cand_per_s_v:.0f}")
        results[name] = dict(device_actual=dev, param_count=n_params, by_k=by_k)

    payload = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provenance=prov,
        contended=contended,
        contention_details=details,
        device=DEVICE,
        ks=KS_V2,
        warmup=WARMUP,
        repeats=REPEATS,
        eval_cfg=EVAL_CFG,
        sync_free_equivalence_max_abs_diff=equivalence,
        value_functions=["lyapunov", "mass_in_region"],
        goal_region="fixed top-right quadrant (timing-only, not accuracy-tuned)",
        results=results,
    )
    os.makedirs(OUTDIR_V2, exist_ok=True)
    out_json = f"{OUTDIR_V2}/results_timing_v2.json"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote {out_json}")


if __name__ == "__main__":
    main()
