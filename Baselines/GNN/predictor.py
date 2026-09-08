"""Baselines/GNN/predictor.py -- BaselinePredictor for
Baselines/common/eval_baseline.py. Wraps the trained
`PropNetDiffDenModel` (model/gnn_dyn.py) and rasterises its particle-space
prediction to occupancy.

Rasterisation note (E1-gnn-improve, batching pass): the harness's own
`Baselines.common.data.rasterize_particles` is a single-transition routine
(it says so in its own docstring) that calls
`PileSweepData._draw_particle_grid` once per row -- and that method itself
loops in Python over the 20 particles issuing ONE `cv2.boxPoints` +
`cv2.fillPoly` pair PER PARTICLE. Timing showed the GNN's per-candidate
cost is flat in K (866->832->820 us, K=32->1024) -- the tell of an
unbatched per-candidate Python/cv2 loop, not network cost (38k params).

`Baselines/common/data.py` is read-only, so the batched path lives here
instead. Two things were TRIED for batching the per-particle work itself
and one of them is a documented dead end, kept here as a warning:

  - TRIED AND REJECTED: collapsing the 20 per-particle `cv2.fillPoly`
    calls into one `cv2.fillPoly(grid, boxes, 1)` call over the whole
    particle list. This LOOKS safe (fillPoly with a list of
    *non-overlapping* polygons and a constant colour is bit-identical to
    calling it once per polygon -- verified directly), but our cube piles
    have ADJACENT/OVERLAPPING boxes, and `cv2.fillPoly` given multiple
    contours in one call fills them under an even-odd winding rule (like
    a single multi-contour path), which XORs out the overlap instead of
    just setting it to 1 -- silently DIFFERENT pixels, caught only by
    `Baselines/GNN/scripts/verify_rasterizer.py`'s real-candidate check
    (batched sum was LOWER than per-row, e.g. 117 vs 122 filled pixels on
    the first real candidate tried). Reverted; do not reintroduce this
    without re-deriving the fill-rule interaction from scratch.
  - KEPT: the quaternion-to-yaw computation is vectorised across the
    WHOLE (B,N) batch at once (plain elementwise arctan2, the identical
    formula to `quaternion_to_yaw` in `Genesis/training/dataset.py`, just
    computed as one numpy array instead of 20*B separate Python/math
    calls) and the world->pixel coordinate transform is likewise done
    once for the whole batch tensor instead of once per row. Both are
    pure elementwise ops with no particle-to-particle interaction, so
    there is no fill-rule-style trap here.
  - KEPT: candidates are rasterised in PARALLEL across a thread pool
    (`concurrent.futures.ThreadPoolExecutor`) instead of a serial Python
    loop. Each candidate's `cv2.boxPoints`/`cv2.fillPoly`/`cv2.circle`
    calls release the GIL (OpenCV's own C implementation), and each
    candidate's grid buffer is independent (own `np.zeros` allocation,
    own list of boxes) -- so this changes nothing about what gets
    computed, only how many candidates are computed at once. This is the
    real point of amortisation for a Python/cv2-bound per-candidate cost
    that could not otherwise fall with K.

Per-particle math (`_draw_particle_grid_fast`) reproduces
`PileSweepData._draw_particle_grid` (Genesis/training/dataset.py) EXACTLY
particle-by-particle (same `cv2.boxPoints` call, same int() truncation,
same PER-PARTICLE `cv2.fillPoly` call, same axis-transpose-at-the-end
convention). Sphere/cylinder particles (never present in our cube cells
but kept for parity) fall back to the original per-particle `cv2.circle`
call, unchanged.

Equivalence with the harness's own `rasterize_particles` was checked
particle-grid-exact (max abs diff == 0.0) on 400 real eval candidates
across both cells before this was wired into `predict_occ` -- see
`Baselines/GNN/LOG.md` and `Baselines/GNN/scripts/verify_rasterizer.py`.

`build_predictor()` takes no arguments (harness contract) and loads
`Baselines/GNN/runs/ckpt_best.pth` by default (override via the
GNN_CKPT env var if scoring a different checkpoint).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from Baselines.GNN.geometry import PARTICLE_DENS, compute_s_delta
from model.gnn_dyn import PropNetDiffDenModel

DEFAULT_CKPT = "Baselines/GNN/runs/ckpt_best.pth"


def _yaw_from_quat_batch(quat: torch.Tensor) -> np.ndarray:
    """Vectorised form of `PileSweepData._draw_particle_grid`'s local
    `quaternion_to_yaw` (Genesis/training/dataset.py) -- identical
    elementwise formula (`atan2(2(wz+xy), 1-2(y^2+z^2))`), computed as a
    numpy array instead of one Python call per particle. `quat`: (...,4)
    w,x,y,z. Widened to float64 first, matching the original's
    `math.atan2` on Python floats (which upconverts the float32 tensor
    values the same way)."""
    q = quat.to(torch.float64).cpu().numpy()
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _draw_particle_grid_fast(raw, particle_states_px: torch.Tensor, config,
                              yaw: np.ndarray) -> torch.Tensor:
    """Drop-in replacement for ONE call of
    `raw._draw_particle_grid(particle_states_px, grid, config)`, with the
    yaw argument precomputed for the whole batch (see module docstring).
    `particle_states_px`: (N,7) tensor already in pixel space
    (world*to_pxl + ctr_in_PXL, same as the harness's own
    `rasterize_particles` produces). `yaw`: (N,) precomputed
    `_yaw_from_quat_batch` output for this row's particles.

    Mirrors `PileSweepData._draw_particle_grid` line for line, including
    issuing ONE `cv2.fillPoly` call PER PARTICLE (see module docstring for
    why collapsing these into a single multi-polygon call is NOT
    equivalent for our overlapping cube piles). Sphere/cylinder particles
    use the original per-particle `cv2.circle` call, unchanged -- never
    exercised by our cube cells but kept so this isn't silently wrong if
    ever pointed at a different one.
    """
    num_particles = config["material"]["n_particles"]
    shape = config["material"]["shape"]
    particle_sizes = config["data_collection"]["sampled"]["particle_sizes"]
    H, W = raw._output_grid.shape
    grid_np = np.zeros((H, W), dtype=np.float32)

    ps = particle_states_px.numpy()
    for idx in range(num_particles):
        center_x = float(ps[idx, 0])
        center_y = float(ps[idx, 1])
        dimensions = particle_sizes[idx]
        upright_cylinder = False

        if shape == "cylinder":
            from scipy.spatial.transform import Rotation as R

            def cylinder_is_standing(quat):
                local_up = np.array([0, 0, 1])
                world_up = np.array([0, 0, 1])
                rot = R.from_quat(quat)
                rotated_axis = rot.apply(local_up)
                alignment = abs(np.dot(rotated_axis, world_up))
                return alignment >= 0.5

            if cylinder_is_standing(ps[idx, 3:]):
                upright_cylinder = True

        if shape == "sphere" or (upright_cylinder and shape == "cylinder"):
            diameter, _, _ = dimensions
            cv2.circle(
                grid_np,
                (int(round(center_x)), int(round(center_y))),
                max(1, int(round(diameter * raw.to_pxl * 0.5))),
                color=1,
                thickness=-1,
            )
            continue

        rotated_rect = (
            (int(center_x), int(center_y)),
            (int(float(dimensions[0]) * raw.to_pxl), int(float(dimensions[1]) * raw.to_pxl)),
            int(float(yaw[idx]) * 180 / math.pi),
        )
        box = np.int32(cv2.boxPoints(rotated_rect))
        cv2.fillPoly(grid_np, [box], 1)

    return torch.from_numpy(grid_np.T.copy())


_MAX_WORKERS = min(32, (os.cpu_count() or 4))


def rasterize_particles_batch(raw, run_idx: list[int], particles_world: torch.Tensor) -> torch.Tensor:
    """Batched form of `Baselines.common.data.rasterize_particles`: rasterise
    a whole (B,20,7) batch of world-frame particle predictions to (B,H,W)
    occupancy grids, instead of the caller looping per row with all of the
    per-row overhead (tensor clone, config lookup, closures) repeated B
    times. `run_idx[i]` indexes `raw.configs`, exactly as the per-row
    routine uses it.

    The coordinate transform and quaternion-to-yaw are vectorised once for
    the whole batch; each candidate's actual rasterisation
    (`_draw_particle_grid_fast`, per-particle `cv2.boxPoints`/`fillPoly`,
    unchanged from the original) is then dispatched across a thread pool
    (`cv2` releases the GIL) so B independent candidates amortise across
    CPU cores instead of running as one long serial Python loop -- this is
    what actually collapses the flat-in-K cost (see module docstring).
    Numerically identical to calling the harness's own
    `rasterize_particles` once per row (see module docstring +
    `Baselines/GNN/scripts/verify_rasterizer.py`)."""
    B, N, _ = particles_world.shape
    px = particles_world.clone()
    px[:, :, :3] = px[:, :, :3] * raw.to_pxl + raw.ctr_in_PXL
    yaw_all = _yaw_from_quat_batch(particles_world[:, :, 3:])  # (B,N), unpixelated quat -> yaw

    H, W = raw._output_grid.shape
    occ = torch.empty((B, H, W), dtype=torch.float32)
    configs = [raw.configs[r] for r in run_idx]

    def _one(i):
        occ[i] = _draw_particle_grid_fast(raw, px[i], configs[i], yaw_all[i])

    if B > 1:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, B)) as pool:
            list(pool.map(_one, range(B)))
    else:
        for i in range(B):
            _one(i)
    return occ


class GNNPredictor:
    name = "gnn"

    def __init__(self, ckpt_path: str = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = PropNetDiffDenModel(ckpt["cfg"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.ckpt_epoch = ckpt.get("epoch")
        self.ckpt_val_mse = ckpt.get("val_mse")
        print(f"[GNNPredictor] loaded {ckpt_path} (epoch={self.ckpt_epoch}, "
              f"val_mse={self.ckpt_val_mse})")

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        states = batch.states.to(self.device)          # (B,20,7)
        s_cur = states[..., :3]
        p_start = batch.p_start.to(self.device)
        p_stop = batch.p_stop.to(self.device)
        B, N, _ = s_cur.shape

        s_delta = compute_s_delta(s_cur, p_start, p_stop)
        a_cur = torch.zeros(B, N, device=self.device)
        dens = torch.full((B,), PARTICLE_DENS, device=self.device)

        s_pred = self.model.predict_one_step(a_cur, s_cur, s_delta, dens)  # (B,20,3)

        # Model has no orientation head (SPEC.md hazard/§6) -- reattach the
        # INPUT frame's quaternion unchanged rather than inventing one.
        quat = states[..., 3:7]
        particles_pred = torch.cat([s_pred, quat], dim=-1).cpu()  # (B,20,7)

        run_idx = batch.run_idx.tolist()
        return rasterize_particles_batch(batch.raw, run_idx, particles_pred)


def build_predictor():
    ckpt_path = os.environ.get("GNN_CKPT", DEFAULT_CKPT)
    return GNNPredictor(ckpt_path=ckpt_path)
