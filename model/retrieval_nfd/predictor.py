"""model/retrieval_nfd/predictor.py -- eval-time predictor for the
"NFD with a retrieved reference" model (EXP-0059 section 8).

REWRITE (2026-09-28, coordinator feedback):
  * channels 0-2 now match `Baselines/NFD/predictor.py::NFDPredictor`'s own
    featurisation EXACTLY (occ0 passed through as-is, r_start/r_stop via
    `draw_plate_soft` with the same geometry constants) -- this predictor's
    `predict_occ(batch)` still plugs into `simple_mpc.adapters.OCC_ADAPTERS`
    the same way every other NFD variant does.
  * retrieval now uses the TRUE particle state when the caller has one
    (`predict_step_particles(occ0, act, states0)`), NOT pseudo-cubes
    extracted from the occupancy image -- pseudo-cube extraction is now only
    a FALLBACK for occupancy-only callers (`predict_occ`, and rollout steps
    after the first, where no true particle state exists -- exactly what the
    design doc's own "extract cubes from the predicted occupancy" rollout
    spec describes).
  * donor channels render with the SAME cv2-box rasteriser that produces
    channel 0 in the TRAINING dataset (`model.retrieval_nfd.render
    .render_cube_boxes_batch`), not a soft approximation.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from scipy import ndimage

from model.retrieval.bank import TransitionBank
from model.retrieval.distance import query_points_and_weights, bank_points_and_weights, topk_search
from model.retrieval.frame import push_frame_to_world, push_angle, wrap_angle
from model.retrieval_nfd.donors import frozen_distance_config
from model.retrieval_nfd.render import render_cube_boxes_batch, cube_dim_px
from model.UNetModels_modular import UNet
from Baselines.NFD.predictor import _plate_geometry_px
from transforms.functional import draw_plate_soft, action_to_pose

_STRUCTURE_5CH = dict(
    features=[4, 8, 16], in_channels=5, out_channels=1, kernel_size=3,
    final_kernel_size=1, activation="relu", residual=True,
    bottleneck_type="None", bottleneck_kwargs={},
)

CKPT_MAIN = "model/retrieval_nfd/runs/retrieval_nfd_ref/unet_best.pth"
CKPT_RANDOM_DONOR = "model/retrieval_nfd/runs/retrieval_nfd_random_donor/unet_best.pth"

_BANK_CACHE = {}


def _get_eval_bank() -> TransitionBank:
    """Eval-time bank = DS-0008 (valid) + DS-0010 (all), NO chain exclusion
    -- DS-0009/DS-0011 query rows are disjoint from this bank by
    construction, so there is no leakage to guard against here."""
    if "bank" not in _BANK_CACHE:
        _BANK_CACHE["bank"] = TransitionBank.build()
    return _BANK_CACHE["bank"]


def _pseudo_cubes_from_occ(occ0: torch.Tensor, threshold: float = 0.15,
                            n_max: int = 20) -> np.ndarray:
    """FALLBACK ONLY (no true particle state available -- rollout steps
    after the first). occ0: (B, H, W) -> (B, n_max, 2) pixel-space
    [x_px, y_px] centroids via connected components, padded with 1e6 (far
    outside any retrieval window, so `window_mask` naturally zero-weights
    the padding)."""
    B, H, W = occ0.shape
    occ_np = occ0.detach().float().cpu().numpy()
    out = np.full((B, n_max, 2), 1e6, dtype=np.float32)
    for b in range(B):
        mask = occ_np[b] > threshold
        lbl, num = ndimage.label(mask)
        if num == 0:
            continue
        idxs = list(range(1, num + 1))
        coms = ndimage.center_of_mass(occ_np[b], lbl, idxs)
        masses = ndimage.sum(occ_np[b], lbl, idxs)
        order = np.argsort(masses)[::-1][:n_max]
        for k, oi in enumerate(order):
            r, c = coms[oi]
            out[b, k] = [r, c]
    return out


class RetrievalRefPredictor:
    """channels=5: [occ0, r_start, r_stop, donor_occ0, donor_occ1].
    `zero_ref=True` implements the "test-time zeroed reference" control
    (iii): the SAME trained checkpoint, donor channels forced to 0 at
    inference regardless of what retrieval would find."""

    def __init__(self, ckpt_path: str, name: str | None = None, zero_ref: bool = False,
                 pseudo_cube_threshold: float = 0.15):
        self.name = name or "retrieval_nfd_ref"
        self.zero_ref = zero_ref
        self.pseudo_cube_threshold = pseudo_cube_threshold
        self.model = UNet(_STRUCTURE_5CH)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.cfg = frozen_distance_config()
        self._bank = None

    def _donor_channels(self, occ0: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor,
                         states0_xy: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        device = occ0.device
        H, W = occ0.shape[-2:]
        if self._bank is None:
            self._bank = _get_eval_bank()
        if states0_xy is None:
            centers_px = _pseudo_cubes_from_occ(occ0, self.pseudo_cube_threshold)  # (B,n,2) numpy
            ctr_xy = torch.tensor([W / 2.0, H / 2.0])
            world_xy = (torch.from_numpy(centers_px) - ctr_xy) / 500.0
        else:
            world_xy = states0_xy.cpu().float()
        p_start_w = p_start.cpu().float()
        p_stop_w = p_stop.cpu().float()
        bank_pts, bank_w, bank_valid = bank_points_and_weights(self._bank, self.cfg)
        q_pts, q_w, _uv, _valid = query_points_and_weights(world_xy, p_start_w, p_stop_w, self.cfg)
        idx, _dist = topk_search(q_pts, q_w, bank_pts, bank_w, self.cfg, k=1,
                                  device=device, bank_valid=bank_valid)
        donor_i = idx[:, 0]
        donor_uv0 = self._bank.uv0[donor_i]
        donor_uv1 = self._bank.uv1[donor_i]
        donor_yaw0 = self._bank.yaw0[donor_i]
        donor_yaw1 = self._bank.yaw1[donor_i]
        wxy0 = push_frame_to_world(donor_uv0, p_start_w, p_stop_w)
        wxy1 = push_frame_to_world(donor_uv1, p_start_w, p_stop_w)
        phi = push_angle(p_start_w, p_stop_w)
        wyaw0 = wrap_angle(donor_yaw0 + phi.unsqueeze(-1))
        wyaw1 = wrap_angle(donor_yaw1 + phi.unsqueeze(-1))
        ctr_xy = torch.tensor([W / 2.0, H / 2.0])
        px0 = wxy0 * 500.0 + ctr_xy
        px1 = wxy1 * 500.0 + ctr_xy
        dim_px = cube_dim_px(500.0)
        donor0 = render_cube_boxes_batch(px0, wyaw0, dim_px, (H, W)).to(device)
        donor1 = render_cube_boxes_batch(px1, wyaw1, dim_px, (H, W)).to(device)
        return donor0, donor1

    def _predict(self, occ0: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor,
                 angle: torch.Tensor, states0_xy: torch.Tensor | None) -> torch.Tensor:
        device = occ0.device
        self.model.to(device)
        H, W = occ0.shape[-2], occ0.shape[-1]
        ctr_xy = torch.tensor([W / 2.0, H / 2.0], device=device)
        start_px = p_start[:, :2].to(device) * 500.0 + ctr_xy
        stop_px = p_stop[:, :2].to(device) * 500.0 + ctr_xy
        angle = angle.to(device)
        plate_x_px, plate_y_px = 0.04 * 500.0, 0.002 * 500.0
        sigma = max(0.5, 1.5 * 0.5)
        r_start = draw_plate_soft(start_px, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_px, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        if self.zero_ref:
            donor0 = torch.zeros_like(occ0)
            donor1 = torch.zeros_like(occ0)
        else:
            donor0, donor1 = self._donor_channels(occ0, p_start, p_stop, states0_xy)
        x = torch.stack([occ0, r_start, r_stop, donor0, donor1], dim=1)
        logits = self.model(x)
        return torch.sigmoid(logits).squeeze(1).cpu()

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        """Occ-adapter contract (`simple_mpc.adapters.OCC_ADAPTERS`) -- no
        true particle state available, so retrieval falls back to
        pseudo-cubes. Kept for MPC/gradient-adapter compatibility; the
        offline eval harness now prefers `predict_step_particles` (see
        `eval_extended.py`'s particle-aware patch) whenever a true state
        exists."""
        return self._predict(batch.occ0, batch.p_start, batch.p_stop, batch.angle, states0_xy=None)

    @torch.no_grad()
    def predict_step_particles(self, occ0: torch.Tensor, act: torch.Tensor,
                                states0: torch.Tensor | None = None) -> torch.Tensor:
        """Particle-aware entry point. occ0: (B,H,W). act: (B,4) world-metre
        [sx,sy,ex,ey]. states0: (B,n,7) TRUE particle state if available
        (1-step scoring, slateN candidates -- all share one true start
        state), else None (rollout steps after the first: no true state,
        pseudo-cube fallback, matching the design doc's own rollout spec)."""
        sx, sy, ex, ey, angle = action_to_pose(act)
        z = torch.zeros_like(sx)
        p_start = torch.stack([sx, sy, z], dim=-1)
        p_stop = torch.stack([ex, ey, z], dim=-1)
        states0_xy = states0[..., :2] if states0 is not None else None
        return self._predict(occ0, p_start, p_stop, angle, states0_xy=states0_xy)


def build_predictor_main() -> RetrievalRefPredictor:
    """--predictor model.retrieval_nfd.predictor:build_predictor_main"""
    ckpt_path = os.environ.get("RETRIEVAL_NFD_CKPT", CKPT_MAIN)
    return RetrievalRefPredictor(ckpt_path, name="retrieval_nfd_ref")


def build_predictor_random_donor() -> RetrievalRefPredictor:
    """The random-donor control (i): same architecture/recipe, trained with
    a bucket-matched random bank donor instead of retrieval."""
    ckpt_path = os.environ.get("RETRIEVAL_NFD_RANDOM_CKPT", CKPT_RANDOM_DONOR)
    return RetrievalRefPredictor(ckpt_path, name="retrieval_nfd_random_donor")


def build_predictor_zeroed_ref() -> RetrievalRefPredictor:
    """Control (iii): the MAIN trained checkpoint, donor channels zeroed at
    test time regardless of retrieval."""
    ckpt_path = os.environ.get("RETRIEVAL_NFD_CKPT", CKPT_MAIN)
    return RetrievalRefPredictor(ckpt_path, name="retrieval_nfd_ref_zeroed", zero_ref=True)
