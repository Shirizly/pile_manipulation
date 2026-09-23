"""model/warped_nfd/predictor.py -- offline eval-time predictor for warped NFD
(the NFD UNet baseline predicting in the canonical PUSH FRAME instead of the
world frame). Split out of ``Baselines/NFD/predictor.py`` (which keeps only
the plain/unwarped ``NFDPredictor``) because this is a new model family with
its own file, not a minor parameter change over the NFD baseline -- see
``model/warped_nfd/lib.py`` (the training-time twin, ``WarpedNFDWrapper``)
for the model this loads.

``build_canonical_stack`` is the ONE function that assembles the UNet's
canonical-frame input (occ0 + plate channels [+ wall channel]), called by
BOTH ``WarpedNFDWrapper.forward`` (training, ``model/warped_nfd/lib.py``) and
``WarpedNFDPredictor.predict_occ`` below, so the two paths cannot drift
apart -- train/eval parity measured at ~1e-6 world-frame prob diff. Living
here (not in ``lib.py``) keeps this module Genesis-free, matching
``Baselines/NFD/predictor.py``'s own design note: ``NFDPredictor`` (and this
module) must stay importable without pulling in ``Genesis.training.dataset``
/ the dataset registry -- `model/` as a whole is Genesis-free, and moving
this file under `model/` must not break that property.
"""
from __future__ import annotations

import os

import torch
import yaml

from Baselines.NFD.predictor import _STRUCTURE_3CH, _plate_geometry_px
from model.UNetModels_modular import UNet
from transforms.functional import (
    canonical_plate_channels,
    draw_plate_soft,
    push_frame_roundtrip,
    to_push_frame,
)


def build_canonical_stack(
    canon_occ0: torch.Tensor,
    start_px: torch.Tensor,
    end_px: torch.Tensor,
    action_world: torch.Tensor,
    occ0_world: torch.Tensor,
    plate_mode: str,
    wall_channel: bool,
    canon_res: int,
    scale: float,
    plate_dim_x_px: float,
    plate_dim_y_px: float,
    sigma_px: float,
    world_res: int,
) -> torch.Tensor:
    """Build the (B, 3-or-4, canon_res, canon_res) UNet input stack in the
    canonical push frame.

    Parameters
    ----------
    canon_occ0    : (B, canon_res, canon_res) -- already-warped occupancy, as
                    handed to `push_frame_roundtrip`'s `fn` callback.
    start_px, end_px : (B, 2) push endpoints, pixels, `push_frame`'s (col,
                    row) order -- NOT the (row, col) order `draw_plate_soft`'s
                    own `center` argument uses (see WARPED_NFD_NOTES.md).
    action_world  : (B, 2, H, W) world-frame [r(p_start), r(p_stop)] renders,
                    full intensity -- only read when `plate_mode == "warp"`.
    occ0_world    : (B, H, W) world-frame occupancy -- only read to size the
                    wall-channel warp (`to_push_frame` needs a same-shaped
                    canvas of ones).
    plate_mode    : "canonical" (render the plates directly in the canonical
                    frame, `canonical_plate_channels`) or "warp" (warp the
                    world-frame plate renders through `to_push_frame`, folding
                    the 2 action channels into the batch dim -- no Python
                    loop).
    wall_channel  : if True, append a 4th channel: the warped workspace-
                    extent indicator (1 inside the workspace, 0 outside),
                    `to_push_frame` applied to a canvas of ones.
    """
    B = canon_occ0.shape[0]
    chans = [canon_occ0.unsqueeze(1)]
    if plate_mode == "canonical":
        push_length_px = (end_px - start_px).norm(dim=-1)
        plates = canonical_plate_channels(
            push_length_px, canon_res, plate_dim_x_px, plate_dim_y_px,
            scale=scale, sigma=sigma_px, world_res=world_res,
        )
    elif plate_mode == "warp":
        H, W = action_world.shape[-2], action_world.shape[-1]
        flat = action_world.reshape(B * 2, H, W)
        sp = start_px.repeat_interleave(2, dim=0)
        ep = end_px.repeat_interleave(2, dim=0)
        warped = to_push_frame(flat, sp, ep, (canon_res, canon_res), scale)
        plates = warped.reshape(B, 2, canon_res, canon_res)
    else:
        raise ValueError(f"unknown plate_mode {plate_mode!r}")
    chans.append(plates)
    if wall_channel:
        wall = to_push_frame(
            torch.ones_like(occ0_world), start_px, end_px,
            (canon_res, canon_res), scale,
        ).unsqueeze(1)
        chans.append(wall)
    return torch.cat(chans, dim=1)


class WarpedNFDPredictor:
    """Warped-NFD offline predictor: same UNet architecture as
    `Baselines.NFD.predictor.NFDPredictor` but predicting in the canonical
    push frame via `push_frame_roundtrip` + `build_canonical_stack` above --
    see `model/warped_nfd/lib.py` (the training-time twin, `WarpedNFDWrapper`)
    for the model this loads."""

    def __init__(self, ckpt_path: str, plate_mode: str, wall_channel: bool,
                 canon_res: int | None = None, scale: float = 1.0,
                 name: str | None = None):
        assert plate_mode in ("canonical", "warp")
        self.plate_mode = plate_mode
        self.wall_channel = wall_channel
        self.canon_res = canon_res
        self.scale = scale
        in_channels = 4 if wall_channel else 3
        structure = dict(_STRUCTURE_3CH, in_channels=in_channels)
        self.model = UNet(structure)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.name = name or f"nfd_unet_warped{'_walls' if wall_channel else ''}"

    def _build_fn(self, batch):
        """Shared setup for `predict_occ` (world-frame, warp->predict->unwarp
        ->blend) and `predict_occ_canonical` (canonical-frame, no unwarp) --
        both must call the SAME `fn` and pixel derivation or the two paths
        can drift apart, exactly the parity concern `build_canonical_stack`'s
        own docstring raises for training vs eval.

        Returns `(fn, occ0, start_px, end_px, canon_res)`. `fn` takes
        `(B, canon_res, canon_res)` canonical occupancy and returns the
        canonical-frame predicted probability, same contract
        `push_frame_roundtrip` requires.
        """
        device = batch.occ0.device
        self.model.to(device)
        H, W = batch.H, batch.W
        raw = batch.raw

        ctr_xy = raw.ctr_in_PXL.to(torch.float32).to(device)[:2]
        # Native (row=x_pixel, col=y_pixel) order -- same formula/order
        # NFDPredictor.predict_occ uses to feed draw_plate_soft directly.
        start_native = batch.p_start[:, :2].to(device) * raw.to_pxl + ctr_xy
        stop_native = batch.p_stop[:, :2].to(device) * raw.to_pxl + ctr_xy
        # push_frame wants (col, row) -- the transpose of draw_plate_soft's
        # own (row, col) center convention (WARPED_NFD_NOTES.md trap #1).
        start_px = start_native[:, [1, 0]]
        end_px = stop_native[:, [1, 0]]
        angle = batch.angle.to(device)
        plate_x_px, plate_y_px, sigma = _plate_geometry_px(raw)

        occ0 = batch.occ0.to(device)
        r_start = draw_plate_soft(start_native, angle, (H, W), plate_x_px, plate_y_px,
                                   intensity=1.0, sigma=sigma)
        r_stop = draw_plate_soft(stop_native, angle, (H, W), plate_x_px, plate_y_px,
                                  intensity=1.0, sigma=sigma)
        action_world = torch.stack([r_start, r_stop], dim=1)
        canon_res = self.canon_res or H

        def fn(canon_occ0):
            stack = build_canonical_stack(
                canon_occ0, start_px, end_px, action_world, occ0,
                self.plate_mode, self.wall_channel, canon_res, self.scale,
                plate_x_px, plate_y_px, sigma, H,
            )
            logits = self.model(stack)
            if logits.dim() == 4:
                logits = logits.squeeze(1)
            return torch.sigmoid(logits)

        return fn, occ0, start_px, end_px, canon_res

    @torch.no_grad()
    def predict_occ(self, batch) -> torch.Tensor:
        fn, occ0, start_px, end_px, canon_res = self._build_fn(batch)
        prob = push_frame_roundtrip(fn, occ0, start_px, end_px, canon_res,
                                     scale=self.scale, blend=True)
        return prob.cpu()

    @torch.no_grad()
    def predict_occ_canonical(self, batch):
        """Native canonical-frame prediction: `fn(to_push_frame(occ0))`,
        with NO final unwarp/blend back to the world frame -- this is the
        model's own working representation, computed with zero extra
        resamplings beyond the one warp every canonical model pays to enter
        its frame at all (EXP-0022 A1: "a warped model produces a canonical
        prediction natively"). Contrast `predict_occ` above, which pays a
        SECOND resampling (the unwarp) plus the validity-mask blend to
        return to the world frame.

        Returns `(pred_canon, start_px, end_px, canon_res, scale)` so a
        caller can build a matching canonical truth/region with the exact
        same warp parameters this prediction used.
        """
        from transforms.functional import to_push_frame
        fn, occ0, start_px, end_px, canon_res = self._build_fn(batch)
        canon_occ0 = to_push_frame(occ0, start_px, end_px, (canon_res, canon_res), self.scale)
        pred_canon = fn(canon_occ0)
        return pred_canon.cpu(), start_px.cpu(), end_px.cpu(), canon_res, self.scale


# Default plate_mode: set from the timing measurement in
# model/warped_nfd/WARPED_NFD_NOTES.md ("plate_mode default" section) -- kept
# as a module constant so the two factories and the pilot configs agree.
WARPED_DEFAULT_PLATE_MODE = "canonical"

CKPT_WARPED = "Baselines/NFD/runs/nfd_warped_L20mm_pilot/unet_best.pth"
CKPT_WARPED_WALLS = "Baselines/NFD/runs/nfd_warped_walls_L20mm_pilot/unet_best.pth"


def _assert_matches_model_card(ckpt_path: str, wall_channel: bool, plate_mode: str,
                                scale: float, canon_res: int | None) -> None:
    """`train_nfd.py` writes `model_card.yaml` beside every checkpoint with
    the exact `plate_mode`/`wall_channel`/`scale`/`canon_res` the model was
    TRAINED with. A silently mismatched setting at eval time (e.g. eval
    defaults drifting from a checkpoint trained with an explicit override)
    would produce a wrong-but-plausible-looking number -- see this
    predictor's eval-report task brief. Assert rather than trust, whenever
    the card is available."""
    card_path = os.path.join(os.path.dirname(ckpt_path), "model_card.yaml")
    if not os.path.exists(card_path):
        return
    card = yaml.safe_load(open(card_path))["model"]
    card_wall = bool(card.get("wall_channel", False))
    card_plate_mode = card.get("plate_mode", "canonical")
    card_scale = float(card.get("scale", 1.0))
    card_canon_res = card.get("canon_res")  # None -> defaults to batch's own H
    assert card_wall == wall_channel, (
        f"{ckpt_path}: model_card wall_channel={card_wall} != requested {wall_channel}")
    assert card_plate_mode == plate_mode, (
        f"{ckpt_path}: model_card plate_mode={card_plate_mode!r} != requested {plate_mode!r}")
    assert abs(card_scale - scale) < 1e-9, (
        f"{ckpt_path}: model_card scale={card_scale} != requested {scale}")
    assert card_canon_res == canon_res, (
        f"{ckpt_path}: model_card canon_res={card_canon_res!r} != requested {canon_res!r}")


def build_predictor_warped(canon_res: int | None = None, plate_mode: str | None = None,
                            scale: float | None = None) -> WarpedNFDPredictor:
    """Warped NFD, no wall channel. --predictor model.warped_nfd.predictor:build_predictor_warped

    Checkpoint path overridable via NFD_WARPED_CKPT (same pattern as
    NFD_CKPT above). `canon_res`/`plate_mode`/`scale` default to this
    checkpoint's own training-time defaults (`None` == batch's own grid
    resolution, `WARPED_DEFAULT_PLATE_MODE`, `1.0`) and are ALWAYS checked
    against `model_card.yaml` next to the checkpoint, not just trusted."""
    ckpt_path = os.environ.get("NFD_WARPED_CKPT", CKPT_WARPED)
    plate_mode = plate_mode or WARPED_DEFAULT_PLATE_MODE
    scale = 1.0 if scale is None else scale
    _assert_matches_model_card(ckpt_path, wall_channel=False, plate_mode=plate_mode,
                                scale=scale, canon_res=canon_res)
    return WarpedNFDPredictor(ckpt_path, plate_mode=plate_mode, wall_channel=False,
                               canon_res=canon_res, scale=scale, name="nfd_unet_warped")


def build_predictor_warped_walls(canon_res: int | None = None, plate_mode: str | None = None,
                                   scale: float | None = None) -> WarpedNFDPredictor:
    """Warped NFD with the wall (workspace-extent) channel.
    --predictor model.warped_nfd.predictor:build_predictor_warped_walls

    Checkpoint path overridable via NFD_WARPED_WALLS_CKPT. See
    `build_predictor_warped` for the `canon_res`/`plate_mode`/`scale`
    contract and the `model_card.yaml` assertion."""
    ckpt_path = os.environ.get("NFD_WARPED_WALLS_CKPT", CKPT_WARPED_WALLS)
    plate_mode = plate_mode or WARPED_DEFAULT_PLATE_MODE
    scale = 1.0 if scale is None else scale
    _assert_matches_model_card(ckpt_path, wall_channel=True, plate_mode=plate_mode,
                                scale=scale, canon_res=canon_res)
    return WarpedNFDPredictor(ckpt_path, plate_mode=plate_mode, wall_channel=True,
                               canon_res=canon_res, scale=scale, name="nfd_unet_warped_walls")
