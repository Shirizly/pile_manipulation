"""EXP-0022 R2 -- residual-formulation analogue of `warp_accuracy_ceiling.py`.

That script measures the accuracy CEILING a warped model predicting an
absolute occupancy can reach: round-trip the TRUE occ1 through
warp->identity->unwarp (+ blend against occ0 outside the validity mask) and
score it as if it were a prediction.

R2's mechanism claims that ceiling should rise substantially, because a
residual field's zero-padding outside the canonical grid's coverage already
reads as "no change" (so bit-exact preservation of the ~98% of the image
that never changes, with NO explicit blend needed) -- see
`Baselines/NFD/residual_nfd_lib.py`'s module docstring. This script tests
that prediction directly, with no trained model:

  1. delta_true = occ1 - occ0            (world frame, signed, [-1,1])
  2. canon_delta_true = to_push_frame(delta_true, ...)   (warp the TRUE residual in)
  3. world_delta_rt = from_push_frame(canon_delta_true, ...)   (unwarp it back out)
  4. occ_pred = clamp(occ0 + world_delta_rt, 0, 1)        (add to PRISTINE occ0, clamp)
  5. score with the ordinary swept-region `accuracy`, exactly as the direct
     ceiling script does.

No `blend_push_prediction`/validity-mask blend anywhere in this pipeline --
that is the point being tested, not an oversight.
"""
import torch

from Baselines.common.data import load_cell
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask
from transforms.functional import from_push_frame, to_push_frame

CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
cell = load_cell(CFG, "train")
occ0, occ1, actions = cell.occ0, cell.occ1, cell.actions
H = W = occ0.shape[-1]
print(f"n={occ0.shape[0]} grid={H}x{W}")

start_px, end_px = actions_to_pixels(actions, cell.workspace_min, cell.workspace_max, (H, W))
plate_px = 0.04 / 0.128 * W
region = swept_region_mask(start_px, end_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

delta_true = (occ1 - occ0).clamp(-1.0, 1.0)

for canon_res in (32, 64, 90, 128, 181):
    canon_delta_true = to_push_frame(delta_true, start_px, end_px, (canon_res, canon_res), 1.0)
    world_delta_rt = from_push_frame(canon_delta_true, start_px, end_px, (H, W), 1.0)
    occ_pred_residual_rt = (occ0 + world_delta_rt).clamp(0.0, 1.0)

    m_ceiling_residual = metrics(occ_pred_residual_rt, occ1, occ0, region)
    m_persist = metrics(occ0, occ1, occ0, region)

    # bit-exactness check outside the validity mask (no blend applied here,
    # so this checks whether zero-padding alone reproduces occ0 there):
    from transforms.functional import push_frame_validity_mask
    mask = push_frame_validity_mask(start_px, end_px, (H, W), (canon_res, canon_res), 1.0)
    outside = mask < 0.5
    max_dev_outside = (occ_pred_residual_rt - occ0)[outside].abs().max().item() if outside.any() else float("nan")
    frac_outside = outside.float().mean().item()

    print(f"\ncanon_res={canon_res}, scale=1.0")
    print(f"  RESIDUAL CEILING (true residual through round trip, no blend) "
          f"accuracy = {m_ceiling_residual['accuracy']:.4f}")
    print(f"  persistence (occ0 as prediction)                              "
          f"accuracy = {m_persist['accuracy']:.4f}")
    print(f"  frac pixels outside validity mask: {frac_outside:.4f}")
    print(f"  max |occ_pred - occ0| outside validity mask (should be ~0, no blend): "
          f"{max_dev_outside:.6f}")
    d = (occ_pred_residual_rt - occ1).abs()
    print(f"  round-trip |err| on occ1: max {d.max():.4f} mean {d.mean():.5f}")
