"""Measure the information the push-frame warp round trip alone destroys.

A warped model predicts in the canonical frame and is unwarped back, so it
pays a double grid_sample resampling the world-frame NFD never pays. This
puts a CEILING on the warped arms' swept-region `accuracy` that has nothing
to do with dynamics. Measured here as: feed the TRUE next occupancy through
the same round trip and score it as if it were a prediction. A perfect
warped predictor scores exactly this and no higher.
"""
import torch
from Baselines.common.data import load_cell
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask
from transforms.functional import push_frame_roundtrip

CFG = "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml"
cell = load_cell(CFG, "train")
occ0, occ1, actions = cell.occ0, cell.occ1, cell.actions
H = W = occ0.shape[-1]
print(f"n={occ0.shape[0]} grid={H}x{W}")

start_px, end_px = actions_to_pixels(actions, cell.workspace_min, cell.workspace_max, (H, W))
plate_px = 0.04 / 0.128 * W
region = swept_region_mask(start_px, end_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

ident = lambda c: c
for canon_res in (32, 64, 90, 128, 181):
    rt_truth = push_frame_roundtrip(ident, occ1, start_px, end_px, canon_res, 1.0).clamp(0, 1)
    rt_occ0  = push_frame_roundtrip(ident, occ0, start_px, end_px, canon_res, 1.0).clamp(0, 1)
    m_ceiling = metrics(rt_truth, occ1, occ0, region)
    m_persist = metrics(occ0, occ1, occ0, region)
    m_rtpers  = metrics(rt_occ0, occ1, occ0, region)
    print(f"\ncanon_res={canon_res}, scale=1.0")
    print(f"  CEILING  (true occ1 through the round trip) accuracy = {m_ceiling['accuracy']:.4f}")
    print(f"  persistence (occ0 as prediction)            accuracy = {m_persist['accuracy']:.4f}")
    print(f"  persistence THROUGH the round trip          accuracy = {m_rtpers['accuracy']:.4f}")
    d = (rt_truth - occ1).abs()
    print(f"  round-trip |err| on occ1: max {d.max():.4f} mean {d.mean():.5f}")
