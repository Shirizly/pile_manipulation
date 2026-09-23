"""Extend `warp_accuracy_ceiling.py` (L20mm only) to L40mm and randlen_test,
so each corpus's warped-arm accuracy is read against its OWN ceiling
(RUN-0009's re-scoring brief). Same method: feed the TRUE next occupancy
through the identity push-frame round trip at canon_res=64 (matching
nfd_warped_randlen's training-time default: canon_res=None -> batch's own
grid resolution, which is 64 at resolution_scale=0.5) and score it as a
prediction.
"""
import torch

from Baselines.common.data import load_cell
from Baselines.common.randlen_data import load_randlen_cell
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask
from transforms.functional import push_frame_roundtrip

CANON_RES = 64
ident = lambda c: c

CELLS = {
    "L20mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L20mm/manifest.json",
    ),
    "L40mm": dict(
        kind="slate",
        eval_cfg="configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml",
        manifest="Genesis/data/slates_multistep/n20_L40mm/manifest.json",
    ),
    "randlen_test": dict(
        kind="randlen",
        cfg="configs/dataset/genesis_overnight_randlen_test_all.yaml",
    ),
}

for name, spec in CELLS.items():
    if spec["kind"] == "slate":
        cell = load_cell(spec["eval_cfg"], "train", manifest_path=spec["manifest"], tag=name)
    else:
        cell = load_randlen_cell(spec["cfg"], "train", tag=name)
    occ0, occ1, actions = cell.occ0, cell.occ1, cell.actions
    H, W = occ0.shape[-2:]
    print(f"\n=== {name}: n={occ0.shape[0]} grid={H}x{W} ===")

    start_px, end_px = actions_to_pixels(actions, cell.workspace_min, cell.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(start_px, end_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    rt_truth = push_frame_roundtrip(ident, occ1.to(torch.float32), start_px, end_px, CANON_RES, 1.0).clamp(0, 1)
    m_ceiling = metrics(rt_truth, occ1.to(torch.float32), occ0.to(torch.float32), region)
    d = (rt_truth - occ1.to(torch.float32)).abs()
    print(f"  canon_res={CANON_RES}  CEILING accuracy = {m_ceiling['accuracy']:.4f}")
    print(f"  round-trip |err| on occ1: max {d.max():.4f} mean {d.mean():.5f}")
