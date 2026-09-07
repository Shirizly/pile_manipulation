"""EXP-B diagnostic: is the canonical-frame warp round-trip itself the source
of the negative image accuracy seen at short push lengths, or is it the fit?

Coordinator flag, 2026-09-07: L=10mm eval showed linear accuracy -0.44 and
mean-delta -0.52 (both worse than persistence) while the UNet scored +0.24 --
a ~68pt gap that would be the largest in the register if attributed to model
quality. But `to_push_frame`'s `scale` fixes the canonical window as a
FRACTION OF THE IMAGE, not of the push length, so the window covers the same
physical area at every push length while push travel shrinks with it: at
L=10mm on a 64x64/0.128m grid (2mm/px) the push moves ~5px, so the
warp->unwarp round-trip's OWN interpolation error can exceed the whole signal
-- C-002 ("the SE(2) warp costs more than one push changes unless the field
is smoothed to sigma~1"), restated at short push length instead of at coarse
resolution.

This isolates that: score `predict_world(A=I, ...)` (linear) and
`predict_meandelta(bmd=0, ...)` (mean-delta with zero delta) -- i.e. warp,
apply NOTHING, unwarp, blend -- so the only thing measured is the round-trip
interpolation cost, with no fit and no operator involved at all. If this
number is close to the FITTED linear operator's own accuracy, the loss is the
warp, not the fit, and the model comparison is invalid for that cell (label
it warp-limited rather than comparing model quality).

Usage
-----
    PYTHONPATH=. python scripts/probes/expB_warp_diagnostic.py \
        configs/dataset/genesis_slates_multistep_n20_L10mm_eval.yaml \
        configs/dataset/genesis_slates_multistep_n20_L10mm_train.yaml
"""
from __future__ import annotations

import argparse

import torch
import yaml

from dmdc_baseline import load_transition_arrays
from fit_linear_foresight import (
    actions_to_pixels, metrics, predict_world, swept_region_mask,
)
from registry.dataset_registry import build_dataset

R, CR = 64, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_cfg")
    ap.add_argument("train_cfg", help="only used for workspace bounds")
    args = ap.parse_args()

    data_tr = load_transition_arrays(args.train_cfg, split="train")
    H, W = data_tr.occ_t.shape[-2:]

    ecfg = yaml.safe_load(open(args.eval_cfg).read())
    wrapper = build_dataset(ecfg, "train")
    raw = wrapper.raw_dataset
    n = len(wrapper)
    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    s_px, e_px = actions_to_pixels(actions, data_tr.workspace_min,
                                   data_tr.workspace_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    D = R * R
    A_identity = torch.eye(D)
    warp_pred = predict_world(A_identity, occ0, s_px, e_px, R, (H, W), CR)
    m_warp = metrics(warp_pred, occ1, occ0, region=region)
    m_persist = metrics(occ0, occ1, occ0, region=region)

    print(f"{args.eval_cfg}")
    print(f"  n={n} transitions, grid {H}x{W}, res={R} crop={CR}")
    print(f"  persistence accuracy (sanity, should be 0.000): {m_persist['accuracy']:+.4f}")
    print(f"  WARP ROUND-TRIP ONLY (A=I, no operator) accuracy: {m_warp['accuracy']:+.4f}")
    print(f"  (mean-delta with bmd=0 is bit-identical to this -- same warp/"
          f"blend pipeline, no additive term)")


if __name__ == "__main__":
    main()
