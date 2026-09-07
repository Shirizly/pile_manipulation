"""C1: is `warp-only`'s positive accuracy a bug, or is it rms rewarding a
smoothed copy of the current state?

The register (EXP-0024_v1) reports `warp-only` (A=identity: warp into the
push frame, apply nothing, warp back, blend) at accuracy +0.0254 (L20mm) and
+0.0442 (L40mm) -- positive means it BEATS raw persistence, which looks
impossible: the identity operator can only add interpolation error to
persistence's exact copy, never remove it, on a pixel-for-pixel basis.

Leading benign explanation: `accuracy` is an RMS ratio (docs/experiments/
METRICS.md), and RMS rewards hedging -- a low-pass-filtered copy of the
current state can score a lower RMS than the sharp current state whenever
material actually moved, because blur partially "predicts" the blur-shaped
component of a translation while a razor-sharp copy commits fully to the
wrong sharp edge. If that is the whole story, an EXPLICIT Gaussian blur of
persistence (no warp involved at all) should reproduce a similarly positive
accuracy on the identical transitions/region. If blurred persistence stays
<= 0 while warp-only stays positive, the warp pipeline is doing something
blur alone does not, and that is a real bug to chase.

Also mechanically re-checks `blend_push_prediction`: outside its validity
mask it must return the ORIGINAL occupancy exactly, not the warped
prediction (transforms/functional.py's own docstring claim) -- confirmed
here by direct comparison rather than by reading the source.

Usage
-----
    PYTHONPATH=. python scripts/probes/warp_blur_diagnostic.py \\
        configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \\
        --tag L20mm --sigmas 0.5,1.0,1.5,2.0,3.0
"""
from __future__ import annotations

import argparse

import torch
import yaml

from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask
from loro_foresight import gaussian_blur
from registry.dataset_registry import build_dataset
from transforms.functional import (
    blend_push_prediction, from_push_frame, push_frame_validity_mask, to_push_frame,
)

R, CR = 64, 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_cfg")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sigmas", default="0.5,1.0,1.5,2.0,3.0")
    args = ap.parse_args()

    ecfg = yaml.safe_load(open(args.eval_cfg).read())
    wrapper = build_dataset(ecfg, "train")
    raw = wrapper.raw_dataset
    n = len(wrapper)
    occ0 = torch.stack([raw[i][0][0][0] for i in range(n)])
    occ1 = torch.stack([raw[i][1] for i in range(n)])
    actions = torch.stack([raw.get_raw_action(i) for i in range(n)])
    ws_min, ws_max = raw.workspace_bounds
    H, W = occ0.shape[-2:]
    s_px, e_px = actions_to_pixels(actions, ws_min, ws_max, (H, W))
    plate_px = 0.04 / 0.128 * W
    region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)

    print(f"=== {args.tag}  ({args.eval_cfg}) ===")
    print(f"n={n} transitions, grid {H}x{W}")

    D = R * R
    A_identity = torch.eye(D)
    canon = to_push_frame(occ0, s_px, e_px, (R, R), CR)
    pred_c = (A_identity @ canon.reshape(n, -1).T).T.reshape(-1, R, R)
    back = from_push_frame(pred_c, s_px, e_px, (H, W), CR)
    mask = push_frame_validity_mask(s_px, e_px, (H, W), (R, R), CR)
    warp_pred = blend_push_prediction(back, occ0, mask).clamp_(0.0, 1.0)

    # --- mechanical check: outside the mask, blend must return the ORIGINAL
    outside = mask < 0.5
    if outside.any():
        d_outside = (warp_pred[outside] - occ0[outside]).abs()
        print(f"blend_push_prediction outside-mask check: "
              f"{100*float(outside.float().mean()):.2f}% of pixels masked out, "
              f"max|blended - original| there = {float(d_outside.max()):.3e} "
              f"(clamp_ to [0,1] can perturb by <1e-6 vs the pre-clamp original "
              f"if occ0 itself has any value outside [0,1]; expect ~0)")
    else:
        print("blend_push_prediction outside-mask check: mask is ~1 everywhere "
              "for this cell (no masked-out pixels to check)")

    m_persist = metrics(occ0, occ1, occ0, region=region)
    m_warp = metrics(warp_pred, occ1, occ0, region=region)
    print(f"\npersistence accuracy (sanity, must be 0.000): {m_persist['accuracy']:+.4f}")
    print(f"WARP-ONLY (A=I round trip) accuracy:           {m_warp['accuracy']:+.4f}")

    print(f"\n{'sigma':>6s} {'blurred-persistence accuracy':>30s}")
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    results = {"warp_only": m_warp["accuracy"], "blur": {}}
    for sigma in sigmas:
        blurred = gaussian_blur(occ0, sigma).clamp(0.0, 1.0)
        m_blur = metrics(blurred, occ1, occ0, region=region)
        results["blur"][sigma] = m_blur["accuracy"]
        flag = ""
        if m_blur["accuracy"] > 0 and m_warp["accuracy"] > 0:
            flag = "  (same sign as warp-only)"
        print(f"{sigma:6.2f} {m_blur['accuracy']:+30.4f}{flag}")

    print("\nverdict inputs: if any blur sigma reproduces a positive accuracy "
          "of similar magnitude to warp-only, rms-rewards-hedging is the "
          "operative explanation and warp-only's positive score is not a "
          "measurement error. If every blurred-persistence accuracy stays "
          "<= 0 while warp-only is positive, the warp pipeline is adding "
          "something blur does not, and that needs its own bug hunt.")


if __name__ == "__main__":
    main()
