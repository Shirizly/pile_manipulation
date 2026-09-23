"""EXP-0022 RUN-0013, second check -- does the canonical action encoding
itself degenerate at short push lengths?

Independent of any trained model: builds the exact canonical two-plate-
channel action encoding `nfd_warped_randlen` receives
(`transforms.functional.canonical_plate_channels`, `plate_mode="canonical"`,
the mode this checkpoint was trained/scored with -- see
`Baselines/NFD/predictor.py::build_canonical_stack`) for a sweep of push
lengths spanning `overnight_randlen`'s corpus range (0.5-79.6mm, per
docs/CODEMAP.md), using the SAME geometry the corpus was collected/trained
with:

  - to_pxl = 1000 * resolution_scale, resolution_scale=0.5 (the randlen
    dataset configs) -> to_pxl=500 px/m
  - plate size (from Genesis/data/overnight_randlen_test/piled_n20/
    _23_config.yaml, representative of the corpus): [0.04, 0.002, 0.0175] m
    -> plate_dim_x_px=20, plate_dim_y_px=1 (px, at to_pxl=500)
  - canon_res = world_res = 64 (resolution_scale=0.5 grid), scale=1.0 --
    nfd_warped_randlen's own training-time defaults (canon_res=None ->
    batch's own H=64; `eval_report.py` MODELS entry for nfd_warped_randlen)
  - sigma = max(0.5, 1.5*resolution_scale) = 0.75 world px (canon==world
    here, so k=1 and sigma_canon=sigma)

For each L, measures how distinguishable the two channels (r_start, r_stop)
are: Pearson correlation (1.0 = identical, i.e. maximally degenerate) and
per-pixel L2 distance (0 = identical), both computed over the full 64x64
canvas. Also reports the "channel diff signal" as a fraction of a single
channel's own norm, since raw L2 scales with the plate's total mass.

No model, no data loading -- pure geometry, runs in <1s.
"""
from __future__ import annotations

import torch

from transforms.functional import canonical_plate_channels

TO_PXL = 500.0  # 1000 * resolution_scale(0.5), overnight_randlen dataset configs
PLATE_DIM_X_M = 0.04
PLATE_DIM_Y_M = 0.002
RESOLUTION_SCALE = 0.5
CANON_RES = 64
WORLD_RES = 64
SCALE = 1.0
SIGMA_PX = max(0.5, 1.5 * RESOLUTION_SCALE)  # = 0.75, world px == canon px here (k=1)

PLATE_X_PX = PLATE_DIM_X_M * TO_PXL  # 20.0
PLATE_Y_PX = PLATE_DIM_Y_M * TO_PXL  # 1.0


def main():
    # overnight_randlen push length range, per docs/CODEMAP.md: 0.5-79.6mm.
    lengths_mm = torch.cat([
        torch.linspace(0.0, 5.0, 26),      # fine resolution where degeneracy is expected
        torch.linspace(5.5, 80.0, 60),
    ])
    lengths_px = lengths_mm * (TO_PXL / 1000.0)  # mm -> m -> px

    chans = canonical_plate_channels(
        lengths_px, CANON_RES, PLATE_X_PX, PLATE_Y_PX,
        scale=SCALE, sigma=SIGMA_PX, world_res=WORLD_RES,
    )  # (B, 2, 64, 64)
    r_start, r_stop = chans[:, 0], chans[:, 1]

    B = r_start.shape[0]
    flat_a = r_start.reshape(B, -1)
    flat_b = r_stop.reshape(B, -1)

    l2 = (flat_a - flat_b).norm(dim=1)
    norm_a = flat_a.norm(dim=1).clamp_min(1e-9)
    rel_l2 = l2 / norm_a  # fraction of one channel's own norm

    corr = torch.empty(B)
    for i in range(B):
        a, b = flat_a[i], flat_b[i]
        if a.std() < 1e-8 or b.std() < 1e-8:
            corr[i] = float("nan")
        else:
            corr[i] = torch.corrcoef(torch.stack([a, b]))[0, 1]

    print(f"{'L(mm)':>8s}  {'L(px)':>8s}  {'corr':>8s}  {'L2':>10s}  {'L2/|a|':>8s}")
    for i in range(B):
        print(f"{lengths_mm[i]:8.2f}  {lengths_px[i]:8.3f}  {corr[i]:8.4f}  "
              f"{l2[i]:10.4f}  {rel_l2[i]:8.4f}")

    # "Effectively indistinguishable" threshold: corr > 0.99 (arbitrary but
    # stated) AND rel_l2 < 0.10 (channels differ by <10% of one channel's own
    # mass). Report the largest L at which this holds, and the smallest L at
    # which it stops holding, since the transition need not be monotonic at
    # the tails from floating-point/threshold noise.
    indist = (corr > 0.99) & (rel_l2 < 0.10)
    if bool(indist.any()):
        thresh_mm = float(lengths_mm[indist].max())
        print(f"\nIndistinguishable (corr>0.99 and L2<10% of channel norm) up to "
              f"L={thresh_mm:.2f}mm")
    else:
        print("\nNever indistinguishable at this threshold across the sweep")

    # Also report where corr crosses below a looser 0.9 (channels clearly
    # separating) and below 0.5 (channels clearly distinct).
    for thr in (0.999, 0.99, 0.95, 0.9, 0.5):
        below = corr < thr
        if bool(below.any()):
            first_below = float(lengths_mm[below][0])
            print(f"corr first drops below {thr}: L={first_below:.2f}mm")
        else:
            print(f"corr never drops below {thr} in this sweep")

    torch.save(dict(lengths_mm=lengths_mm, lengths_px=lengths_px, corr=corr,
                     l2=l2, rel_l2=rel_l2), OUT)


OUT = "experiments/EXP-0022-warped-nfd-push-frame/artifacts/RUN-0013-length-stratified-canonical/encoding_degeneracy.pt"

if __name__ == "__main__":
    main()
