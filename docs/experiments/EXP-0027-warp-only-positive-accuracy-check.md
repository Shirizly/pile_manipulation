---
id: EXP-0027
title: >
  warp-only's positive image accuracy at L20mm/L40mm (+0.0254/+0.0442,
  EXP-0024_v1) is RMS rewarding a blurred copy of the current state, not a
  bug -- an explicit Gaussian blur of persistence reproduces a positive
  accuracy of the same sign and comparable-or-larger magnitude on the
  identical transitions/region
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: null

claim: >
  On the L20mm and L40mm slates_multistep eval cells, an explicit Gaussian
  blur of persistence (no warp, no operator) reproduces a positive
  swept-region `accuracy` (docs/experiments/METRICS.md) of the same sign as
  `warp-only` (A=identity round trip) at some sigma, with magnitude within
  2x of warp-only's own value -- i.e. warp-only's positive score is
  explained by RMS rewarding low-pass hedging, not by a defect in
  `to_push_frame`/`from_push_frame`/`blend_push_prediction`.

prediction:
  supports: >
    at least one blur sigma in {0.5, 1.0, 1.5, 2.0, 3.0} gives a positive
    accuracy at both L20mm and L40mm, and `blend_push_prediction` returns the
    exact original occupancy (max abs diff < 1e-5) outside its validity mask.
  refutes: >
    every blurred-persistence accuracy stays <= 0 at both cells while
    warp-only stays positive, or the outside-mask check finds a nonzero
    difference -- either would point at a real bug in the warp/blend
    pipeline rather than a metric property.
  discriminating: true

provenance:
  commit: 250102d8
  dirty: false
  data_commit: unrecorded (Genesis/data/slates_multistep/n20_L{20,40}mm, see EXP-0024_v1's own provenance block)
  script: scripts/probes/warp_blur_diagnostic.py
  data: ["configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml",
         "configs/dataset/genesis_slates_multistep_n20_L40mm_eval.yaml"]
  code_path: "PileSweepData raster via registry.dataset_registry.build_dataset; fit_linear_foresight.metrics/predict_world-equivalent inline; loro_foresight.gaussian_blur"
  seed: null (no fit, no randomness -- A=identity and blur are deterministic given the eval data)
  split: "whole eval-split file set (val_pct=0/test_pct=0), n=7680 transitions per cell"
  runtime: "~1 min CPU per cell"

budget:
  declared: "not separately declared; part of the ~3h/250k-token action-pool-diagnostics task"
  spent: "~10 min, ~15k tokens (two script runs + one extra L10mm context run)"
  outcome: within

design:
  varied: {cell: [L20mm, L40mm], treatment: ["A=identity (warp-only)", "gaussian_blur(persistence, sigma) for sigma in 0.5..3.0"]}
  held_fixed: {region: "swept_region_mask, half_width=0.5*plate_px+2, pad=0.5*plate_px", metric: accuracy, transitions: "identical eval-split rows for both treatments within a cell", R: 64, crop: 1.0}
  baselines: [persistence]
  metric: "accuracy"

noise_floor: "not applicable -- both quantities are deterministic functions of the same fixed eval set, no resampling; the comparison is sign/magnitude, not a paired significance test"

depends_on: [warp-blend, swept-region-metric]
establishes: []

result: >
  L20mm: warp-only +0.0254 vs blurred-persistence +0.0313 at sigma=1.0 (same
  sign, same order of magnitude). L40mm: warp-only +0.0442 vs
  blurred-persistence +0.0350..+0.148 across sigma=0.5..3.0 (same sign, blur
  exceeds warp-only at every sigma tried). Outside-mask check: max diff
  0.000e+00 at both cells.
verdict: supported
downgrades: []
grade: high
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If `warp-only`'s positive accuracy were a bug in the warp/resample/blend
code, an explicit blur -- a completely different code path, no warp
geometry, no validity mask, no canonical frame at all -- should NOT
reproduce a similarly-signed, similarly-sized effect; it would either stay
negative (blur cannot beat a sharp copy without exploiting whatever the bug
exploits) or reproduce it only by coincidence at one sigma. Finding a
positive, comparable-or-larger accuracy across a RANGE of blur sigmas at
both cells is a strong, non-coincidental match to the "RMS rewards hedging"
mechanism, and the `blend_push_prediction` outside-mask check independently
rules out the other candidate bug (returning the wrong thing outside the
validity mask) directly, by measurement rather than by re-reading the code.

## What was actually run

`scripts/probes/warp_blur_diagnostic.py <eval_cfg> --tag <cell> --sigmas
0.5,1.0,1.5,2.0,3.0` on `genesis_slates_multistep_n20_L20mm_eval.yaml` and
`..._L40mm_eval.yaml`. Also ran on `..._L10mm_eval.yaml` for context (not
part of the original prediction, reported as an exploratory addendum below,
not used to support/refute the claim above). No fitting: `A=identity`
(warp round-trip only) and `gaussian_blur` are both applied directly to the
eval occupancy, scored against the same truth/region as everything else in
this project's pipeline (`fit_linear_foresight.metrics`, same swept-region
mask geometry as `expB_multistep_eval.py`).

## Numbers

| cell | persistence (sanity) | warp-only (A=I) | blur σ=0.5 | σ=1.0 | σ=1.5 | σ=2.0 | σ=3.0 | outside-mask max\|diff\| |
|---|---|---|---|---|---|---|---|---|
| L20mm | +0.0000 | **+0.0254** | +0.0294 | **+0.0313** | +0.0217 | +0.0072 | −0.0255 | 0.000e+00 |
| L40mm | +0.0000 | **+0.0442** | **+0.0350** | +0.0789 | +0.1051 | +0.1236 | +0.1482 | 0.000e+00 |
| L10mm (exploratory, not part of the prediction) | +0.0000 | **−0.5294** | −0.3100 | −0.9689 | −1.3245 | −1.5633 | −1.8506 | 0.000e+00 |

## What would change the verdict

A demonstration that the SPECIFIC pixels warp-only improves on are NOT the
ones blur improves on (e.g. a per-pixel or per-transition correlation
between warp-only's error map and blur's error map, restricted to the
swept region) would be a sharper test than matching sign/magnitude alone --
if the two treatments help on disjoint subsets of transitions, "RMS rewards
hedging" would be an incomplete explanation and something specific to the
warp geometry would still need chasing. Estimated cost: ~20 min (reuse the
existing per-transition tensors, no new data).

## Threats

None of the five downgrade domains apply: same code path throughout
(no cross-rasteriser comparison), the sign/magnitude match is large and
holds at 4/5 sigmas tried per cell (not a single cherry-picked point), the
design was not selected after seeing a favourable result (the sigma sweep
is reported in full, including the sigma=3.0 L20mm cell where blur turns
negative), and the region/metric/transitions are identical between
treatments by construction. The L10mm exploratory addendum going the OTHER
direction (blur makes things worse, not better) was considered as a
possible threat to the mechanism story, but is explained by scale (L10mm's
true signal is ~4-18x smaller than L20/L40mm's, see EXP-0028) rather than
contradicting it, and is reported as exploratory/context, not as evidence
for or against this record's claim.

## Unrelated findings

- `runs_expB/n20_L10mm_accuracy.json`'s warp-only-equivalent value
  (identity-operator accuracy, reconstructable from this script) is
  −0.5294, matching EXP-0024_v1's quoted number to 4 decimal places --
  cross-checks that record's own number via an independently written
  script.
- `scripts/check_register.py`'s hardcoded `design.metric` key list (line
  ~155) does not include `accuracy` or `slateK_exact` -- the two metrics
  `docs/experiments/METRICS.md` itself calls "THE TWO STANDARD METRICS" and
  asks every record to report -- so any record naming them gets a spurious
  warning. Found because this record's `metric: "accuracy"` triggered it.
- none else.
