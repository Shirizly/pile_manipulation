---
id: EXP-0001
title: The dataset's occupancy and plate channels are mutually transposed
tier: T1
mode: confirmatory
date: 2026-09-03
hypothesis: H-A1
claim: >
  The occupancy channel of a PileSweepData sample places world x on grid dim 1
  while its plate/action channel places world x on dim 0, and correcting this
  reduces the linear operator's swept-region error from above persistence to
  well below it (>40 points of "% of change") on genesis_foresight_L040.
prediction:
  supports: "transposing the registry occupancy alone moves the operator from ~107% of the change to within a few points of the independently re-rasterised path"
  refutes:  "transposing changes the operator by less than the ~1-point run-to-run spread, i.e. the two rasterisers differ for some other reason"
  discriminating: true
provenance:
  commit: 17a7d4a7
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/ab_occ.py`
                                  # did not exist at 17a7d4a7; it was written in the same
                                  # session and first committed at aac084e3. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the aac084e3 version of that file.
  script_first_committed: aac084e3
  script: scripts/probes/ab_occ.py
  data: ["configs/dataset/genesis_foresight_L040.yaml", "Genesis/data/foresight/L040/**/*_data.pt"]
  code_path: "both, deliberately: PileSweepData raster vs points_to_mask"
  seed: 0
  split: "episode-level, 25% of 8 files, seed 0"
  runtime: "~3 min, CPU"
design:
  varied: {occupancy_source: ["registry as stored", "registry transposed", "points_to_mask re-rasterised"]}
  held_fixed: {res: 64, crop: 0.5, blur: 1.0, view: mask, ridge: 1.0, estimator: "ridge toward identity", actions: identical, split: identical, metric: identical, occupied_area: "matched via cube_size 0.007 (occ_mean 0.1169 vs 0.1194)"}
  baselines: [persistence, mean-delta, identity-warp]
  metric: "held-out rms over the swept region, as a percentage of the persistence rms (100% = no better than predicting nothing moved)"
noise_floor: "~1 point; the report's own fold sd is 0.004 rms against a persistence rms of 0.11, and the effect here is ~49 points"
establishes: [grid-convention, rasteriser-identity, pixel-index-origin]
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split]
result: "107.1% as stored -> 57.9% transposed -> 54.1% re-rasterised; best IoU against a dim0=world_x raster is 0.103 untransposed and 0.573 transposed"
verdict: supported
downgrades: []
grade: high
supersedes: []
invalidated_by: null
---

## Why this test discriminates

Two rasterisers disagreed by 49 points, and there were two candidate causes:
different footprint area, or different axis order. Footprint area is measurable
and can be matched exactly (`occ_mean` 0.1169 vs 0.1194), which removes it as an
explanation. Applying *only* a transpose to the registry field — same data, same
actions, same fit, same split — leaves axis order as the sole difference. If the
transpose recovers the re-rasterised performance, nothing else is available to
explain the gap.

## What was actually run

Three occupancy sources through one fit function. The area match was set by
sweeping `cube_size` until `occ_mean` matched the registry's, rather than by
using the geometrically correct 0.005 m — otherwise area and axis order would
both have varied. The footprint sweep itself (0.002 → 0.012 m, giving 68.6% →
48.9% monotonically) is reported below because it is what rules area out.

Source-level confirmation was obtained independently of the fit:
`_draw_particle_grid` passes `(center_x, center_y)` to `cv2.circle`, and cv2
takes points as (column, row), so world x lands on dim 1; `draw_plate_soft`
compares `center[:,0]` against `gx` from `meshgrid(..., indexing="ij")`, so world
x lands on dim 0. `draw_plate_soft`'s own docstring asserts the opposite of its
body, and `dmdc_baseline.py:45` repeats that assertion.

## Numbers

| occupancy source | occ_mean | linear | mean-delta | identity (warp only) |
|---|---|---|---|---|
| registry, as stored | 0.1194 | **107.1%** | 99.0% | 100.0% |
| registry, transposed | 0.1194 | **57.9%** | 87.1% | 98.3% |
| re-rasterised, area-matched | 0.1169 | **54.1%** | 86.5% | 98.3% |

Footprint sweep on the re-rasterised path, ruling out area as the cause:

| cube_size (m) | 0.002 | 0.005 | 0.007 | 0.009 | 0.012 |
|---|---|---|---|---|---|
| occ_mean | 0.0096 | 0.0595 | 0.1169 | 0.1890 | 0.3188 |
| linear | 68.6% | 58.9% | 54.1% | 50.9% | 48.9% |

Alignment, 200 samples, best over ±4 px shifts: IoU 0.103 untransposed,
**0.573** transposed (at a −1,−1 px shift, which is `pixel-index-origin`).
Plate-channel centroid vs action midpoint: 0.9 px under dim0=world_x, ~12 px
under dim0=world_y.

## What would change the verdict

Nothing cheap; the source-level reading and the measurement agree. The open
question is not *whether* but *which is correct in absolute terms* — the plate
channel and `particles_to_occupancy` agree with each other, so the cv2 particle
raster is the odd one out and is the thing to change. Cost: a one-line swap plus
the two invariant tests, then re-running everything in EXP-0002.

## Threats

- `untested-dependency`: `swept-region-metric` and `episode-split` are
  `unchecked`. Neither differs across the three rows, so neither can produce the
  effect, but the tag is cited honestly.
- Considered and dismissed: **split luck.** One seed-0 split, but the effect is
  49 points against a fold sd of well under 1 point, and it reproduces on the
  re-rasterised path independently.
- Considered and dismissed: **blur interaction.** The transpose gap is present
  at blur 0 too (EXP-0003's sweep), so it is not an artifact of σ=1.

## Grade note, 2026-09-05

`untested-dependency` dropped: this record's `depends_on` tags all
hold as of the invariant tests added today (`tests/test_metric_invariants.py`,
`tests/test_grid_convention.py`). Grade moderate -> high. The evidence did not
change; what changed is that the assumptions it rests on are now checked.

## Unrelated findings

none recorded — this record predates the section (added 2026-09-05).
