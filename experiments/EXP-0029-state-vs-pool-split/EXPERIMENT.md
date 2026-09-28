---
id: EXP-0029
title: >
  On DS-0001, a model's per-state advantage partly survives a fresh action pool
  (split-half r ~0.4 at 500 actions, ~0.16 at 128), but choosing the model per
  state does not beat the best single model; the randlen +0.020 switching gain
  was likely inflated by scoring on the same pool it was chosen on
tier: T1
mode: confirmatory              # C1/C2 pre-registered in DESIGN.md; the two-model switch is exploratory
date: 2026-09-24
hypothesis: null
claim: >
  On DS-0001 (20 n20 scatter states x 1000 actions; 5 OCC_ADAPTERS models; 30
  goals; soft ground-truth scoring), a model's per-state advantage is a
  property of the STATE, not of the particular action pool: (C1, K=128) the
  cross-fitted per-state switching gain (choose on pool 1, score on disjoint
  pool 2) is > 0 with its 95% interval excluding 0 AND the median split-half
  correlation of per-state pairwise advantage exceeds 0.3; (C2, K=500) same.
prediction:
  supports: "gain interval excludes 0 (>0) AND median split-half r > 0.3"
  refutes: "gain interval includes 0 AND median split-half r < 0.1"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true                   # soft ground-truth scoring code added this session (uncommitted)
  script: code/split_test.py; code/pair_switch_exploratory.py (exploratory)
  data: ["DS-0001 (Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm)"]
  code_path: >
    OCC_ADAPTERS predict_step on occ_from_particles(state) -> lyapunov /
    mass_in_region per goal; truth: occ_for_scoring(states_) (soft) -> same
    value fns; slateN capture per disjoint pool
  seed: "split rng seed 0 (split_test.py), 1 (pair_switch_exploratory.py)"
  split: "200 random disjoint pool pairs per state, K=128 and K=500"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "~2 min"
budget:
  declared: "not separately declared; part of the user's Phase A follow-up"
  spent: "~20 min wall-clock"
  outcome: within
design:
  varied:
    pool_size_K: "128, 500"
    value_fn: "lyapunov (primary), mass_in_region"
  held_fixed:
    models: "nfd_3ch_randlen, nfd_3ch_finetuned, nfd_warped_randlen, nfd_residual_warped (L20mm pilot), linear_switched_hard"
    goals: "26 letters + 4 quadrants, 64x64 slate grid"
    truth_scoring: "soft (occ_for_scoring)"
  baselines: >
    best single model chosen on pool 1 (the comparison the switch must beat);
    random pick = capture 0 by construction.
  metric: slateN (per pool), split-half reliability r, cross-fitted switching gain
noise_floor: >
  Split-to-split spread of the gain (95% interval over 200 splits) and a state
  bootstrap (20 states) are both reported; the state bootstrap is the wider
  (e.g. K=128 lyapunov [-0.130, +0.031]).
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x,
             occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  C1 (K=128) neither: gain -0.029 [-0.090, +0.024], median r +0.16 (between
  thresholds). C2 (K=500) not supported: median r +0.40 (6/10 pairs with r
  interval > 0, up to +0.55) but gain -0.006 [-0.055, +0.025]. Exploratory,
  the two closest models only: gain -0.008 (K=128) to +0.009 (K=500), no
  interval excludes 0.
verdict: inconclusive
downgrades: [indirectness, imprecision]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The randlen analysis (experiments/temp/state-specialisation) chose a model per
state and scored it on the SAME 128 actions (only the goals were split), so
luck of the pool counted as a gain. Here the two pools are disjoint samples of
the same state's action distribution, so only a state-level advantage can
transfer.

## What was actually run

RUN-0001 as designed (DESIGN.md). Soft ground-truth scoring throughout. A
follow-up restricted to the two closest models (`nfd_3ch_randlen`,
`nfd_warped_randlen`, mean gap 0.06) was added AFTER seeing the result, because
with a 0.3-0.4 capture gap to the weakest models the 5-model switch mostly
measures the cost of ever leaving the best model; it is exploratory.

## Numbers (results/split_test.json)

| K / value fn | median split-half r | switching gain [95% over splits] | state bootstrap | best single -> switch (hindsight) |
|---|---|---|---|---|
| 128 lyapunov | +0.16 | -0.029 [-0.090, +0.024] | [-0.130, +0.031] | 0.684 -> 0.656 (0.732) |
| 128 mass_in_region | +0.22 | -0.001 [-0.041, +0.033] | [-0.054, +0.052] | 0.681 -> 0.680 (0.733) |
| 500 lyapunov | +0.40 | -0.006 [-0.055, +0.025] | [-0.083, +0.051] | 0.674 -> 0.668 (0.715) |
| 500 mass_in_region | +0.34 | -0.002 [-0.029, +0.027] | [-0.045, +0.039] | 0.664 -> 0.662 (0.704) |

Most reliable pairs at K=500 lyapunov: nfd_3ch_randlen|nfd_warped_randlen r
+0.55 [+0.31, +0.74]; nfd_warped_randlen|nfd_residual_warped +0.47; nfd_3ch_randlen|
nfd_residual_warped +0.44. Pairs involving linear_switched_hard are the least
reliable (r 0.06-0.26 for lyapunov).

Exploratory two-model switch (nfd_3ch_randlen vs nfd_warped_randlen): K=128
lyapunov -0.008 [-0.046, +0.026]; mass +0.007 [-0.016, +0.037]; K=500
lyapunov +0.007 [-0.024, +0.031] (positive in 75% of splits); mass +0.009
[-0.013, +0.038] (78%).

## Reading

- A state-level component of "which model is better" exists (r ~0.4-0.55 at
  K=500 for NFD-family pairs), and most of the per-state advantage seen in a
  128-action pool is pool noise (r ~0.16).
- Turning it into a better controller by picking a model per state from one
  pool's evidence does not work at this scale: the per-state signal is weaker
  than the pool noise at K=128, and at K=500 the gain is small and unresolved.
- The randlen result (+0.020) should be re-read as an upper bound inflated by
  pool reuse; a clean version needs disjoint pools per state (the planned
  DS-0005 collects two per state).
- A per-state switch chosen from STATE FEATURES (the user's actual goal --
  descriptor or linear map) is untested here; this record only tests whether
  per-state winners transfer across pools at all.

## What would change the verdict

More states (20 is the binding limit -- the state bootstrap interval is ~2x
the split interval), two disjoint pools per state collected by design, and a
closer-matched model set. DS-0005 provides the first two.

## Threats

n20 scatter only; 5 models, one of which (nfd_residual_warped) is an L20mm
pilot checkpoint far weaker than the rest.

## Unrelated findings

The corpus rasteriser draws each 5 mm cube as a 2x2 px square rotated by its
integer-degree yaw with pixel-snapped corners: zeroing every cube's yaw changes
56% of occupied pixels on randlen_test step-0 rows (occupied pixels 219 vs 199
per image). So models are trained on images whose pixels are dominated by
quantised orientation artifacts (see TODO).
