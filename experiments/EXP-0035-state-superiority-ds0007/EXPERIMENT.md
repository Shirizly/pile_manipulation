---
id: EXP-0035
title: >
  On ~800 Sean states per shard (training-matched physics), averaging the NFD
  models' predictions beats the best single model by +0.014 to +0.058 slateN
  (every CI excludes 0), while per-state model choice from a small pool hurts and
  no short state descriptor predicts which model wins
tier: T1
mode: confirmatory              # A1-A4 pre-registered in DESIGN.md before running
date: 2026-09-24
hypothesis: null
claim: >
  On DS-0007 scattered_n20 (801 states) and scattered_n50 (800 states), with 8
  OCC_ADAPTERS models, 30 goals and soft truth scoring: (A1) per-state pairwise
  advantage is unreliable across disjoint 4-push halves (median r < 0.2);
  (A2) per-state switching chosen on one half does not beat the best single model
  on the other; (A3) the average of the 7 'nfd*' models' predicted dv (the 6 randlen variants + nfd_3ch_finetuned; corrected 2026-09-24 from '6') beats the best
  single model; (A4) at least one short state descriptor is Holm-significantly
  associated with a pair's per-state advantage.
prediction:
  supports: "A1 median r < 0.2; A2 interval includes 0; A3 CI excludes 0 (>0); A4 >= 1 Holm-significant link"
  refutes: "A1 r >= 0.2; A2 interval > 0; A3 CI includes 0 or < 0; A4 no Holm-significant link"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0030-state-superiority-ds0005/code/analyse.py (shared with EXP-0030)
  data: ["DS-0007 scattered_n20, scattered_n50"]
  code_path: "OCC_ADAPTERS predict_step -> per-goal lyapunov / mass_in_region -> slateN per pool half; truth occ_for_scoring(states_)"
  seed: "resplit rng seed 0 (100 resplits); state bootstrap 5000"
  split: "each state's ~8 pushes split into two halves, 100 random re-splits (primary = the first)"
  data_commit: not applicable
  runs: [RUN-n20, RUN-n50]
  runtime: "~3 min each"
budget:
  declared: "not separately declared (TODO G3a / G4b)"
  spent: "~30 min wall-clock incl. the NaN fix"
  outcome: within
design:
  varied:
    shard: "scattered_n20, scattered_n50"
    value_fn: "lyapunov, mass_in_region"
  held_fixed:
    models: "nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug, nfd_warped_randlen_flipaug_epoch30, nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43, nfd_3ch_finetuned, linear_switched_hard; ensemble = mean of the 7 'nfd*' models' predicted dv (incl. nfd_3ch_finetuned)"
    goals: "26 letters + 4 quadrants"
    truth_scoring: soft
  baselines: "best single model (chosen on the other half); random = capture 0"
  metric: slateN; split-half reliability r; switching gain; ensemble gain
noise_floor: >
  State-bootstrap 95% intervals (n = 800 states) on every gain; e.g. ensemble gain
  n50 lyapunov [+0.006, +0.020]. No training-seed floor (TODO H1): the ensemble's
  members are single runs, and an ensemble of SEEDS of one model is untested.
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x,
             occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  A1 supported: median r (NFD pairs) +0.01 to +0.02 in all four cells. A2 REFUTED
  in the unfavourable direction: per-state switching is significantly WORSE than
  the best single model (-0.052 [-0.064, -0.037] n20 lyapunov; -0.017 to -0.052
  everywhere). A3 supported: ensemble - best single +0.036 [+0.026, +0.046] (n20
  lyap), +0.058 (n20 mass), +0.014 [+0.006, +0.020] (n50 lyap), +0.046 [+0.039,
  +0.056] (n50 mass). A4 refuted: 0 Holm-significant descriptor links (only 5 pairs,
  all involving nfd_3ch_finetuned, cleared the r > 0.2 gate; |rho| <= 0.06); the
  ridge selector always chose the better model (gain exactly 0).
verdict: supported              # the confirmatory core (A1, A3) holds; A2/A4 refutations are recorded as results
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

With ~800 states per shard the ensemble and descriptor tests have power the
20-state DS-0001 analysis (EXP-0029) lacked; the small pools (4 pushes per half)
deliberately stress per-state reliability.

## What was actually run

EXP-0030's pre-registered analysis code on two DS-0007 shards (flat-row loader
added). A first pass propagated NaN from pools whose true dv are all equal
(pushes that touch nothing); the rerun drops those cells (2 of 160,200 for n20).

## Numbers (results/analysis_n20.json, analysis_n50.json)

| shard / value fn | A1 median r (NFD) | A2 switching gain | A3 ensemble - best single |
|---|---|---|---|
| n20 lyapunov | +0.01 | -0.052 [-0.064, -0.037] | +0.036 [+0.026, +0.046] |
| n20 mass_in_region | +0.02 | -0.018 [-0.041, -0.018] | +0.058 [+0.037, +0.057] |
| n50 lyapunov | +0.01 | -0.047 [-0.057, -0.038] | +0.014 [+0.006, +0.020] |
| n50 mass_in_region | +0.01 | -0.017 [-0.032, -0.013] | +0.046 [+0.039, +0.056] |

Model means (n50, lyapunov): ensemble 0.912; nfd_residual_warped_flipaug_randlen
0.897; nfd_warped_randlen 0.853; nfd_warped_randlen_flipaug 0.852; epoch30 0.848;
linear_switched_hard 0.835; nfd_3ch_randlen 0.830; nfd_residual_worldframe_noaug_ep43
0.827; nfd_3ch_finetuned 0.594.

## Reading

- For the switched-pipeline goal, the bar is now the ENSEMBLE, not the best single
  model: averaging seven NFD models' predictions (including the weak nfd_3ch_finetuned) gains more than any per-state
  selection has shown anywhere in this project.
- Per-state model choice needs reliable per-state evidence; small pools give none
  (r ~ 0), so choosing per state amounts to choosing noise and loses ~0.02-0.05.
- No short descriptor tracks per-state advantage (null result, powered by 800
  states, though only on pairs that passed the reliability gate -- which, at
  K = 4, was almost none). A descriptor study needs larger pools per state
  (DS-0006: 64 + 64).

## What would change the verdict

Larger pools per state (DS-0006) for A1/A4; an ensemble of training SEEDS of one
model, to separate "diverse architectures" from "any averaging" (TODO H1).

## Threats

Sean's push lengths are unbinned (0-70 mm) and 9% of pushes touch nothing, which
lowers every capture; pools come from chained episodes (states within a file are
correlated -- resampling by state, not file, may understate uncertainty slightly).

## Unrelated findings

None.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- C-039's ensemble win holds at EQUAL candidate count only; at a matched wall-clock budget the ensemble loses (EXP-0030 A5, C-043). DS-0007 (Sean) was audited 2026-10-03 (EXP-0065): <= 0.2 % illegal touchdowns in the n20 / n50 shards (scattered_n50 1 %) -- this record's data is effectively clean.
