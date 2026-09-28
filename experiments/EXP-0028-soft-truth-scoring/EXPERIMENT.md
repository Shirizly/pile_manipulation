---
id: EXP-0028
title: >
  Switching the harness's ground truth from hard images to the soft,
  mass-conserving scoring barely changes model rankings (Kendall tau 0.91-0.97
  for lyapunov) or paired noise: on these corpora the slate-to-slate noise is
  model x state / pool noise, not truth aliasing
tier: T1
mode: exploratory               # no prediction written before the run; a descriptive re-issue
date: 2026-09-24
hypothesis: null
claim: >
  On the eval_report harness (12 models; L20mm, L40mm, randlen_test; 3-goal and
  30-goal sets), re-scoring ground truth with the soft splat instead of the
  dataset's hard occ1 leaves the lyapunov model ranking essentially unchanged
  and does not materially reduce the paired per-slate noise between models.
provenance:
  commit: 3bae8cd7
  dirty: true                   # soft-scoring code added this session, uncommitted
  script: code/run_both.sh (eval_report --truth-scoring soft); code/compare.py
  data: ["eval_report CORPORA L20mm, L40mm, randlen_test"]
  code_path: "eval_report.truth_for_scoring -> _capture_report(truth_s0) -> paired_stats"
  seed: "bootstrap seed 0"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "~75 min CPU"
budget:
  declared: "part of the scoring change (user request 2026-09-24)"
  spent: "~75 min CPU, background"
  outcome: within
design:
  varied:
    truth_scoring: "image (EXP-0026 RUN-0001 / EXP-0027 RUN-0001) vs soft (this record)"
    goal_set: "default (3) and many (30)"
  held_fixed:
    models_slates: "same 12 models, same slates"
    predictions: "model images, unchanged"
  baselines: "random (capture 0) and persistence (degenerate), both reported by the harness"
  metric: slateN
noise_floor: >
  Paired per-slate sd between models (median over 66 pairs), soft scoring,
  lyapunov: 0.120 / 0.067 / 0.193 (3 goals) and 0.064 / 0.051 / 0.143 (30
  goals) on L20mm / L40mm / randlen_test. No seed-level floor (TODO H1).
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x, randlen-step0-pool-size-128]
establishes: []
result: >
  lyapunov rank agreement image vs soft: tau 0.91-0.97 in all 6 corpus x goal-set
  cells; mean |shift| of model means 0.009-0.014; paired sd moves by <= 0.016
  either way; resolved pairs change by 0-7 of 66. mass_in_region /
  signed_mass move more (tau 0.70-1.00; L20mm signed_mass shifts +0.049).
verdict: supported
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0027 RUN-0005 showed hard-image truth adds noise of ~12% of the
between-action spread to every true dv on a slate-grid state. If that noise
were a large part of what separates models slate-by-slate, soft scoring would
shrink the paired sd and reorder close models; it does neither here.

## What was actually run

RUN-0001: the harness with `--truth-scoring soft`, 12 models, 3 corpora, both
goal sets, compared against the image-scored reports of EXP-0026 RUN-0001 and
EXP-0027 RUN-0001 (same models and slates). Predictions unchanged.

## Numbers (results/compare.json)

| goal set | corpus | lyapunov tau | mean shift | paired sd image -> soft | resolved pairs |
|---|---|---|---|---|---|
| 3 goals | L20mm | 0.94 | +0.004 | 0.136 -> 0.120 | 38 -> 42 |
| 3 goals | L40mm | 0.91 | +0.009 | 0.070 -> 0.067 | 22 -> 27 |
| 3 goals | randlen_test | 0.91 | +0.001 | 0.185 -> 0.193 | 47 -> 47 |
| 30 goals | L20mm | 0.94 | +0.001 | 0.064 -> 0.064 | 45 -> 51 |
| 30 goals | L40mm | 0.91 | +0.009 | 0.055 -> 0.051 | 51 -> 51 |
| 30 goals | randlen_test | 0.97 | -0.001 | 0.141 -> 0.143 | 47 -> 54 |

**Reference table, randlen_test, lyapunov, SOFT scoring, 30 goals** (mean,
95% rank interval, P(first)): nfd_residual_warped_flipaug_randlen 0.930 (1-4,
0.61); nfd_warped_randlen_flipaug 0.926 (1-3, 0.31); nfd_warped_randlen 0.915
(1-6); nfd_warped_randlen_flipaug_epoch30 0.915 (2-5); nfd_residual_worldframe_noaug_ep43
0.902 (3-6); nfd_randlen 0.894 (4-6); linear_switched_res32 0.853 (7);
linear_switched_res64 0.702 (8-9); gnn_randlen_n30 0.690 (8-9);
linear_single_res32 0.600 (10); gnn_l20l40 0.490 (11-12); linear_single_res64
0.478 (11-12). 3-goal version in results/compare.json.

## Reading

The scoring fix is correct and matters where true outcomes are compared
directly (repeats, gradient benchmarks, closed-loop progress), but it is not
what limits the harness's model comparisons: those are limited by genuine
model x state variation and pool sampling (EXP-0029). The reference table's
conclusions stand under either scoring; numbers from the two must still not be
mixed.

## What would change the verdict

A corpus with near-tied models where truth aliasing is a larger share of the
per-slate spread (e.g. small pools of near-identical pushes).

## Threats

Single training run per model (no seed floor).

## Unrelated findings

None.
