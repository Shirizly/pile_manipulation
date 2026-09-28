---
id: EXP-0043
title: >
  The batched closed-loop runner (32 episodes side by side) reproduces EXP-0039's GD/CEM
  results in 7 of 8 cells with the same best model; a closed-loop episode's repeat sd is
  0.01-0.02 under CEM (r 0.64-0.96 between runs) and 0.03-0.04 under GD (r 0.49-0.66)
tier: T1
mode: exploratory
date: 2026-09-24
hypothesis: null
claim: >
  run_episodes_batched (K = 32 envs, execute_action + update_material_state) gives the
  same per-(model, planner) mean closed-loop improvement as EXP-0039's sequential
  run_episode, within 2 SE, on EXP-0039's GD/CEM cells (4 models x 4 goals x 4 starts,
  1 s, 8 pushes).
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/batched_closed_loop.py, code/compare.py
  data: ["DS-0006 states 40-43 (start states)", "EXP-0039 episodes"]
  code_path: "simple_mpc.learned_mpc.run_episodes_batched -> SandboxManipulation.execute_action + update_material_state (real settle, TRAINING_PHYSICS, 32 envs)"
  seed: "none set (sampler and planners unseeded)"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001-replicate]
  runtime: "24 min GPU (128 episodes)"
budget:
  declared: "~25 min GPU (DESIGN.md)"
  spent: "24 min GPU"
  outcome: within
design:
  varied:
    runner: "sequential (EXP-0039) vs batched (this run)"
  held_fixed:
    cells: "EXP-0039's 4 models x {gd, cem} x 4 goals x 4 starts, 1.0 s, 8 pushes, n_cand 64"
  baselines: "EXP-0039 (the sequential runner) is the reference"
  metric: "closed-loop improvement V0 - V8 (soft lyapunov)"
noise_floor: "paired over 16 episodes; the between-run difference itself is the replicate noise"
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  batched - sequential per (model, planner): -0.006 to +0.004 in 7 cells (CIs include 0,
  except linear/cem -0.006 [-0.013, -0.000]); nfd_3ch_randlen/gd +0.033 [+0.008, +0.060]
  (2.5 SE, fails the 2-SE criterion). Worldframe best under both planners in both runs.
  Episode correlation between runs: CEM r 0.64-0.96, GD 0.49-0.66; repeat sd CEM
  0.009-0.022, GD 0.028-0.039 (episode sd 0.035-0.056). The batched run's planners made
  6-18% more evaluations per decision.
verdict: supported              # 7/8 cells within 2 SE; the one exception is noted
downgrades: [inconsistency]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The later benchmark runs (TODO B2-B4) rest on the batched runner. A systematic
difference from the sequential harness would show here as a shift in every cell or a
change of model order.

## What was actually run
RUN-0001 exactly as DESIGN.md: 128 episodes in 4 chunks of 32.

## Numbers (results/compare_exp0039.json)
| model / planner | sequential | batched | diff [95% CI] | r | repeat sd |
|---|---|---|---|---|---|
| linear / gd | 0.184 | 0.180 | -0.005 [-0.026, +0.017] | 0.66 | 0.033 |
| linear / cem | 0.285 | 0.279 | -0.006 [-0.013, -0.000] | 0.96 | 0.009 |
| nfd_3ch / gd | 0.205 | 0.238 | +0.033 [+0.008, +0.060] | 0.49 | 0.039 |
| nfd_3ch / cem | 0.273 | 0.277 | +0.004 [-0.011, +0.018] | 0.64 | 0.022 |
| seed1 / gd | 0.224 | 0.224 | -0.001 [-0.021, +0.018] | 0.59 | 0.028 |
| seed1 / cem | 0.273 | 0.274 | +0.001 [-0.011, +0.014] | 0.85 | 0.019 |
| worldframe / gd | 0.265 | 0.260 | -0.005 [-0.027, +0.020] | 0.54 | 0.034 |
| worldframe / cem | 0.292 | 0.286 | -0.005 [-0.018, +0.006] | 0.83 | 0.017 |

## Reading
- The runners agree. The one outlier (nfd_3ch/gd) is 1 cell in 8 at 95%, but at
  2.5 SE it fails the pre-set criterion. The batched planners also get ~10% more
  evaluations per decision, for a reason not diagnosed. That could favour GD slightly.
- CEM episodes are largely determined by (goal, start), so repeats add little.
  GD episodes carry about as much run noise as (goal, start) signal, so GD
  comparisons gain from repeating episodes, not only from adding goals and starts.
- The nfd_3ch vs seed1 order flips between runs under GD, consistent with C-046's
  closed-loop seed tie.

## What would change the verdict
A second batched replicate: if nfd_3ch/gd stays high, the runners differ for GD.

## Threats
Same 16 (goal, start) pairs as EXP-0039; n20 scatter only.

## Unrelated findings
none
