---
id: EXP-0054
title: >
  In closed loop with pushes fixed at 20 mm, the narrow-domain NFDs (most accurate offline)
  complete letter goals WORSE than the broad models: in-goal mass after 24 pushes 0.45 / 0.52
  vs 0.56-0.57, lyapunov progress per push about half; accuracy again fails to predict control
tier: T1
mode: confirmatory              # N1, N2 in DESIGN.md before the narrow models finished training
date: 2026-09-25
hypothesis: null
claim: >
  With actions restricted to exact 20 mm perpendicular pushes (1 s GD, lyapunov - 3 x in-goal
  mass, 24 pushes, letters O T S L X Z + two_squares, DS-0006 starts 40-43), the better narrow
  NFD beats the broad nfd_3ch_randlen on in-goal mass at 24 pushes by >= 0.03 (N1), and model
  order follows EXP-0053 accuracy_1 (N2).
prediction:
  supports: "N1 narrow - broad >= +0.03; N2 order = accuracy order"
  refutes: "N1 < +0.03; N2 order differs"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, code/analyse.py
  data: ["DS-0006 starts 40-43", "EXP-0053 models", "EXP-0046 optima"]
  code_path: "run_episodes_batched(record_states) -> plan(gd, push_len=0.02) on ModelObjective(mass_weight=3) -> execute_action"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "61 min GPU (exclusive)"
budget:
  declared: "~70 min GPU"
  spent: "61 min GPU"
  outcome: within
design:
  varied:
    model: "nfd_3ch_narrow_l20, nfd_3ch_narrow_l20_wide, nfd_3ch_randlen, nfd_residual_worldframe_noaug_ep43"
  held_fixed:
    planner: "gd lr 5e-3, 32 restarts, objective lyapunov - 3 x in-goal mass, push length exactly 20 mm, 1 s, 24 pushes"
    cells: "letters O T S L X Z + two_squares x starts 40-43 (28 episodes per model)"
  baselines: "broad models under the same restriction; EXP-0052 free-length run on the same objective"
  metric: "in-goal mass fraction at k = 8, 24; completion at 0.8 / 0.9 x optimum; lyapunov achieved_fraction"
noise_floor: "28 paired episodes per model; EXP-0043 repeat sd 0.03-0.04 (GD, lyapunov)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  In-goal mass at 8 / 24 pushes (start 0.04): narrow NFD 0.37 / 0.45, narrow wide 0.38 / 0.52, broad
  nfd_3ch_randlen 0.36 / 0.56, broad worldframe 0.35 / 0.57. Lyapunov achieved at 24: 0.27 / 0.43 /
  0.59 / 0.61. Completion at 0.8 x optimum: 1/112 episodes. True lyapunov change per push: -0.004
  (narrow), -0.006 (wide), -0.008 / -0.009 (broad). N1 refuted (narrow - broad = -0.11 / -0.04);
  N2 refuted (the most accurate model is last).
verdict: refuted
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The same planner and objective run on each model in the action space the narrow models were
trained on. If better offline accuracy translated into control, the narrow NFD should lead.

## What was actually run
RUN-0001 as DESIGN.md (112/112 episodes; the GPU was not shared).

## Numbers (results/analysis.json)
| model | EXP-0053 accuracy_1 / slateN | in-goal mass k=8 / k=24 | lyapunov achieved k=24 | true lyapunov change per push |
|---|---|---|---|---|
| nfd_3ch_narrow_l20 | 0.506 / 0.774 | 0.37 / 0.45 | 0.27 | -0.0039 |
| nfd_3ch_narrow_l20_wide | 0.486 / 0.803 | 0.38 / 0.52 | 0.43 | -0.0061 |
| nfd_3ch_randlen | 0.453 / 0.695 | 0.36 / 0.56 | 0.59 | -0.0084 |
| nfd_residual_worldframe_noaug_ep43 | 0.466 / 0.721 | 0.35 / 0.57 | 0.61 | -0.0087 |

For reference, EXP-0052 (free push length 20-70 mm, same objective, worldframe GD, starts
40-41) reached 0.74 in-goal mass for letters at 24 pushes.

## Reading
- **Offline accuracy and slateN on the narrow test set do not predict closed-loop success
  here.** The order reverses: the most accurate (and a top-slateN) model makes the least
  progress. This is the "accuracy is a bad control predictor" result again, now on a clean
  narrow domain and scored on task success.
- A candidate mechanism, not tested: GD follows the model's gradient. A model trained only on
  exact 20 mm pushes from pile-aware starts may have a flatter or misleading landscape over push
  START position (the only free parameters under the restriction), even while its predictions
  at sampled pushes are accurate. The broad models saw varied starts and lengths.
- **Fixing pushes at 20 mm makes the task far slower:** 24 short pushes move much less material
  than 20-70 mm ones (in-goal mass 0.45-0.57 vs 0.74), so completion time, the true utility,
  gets much worse. Long pushes matter for gathering material; short ones could matter for
  final placement. A mixed-length action space is the natural next step.

## What would change the verdict
- CEM instead of GD under the same restriction (it does not rely on the model's gradient).
- Longer episodes; more starts; model seeds.
- A two-phase strategy (long gathering pushes, then 20 mm placement pushes using the narrow model).

## Threats
- One planner (GD) and one objective.
- 4 starts.
- Pushes near walls were clipped below 20 mm by the projection (min 7 mm).
- The narrow models' training sampler (pile-aware starts) differs from the planner's search space.

## Unrelated findings
none
