---
id: EXP-0051
title: >
  Lyapunov says the tasks are 96% done, but letter goals are not: after 24 pushes only
  0.62-0.68 of the mass is inside the letter (optimum 0.92-0.97) and 0/48 letter episodes
  reach 90% of the optimum; quadrants complete in ~5 pushes and two_squares in 75% of episodes
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  With tuned GD/CEM and two learned models (1 s per push, 24 pushes, n20 scatter starts),
  lyapunov-optimising MPC places most of the mass inside quadrant and two-square goals,
  but not inside thin letter goals: in-goal mass fraction and completion rate stay far
  below the per-goal optimum although the lyapunov achieved fraction is ~0.95.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py --record-states, code/analyse.py
  data: ["DS-0006 states 40-41 (start states)", "EXP-0046 per-goal optima (V*, mass_frac_best_placement)"]
  code_path: "run_episodes_batched(record_states) -> occ_for_scoring(states) -> mass_in_region / signed_mass_in_region / total mass"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "1.4 h GPU (shared)"
budget:
  declared: "~75 min GPU"
  spent: "84 min GPU"
  outcome: exceeded
design:
  varied:
    model: "nfd_residual_worldframe_noaug_ep43, linear_switched_soft"
    planner: "gd (lr 5e-3, 32 restarts), cem (1024, elite 0.25)"
    goal: "quadrant_0, quadrant_3, letter O T S L X Z, two_squares"
  held_fixed:
    budget: "1.0 s per push, 24 pushes"
    starts: "DS-0006 40, 41"
  baselines: "per-goal optimum placement (EXP-0046); start state (k = 0)"
  metric: "achieved_fraction (lyapunov), in-goal mass fraction, signed-mass fraction, completion pushes/time at theta x optimum in-goal mass"
noise_floor: "18 episodes per (model, planner); EXP-0043 repeat sd 0.01-0.04 (lyapunov)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  All 72 episodes: in-goal mass 0.10 at start, 0.50 / 0.68 / 0.75 after 4 / 8 / 24 pushes
  (0.52 / 0.70 / 0.77 of the optimum); lyapunov achieved 0.74 / 0.93 / 0.96. Letters: in-goal
  mass at 24 pushes 0.62-0.68 vs optimum 0.92-0.97, completion at 0.9 x optimum 0/48.
  Quadrants: 0.99-1.00 by push 8, all complete. two_squares: 0.89 at 24, 75% complete.
  The four model/planner cells differ by <= 0.03 in in-goal mass.
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
EXP-0046's "95% achieved" is in lyapunov terms. Mass inside the goal is what task
success means, so a gap between the two would show that lyapunov-optimising MPC can
look done without being done.

## What was actually run
RUN-0001 as DESIGN.md (72/72 episodes). GPU shared with other Genesis jobs, so the
planners likely got fewer evaluations per second than in EXP-0044/0045.

## Numbers (results/analysis.json)
| goal | optimum in-goal mass | in-goal mass k=8 | k=24 | lyapunov achieved k=24 | complete at 0.9 x opt |
|---|---|---|---|---|---|
| letter_L | 0.92 | 0.55 | 0.65 | 0.96 | 0.00 |
| letter_O | 0.96 | 0.58 | 0.62 | 0.92 | 0.00 |
| letter_S | 0.94 | 0.54 | 0.63 | 0.95 | 0.00 |
| letter_T | 0.97 | 0.54 | 0.62 | 0.94 | 0.00 |
| letter_X | 0.95 | 0.57 | 0.64 | 0.94 | 0.00 |
| letter_Z | 0.95 | 0.59 | 0.68 | 0.94 | 0.00 |
| quadrant_0 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |
| quadrant_3 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 |
| two_squares | 1.00 | 0.77 | 0.89 | 0.97 | 0.75 |

Completed episodes (quadrants, most two_squares) finished in a median of 5-8 pushes. At
1 s planning plus a 2 s push, that is ~15-24 s of task time. The median pushes are computed
only over completed episodes, so they are not comparable across cells with different
completion rates.

## Reading
- **Lyapunov is a poor success measure for thin goals.** The distance field is near zero
  next to a stroke, so cubes resting just outside a letter cost almost nothing. MPC drives
  the mass to the letter's neighbourhood (lyapunov 0.95), then stops improving what
  matters (in-goal mass stuck at ~2/3 of possible).
- **Letters are therefore NOT solved,** and they are exactly where better planning
  objectives, precision and models can matter. Quadrants are easy (done in ~5 pushes) and
  two_squares is in between.
- Model and planner barely matter here either (<= 0.03). The bottleneck is what is being
  optimised, and possibly precision, not which of these models plans.
- From now on every result reports in-goal mass / signed mass beside lyapunov (user rule,
  2026-09-25).

## What would change the verdict
Planning on a success-type objective (EXP-0052) reaching completion would show the
objective, not the dynamics precision, is what stalls. A perfect-model (simulator) planner
on the same objective would bound what precision could still add.

## Threats
- 2 starts; one tuned setting per planner; GPU shared.
- Completion threshold relative to a soft-splat optimum (letters cannot reach 1.0).

## Unrelated findings
CEM/GD-chosen pushes start a median 13 mm (90th pct 33 mm) before the first cube in the
swath, with lengths 20-63 mm. Pile-aware starts ~one cube width from the pile face. That
hints the pile-aware start rule excludes pushes the planners prefer; not tested against
pile-aware statistics from the same states.
