---
id: EXP-0050
title: >
  Simulator-as-model planning is not a ceiling at feasible budgets: greedy best-of-16
  pile-aware reaches 0.08 improvement after 8 pushes, and simulator CEM with 96 simulated
  pushes per decision (0.29 at 8) is slightly WORSE than learned-model CEM/GD with thousands
  of cheap evaluations (0.31-0.33); even the simulator planner leaves letters 0.43-0.65 filled
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  On EXP-0045's 8 goals x DS-0006 start 40 (n20 scatter, TRAINING_PHYSICS, lyapunov), a planner
  that uses the simulator itself as the model, at the budgets that fit (16 greedy / 96 CEM
  simulated pushes per decision), does not exceed learned-model MPC with tuned GD/CEM at 1 s.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/oracle.py
  data: ["DS-0006 state 40 (start)", "EXP-0045 learned-model episodes (b1.0, start 40)"]
  code_path: "Genesis pile-aware sampler -> SandboxManipulation.execute_action per env (episodes batched across 32 envs) -> lyapunov(occ_for_scoring); greedy pick or CEM refit on true outcomes"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "greedy16 30 min; CEM 1 h 50 min (GPU shared)"
budget:
  declared: "~90 min GPU (overnight stage E)"
  spent: "~2.3 h GPU (shared)"
  outcome: exceeded
design:
  varied:
    planner: "simulator greedy best-of-16 (pile-aware); simulator CEM 32 + 2 x 32 refits"
  held_fixed:
    cells: "8 goals (quadrant_0/3, letter O T S L X Z) x start 40; 8 (greedy) / 12 (CEM) pushes"
    physics: TRAINING_PHYSICS
  baselines: "EXP-0045 learned worldframe CEM / GD at 1 s on the same cells"
  metric: "improvement V0 - V_k (soft lyapunov); final in-goal mass fraction"
noise_floor: "8 episodes, one per goal; EXP-0043 repeat sd 0.01-0.04"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  Greedy best-of-16: 0.027 / 0.060 / 0.080 after 1 / 4 / 8 pushes. Simulator CEM: 0.093 / 0.221 /
  0.292 / 0.309 after 1 / 4 / 8 / 12; learned worldframe CEM 0.271 / 0.312 / 0.321 (k 4/8/12),
  learned GD 0.324 / 0.328 (k 8/12). Final in-goal mass of the simulator-CEM episodes: quadrants
  1.00, letters 0.43-0.65 (optimum 0.92-0.97). Addendum 3 (success objective, lyapunov - 1 x
  in-goal mass; letters O T S L + two_squares, start 40, 12 pushes): simulator CEM in-goal mass
  0.54-0.70 (letters) / 0.68 (two_squares) vs learned worldframe on the same objective (EXP-0052)
  GD 0.50-0.72 / 0.95, CEM 0.57-0.72 / 0.89.
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If model error were what limits MPC, a planner with a perfect model should do clearly better.
If it does not, at budgets that can actually be run, model accuracy is not the binding
constraint here.

## What was actually run
See RUN.md and DESIGN.md addenda 1-3. The lookahead check and greedy-64 were cancelled
(greedy from pile-aware candidates is weak, and the cancellation reason was CORRECTED
in addendum 2). The success-objective simulator run is queued.

## Numbers
| goal | simulator CEM k4 / k8 / k12 | learned CEM k4 / k8 / k12 | learned GD k8 / k12 | simulator final in-goal mass |
|---|---|---|---|---|
| quadrant_0 | 0.278 / 0.294 / 0.294 | 0.283 / 0.294 / 0.294 | 0.294 / 0.294 | 1.00 |
| quadrant_3 | 0.236 / 0.290 / 0.290 | 0.253 / 0.290 / 0.290 | 0.287 / 0.290 | 1.00 |
| letter_O | 0.134 / 0.249 / 0.276 | 0.252 / 0.271 / 0.283 | 0.325 / 0.329 | 0.56 |
| letter_T | 0.261 / 0.362 / 0.372 | 0.323 / 0.341 / 0.350 | 0.367 / 0.371 | 0.65 |
| letter_S | 0.188 / 0.276 / 0.302 | 0.247 / 0.285 / 0.321 | 0.306 / 0.329 | 0.43 |
| letter_L | 0.155 / 0.234 / 0.274 | 0.228 / 0.278 / 0.284 | 0.277 / 0.286 | 0.53 |
| letter_X | 0.235 / 0.298 / 0.330 | 0.282 / 0.356 / 0.366 | 0.370 / 0.368 | 0.55 |
| letter_Z | 0.281 / 0.329 / 0.335 | 0.297 / 0.379 / 0.376 | 0.365 / 0.357 | 0.61 |
| mean | 0.221 / 0.292 / 0.309 | 0.271 / 0.312 / 0.321 | 0.324 / 0.328 | |

## Reading
- **At feasible budgets, search beats model accuracy.** A decision costs ~96 simulated
  pushes for the simulator planner vs ~26k model evaluations for learned CEM. The learned
  models' errors cost less than the lost search, so learned MPC is at or above the simulator
  planner on every goal after 4 pushes.
- Pure sampling (greedy best-of-16) is far weaker than refinement, as expected. That result
  does NOT isolate the pile-aware sampler (see the correction in DESIGN.md addendum 2).
- **Even a perfect-model planner does not fill letters when it optimises lyapunov** (in-goal
  mass 0.43-0.65). The objective, not the dynamics model, is what stalls letter completion.
  That points to EXP-0052 (a success-type objective) and to the queued simulator run on that
  objective.
- A true ceiling would need a much larger simulated search (not affordable per decision), or
  a different reference (e.g. a planner on a perfect model at learned-model speed).

## Addendum 3 result: the success-objective simulator planner (results/cem_mass_s40.json)
| goal | simulator CEM in-goal mass k4 / k8 / k12 | learned GD (w1) | learned CEM (w1) | optimum |
|---|---|---|---|---|
| letter_O | 0.40 / 0.51 / 0.54 | 0.42 / 0.69 / 0.71 | 0.41 / 0.62 / 0.71 | 0.96 |
| letter_T | 0.44 / 0.53 / 0.65 | 0.50 / 0.33 / 0.50 | 0.40 / 0.50 / 0.57 | 0.97 |
| letter_S | 0.36 / 0.56 / 0.70 | 0.49 / 0.60 / 0.57 | 0.49 / 0.65 / 0.72 | 0.94 |
| letter_L | 0.32 / 0.47 / 0.55 | 0.57 / 0.62 / 0.72 | 0.46 / 0.52 / 0.60 | 0.92 |
| two_squares | 0.46 / 0.63 / 0.68 | 0.69 / 0.90 / 0.95 | 0.61 / 0.84 / 0.89 | 1.00 |

With a perfect model but ~96 simulated pushes per decision, the success-objective planner fills
letters no better than learned models with thousands of cheap evaluations, and does worse on
two_squares. At feasible budgets, model accuracy is not what limits task success.
Search (budget, and the one-push horizon) is the stronger candidate. One episode per goal, so
per-goal differences are noisy.

## What would change the verdict
A simulator planner with a search budget comparable to the learned one.

## Threats
- One start, 8 episodes.
- CEM initialised from pile-aware samples.
- GPU shared (affects only wall time, not outcomes).

## Unrelated findings
none
