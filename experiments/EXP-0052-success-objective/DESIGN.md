# EXP-0052 — planning on a success-type objective for letter goals

Written 2026-09-25 BEFORE running.

## Why
EXP-0051: for letters, planning on lyapunov reaches 96% of the achievable lyapunov
improvement but only 0.62-0.68 in-goal mass (optimum 0.92-0.97), and 0/48 letter episodes
reach 90% of the optimum within 24 pushes (quadrants 100%, two_squares 75%). Lyapunov is
nearly satisfied when cubes sit NEAR thin strokes, so the planner stops improving task
success. Test: add the in-goal mass fraction of the predicted image to the planning cost.

## Design
- Batched runner, states recorded; nfd_residual_worldframe_noaug_ep43; tuned gd / cem; 1 s;
  24 pushes; goals letter_O, T, S, L, X, Z + two_squares; starts 40, 41.
- Objective cells: cost = lyapunov - w x in-goal mass fraction (predicted image), w in {1, 3}.
  Control: EXP-0051's lyapunov-only episodes for the same (model, planner, goal, start),
  run with identical code and settings (paired across runs; EXP-0043 repeat sd 0.01-0.04).
- Scored by EXP-0051's analysis: in-goal mass fraction vs optimum at k = 8, 24;
  completion (0.8 / 0.9 of optimum) rate, pushes and time.

## Predictions
O1: w = 1 raises letter in-goal mass at k = 24 by >= 0.05 over lyapunov-only.
O2: completion at 0.9 x optimum rises above 0/12 letter episodes per planner.
O3: lyapunov achieved fraction falls by <= 0.03 (the terms are compatible).
