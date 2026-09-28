# EXP-0044 — closed-loop goal breadth with tuned planners (TODO B2)

Written 2026-09-24 BEFORE running.

## Why
EXP-0039 used 4 goals x 4 starts and untuned planners. The variance decomposition of its
episodes found goal x model and start x model variance each about as large as the
residual. And EXP-0042 found the planner defaults far from their optima. This run widens
goals and starts and uses the tuned settings. It answers: (a) do model rankings hold
across goal shapes; (b) how much resolving power a 12 x 8 design has (the power table for
the final benchmark); (c) whether tuned GD still trails CEM over 8 pushes.

## Design
- Runner: EXP-0043's batched runner (C-048), 32 envs.
- Models: the EXP-0039 four (nfd_3ch_randlen, nfd_3ch_randlen_seed1, linear_switched_soft,
  nfd_residual_worldframe_noaug_ep43).
- Planners, tuned (EXP-0042 sweep): GD lr 5e-3, 32 restarts, pool 64; CEM pool =
  population 1024, elite 0.25. Budget 1.0 s; 8 pushes.
- Goals (12): quadrant_0, quadrant_1, quadrant_3, letter_O, T, S, L, C, X, H, Z, I.
  Starts: DS-0006 states 40-47. EXP-0039's 4 goals x starts 40-43 are a subset.
  768 episodes, ~2.4 h.
- Closed-loop score: improvement V0 - V8 (soft lyapunov).

## Predictions
P1: worldframe has the best mean improvement under both planners, and is at least tied for
    best on >= 9 of the 12 goals.
P2: >= 4 of 6 model pairs are Holm-resolved under CEM, versus 2/6 at 4 x 4 in EXP-0039.
P3: the tuned GD - tuned CEM gap is within +-0.02 (averaged over models). EXP-0039's
    default GD trailed default CEM by 0.03-0.10.
P4: on EXP-0039's 16 (goal, start) cells, tuned settings improve on EXP-0039/0043's
    default-setting episodes for GD by >= 0.03.
Also reported: variance components (model x goal, model x start, residual) and the
power table.
