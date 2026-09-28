# EXP-0042 — one-step planner benchmark: MPC-parameter ablation and planning-time curve (pilot)

Written 2026-09-24 BEFORE running. Mode: EXPLORATORY pilot (sizes the full design;
no prediction is tested here).

## Why
User direction (2026-09-24): strengthen the closed-loop benchmark before growing the
model population -- more goal shapes, ablate MPC parameters (initial pool size, ...),
and later the trade-off between time spent deciding and number of actions. A full
closed-loop cell costs 8 sequential pushes. A ONE-STEP trial costs one push:
plan from a banked DS-0006 state with a learned model under a wall-clock budget,
then simulate only the chosen push. That makes it a cheap ablation bench.
It also checks the closed-loop results, since EXP-0039 has default-setting orderings
it should reproduce (CEM > GD > rank).

## Design
- Model: nfd_residual_worldframe_noaug_ep43 (the best closed-loop model, EXP-0039).
- States: DS-0006 states 0-9; goals: EXP-0037's 6 (quadrant_0, quadrant_3, letter_O,
  letter_T, letter_L, letter_S), lyapunov.
- Candidates: per state, a bank of 32 x 1024 pile-aware pushes drawn once from the
  Genesis sampler (outside the budget; sampling costs milliseconds). Each trial takes
  the first n_cand of its own random permutation of the bank; rank's resampling continues
  down the same permutation.
- Planner: `simple_mpc.learned_mpc.plan` (exactly the closed-loop planner).
  Defaults as EXP-0039: budget 1.0 s, n_cand 64, GD 8 restarts lr 1.5e-3, CEM
  population = 64, elite 0.125.
- Cells (one factor at a time around the defaults; 29 per (state, goal)):
  - budget {0.1, 0.3, 1, 3} s x {rank, gd, cem};
  - n_cand {16, 256} x {rank, gd, cem} at 1 s;
  - GD restarts {1, 32}, GD lr {5e-4, 5e-3};
  - CEM population {256} (pool 64), elite fraction {0.06, 0.25};
  - a REPEAT of the three 1 s default cells (fresh candidate draw), for the
    trial-to-trial noise floor.
- Truth: the chosen push simulated through GenesisOracleEnv.rollout_candidates (full
  fidelity, TRAINING_PHYSICS, 32 envs), banked in DS-0004 under EXP-0037's
  fingerprint, so the 128-push pools already banked for these states are reused.
  Score = capture vs that pool (EXP-0037's grad_capture scale: 0 = pool mean,
  1 = pool best, > 1 = better than every pool push).
- Checkpoint after every state (atomic).

## Quantities (pilot)
Per cell: mean capture (paired over the 60 (state, goal) trials) and the sd of
differences between cells. Also the repeat-cell noise floor, the budget curve per
planner, and whether the default ordering is CEM > GD > rank as in closed loop.
The full design is sized from these numbers.

## Cost
Planning ~29 s per (state, goal) -> ~29 min; simulation ~10 batches of 32 per state
-> ~15 min. About 45 min GPU in total.

## Addendum 1 (2026-09-24, after the pilot): tuning sweep on FRESH states, 2 models
The pilot chose and read the knobs on states 0-9, so the sweep runs on states
10-19 (their pools are banked too). Models: nfd_residual_worldframe_noaug_ep43 and
nfd_3ch_randlen. Cells (1 s unless named): rank default; GD lr {1.5e-3 (default),
5e-3, 1.5e-2, 5e-2} x restarts {8, 32}; CEM (pool = population) {64 (default), 256,
1024} x elite {0.125, 0.25}; plus two decision-time points, GD lr 5e-3 / 32 restarts
at 0.3 s and CEM 256 / elite 0.25 at 0.1 s.
Predictions, written before the sweep:
- T1: GD lr 5e-3 with 32 restarts beats the GD default by >= 0.2 capture for both
  models (the pilot's effects replicate on fresh states);
- T2: CEM 256 beats CEM 64 for both models;
- T3: the best tuned cell per planner has worldframe >= nfd_3ch_randlen, as at the
  defaults, i.e. tuning does not reverse the model order.
