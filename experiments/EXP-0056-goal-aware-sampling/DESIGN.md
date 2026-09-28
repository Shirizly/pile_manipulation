# EXP-0056 — goal-aware selection of the planner's starting candidates

Written 2026-09-25 after the run, from the design proposed to and approved by the user BEFORE
the run (see EXP-0055 DESIGN.md for the timeline); predictions in EXPERIMENT.md written
minutes after launch, before any result was read.

## Why
The pile-aware sampler (the best sampler in EXP-0049) aims the blade at the nearest pile face
whatever the goal: it proposes pushes that move correctly placed cubes, or carry cubes into goal
parts already full. User hypothesis: sampling should weigh down motions that end in full regions
of the goal (related to the earlier EMD-inspired sampler, `simple_mpc/action_sampler.py::
OTGuidedActionSampler`).

## Design
- Same run, model, planner, goals, starts, pushes and metrics as EXP-0055 (cells of EXP-0055
  RUN-0001); value held at lyapunov.
- The planner's 64 starting candidates per push are chosen from a 2048-push pile-aware bank
  (`simple_mpc/goal_aware_sampling.py`, geometric carry model: cubes in the blade swath end at
  the stop line):
  - control `lyap` (pile): the first 64 of the bank;
  - S1 `s_misplaced`: sampled in proportion to the misplaced mass in the swath (cubes outside
    the goal = 1, crowded goal pixels = their over-capacity share), + 0.05 floor;
  - S2 `s_deposit`: softmax (tau 0.5) of the change in placement quality of the carried cubes,
    q = 1 - over-capacity share inside the goal, -min(1, 5 x distance field) outside;
  - S3 `s_otmix`: 32 pushes along the entropic-OT displacement of misplaced cubes (blade one
    cube-width behind the cube; starts landing on a cube rejected) + 32 pile-aware.
- Re-implements the OT sampler's idea rather than using `OTGuidedActionSampler`, whose action
  convention predates the blade-centre perpendicular pushes.
- Do-nothing baseline: k = 0 values. Paired per (goal, start) against `lyap`.

## Predictions (EXPERIMENT.md)
O1: some S with paired thin-goal in-goal mass >= +0.05 over pile-aware at k = 20.
O2: its mean censored completion time lower. Prior stated in the record: small effects, because
GD refinement under lyapunov pulls candidates back to capacity-blind pushes.

## Deviations
- GD only: CEM (where the starting pool matters most) cut for the 3 h cap.
- The crossed cell with the best value function was not run (no sampler won).
