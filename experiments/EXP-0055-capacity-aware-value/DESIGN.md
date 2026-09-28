# EXP-0055 — capacity-aware value functions for thin goal shapes

Written 2026-09-25 after the run, from the design proposed to and approved by the user BEFORE
the run (plan at ~08:40, cut by the user to a 3 h cap at ~08:50; predictions in EXPERIMENT.md
written ~09:05, minutes after launch, before any result was read). RUN-0002 was added after
seeing RUN-0001 and is exploratory.

## Why
EXP-0051/0052: thin letters end at ~0.65-0.74 in-goal mass (optimum 0.92-0.97) under lyapunov,
and adding in-goal mass to the cost helps little. Both are LINEAR in the image: each cube is
scored alone, so a cube arriving in an already-full stroke scores like one filling an empty
part (lyapunov = transport cost to the goal set with unlimited capacity). User hypothesis: the
value function must weigh uniform covering of the goal, e.g. by EMD or a potential-field
repulsion. Also needed when the task is to match the shape, not only to end up inside it.

## Design
- Shared batched run with EXP-0056 (`experiments/EXP-0043-*/code/batched_closed_loop.py`,
  states recorded), one cell = one 32-env chunk.
- Model nfd_residual_worldframe_noaug_ep43; planner GD, lr 5e-3, 32 restarts from the best of
  64 pile-aware candidates (bank 2048/env), 1 s per push; 20 free-length 20-70 mm pushes.
- Goals: letter O T S X L I, two_squares (thin; the O1/O2 test set, 28 episodes per cell) and
  quadrant_0 (regression check, O3). Starts: DS-0006 slates 40-43 (scatter).
- Target: uniform distribution over the goal mask (`value_functions.uniform_target`). Ceiling
  for coverage: best of 8 Lloyd layouts of 20 cube centres over the mask (`uniform_layout`).
- Cells (value varied, everything else fixed):
  - `lyap` control (also EXP-0056's control);
  - V1 `emd`: sliced W1 to the target, 32 px, 16 directions (cost measured +11% per candidate
    vs NFD forward);
  - V2 `emd_lin`: lyapunov with the distance field replaced by the entropic-OT dual potential
    from the current observed state to the target, recomputed every push (~35 ms, outside the
    planning budget);
  - V3 `crowd`: lyapunov + 0.1 x sum_goal relu(blur_1.25px(p) - rho*)^2 / rho*, rho* = 1/|mask|;
    lam 0.1 set a priori, not swept (3 h cap).
  - RUN-0002 (exploratory): `crowd_floor` (rho* floored at one isolated cube's blurred density)
    at lam 0.1 and 0.3.
- Do-nothing baseline: the start state (k = 0) of every episode.
- Metrics (METRICS.md): completion_time (headline; in-goal mass >= 0.9 x EXP-0046 optimum, or
  covered_frac >= 0.8 x ceiling; t_act 2 s, censored at 60 s), in-goal mass (mass_in_region)
  rel. optimum, signed_mass fraction, achieved_fraction (lyapunov), coverage_emd (as emd_ach),
  covered_frac (rel. ceiling). Every cell reports all of them, whatever it optimised.
- Paired per (goal, start) against `lyap`, bootstrap 95% CI.
- Budget: 3 h wall for EXP-0055 + EXP-0056 including implementation (user).

## Cut to fit the 3 h cap (user: keep per-cell sample size, cut variants)
Originally proposed: 12 goals x 4 starts, 24 pushes, GD and CEM, a success-objective reference
cell, a 3-value lam screen for V3. Kept per-cell n (8 goals x 4 starts = one full chunk); dropped
CEM, the success-objective cell (EXP-0052 has it, unpaired), the lam screen and clump starts.

## Predictions (EXPERIMENT.md)
O1: some V with paired thin-goal in-goal mass >= +0.05 AND coverage >= +0.05 at k = 20.
O2: that V has a lower mean censored completion time. O3: quadrant_0 in-goal mass drop <= 0.05.

## Deviations
- The planned crossed cell (best V x best S) was not run: no sampler beat pile-aware (EXP-0056).
- RUN-0002 replaced it (exploratory follow-up of the V3 quadrant failure).
