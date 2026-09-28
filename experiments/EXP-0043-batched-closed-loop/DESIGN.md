# EXP-0043 — batched closed-loop runner: equivalence with EXP-0039 and episode replicate noise

Written 2026-09-24 BEFORE running. Mode: exploratory (a harness check that doubles as a
replicate measurement).

## Why
TODO B1: goal breadth (B2), closed-loop ablation (B3) and the decision-time trade-off
(B4) all need many more episodes. Running 32 episodes side by side in 32 envs gives
~12x simulator throughput (temp/batched-exec-timing).
Before any benchmark result rests on the batched runner, it must produce the same
closed-loop numbers as the sequential EXP-0039 harness. Rerunning EXP-0039's cells
also gives a second, independent replicate of every episode. From that we get the
REPEAT noise of a closed-loop episode (same model, planner, goal and start), which
the power tables need.

## Design
- Runner: `simple_mpc.learned_mpc.run_episodes_batched`, execution via
  `SandboxManipulation.execute_action` + `update_material_state` (real settle), 32
  envs; code/batched_closed_loop.py.
- Cells: EXP-0039's 4 models x {gd, cem} x 4 goals x 4 starts, 1.0 s, 8 pushes,
  n_cand 64 (128 episodes). Rank is left out: its resampling differs by design
  (candidate bank drawn outside the budget).
- Compare with EXP-0039, per (model, planner): mean improvement, paired over the 16
  (goal, start) episodes; Pearson r of the per-episode improvements across the two
  runs; the repeat sd, sd(run2 - run1)/sqrt(2).
- Pass: every (model, planner) mean within 2 SE of EXP-0039, and the same resolved
  model orderings (worldframe best under GD).

## Cost
4 chunks of 32 episodes x 8 steps x (32 x 1 s planning + ~12 s execution) ~ 25 min GPU.
