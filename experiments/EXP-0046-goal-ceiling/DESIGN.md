# EXP-0046 -- goal ceilings V*(g), achieved fraction and goal redundancy (overnight stage A)

Written retrospectively 2026-09-25 from EXPERIMENT.md, code/vstar.py and code/analyse.py, runs/
RUN-0001/RUN-0002, the command ledger and the TODO.md overnight plan (stage A), not before the run.
The pre-run plan is quoted where TODO.md records it.

## Why
EXP-0044/0045: every tuned model and planner saturates at ~0.31 lyapunov improvement. Is that the
task's limit (goals unreachable in principle) or a planner/model shortfall? Also: does the
lyapunov optimum also satisfy mass-based success, and which of the 30 goals are redundant?
Pre-run plan (TODO.md stage A, "CPU agent, 45 min"): "goal ceiling V*(g) for every goal
(lyapunov-optimal placement of 20 cube footprints, scored with the soft scorer); check the
lyapunov-optimal state also fully satisfies mass_in_region and signed_mass; add a
two-disconnected-squares goal; goal redundancy analysis (shape + per-goal performance similarity)
-> droppable goals; normalised re-score (V0-V)/(V0-V*) of EXP-0039/0044/0045."

## Design
- Goals: eval_report `many_plus` = the 30 `many` goals (letters A-Z, quadrants 0-3) + the new
  `two_squares` (goals.py `two_squares_mask`).
- V* (RUN-0001, CPU): 20 single-layer, non-overlapping, axis-aligned 5 mm cubes on a 0.5 mm grid
  inside the tray; cost = lyapunov under occ_for_scoring (sigma 1 px, 64 px / +-64 mm), computed
  as a per-cube separable cost map (checked against the real scorer to 1e-9). Packer: greedy +
  lattice offsets + 12 noisy-greedy restarts + coordinate descent (an achievable upper bound on
  the optimum). Same packer also maximises mass_in_region. Also capacity and stroke width.
- Achieved fraction (RUN-0002): METRICS.md `achieved_fraction` = (V0 - V_k)/(V0 - V*) per episode,
  on EXP-0044 (768 episodes, tuned, 1 s) and EXP-0045 (512 episodes, budgets 0.03-1 s), DS-0006
  starts 40-47, k = 8 and 24.
- Redundancy (RUN-0002): EXP-0030 RUN-0001 truth/pred dv (DS-0006, 160 states x 128 pushes,
  8 models): per-state Spearman of true dv between goals, mask IoU, distance-field correlation,
  per-goal slateN and model-ranking Kendall; average-linkage clustering on 1 - push-rank corr.
- Baselines: do-nothing (V0, achieved_fraction 0); theoretical max mass (all mass inside).
- Metrics: `achieved_fraction`, `lyapunov`, `mass_in_region` / `signed_mass_in_region` over total.
- Budget: declared 45 min CPU; spent ~35 min.

## Predictions
Exploratory: no pre-registered prediction block. The frontmatter claim states the expected outcome
(V* <= 0.01 for all goals; mass_in_region < 100% at the optimum for every letter; achieved_fraction
>= 0.8 after 8 pushes at budgets >= 0.1 s), but it is not recorded as written before the run.

## Deviations
- The planned normalised re-score covered EXP-0044 and EXP-0045 only; EXP-0039 was not re-scored.
- RUN-0001 was run three times (ledger 01:37-01:38): noisy-greedy restarts and padded EDT added,
  then the mass optimiser seeded from the lyapunov optimum. The final run is the one reported.
- Redundancy uses one-step pools from EXP-0030, not closed-loop per-goal performance.
