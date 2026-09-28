# EXP-0040 -- smoothness of each model's predicted objective along action-space lines

Written retrospectively 2026-09-25 from EXPERIMENT.md (frontmatter), code/smoothness.py (docstring and
constants) and results/smoothness.json, not before the run. No pre-run plan exists in TODO.md; no
command-ledger entry was found for this experiment.

## Why
EXP-0037 (gradient benchmark on DS-0006) found the residual NFD arms optimise best under gradient
descent. Candidate explanation tested here: those arms have a smoother predicted objective (lower
roughness, fewer local minima) along lines through good pushes.

## Design
- States: DS-0006 states 0-9 (the first 10 of EXP-0037's stage-1 states, `artifacts/RUN-0001/stage1.pt`).
- Goals: EXP-0037's stage-1 goal set (read from stage1.pt; 6 goals per the TODO.md G1a note).
- Arms (7, all of EXP-0037's): nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug,
  nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43, linear_switched_soft,
  ensemble_nfd (mean of the member predictions).
- Lines: per state 16 random unit directions in the 4-D action (rng seed 0, shared by all arms),
  centred on the arm's own top rank pick per goal (EXP-0037 `a_rank`), +/-8 mm, 41 points,
  projected with `learned_mpc.project_push`.
- Per line: predicted lyapunov dv (OCC_ADAPTERS predict_step); roughness = mean |2nd difference| /
  profile range; number of interior local minima. Averaged per arm.
- Varied: arm. Held fixed: states, goals, line directions, amplitude.
- Outcome: Spearman across the 7 arms of roughness / local minima vs EXP-0037 gain_capture,
  grad_capture, rank_capture (definitions in EXP-0027/EXP-0023 DESIGN.md; METRICS.md
  `gradient_gain` family).
- Baselines: none (no do-nothing or random-direction control; the comparison is across arms).
- Sample size: 10 states x 16 lines x G goals per arm; n = 7 arms for the correlation.
- Budget: not recorded.

## Predictions
Exploratory: no pre-registered prediction. The frontmatter claim (residual arms' GD advantage is
explained by a smoother objective) is the hypothesis the result refutes.

## Deviations
- No pre-run plan or budget was recorded, so deviations from plan cannot be assessed.
- Only 10 of EXP-0037's 40 states were used (the script's STATES constant); the reason is not recorded.
