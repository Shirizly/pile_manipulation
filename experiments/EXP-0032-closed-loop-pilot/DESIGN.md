# EXP-0032 — closed-loop MPC pilot with learned models

Pilot (no claim tested; outcome used only to size TODO G1b). Written before running.
- 3 models of different offline tiers (nfd_warped_randlen_flipaug, linear_switched_hard,
  nfd_3ch_finetuned) x 3 planners (rank, gd, cem) x 2 goals (letter_T, quadrant_0) x
  4 DS-0005 start states; 8 pushes per episode; 1.0 s planning budget per decision;
  64 pile-aware candidates per decision. Truth: soft scoring. Execution: GenesisOracleEnv.step.
- Record: value trajectory, per-step predicted vs true dv, evals per step, wall time.
- Questions: does it run end to end; seconds per episode; episode-to-episode sd of the
  final improvement (sizes the number of episodes the real study needs); are the three
  tiers separated at all under closed loop.
