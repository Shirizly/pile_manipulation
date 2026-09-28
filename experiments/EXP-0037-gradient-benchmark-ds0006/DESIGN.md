# EXP-0037 — trusted gradient benchmark (TODO G1a): models as OBJECTIVES to optimise

Written 2026-09-24 BEFORE running (plan gate). Supersedes EXP-0027 item 2's design.

## Fixes relative to EXP-0023 / EXP-0027
training-matched physics everywhere (DS-0006 states; env via TRAINING_PHYSICS);
pool AND optimised pushes evaluated through ONE path (rollout_candidates, full
fidelity, batched by push length -- EXP-0031); soft truth scoring; several GD
restarts per cell (EXP-0027: one GD run is a noisy sample).

## Design
- States: DS-0006 slates 0-39 (40). Pool: each state's 128 corpus pushes,
  RE-SIMULATED through the rollout path (their corpus outcomes are not used).
- Goals (6): quadrant_0, quadrant_3, letter_O, letter_T, letter_L, letter_S (lyapunov).
- Arms (7): nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug,
  nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43,
  linear_switched_soft, ensemble_nfd (mean predicted dv of the 5 NFD arms --
  differentiable, so it can be optimised against).
- Per (state, goal, arm): a_rank = best pool push by predicted dv; 3 GD restarts
  (EXP-0023's projected Adam, 120 steps, lr 1.5e-3) from the top-3 pool pushes by
  predicted dv; a_grad = the restart endpoint with the best PREDICTED dv (what an
  MPC controller would execute).
- Scores (slateN scale, higher = better, per (state, goal)):
  rank_capture = capture of a_rank in the re-simulated pool;
  grad_capture = capture of a_grad (> 1 = beat every pool push);
  gain_capture = grad_capture - rank_capture;
  restart spread = sd over the 3 restart endpoints' TRUE captures.

## Predictions
P1 power: >= 5 of 21 arm pairs Holm-significant on grad_capture (goal-averaged
   per state, paired over 40 states).
P2 ranking vs optimisation: Kendall tau between arms' mean rank_capture (slateN of
   the pick) and mean grad_capture >= 0.6 (slateN would then be a usable proxy for
   gradient-based control). Refuted if <= 0.2.
P3 ensemble: ensemble_nfd has the highest mean grad_capture (and beats the best
   single arm, paired CI excluding 0).
P4 optimiser noise: median restart spread >= 20% of the between-arm sd of
   grad_capture (i.e. restarts matter).

## Cost and checkpoints
Stage 1 (GPU, models only): 40 x 7 arms, goals and restarts batched, ~10 min.
Stage 2 (Genesis): ~128 + 7 x 6 x 4 = 296 pushes per state (deduplicated), ~10
batches x ~20 s -> ~3.5 min/state, ~2.3 h. Every simulated outcome banked in
DS-0004 (sim_path oracle_rollout_full, fingerprint includes the physics); stage 2
checkpoints per state and resumes from the bank if cut off.
