# EXP-0050 — perfect-model MPC ceiling: greedy oracle and 2-push lookahead (overnight stage E)

Written 2026-09-25 BEFORE running. Mode: exploratory (a ceiling measurement).

## Why
EXP-0044/0045: every learned model and planner saturates at ~0.31 improvement. Is that
the task's limit or the models'? An ORACLE planner uses the simulator itself as the model:
each push, it simulates N candidates from the true state and executes the truly best one
(greedy). This isolates search limits (candidate distribution, horizon 1) from model error.
A 2-push lookahead oracle asks whether one-push-at-a-time planning is myopic here.
(EXP-0046 supplies the geometric ceiling V*(g) for normalisation.)

## Design
- Cells: EXP-0045's 8 goals (quadrant_0, quadrant_3, letter_O, T, S, L, X, Z) x DS-0006
  starts 40, 41 (16 episodes), 12 pushes, TRAINING_PHYSICS, full fidelity. Candidates from
  the pile-aware sampler (as in EXP-0045's planners), N = 64 per push.
- Greedy oracle: simulate the 64 candidates (32 envs, batched across episodes), pick the
  lowest true lyapunov; the chosen candidate's simulated outcome IS the next state.
- Lookahead oracle (a subset, starts 40 only: 8 episodes, 8 pushes): K1 = 16 first pushes,
  each followed by K2 = 8 second pushes from its outcome. Choose the first push whose best
  second outcome is lowest; execute only the first push (receding horizon). Baseline:
  greedy with the SAME first-push candidate count (best of the 16).
- Compare with EXP-0045's learned-model curves on the same (goal, start) cells.
- Checkpoint after every push (results JSON, atomic).

## Quantities
Per push k: mean improvement V0 - V_k (oracle vs learned); lookahead - greedy(16) at k = 4, 8.
If the greedy oracle clearly passes ~0.31, the plateau is a MODEL limit; if not, it is a
search / candidate or task limit (then compare with V*).

## Addendum (2026-09-25 01:45, cost only, before any outcome beyond push 1 of greedy16)
With three Genesis jobs sharing the GPU a 32-env batch takes ~22 s. Reduced to fit the
night: greedy N=64 on starts 40 only (8 episodes, 12 pushes); lookahead on 4 goals
(quadrant_0, letter_O, letter_T, letter_S) x start 40, 6 pushes, K1 = 16, K2 = 8. The
greedy best-of-16 baseline (8 goals x start 40, 8 pushes) is unchanged.

## Addendum 2 (2026-09-25 01:50, after greedy16 push 4 = 0.060 mean improvement)
Greedy selection among PILE-AWARE candidates is not a ceiling: even with the true
outcome it reaches only 0.06 after 4 pushes (learned models with CEM refinement: ~0.24).
(Correction 2026-09-25 02:40, user: this does NOT isolate the candidate distribution -- best-of-16 vs CEM's thousands of
refined evaluations confounds pool size with refinement; what it shows is that pure sampling of a small pile-aware pool,
even with a perfect model, is far weaker than refinement. Whether the sampler's SUPPORT is the problem is untested.)
The lookahead and greedy-64 runs (same candidates) are cancelled. Replacement:
simulator-as-model CEM (--mode cem): per push, iteration 0 = 32 pile-aware samples, then
2 Gaussian refits to the 8 true-best, 32 samples each (96 simulated pushes per decision),
the best ever simulated is executed. Cells: the 8 goals x start 40, 8 pushes, compared with
EXP-0045's learned CEM/GD at the same cells. The 2-push lookahead check is deferred (it
needs a good candidate generator first, EXP-0049).

## Addendum 3 (2026-09-25 03:40): the success-objective ceiling
EXP-0051 found letters are not completed under lyapunov. A perfect-model (simulator) CEM
selecting on cost = lyapunov - 1.0 x in-goal mass fraction (EXP-0052's objective), on letter_O,
T, S, L + two_squares, start 40, 12 pushes. It records the in-goal mass and states after every
push. Compared with EXP-0052's learned-model runs on the same objective: this is the ceiling for
"better models".
