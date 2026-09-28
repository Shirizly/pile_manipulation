# EXP-0051 — task success and task-completion time (user 2026-09-25)

Written 2026-09-25 BEFORE running.

## Why
EXP-0046: learned-model MPC reaches 87-96% of the achievable LYAPUNOV improvement. But
lyapunov is lenient: it can be mostly satisfied while cubes still sit just outside the
goal. User: the benchmark's one true utility is TASK COMPLETION TIME -- total time
including action execution time, not simulation time -- and success-type value
functions (mass_in_region, signed_mass) must always be reported beside lyapunov.

## Design
- Batched runner with final/per-push particle states recorded (--record-states).
- Models x planners (tuned): {nfd_residual_worldframe_noaug_ep43, linear_switched_soft} x
  {gd (lr 5e-3, 32 restarts), cem (1024, elite 0.25)}; budget 1.0 s; 24 pushes.
- Goals (9): quadrant_0, quadrant_3, letter_O, T, S, L, X, Z, two_squares (new, EXP-0046);
  starts: DS-0006 40, 41. 72 episodes.
- Success per push k (soft scoring, occ_for_scoring): mass fraction inside the goal mask
  (mass_in_region / total mass), and signed-mass fraction; both relative to the
  per-goal optimum from EXP-0046 (letters cannot reach 100%: soft-splat spill).
- Task completion: the first push k at which the in-goal mass fraction reaches theta x its
  optimum, theta in {0.8, 0.9, 0.95}; censored at 24 pushes. Completion time =
  sum over pushes up to k of (actual planning time + t_act), t_act in {1, 2, 5, 10} s.
  Median and censoring rate per (model, planner, goal).
- GPU is shared with other Genesis jobs during this run, so planners may get fewer
  evaluations per second than in EXP-0044/0045; recorded.

## Questions (descriptive)
S1: at 8 and 24 pushes, what fraction of the mass is inside the goal (vs optimum)?
S2: what fraction of episodes completes (theta = 0.9) within 24 pushes, and how fast?
S3: does lyapunov's "95% achieved" correspond to near-complete tasks, or not?
