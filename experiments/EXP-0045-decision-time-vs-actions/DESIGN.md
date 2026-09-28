# EXP-0045 — decision time vs number of actions (TODO B4)

Written 2026-09-24 BEFORE running.

## Question (user, 2026-09-24)
In a fixed total time, is it better to plan each push longer or to make more pushes?
With planning budget b per push and execution time t_act per push, a total time T
allows k = floor(T / (b_actual + t_act)) pushes. The quantity is V0 - V_k (soft
lyapunov) as a function of b, for given (T, t_act).

## Design
- Batched runner (C-048). Models: nfd_residual_worldframe_noaug_ep43 and
  linear_switched_soft (about 1.8x the evaluations per second under CEM, EXP-0044).
- Planners, tuned (EXP-0042/0044): GD lr 5e-3 / 32 restarts; CEM pool = pop 1024, elite 0.25.
- Budgets b in {0.03, 0.1, 0.3, 1.0} s; 24 pushes per episode; the value after EVERY push
  is recorded, so a single set of episodes gives every (T, t_act) pair after the fact.
- Goals: quadrant_0, quadrant_3, letter_O, T, S, L, X, Z; starts: DS-0006 40-43.
  32 episodes per (model, planner, budget) -> 512 episodes.
- Actual planning time per push is recorded (a planner may overrun a tiny budget by one
  iteration). The trade-off analysis uses the ACTUAL mean time, not the nominal b.
- t_act is a parameter of the analysis, not of the run: reported for t_act in
  {0 (compute-bound), 0.5, 2, 5, 10} s. The real-robot value is for the user to choose.

## Predictions
Q1: the value curve per push V0 - V_k is concave in k for every cell (diminishing returns).
Q2: at one step and at k = 8, b = 0.1 s loses <= 0.02 against b = 1 s under CEM (EXP-0042
    saw no budget effect at one step); under GD b = 0.03 s loses >= 0.03 against 0.3 s.
Q3: for t_act >= 2 s and T = 8 x (1 + t_act), the best b is <= 0.3 s for both planners:
    more pushes beat longer planning.
Q4: at b = 0.03 s the model gap (worldframe - linear) differs from its value at 1 s by >= 0.01,
    i.e. tight budgets change which model wins.

## Cost
Planning 512 x 24 x ~0.36 s ~ 1.2 h; execution 16 chunks x 24 pushes x ~12 s ~ 1.3 h.
About 2.5 h GPU.
