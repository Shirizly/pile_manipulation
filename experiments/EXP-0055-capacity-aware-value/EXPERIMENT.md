---
id: EXP-0055
title: >
  Capacity-aware value functions for thin goals: only the over-capacity repulsion (V3) helps,
  and modestly (+0.03 in-goal mass, +0.075 coverage vs lyapunov, completion time unchanged);
  exact EMD trades in-goal mass for its own EMD; the linearised EMD field fails (overshoot)
tier: T1
mode: confirmatory              # predictions written 2026-09-25 ~09:05, minutes after launch, before any result was seen
date: 2026-09-25
hypothesis: null
claim: >
  On thin goals (letters O T S X L I + two_squares; DS-0006 scatter starts 40-43; worldframe NFD;
  tuned GD, 1 s per push; 20 free-length 20-70 mm pushes; pile-aware candidates), replacing
  lyapunov -- which is linear in the image and so blind to goal capacity -- by a capacity-aware
  value function raises final in-goal mass (relative to the EXP-0046 optimum) by >= 0.05 and
  final covered-area fraction (relative to the uniform-layout ceiling) by >= 0.05, paired per
  (goal, start), for at least one of V1 emd / V2 emd_lin / V3 crowd (O1); that variant also
  shortens the mean censored completion time (in-goal mass >= 0.9 x optimum, t_act = 2 s) (O2);
  and on quadrant_0 it costs <= 0.05 in-goal mass (O3).
prediction:
  supports: "O1: some V with paired mass_rel_k20 >= +0.05 and cover_rel_k20 >= +0.05 over lyap on the 28 thin-goal episodes; O2: its time_mass_tact2 < 0; O3: quadrant_0 mass_rel drop <= 0.05"
  refutes: "no V reaches +0.05 on both mass_rel and cover_rel"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, code/analyse.py
  data: ["DS-0006 starts 40-43", "EXP-0046 optima (vstar.json)"]
  code_path: "simple_mpc/value_functions.py -> learned_mpc.ModelObjective(value=...) -> plan(gd) -> run_episodes_batched(record_states)"
  seed: "none set (Genesis sim deterministic per start; planner restarts from the candidate bank)"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001, RUN-0002]
  runtime: "RUN-0001 1.8 h GPU (7 cells, shared with EXP-0056); RUN-0002 0.5 h (2 exploratory cells)"
budget:
  declared: "3 h wall total for EXP-0055 + EXP-0056 (user, 2026-09-25)"
  spent: "~3 h wall (08:50-11:50) including implementation, both experiments and the visualisations"
  outcome: within
design:
  varied:
    value: "lyap (control); emd = V1 sliced W1 (32 px, 16 dirs) to the uniform goal target; emd_lin = V2 lyapunov with the distance field replaced by the entropic-OT dual potential from the current observed state to the target (recomputed every push); crowd = V3 lyapunov + 0.1 x sum_goal relu(blur_1.25px(p) - rho*)^2 / rho*"
  held_fixed:
    model: nfd_residual_worldframe_noaug_ep43
    planner: "gd, lr 5e-3, 32 restarts from the best of 64 pile-aware candidates, 1 s"
    goals: "letter O T S X L I, two_squares, quadrant_0 (regression check); starts 40-43; 20 pushes"
    crowd_lam: "0.1, set a priori (a cube crowding a full part costs about as much as sitting ~7 px outside the goal); not swept -- the user's 3 h cap"
  baselines: "lyap cell of the same run (do-nothing = the start state, k = 0, reported as the starting values of every curve)"
  metric: "completion_time (headline), mass_in_region fraction rel. optimum, signed_mass fraction, achieved_fraction (lyapunov), coverage_emd, covered_frac"
noise_floor: "4 starts x 7 thin goals = 28 paired episodes per cell; paired bootstrap CI; EXP-0043 repeat sd 0.01-0.04 lyapunov"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  Thin goals, 28 paired episodes, k = 20, variant minus lyapunov [95% CI]: V3 crowd in-goal mass
  +0.033 [+0.003, +0.062], coverage +0.075 [+0.041, +0.105], coverage EMD achieved +0.066, signed
  +0.061, lyapunov achieved +0.009; completion time (mass, t_act 2 s, censored) +1.7 s
  [-0.5, +4.4]. V1 emd: its own EMD +0.244 but in-goal mass -0.081, coverage -0.002. V2 emd_lin:
  in-goal mass -0.52, coverage -0.45 (median push 67 mm vs 40, knocks cubes out through thin
  strokes). O3 fails for V3: quadrant_0 in-goal mass 0.22 vs 1.00 (capacity below one cube).
  Exploratory follow-up (RUN-0002, chosen after seeing RUN-0001): capacity floored at one cube's
  density, lam 0.1: thin in-goal mass +0.017 [-0.024, +0.058], coverage +0.054 [+0.003, +0.099],
  EMD +0.128, quadrant_0 0.79; lam 0.3 worse (-0.034 mass). Repeat noise between two cells
  identical on 5 goals (crowd vs crowd_floor): per-episode sd 0.09 in-goal mass, per-goal means
  differ by up to 0.12.
verdict: refuted                # O1: no variant reaches +0.05 on BOTH in-goal mass and coverage; O2 not met; O3 fails for V3
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Lyapunov and the EXP-0052 in-goal-mass term are both linear in the image: every cube is scored
independently, so no amount of planning makes the controller prefer an empty part of a stroke
over a full one. Each variant here makes goal capacity visible to the planner. If thin letters
stay at ~0.65 in-goal mass under all three, capacity-blindness of the value is not what limits
them.

## What was actually run
RUN-0002 (exploratory, chosen after RUN-0001): crowd_floor at lam 0.1 and 0.3, same
settings; merged with RUN-0001 into results/all.json for analysis.
RUN-0001 (runs/RUN-0001): one batched run, 7 cells x 32 episodes, shared with EXP-0056 (cells
s_misplaced, s_deposit, s_otmix use the lyap value with goal-aware candidate selection).
Predictions written after launch, before any result was read. Chunk overhead: the planning
budget is 1 s for all 32 envs, i.e. ~32 s per step + ~14 s simulation.

## Cost measurements (bench on this GPU, 1024 candidates, forward + backward)
lyapunov 0.5 ms; sliced EMD 16 dirs 47 ms at 64 px, 10 ms at 32 px; Sinkhorn 30 it 134 ms at
64 px, 37 ms at 32 px; NFD predict 90 ms; linear 38 ms. The V2 potential costs ~35 ms per push
per env, outside the planning budget (as the candidate bank is).

## Numbers (results/analysis_all.json; `code/analyse.py all`)
All 32 episodes per cell, k = 20 (thin + quadrant_0):

| cell | lyap achieved | in-goal mass rel. opt | signed | EMD achieved | coverage rel. ceiling | complete 0.9 (mass) | mean time t_act 2 s |
|---|---|---|---|---|---|---|---|
| lyap | 0.97 | 0.82 | +0.54 | 0.73 | 0.66 | 0.31 | 52 s |
| crowd (V3) | 0.96 | 0.75 | +0.40 | 0.76 | 0.69 | 0.06 | 59 s |
| emd (V1) | 0.95 | 0.74 | +0.40 | 0.98 | 0.69 | 0.19 | 56 s |
| emd_lin (V2) | 0.56 | 0.33 | -0.36 | 0.35 | 0.27 | 0.06 | 58 s |
| crowd_floor (expl.) | 0.96 | 0.80 | +0.52 | 0.83 | 0.71 | 0.19 | 53 s |
| crowd_floor lam 0.3 (expl.) | 0.96 | 0.77 | +0.45 | 0.78 | 0.69 | 0.06 | 59 s |

Per goal (in-goal mass / coverage, rel.): V3 gains on O (0.83/0.71 vs 0.74/0.54), X (0.82/0.68
vs 0.75/0.55), T (0.82/0.79 vs 0.77/0.73); per-goal differences below ~0.12 are within repeat
noise (4 starts per goal).

## Reading
- Capacity-blindness of the value is real but explains only a small part of the thin-goal gap:
  the one variant that makes crowding costly adds ~0.03 in-goal mass and ~0.05-0.08 coverage,
  and does not complete goals sooner.
- Lyapunov itself churns: after push 5 it brings ~2.2 cubes per push into the goal and knocks
  ~1.8 out (recorded states); V2's field makes that worse, because a first-order potential keeps
  decreasing OUTSIDE the goal toward the empty part, so pushing through a stroke looks good.
- Exact EMD (V1) optimises uniform spreading, which on thin letters costs "all mass inside".
  Coverage and in-goal mass are different tasks; the metric has to be chosen with the task.
- A capacity below one cube (1/|mask| for goals much larger than the cubes' area) makes V3
  repel cubes from the goal; the floor fixes most of it (quadrant 0.79, not 1.00).

## What would change the verdict
- CEM with these values (pool-based, not gradient) -- ~16 min per cell.
- lam and blur sweep for crowd_floor on more starts (noise per episode sd 0.09 needs ~4x the
  episodes to resolve +0.03) -- ~1 h.
- A model better at multi-cube placement: with lyapunov churn at ~1.8 cubes out per push the
  planner may be limited by prediction of where cubes stop, not by the value.

## Unrelated findings
- Every chunk plans for all 32 envs including padding, so a partial chunk costs a full one's
  planning time (now noted in CODEMAP).
