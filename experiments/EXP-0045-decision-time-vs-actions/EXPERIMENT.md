---
id: EXP-0045
title: >
  Decision time vs number of pushes: past ~0.1 s (CEM) / ~0.3 s (GD) of planning per push
  nothing more is gained, while extra pushes keep paying until the task saturates at
  ~0.31 improvement after ~16 pushes; below that, planning time is critical and model
  speed decides the winner (linear beats worldframe by 0.20 at 0.03 s under CEM)
tier: T1
mode: confirmatory              # Q1-Q4 in DESIGN.md before running
date: 2026-09-25
hypothesis: null
claim: >
  On 32 (goal, start) episodes of 24 pushes (8 goals, DS-0006 starts 40-43, n20 scatter,
  TRAINING_PHYSICS), with tuned GD / CEM and budgets 0.03-1 s per push, for worldframe and
  linear_switched_soft: (Q1) value curves are concave in the number of pushes; (Q2) CEM
  loses <= 0.02 going 1 s -> 0.1 s, GD loses >= 0.03 going 0.3 s -> 0.03 s; (Q3) for
  t_act >= 2 s and T = 8 x (1 + t_act) the best budget is <= 0.3 s; (Q4) the worldframe -
  linear gap at 0.03 s differs from that at 1 s by >= 0.01.
prediction:
  supports: "Q1 concave; Q2 as stated; Q3 best b <= 0.3 s in every (model, planner, t_act >= 2) cell; Q4 |gap(0.03) - gap(1)| >= 0.01"
  refutes: "the reverse of each"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, code/analyse.py
  data: ["DS-0006 states 40-43 (start states)"]
  code_path: "run_episodes_batched (32 envs) -> execute_action + update_material_state; progress lyapunov(occ_for_scoring) after every push"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "2.5 h GPU"
budget:
  declared: "~2.5 h GPU (DESIGN.md)"
  spent: "2.5 h GPU"
  outcome: within
design:
  varied:
    budget: "0.03, 0.1, 0.3, 1.0 s per push (actual mean plan time recorded)"
    model: "nfd_residual_worldframe_noaug_ep43, linear_switched_soft"
    planner: "gd (lr 5e-3, 32 restarts), cem (1024, elite 0.25)"
  held_fixed:
    episodes: "24 pushes; 8 goals x 4 starts, identical across cells"
    physics: TRAINING_PHYSICS
  baselines: "each cell's own 1 s budget; improvement 0 = do nothing"
  metric: "improvement V0 - V_k after k pushes (soft lyapunov); fixed-total-time value V0 - V_k(b), k(b) = floor(T / (plan_time(b) + t_act))"
noise_floor: "paired over 32 episodes (bootstrap CIs); EXP-0043 repeat sd 0.01-0.04 at k = 8"
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  Q1 supported: every curve rises steeply, then flattens at ~0.30-0.315 by k ~ 12-16 for all
  cells with b >= 0.1 s (CEM) / 0.3 s (GD). Q2 supported: CEM 1 s -> 0.1 s loses 0.002-0.013;
  GD 0.3 s -> 0.03 s loses 0.05 (k=1) to 0.16 (k=8). Q3 mostly supported: best b <= 0.3 s in
  7 of 12 cells with t_act >= 2 s; in the other 5 (t_act 5-10 s, where a shorter budget buys
  <= 1 extra push) 1 s wins by <= 0.004. With a shorter total time (4 pushes at 1 s),
  0.1-0.3 s beats 1 s by +0.02-0.05 for t_act <= 2 s. Q4 supported: worldframe - linear is
  -0.20 (CEM) / -0.016 to -0.028 (GD) at 0.03 s vs ~0 at 1 s.
verdict: supported
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
One set of 24-push episodes at four budgets gives the value after every push, so every
(total time, execution time) pair can be evaluated after the fact. The two models differ
in speed (linear gets ~1.8x CEM evaluations), so tight budgets test whether speed changes
the winner.

## What was actually run
RUN-0001 exactly as DESIGN.md (512/512 episodes).

## Numbers (results/analysis.json)
Improvement after k pushes (mean of 32 episodes); actual plan time per push:

| model / planner / budget | plan s | k=1 | k=4 | k=8 | k=16 | k=24 |
|---|---|---|---|---|---|---|
| worldframe / gd / 0.03 | 0.024 | 0.041 | 0.105 | 0.147 | 0.185 | 0.198 |
| worldframe / gd / 0.1 | 0.094 | 0.081 | 0.237 | 0.291 | 0.309 | 0.310 |
| worldframe / gd / 0.3 | 0.295 | 0.090 | 0.243 | 0.304 | 0.313 | 0.314 |
| worldframe / gd / 1 | 0.995 | 0.092 | 0.246 | 0.302 | 0.314 | 0.316 |
| worldframe / cem / 0.03 | 0.037 | 0.032 | 0.061 | 0.068 | 0.064 | 0.063 |
| worldframe / cem / 0.1 | 0.075 | 0.081 | 0.225 | 0.286 | 0.305 | 0.309 |
| worldframe / cem / 0.3 | 0.275 | 0.090 | 0.240 | 0.295 | 0.308 | 0.312 |
| worldframe / cem / 1 | 0.980 | 0.088 | 0.249 | 0.299 | 0.310 | 0.312 |
| linear / gd / 0.03 | 0.027 | 0.038 | 0.117 | 0.163 | 0.207 | 0.226 |
| linear / gd / 0.1 | 0.096 | 0.074 | 0.203 | 0.274 | 0.303 | 0.305 |
| linear / gd / 0.3 | 0.296 | 0.084 | 0.241 | 0.301 | 0.313 | 0.315 |
| linear / gd / 1 | 0.996 | 0.086 | 0.246 | 0.304 | 0.314 | 0.315 |
| linear / cem / 0.03 | 0.041 | 0.084 | 0.220 | 0.270 | 0.294 | 0.295 |
| linear / cem / 0.1 | 0.083 | 0.089 | 0.242 | 0.292 | 0.304 | 0.309 |
| linear / cem / 0.3 | 0.291 | 0.087 | 0.251 | 0.296 | 0.304 | 0.308 |
| linear / cem / 1 | 0.990 | 0.085 | 0.245 | 0.300 | 0.309 | 0.309 |

Fixed total time T = 4 x (1 s + t_act), value for budget 0.03 / 0.1 / 0.3 / 1 s (pushes made):
- t_act 0.5 s, worldframe GD: 0.162 (11) / **0.299 (10)** / 0.295 (7) / 0.246 (4);
- t_act 2 s, worldframe GD: 0.116 / 0.261 / **0.269** / 0.246 (5, 5, 5, 4 pushes);
- t_act 10 s: every budget gets 4 pushes, so 1 s is best or tied (0.246 / 0.249).
Full grid (t_act 0-10 s; T = 4, 8, 16 pushes at 1 s): results/analysis.json "tradeoff".

## Reading
- **Planning time has a knee.** At 0.1 s (CEM) or 0.3 s (GD) the push is already about as
  good as at 1 s. So "more time per push" buys almost nothing beyond ~0.3 s here.
- **More pushes pay only until the task saturates.** Every good cell flattens at ~0.31 by
  about 12-16 pushes. With a real execution time of 0.5-2 s and a tight total time,
  cutting planning to 0.1-0.3 s gains 0.02-0.05 through extra pushes. With 10 s pushes the
  saved time buys no extra push, so the budget does not matter.
- **Tight budgets separate the models, by speed.** At 0.03 s the worldframe NFD fits only
  the initial 1024-candidate scoring into CEM, which is plain ranking, and stalls at 0.06.
  The faster linear model gets one CEM refinement in and reaches 0.30. A single CEM
  iteration is therefore worth ~0.23: ranking pile-aware candidates is what fails (as
  rank did in EXP-0039 and EXP-0042).
- **The ~0.31 plateau explains EXP-0044.** At k = 8, every model is within ~0.01 of its
  plateau. A benchmark that scores the value after 8 pushes at 1 s sits in the saturated
  regime, where models cannot differ. Discriminating settings are: early pushes (k = 1-4),
  tight budgets, or harder goals and states where the plateau is not reached.
  Whether 0.31 is the physical best achievable for these goals, or a limit shared by
  these models, needs an oracle ceiling.

## What would change the verdict
- An oracle (Genesis-as-model) run on the same cells: if it passes 0.31 clearly, the plateau is a model limit, not the task's.
- n50 states or harder goals, where saturation comes later.

## Threats
- 2 models; 8 goals x 4 starts; n20 scatter; lyapunov only.
- Planning time was measured with the simulator idle.
- Real execution time is a free parameter (t_act).

## Unrelated findings
At a 0.03 s budget the worldframe/CEM plateau (~0.063) matches EXP-0039's rank planner
(~0.06). Picking the best of pile-aware candidates without refinement stalls whatever the
model, which points at the candidate distribution rather than the models.
