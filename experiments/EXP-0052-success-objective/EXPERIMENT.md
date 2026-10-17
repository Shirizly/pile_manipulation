---
id: EXP-0052
title: >
  Adding in-goal mass to the planning cost helps letter goals only a little: +0.03 to +0.09
  in-goal mass after 24 pushes (best: GD, weight 3, 0.74 vs 0.65), more episodes reach 80% of
  optimum, but 90% of optimum is still reached in only 1 of 48 letter episodes
tier: T1
mode: confirmatory              # O1-O3 in DESIGN.md before running
date: 2026-09-25
hypothesis: null
claim: >
  For letter goals (O T S L X Z; DS-0006 starts 40-41; worldframe NFD; tuned GD/CEM; 1 s; 24
  pushes), planning on lyapunov - w x in-goal mass fraction (w = 1) raises the in-goal mass at
  24 pushes by >= 0.05 over lyapunov-only (O1), lifts completion at 0.9 x optimum above 0/12 per
  planner (O2), and costs <= 0.03 of lyapunov achieved fraction (O3).
prediction:
  supports: "O1 >= +0.05; O2 > 0/12; O3 drop <= 0.03"
  refutes: "O1 < +0.05; O2 = 0; O3 drop > 0.03"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, code/analyse.py
  data: ["DS-0006 starts 40-41", "EXP-0051 lyapunov-only control episodes", "EXP-0046 optima"]
  code_path: "learned_mpc.ModelObjective(mask, mass_weight) -> plan (gd / cem) -> run_episodes_batched(record_states)"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "46 min GPU (shared)"
budget:
  declared: "~45 min GPU"
  spent: "46 min GPU"
  outcome: within
design:
  varied:
    objective: "lyapunov (EXP-0051 control), lyapunov - 1 x in-goal mass, lyapunov - 3 x in-goal mass"
    planner: "gd, cem (tuned)"
  held_fixed:
    model: nfd_residual_worldframe_noaug_ep43
    goals: "letter O T S L X Z + two_squares; starts 40, 41; 24 pushes; 1 s"
  baselines: "lyapunov-only episodes (EXP-0051, same cells)"
  metric: "in-goal mass fraction at k = 8, 24; completion at 0.8 / 0.9 x optimum; achieved_fraction"
noise_floor: "12 letter episodes per (planner, objective), paired with the control; control from a separate run (EXP-0043 repeat sd 0.01-0.04 lyapunov)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  Letters, in-goal mass at 24 pushes (objective vs control): w1 GD 0.70 vs 0.65 (+0.056
  [-0.002, +0.115]), w1 CEM 0.69 vs 0.66 (+0.033), w3 GD 0.74 vs 0.65 (+0.090 [+0.008, +0.170]),
  w3 CEM 0.70 vs 0.66 (+0.035). Completion at 0.8 x optimum rises (e.g. w1 CEM 0.58 vs 0.25,
  w3 GD 0.67 vs 0.42); at 0.9 x optimum 1/48 letter episodes (w1 CEM) vs 0/24 control. Lyapunov
  achieved drops 0.00-0.05. two_squares (n = 2 per cell) mixed, w3 GD worse.
verdict: refuted                # 2026-10-15 corrected: pre-registered w=1 cells: GD meets O1 (+0.056 >= 0.05, CI [-0.002, +0.115] incl. 0) and O3 (drop 0.006) but fails O2 (0/12); CEM fails O1 (+0.033) and O3 (drop 0.0302 > 0.03), meets O2 (1/12). No w=1 planner meets O1-O3 together -> refuted stands (results/analysis.json)
downgrades: [imprecision, provenance]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If the lyapunov objective alone were what stops letter completion (EXP-0051), putting
in-goal mass into the cost should close most of the gap to the optimum.

## What was actually run
RUN-0001 as DESIGN.md (56/56 episodes). The control comes from a different run (EXP-0051)
with identical code and settings, hence the `provenance` downgrade.

## Numbers (results/analysis.json)
| objective / planner | in-goal mass k=8 (ctrl) | k=24 (ctrl) | diff k=24 [95% CI] | complete 0.8 (ctrl) | complete 0.9 |
|---|---|---|---|---|---|
| w=1 / gd | 0.61 (0.59) | 0.70 (0.65) | +0.056 [-0.002, +0.115] | 0.50 (0.42) | 0.00 |
| w=1 / cem | 0.59 (0.56) | 0.69 (0.66) | +0.033 [-0.028, +0.091] | 0.58 (0.25) | 0.08 |
| w=3 / gd | 0.68 (0.59) | 0.74 (0.65) | +0.090 [+0.008, +0.170] | 0.67 (0.42) | 0.00 |
| w=3 / cem | 0.59 (0.56) | 0.70 (0.66) | +0.035 [-0.011, +0.083] | 0.25 (0.25) | 0.00 |

## Reading
- The objective is part of the problem but not most of it. With in-goal mass in the cost,
  letters still end at 0.69-0.74 in-goal mass (optimum 0.92-0.97).
- GD makes the most of the success term (w = 3: +0.09). CEM gains less, which fits CEM
  sampling around pile-aware candidates while GD follows the term's gradient.
- What remains: precision (placing 5 mm cubes into ~8 mm strokes), the one-step planning
  horizon, and model accuracy near the strokes. The simulator-as-model planner on the same
  objective (EXP-0050 addendum 3) separates model accuracy from the rest.

## What would change the verdict
More starts (only 2); a signed-mass or completion-shaped objective; a horizon > 1.

## Threats
2 starts; the control is a separate run; one model; GPU shared.

## Unrelated findings
none

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- ISS-013: planner candidates / initialisations in this record came from the pre-fix pile-aware sampler (ISS-010 class, ~half illegal touchdowns in audited banks); executed pushes have not been audited for touchdown legality.
- EXP-0065 RUN-0003 (C-067): executed closed-loop pushes include illegal touchdowns (blade on a cube); measured lower bounds per record in `experiments/EXP-0065-*/results/audit_closed_loop_actions.json`.

## Verdict check (2026-10-15)
Re-checked against DESIGN.md O1-O3 (pre-registered for w = 1) and `results/analysis.json`. The old verdict comment said w = 1 missed
+0.05 on both planners; that was wrong for GD (+0.056, point estimate; 95% CI [-0.002, +0.115] includes 0). Per planner at w = 1:
GD O1 met / O2 not met (0/12 at 0.9 x optimum) / O3 met (achieved 0.939 -> 0.933); CEM O1 not met (+0.033) / O2 met (1/12) /
O3 not met (0.947 -> 0.916, drop 0.030 just over 0.03). The prediction is a conjunction and no w = 1 planner satisfies it, so the
verdict stays `refuted`; only the comment was corrected. w = 3 GD was not pre-registered (O1-O3 name w = 1).

