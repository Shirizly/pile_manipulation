---
id: EXP-0037
title: >
  A model's skill at RANKING pushes does not predict its quality as an OBJECTIVE to
  optimise (Kendall +0.14 over 7 arms): warped NFD ranks 3rd but optimises worst,
  world-frame residual NFD ranks 6th but optimises 2nd; the NFD ensemble is best at
  both; the gradient benchmark now resolves 13/21 arm pairs
tier: T1
mode: confirmatory              # P1-P4 pre-registered in DESIGN.md before running
date: 2026-09-24
hypothesis: null
claim: >
  On 40 DS-0006 states x 6 goals (training-matched physics, one simulation path,
  soft truth, 3 GD restarts, EXP-0023's optimiser): (P1) >= 5/21 arm pairs are
  Holm-significant on grad_capture; (P2) arms' mean rank_capture and grad_capture
  agree (Kendall >= 0.6); (P3) the NFD ensemble has the best grad_capture and beats
  the best single arm; (P4) restart-to-restart spread is >= 20% of the between-arm sd.
prediction:
  supports: "P1 >= 5/21; P2 tau >= 0.6; P3 CI > 0; P4 ratio >= 0.2"
  refutes: "P1 <= 1/21; P2 tau <= 0.2; P3 CI includes 0 or another arm is best; P4 ratio < 0.2"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/stage1.py, code/stage2.py, code/stage3.py
  data: ["DS-0006 slates 0-39; pools re-simulated through rollout_candidates"]
  code_path: >
    stage1: OCC_ADAPTERS predict_step, row-wise lyapunov per goal, projected Adam
    (EXP-0023) from the top-3 pool pushes; stage2: GenesisOracleEnv.rollout_candidates
    full fidelity, TRAINING_PHYSICS, batched by push length, banked in DS-0004;
    stage3: occ_for_scoring -> lyapunov per goal -> capture vs the re-simulated pool
  seed: "none needed (deterministic top-k starts; Adam deterministic up to float noise)"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001-stage1, RUN-0002-stage2, RUN-0003-stage3]
  runtime: "stage1 ~25 min GPU; stage2 52 min GPU (75 s/state); stage3 ~1 min CPU"
budget:
  declared: "~2.5 h GPU (DESIGN.md)"
  spent: "~1.3 h GPU"
  outcome: within
design:
  varied:
    arm: "nfd_3ch_randlen, nfd_warped_randlen, nfd_warped_randlen_flipaug, nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43, linear_switched_soft, ensemble_nfd (mean dv of the 5 NFD arms)"
  held_fixed:
    goals: "quadrant_0, quadrant_3, letter_O, letter_T, letter_L, letter_S (lyapunov)"
    optimiser: "projected Adam 120 steps lr 1.5e-3, 20-70 mm, 4 mm margin; 3 restarts; a_grad = restart with best PREDICTED dv"
    physics: TRAINING_PHYSICS
    pool: "each state's 128 DS-0006 pushes, re-simulated through the same path"
  baselines: "random pool pick (capture 0), pool best (capture 1), each arm's own rank-only pick"
  metric: grad_capture, rank_capture, gain_capture (DESIGN.md; slateN scale)
noise_floor: >
  Paired over 40 states (goal-averaged); restart-to-restart sd of a cell's true
  capture 0.233 (median). No training-seed floor (EXP-0036 running).
depends_on: [score-occupancy-subpixel-stable,
             occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  P1 supported: 13/21 pairs Holm-significant on grad_capture (Friedman p 2.5e-18).
  P2 REFUTED: Kendall(rank_capture, grad_capture) = +0.14 (p 0.77). P3 supported:
  ensemble grad_capture 1.374, best single (nfd_residual_worldframe_noaug_ep43) 1.301,
  difference +0.073 [+0.021, +0.128]. P4 supported: median restart sd 0.233 vs
  between-arm sd 0.110 (ratio 2.1). GD beat every pool push in 55-85% of cells and
  ended worse than the arm's own rank pick in 10-32%.
verdict: supported              # the pre-registered claim's conjunction fails on P2; P2's refutation IS the main finding -- see body
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
P2 is the question the user needs answered: if ranking skill (slateN-like) and
optimisation quality agreed, slateN could stand in for gradient-based MPC. Every
confound of EXP-0023/0027 (physics, path, scoring, single restart, 10 states) is
removed, and P1 checks there is enough power to see the answer.

## What was actually run
All three stages as designed. Stage 2 simulated 248-254 unique pushes per state
(pool + rank picks + 3 restarts x 7 arms x 6 goals, deduplicated) in ~75 s/state.

## Numbers (results/analysis.json)
| arm | rank_capture | grad_capture | gain | GD beats pool | GD worse than rank |
|---|---|---|---|---|---|
| ensemble_nfd | 0.875 (1) | 1.374 (1) | +0.499 | 0.85 | 0.10 |
| nfd_residual_worldframe_noaug_ep43 | 0.785 (6) | 1.301 (2-3) | +0.516 | 0.75 | 0.10 |
| nfd_residual_warped_flipaug_randlen | 0.834 (2-4) | 1.271 (2-3) | +0.437 | 0.75 | 0.16 |
| nfd_warped_randlen_flipaug | 0.795 (4-7) | 1.152 (4-6) | +0.357 | 0.66 | 0.20 |
| nfd_3ch_randlen | 0.772 (4-7) | 1.145 (4-6) | +0.373 | 0.61 | 0.25 |
| linear_switched_soft | 0.819 (2-6) | 1.119 (4-7) | +0.300 | 0.63 | 0.23 |
| nfd_warped_randlen | 0.822 (2-6) | 1.078 (6-7) | +0.256 | 0.55 | 0.32 |
(rank intervals from state bootstrap in parentheses)

## Reading
- For gradient-based MPC, a model's slateN-style ranking score is NOT a proxy for
  its value as an objective. The two push-frame-WARPED arms without a residual head
  optimise worst; both RESIDUAL arms optimise best -- a candidate mechanism
  (residual prediction = occ0 + small correction may give a smoother,
  better-conditioned objective) worth testing directly.
- The ensemble is the best objective as well as the best ranker.
- One GD run is a coin toss relative to model differences (restart sd 2x the arm
  sd); benchmarks and controllers should use restarts.
- Whether grad_capture (or rank_capture) predicts CLOSED-LOOP control is EXP-0039.

## What would change the verdict
Training-seed variation (EXP-0036) of the same size as the arm gaps; a different
optimiser (lr, steps) reordering the arms.

## Threats
One optimiser setting; 40 states; the grad pick uses the model's own prediction
(as MPC would), so optimistic models are penalised by design.

## Unrelated findings
None.
