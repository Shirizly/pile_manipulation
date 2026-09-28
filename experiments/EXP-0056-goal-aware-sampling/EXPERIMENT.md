---
id: EXP-0056
title: >
  Goal-aware candidate selection (misplaced mass, deposit-aware carry model, OT-guided mix)
  does not beat pile-aware for thin goals under lyapunov + GD: all three lose 0.03-0.04
  in-goal mass
tier: T1
mode: confirmatory              # predictions written 2026-09-25 ~09:05, minutes after launch, before any result was seen
date: 2026-09-25
hypothesis: null
claim: >
  Under the same conditions as EXP-0055 (letters O T S X L I + two_squares, quadrant_0; DS-0006
  starts 40-43; worldframe NFD; tuned GD 1 s; 20 pushes) with the lyapunov value held fixed,
  choosing the planner's 64 starting candidates goal-aware -- S1 by misplaced mass in the swath,
  S2 by a geometric carry model's change in placement quality, S3 half from entropic-OT
  displacement pushes -- instead of the first 64 pile-aware draws raises final in-goal mass
  (relative to the EXP-0046 optimum) by >= 0.05 on the 28 thin-goal episodes, paired, for at
  least one sampler (O1), with a lower mean censored completion time (O2). Prior expectation:
  small effects, because GD refinement under lyapunov pulls candidates back toward
  capacity-blind pushes; a sampler can help mainly by filtering.
prediction:
  supports: "O1: some S with paired mass_rel_k20 >= +0.05 over the lyap/pile cell; O2: its time_mass_tact2 < 0"
  refutes: "no S reaches +0.05 mass_rel_k20"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, EXP-0055 code/analyse.py
  data: ["DS-0006 starts 40-43", "EXP-0046 optima (vstar.json)"]
  code_path: "pile-aware bank (2048/env) -> simple_mpc/goal_aware_sampling.py select_* / ot_proposals -> plan(gd) -> run_episodes_batched"
  seed: "selection generator seeded per env"
  split: "not applicable"
  data_commit: not applicable
  runs: ["EXP-0055 RUN-0001 (shared run)"]
  runtime: "shared with EXP-0055"
budget:
  declared: "3 h wall total for EXP-0055 + EXP-0056 (user, 2026-09-25)"
  spent: "shared run, see EXP-0055"
  outcome: within
design:
  varied:
    sampler: "pile (control: first 64 of the pile-aware bank); misplaced (S1); deposit (S2, softmax tau 0.5); ot_mix (S3: 32 OT proposals + 32 pile-aware)"
  held_fixed:
    value: lyapunov
    model: nfd_residual_worldframe_noaug_ep43
    planner: "gd, lr 5e-3, 32 restarts, 1 s (CEM not run -- user's 3 h cap; CEM is where the pool matters most, see 'What would change the verdict')"
    goals: "as EXP-0055"
  baselines: "lyap cell of EXP-0055 RUN-0001 (pile-aware); do-nothing = k = 0 values"
  metric: "completion_time (headline), mass_in_region fraction rel. optimum, signed_mass fraction, achieved_fraction (lyapunov), coverage_emd, covered_frac"
noise_floor: "28 paired thin-goal episodes per cell; paired bootstrap CI"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  Thin goals, 28 paired episodes, k = 20, sampler minus pile-aware [95% CI]: in-goal mass
  misplaced -0.031 [-0.061, -0.002], deposit -0.035 [-0.072, +0.005], ot_mix -0.042 [-0.076,
  -0.007]; coverage -0.026 to -0.033; lyapunov achieved -0.013 to -0.019; completion time (mass,
  t_act 2 s) +0.9 / +3.0 / +1.4 s. quadrant_0 unchanged (1.00). Selection overhead ~3 ms/env
  (S1, S2), ~0.2 s/env (S3, CPU OT).
verdict: refuted
downgrades: [indirectness]      # GD only; CEM, where the starting pool matters most, not run
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The pile-aware sampler aims the blade at the nearest pile face whatever the goal. If goal
blindness of the starting candidates limits thin-goal filling, selecting them by where their
cubes come from and go to should raise in-goal mass even with the value unchanged.

## What was actually run
Cells s_misplaced, s_deposit, s_otmix of EXP-0055 RUN-0001. The OT sampler is a GPU/CPU
re-implementation of the idea of `simple_mpc/action_sampler.py::OTGuidedActionSampler`
(W1 entropic plan on the scoring grid, blade one cube-width behind a misplaced cube, along its
barycentric displacement), not that class: its action convention predates the blade-centre
perpendicular pushes used here.

## Reading
- Under lyapunov + GD, choosing the starting pushes by goal state does not help, and slightly
  hurts: GD refines the best 32 of 64 starts toward what lyapunov rewards, so goal-aware starts
  mostly lose the geometric diversity of the pile-aware draw.
- Per-goal gains exist (ot_mix on O +0.15, X +0.10 on single starts; GIFs in EXP-0055
  artifacts/demos/lyap_vs_s_*), but they are within repeat noise (per-episode sd 0.09).

## What would change the verdict
- CEM (pool-driven) with the same samplers, ~16 min per cell; a sampler paired with the V3
  value; clump starts.

## Unrelated findings
none
