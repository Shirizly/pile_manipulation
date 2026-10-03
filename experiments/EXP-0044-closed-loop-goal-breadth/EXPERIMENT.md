---
id: EXP-0044
title: >
  With tuned planners and 12 goals x 8 starts, the four models are nearly equivalent in
  closed loop: every model gap is <= 0.016, about the size of the gap between two training
  seeds. Tuning GD adds +0.03-0.12 (tuned GD = tuned CEM), so the planner settings matter
  far more than the model on this task.
tier: T1
mode: confirmatory              # P1-P4 in DESIGN.md before running
date: 2026-09-24
hypothesis: null
claim: >
  On 96 (goal, start) episodes of 8 pushes (12 goals, DS-0006 starts 40-47, n20 scatter,
  TRAINING_PHYSICS, 1 s per decision, tuned planners from EXP-0042): (P1) worldframe is
  best under both planners and tied-or-best on >= 9/12 goals; (P2) >= 4/6 model pairs are
  Holm-resolved under CEM; (P3) tuned GD - tuned CEM is within +-0.02; (P4) tuned GD beats
  EXP-0039/0043's default GD by >= 0.03 on their 16 cells.
prediction:
  supports: "P1 best under both + >= 9/12 goals; P2 >= 4/6; P3 |GD-CEM| <= 0.02; P4 >= +0.03"
  refutes: "P1 another model best under a planner; P2 <= 2/6; P3 |GD-CEM| > 0.02; P4 < 0.03"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py, code/analyse.py
  data: ["DS-0006 states 40-47 (start states)", "EXP-0039 + EXP-0043 default-setting episodes (P4)"]
  code_path: "simple_mpc.learned_mpc.run_episodes_batched (32 envs) -> execute_action + update_material_state; progress lyapunov(occ_for_scoring)"
  seed: "none set"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "2.4 h GPU"
budget:
  declared: "~2.4 h GPU (DESIGN.md)"
  spent: "2.4 h GPU"
  outcome: within
design:
  varied:
    model: "nfd_3ch_randlen, nfd_3ch_randlen_seed1, linear_switched_soft, nfd_residual_worldframe_noaug_ep43"
    planner: "gd (lr 5e-3, 32 restarts, pool 64), cem (pool = pop 1024, elite 0.25)"
    goal: "quadrant_0/1/3, letter O T S L C X H Z I"
    start: "DS-0006 states 40-47"
  held_fixed:
    budget: "1.0 s per decision; 8 pushes"
    physics: TRAINING_PHYSICS
  baselines: "the seed pair; EXP-0039/0043 default-setting episodes (P4); improvement 0 = do nothing"
  metric: "closed-loop improvement V0 - V8 (soft lyapunov)"
noise_floor: >
  Paired over 96 episodes; model-difference residual sd ~0.031 (GD) / 0.024 (CEM) per
  episode; seed-pair gap +0.0105 GD (Holm 0.02), +0.002 CEM.
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  P1 refuted in part: under GD linear 0.294 ~ worldframe 0.293 ~ nfd_3ch 0.288 > seed1 0.278;
  under CEM worldframe 0.291 best; worldframe tied-or-best on 11/12 (GD) and 12/12 (CEM) goals.
  P2 refuted: 2/6 CEM pairs Holm-resolved (worldframe > both NFD seeds, by 0.010-0.012);
  4/6 CI-resolved. GD: 3/6 Holm-resolved, all involving seed1 (the worst). P3 supported:
  GD - CEM +0.004 [+0.000, +0.008]. P4 supported: tuned - default GD +0.033 (worldframe) to
  +0.116 (linear); CEM tuning adds only +0.000-0.005 in closed loop.
verdict: supported              # P3, P4 hold; P1 fails under GD, P2 fails -- see body
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
It is the first closed-loop comparison with the planners tuned (EXP-0042) and with enough
goals and starts to resolve 0.015-level gaps. If EXP-0039's model differences were real
control differences, they should survive tuning and widen in resolution.

## What was actually run
RUN-0001 exactly as DESIGN.md (768/768 episodes). Default-setting references for P4:
the mean of EXP-0039 (sequential) and EXP-0043 (batched) per (model, planner, goal, start).

## Numbers (results/analysis.json)
| model | GD (tuned) | CEM (tuned) | GD tuned - default (16 cells) | CEM tuned - default |
|---|---|---|---|---|
| nfd_3ch_randlen | 0.288 | 0.281 | +0.074 [+0.057, +0.093] | +0.000 |
| nfd_3ch_randlen_seed1 | 0.278 | 0.279 | +0.060 [+0.042, +0.079] | +0.002 |
| linear_switched_soft | 0.294 | 0.285 | +0.116 [+0.100, +0.132] | +0.004 |
| nfd_residual_worldframe_noaug_ep43 | 0.293 | 0.291 | +0.033 [+0.021, +0.044] | +0.005 |

Holm-resolved model pairs (of 6): GD 3 -- nfd_3ch > seed1 +0.0105, linear > seed1 +0.016,
worldframe > seed1 +0.015; CEM 2 -- worldframe > nfd_3ch +0.010, worldframe > seed1 +0.012.

Resolvable gap (2.8 SE of a model-pair difference), from these variance components:

| design (goals x starts) | GD | CEM |
|---|---|---|
| 4 x 4 | 0.032 | 0.025 |
| 12 x 8 (this run) | 0.017 | 0.014 |
| 24 x 8 | 0.014 | 0.011 |
| 24 x 16 | 0.011 | 0.009 |
| 48 x 16 | 0.010 | 0.007 |

With tuned planners, residual (episode) noise dominates: pair-difference variance 9.7e-4
residual vs 1.6e-4 goal x model and 1.2e-4 start x model (GD).

Offline metrics against the resolved pairs (tuned):
- accuracy (3ch 0.456, seed1 0.462, worldframe 0.484) agrees on 3/4 (it misses nfd_3ch > seed1 under GD);
- DS-0006 slateN (3ch 0.745, seed1 0.787, linear 0.758, worldframe 0.761) agrees on 1/5;
- grad_capture agrees on 1/1.

## Reading
- **Tuning erased most of the model differences EXP-0039 found.** Under the default GD
  settings worldframe led by 0.04-0.08. Tuned, all four models are within 0.016, and the
  linear model (last under default GD) ties for first. The default GD settings had
  penalised models unevenly (EXP-0042 saw the same at one step). A benchmark that fixes
  one planner setting for every model measures model-planner fit, not model quality.
- **The planner matters far more than the model.** Tuning GD adds 0.03-0.12; the largest
  model gap is 0.016. Tuned GD equals tuned CEM over 8 pushes. CEM's one-step tuning gain
  (EXP-0042, +0.13) does not carry over to closed loop.
- **Model gaps are at the training-seed level.** The seed pair differs by 0.0105 under GD,
  the same size as the model gaps. It differs in the OPPOSITE direction to slateN, which
  put seed1 ahead by 0.04.
- **Accuracy still orders the resolved pairs better than slateN** (3/4 vs 1/5), but
  every gap is now 0.01-0.016. That is weak evidence either way.
- For the benchmark: on this task (n20 scatter, 8 pushes, 1 s), closed-loop performance
  hardly separates these models. Separating models 0.01 apart needs about 24 x 16
  episodes per cell. More useful is probably a HARDER setting where models can differ:
  tighter budgets (where speed and gradient quality matter), longer horizons, more
  particles, or a setting with measured headroom (an oracle ceiling).

## What would change the verdict
- An oracle ceiling (Genesis-as-model CEM) on the same cells, to tell "all models near the
  achievable maximum" from "the task can't separate them".
- The same comparison at 0.1 s budgets or with n50 states.
- Per-model tuning (one planner setting for all models may still favour some).

## Threats
- One tuned setting per planner, chosen on 2 models (EXP-0042 sweep; linear and seed1 not in the sweep).
- n20 scatter; 8 pushes; lyapunov only.
- The P4 default reference is the mean of two runs; the tuned side is one run.

## Unrelated findings
none

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- EXP-0045/0046/0051: the ~0.31 lyapunov plateau is near the achievable ceiling and lyapunov saturation hides incomplete letter tasks -- the model equivalence here is at least partly task/objective saturation (C-049 note).
- ISS-013: planner candidates / initialisations in this record came from the pre-fix pile-aware sampler (ISS-010 class, ~half illegal touchdowns in audited banks); executed pushes have not been audited for touchdown legality.
- EXP-0065 RUN-0003 (C-067): executed closed-loop pushes include illegal touchdowns (blade on a cube); measured lower bounds per record in `experiments/EXP-0065-*/results/audit_closed_loop_actions.json` -- this record's episodes did not record states, so not measurable directly; sibling records show 16-53 %.
- EXP-0065 RUN-0005: re-scoring the four models' DS-0006 slateN on LEGAL-only candidates raises slateN's agreement with this record's 5 resolved pairs from 1/5 to 4/5 (by ~0.003 margins), but 0/5 when restricted to the 8 closed-loop start states; optimism at the pick also agrees 4/5. The 'accuracy beats slateN' reading here does not survive the legality correction -- neither metric is separated with power.
