---
id: EXP-0039
title: >
  In closed-loop MPC (4 models x rank / GD / CEM, 1 s per decision, 16 paired episodes),
  the world-frame residual NFD is the best model under GD and CEM, and only accuracy and
  grad_capture order every resolved model pair correctly (4/4 and 3/3). slateN gets 4/6 and
  pool spearman 1/6. The two seeds of one recipe tie in closed loop despite a 0.04 slateN gap.
tier: T1
mode: confirmatory              # H1-H6 pre-registered in DESIGN.md; population cut to 4 models before running (addendum 3)
date: 2026-09-24
hypothesis: null
claim: >
  On 16 (goal, start) episodes of 8 pushes (DS-0006 starts 40-43; quadrant_0, letter_O,
  letter_T, letter_S; TRAINING_PHYSICS executor; soft lyapunov; 1.0 s planning budget
  per decision), for nfd_3ch_randlen, its seed1 retrain, linear_switched_soft and
  nfd_residual_worldframe_noaug_ep43: (H1) slateN orders the models under rank as
  closed loop does; (H2) under CEM pool spearman orders them better than slateN;
  (H3) under GD grad_capture orders them better than slateN; (H4) accuracy is
  uninformative within the NFD models; (H6) under GD accuracy orders them at least as
  well as slateN. With n = 4 models all of these are DESCRIPTIVE (addendum 3); the
  primary quantity is agreement with the Holm-resolved closed-loop model pairs.
prediction:
  supports: "H1 slateN agrees with the rank-planner pairs; H2 spearman > slateN under CEM; H3 grad_capture > slateN under GD; H4 accuracy at chance within NFD; H6 accuracy >= slateN under GD"
  refutes: "the reverse of each (H4 and H6 cannot both hold under GD)"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/closed_loop.py, code/analyse.py
  data: ["DS-0006 states 40-43 (start states)", "DS-0006 whole pools + EXP-0030/0036 cached predictions (offline slateN, spearman)", "EXP-0037 grad_capture", "randlen_test accuracy (EXP-0038 / EXP-0036)"]
  code_path: >
    simple_mpc.learned_mpc.run_episode (plan: rank resampling 64-candidate batches /
    gd 8 restarts projected Adam / cem pop 64, each until 1.0 s) -> GenesisOracleEnv.step
    with TRAINING_PHYSICS; progress = lyapunov(occ_for_scoring)
  seed: "none set (Genesis candidate sampling and CEM noise unseeded; episodes paired by (goal, start))"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "2.8 h GPU (192 episodes, ~52 s each)"
budget:
  declared: "~3-4 h total (user, addendum 3)"
  spent: "2.8 h GPU + ~5 min analysis"
  outcome: within
design:
  varied:
    model: "nfd_3ch_randlen, nfd_3ch_randlen_seed1, linear_switched_soft, nfd_residual_worldframe_noaug_ep43"
    planner: "rank (budget-matched resampling), gd, cem"
  held_fixed:
    budget: "1.0 s planning per decision (one level)"
    episodes: "8 pushes; 4 goals x 4 DS-0006 starts, identical across models"
    physics: TRAINING_PHYSICS
  baselines: "the seed pair (closed-loop seed floor); do-nothing = improvement 0; each model's rank planner"
  metric: "closed-loop improvement V0 - V8 (soft lyapunov); offline slateN, spearman (DS-0006), grad_capture (EXP-0037), accuracy"
noise_floor: >
  Paired over 16 episodes (bootstrap CI, sign-flip p, Holm over 6 pairs per planner);
  episode sd 0.03-0.05; closed-loop seed gap (nfd_3ch_randlen vs seed1) 0.005 rank,
  0.019 GD, 0.000 CEM (all n.s.).
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  Closed-loop improvement (rank / GD / CEM): nfd_3ch_randlen .058/.205/.273, seed1
  .062/.224/.273, linear .063/.184/.285, residual_worldframe .055/.265/.292. CEM > GD
  > rank for every model (Holm < .005). Rank: no pair resolved. GD: 4/6 pairs resolved,
  residual_worldframe best by +0.04-0.08. CEM: 2/6 resolved, residual_worldframe
  beats both NFD seeds by +0.019. On the 6 resolved (planner, pair) cells, accuracy agrees
  4/4, grad_capture 3/3, slateN 4/6, spearman 1/6. H1 untestable (rank resolves
  nothing); H2 refuted; H3 and H6 supported (descriptive); H4 refuted (descriptive).
verdict: inconclusive           # n = 4 models: every H is descriptive; the pattern rests mostly on one model -- see body
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
It is the first time-limited, closed-loop test of the offline metrics on the same
models. The world-frame residual is the model where slateN (6th of 7 in EXP-0037)
and grad_capture (2nd) disagree most. The seed pair tells whether a slateN gap of
0.04 between two runs of one recipe means anything in closed loop.

## What was actually run
RUN-0001 exactly as addendum 3 (192/192 episodes complete, resumable episodes.json).
The analysis followed addendum 3's paired plan. linear_switched_soft's DS-0006
predictions were computed here (EXP-0030 had only the hard-gated variant).

## Numbers (results/analysis.json)
Improvement V0 - V8, mean (sd) [evaluations per decision]:

| model | rank | gd | cem |
|---|---|---|---|
| nfd_3ch_randlen | +0.058 (0.035) [10048] | +0.205 (0.050) [764] | +0.273 (0.036) [14030] |
| nfd_3ch_randlen_seed1 | +0.062 (0.028) [10631] | +0.224 (0.037) [809] | +0.273 (0.052) [15000] |
| linear_switched_soft | +0.063 (0.037) [13074] | +0.184 (0.055) [955] | +0.285 (0.039) [21210] |
| nfd_residual_worldframe_noaug_ep43 | +0.055 (0.026) [10246] | +0.265 (0.044) [771] | +0.292 (0.044) [14472] |

Resolved model pairs (Holm < 0.05):
- GD: worldframe beats 3ch by 0.060 (Holm 0.002), beats seed1 by 0.041 (0.001) and beats linear by 0.081 (<0.001); seed1 beats linear by 0.040 (0.044).
- CEM: worldframe beats 3ch by 0.019 (0.035) and beats seed1 by 0.019 (0.024).
- Rank: none; all pairs are within 0.008.

Offline metrics:

| model | DS-0006 slateN | spearman | grad_capture | accuracy |
|---|---|---|---|---|
| nfd_3ch_randlen | 0.745 | 0.796 | 1.145 | 0.456 |
| nfd_3ch_randlen_seed1 | 0.787 | 0.832 | -- | 0.462 |
| linear_switched_soft | 0.758 | 0.808 | 1.119 | -- |
| nfd_residual_worldframe_noaug_ep43 | 0.761 | 0.793 | 1.301 | 0.484 |

Agreement of each offline metric with the sign of a resolved closed-loop pair:
accuracy 4/4, grad_capture 3/3, slateN 4/6, spearman 1/6. slateN's two misses are the
same fact twice: it ranks seed1 above worldframe (0.787 vs 0.761), while worldframe
wins in closed loop under both GD and CEM.

## Reading
- **The planner matters far more than the model.** CEM > GD > rank for all four
  models, by 0.03-0.24. Under a 1 s budget the rank planner (about 10,000 random
  pile-aware candidates per decision) barely moves the goal. It separates no models.
- **One model is really better in closed loop, and slateN misses it.** Accuracy and
  grad_capture both rank the world-frame residual first. Under GD it leads by
  0.04-0.08, and under CEM by 0.019 over both NFD seeds.
- **The seed gap:** the two seeds differ by 0.042 slateN on DS-0006, but they tie in
  closed loop under all three planners (0.000-0.019, all n.s.). Offline slateN gaps of
  that size between models should not be read as control differences.
- **Speed:** linear gets 1.5x the CEM evaluations and moves from last under GD to
  second under CEM (n.s. against the NFDs).
- This matches EXP-0036 (accuracy is steady across seeds and separates worldframe
  from the seed spread; slateN does not) and EXP-0041 (accuracy ordered EXP-0037's
  gain_capture). It is still 4 models and mostly one decisive one: a pattern, not a
  law.

## What would change the verdict
- A 6-8 model population. Adding seeds 2-3 and one warped model would give pairs
  where accuracy and slateN disagree without involving worldframe. That is about
  45 min per model.
- A second budget level.
- More episodes for CEM (its resolved gaps are small, about 0.019).

## Threats
- 4 goals, 4 starts, n20 scatter only, one 1 s budget.
- Timing shares the GPU with Genesis execution.
- Offline slateN averages all 160 DS-0006 states (the 4 closed-loop starts are among them, but are 4 of 160).
- grad_capture is missing for seed1, and accuracy for linear.

## Unrelated findings
none

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- **Narrowed / superseded in its main reading:** EXP-0042 found the default GD settings used here far from optimum (gaps inflated); EXP-0044 (tuned planners, 12 goals x 8 starts, 768 episodes) erased most model differences (<= 0.016) and the CEM > GD ordering (tuned GD = CEM); EXP-0045/0046 showed lyapunov saturates near its ceiling by ~8-16 pushes and EXP-0051 that it misreports letter completion. The metric-agreement pattern (accuracy > slateN > spearman) persists in EXP-0044 but on seed-sized gaps (C-045 / C-046 narrowed 2026-10-03).
- ISS-013: planner candidates / initialisations in this record came from the pre-fix pile-aware sampler (ISS-010 class, ~half illegal touchdowns in audited banks); executed pushes have not been audited for touchdown legality.
- EXP-0065 RUN-0003 (C-067): executed closed-loop pushes include illegal touchdowns (blade on a cube); measured lower bounds per record in `experiments/EXP-0065-*/results/audit_closed_loop_actions.json` -- this record's episodes did not record states, so not measurable directly; sibling records show 16-53 %.

## Pre-registered re-run (2026-10-04, RUN-0002): legal actions, headroom, across families

Registered BEFORE any cell runs (committed first); design in `runs/RUN-0002-legal-headroom-rerun/DESIGN.md`
(written by an advisor agent, reviewed by the coordinator). Why: this record's metric-validity result was measured
on a saturated lyapunov task, 4 NFD-family-heavy models, mistuned planners, illegal touchdowns in both the offline
pools and the executed pushes (EXP-0065, C-067). Re-run: 8 OCC models across families (narrow v2 NFD seeds 0/1/2,
soft NFD sigma 2, linear narrow v2 res64, linear_switched_soft, nfd_3ch_randlen, a deliberately weak epoch-10 NFD),
8 letter goals (O T S L X Z C H), DS-0006 starts 40-55, 20 mm pushes, `--legalize`, 16 pushes, plain lyapunov
objective, SCORE = AUC over pushes 4-16 of in-goal mass / per-goal optimum (EXP-0046) -- the objective is not the score.
Arm A: tuned CEM (pop 1024, elite 0.25), 0.5 s. Arm B: matched-evaluation rank-128 (`--budget 0.0`), starts 40-47.
Offline metrics for every member on ONE legal corpus (DS-0016 test pools via EXP-0059 `test_v2.py`): accuracy_1,
slateN, slateN_tough, within-pool Spearman, optimism at the pick, top-1 regret.

Prediction (pre-registered): **P1** slateN_tough has the highest Spearman with the closed-loop score, rho >= 0.7, and
agrees with >= 80 % of Holm-resolved closed-loop pairs; **P2** accuracy_1 agrees with < 70 % of them; **P3** optimism
correlates negatively (rho <= -0.5). Refutation of P1: no metric reaches rho >= 0.5 / 70 % pair agreement.
Uninformative-outcome guard: at least 8 of 28 pairs must be Holm-resolved, else the verdict is inconclusive.
Retrieval (no OCC adapter yet) is excluded from this registration; it may be added to arm B later as an
exploratory row.

