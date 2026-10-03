---
id: EXP-0042
title: >
  One-step planner benchmark (pilot): the closed-loop GD defaults are badly tuned
  (learning rate 5e-3 instead of 1.5e-3: +0.41 capture; 32 restarts instead of 8: +0.32),
  CEM depends on pool/population size (16: -0.30, 256: +0.17) but not on budget
  (0.1 s is as good as 3 s), and rank gains nothing from 1.5k -> 48k candidates
tier: T1
mode: exploratory               # pilot: sizes the full design, no prediction tested
date: 2026-09-24
hypothesis: null
claim: >
  On 10 DS-0006 states x 6 goals (lyapunov), one push planned by learned_mpc.plan with
  nfd_residual_worldframe_noaug_ep43 and simulated once (training physics, full
  fidelity), single-factor changes of the MPC parameters around the EXP-0039 defaults
  move one-step capture by more than the between-planner gaps.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/one_step.py, code/analyse.py
  data: ["DS-0006 states 0-9 (start states + banked 128-push pools, DS-0004)"]
  code_path: "Genesis pile-aware sampler (bank 32x1024, outside the budget) -> learned_mpc.plan -> rollout_candidates full fidelity TRAINING_PHYSICS (banked) -> lyapunov(occ_for_scoring) capture vs the pool"
  seed: "candidate permutation rng 1000+state; planners otherwise unseeded"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001-pilot, RUN-0002-sweep]
  runtime: "38 min GPU pilot + 48 min GPU sweep"
budget:
  declared: "~45 min GPU (DESIGN.md)"
  spent: "86 min GPU (pilot + addendum-1 sweep)"
  outcome: within
design:
  varied:
    cell: "budget {0.1,0.3,1,3} s x {rank,gd,cem}; n_cand {16,256}; GD restarts {1,32}, lr {5e-4,5e-3}; CEM pop 256, elite {0.06,0.25}; repeat of the 1 s defaults"
  held_fixed:
    model: nfd_residual_worldframe_noaug_ep43
    defaults: "budget 1.0 s, n_cand 64, GD 8 restarts lr 1.5e-3, CEM pop = n_cand, elite 0.125 (EXP-0039)"
  baselines: "each planner's 1 s default; pool mean (capture 0) / pool best (capture 1)"
  metric: "one-step capture vs the banked 128-push pool (EXP-0037 grad_capture scale)"
noise_floor: >
  Repeat of the default cells (fresh candidate draw): single-trial repeat sd 0.08 rank,
  0.27 GD, 0.39 CEM; state-bootstrap CIs over 10 states (goal-averaged).
depends_on: [score-occupancy-subpixel-stable, occ-gradient-adapter-matches-offline-predictor]
establishes: []
result: >
  1 s defaults: rank 0.38, GD 0.96, CEM 1.29 (the closed-loop order). GD: lr 5e-3 +0.41
  [+0.28, +0.54] (1.365, above CEM's default); 32 restarts +0.32 [+0.16, +0.48]; 1 restart
  -0.33; lr 5e-4 -0.20; budget 0.1 s -0.43, 3 s +0.11 [+0.03, +0.19]. CEM: pool 16 -0.30
  [-0.46, -0.13], pool 256 +0.17 [+0.07, +0.26]; pop 256 (pool 64) +0.14 [-0.01, +0.27];
  elite 0.06 -0.18; budget 0.1-3 s within +-0.04 (n.s.). Rank: 0.37-0.41 from 1.6k to 48k
  evaluations. SWEEP (addendum 1, fresh states 10-19, 2 models): T1 supported (GD lr 5e-3 /
  32 restarts +0.29 [+0.10, +0.47] worldframe, +0.44 [+0.35, +0.52] nfd_3ch); T2 partly
  (CEM 256 vs 64 +0.13 / +0.08, n.s.; 1024 / elite 0.25 +0.14 / +0.13, CIs > 0); T3
  supported but narrowed (worldframe >= nfd_3ch in all 17 cells, yet its GD lead shrinks
  from +0.28 at the default to +0.13 [-0.02, +0.29] tuned). Tuned GD ~ tuned CEM (1.29-1.33
  vs 1.24-1.27 worldframe; 1.15-1.19 vs 1.19 nfd_3ch); tuned GD at 0.3 s >= at 1 s.
verdict: supported
downgrades: [indirectness, selection]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
A one-step trial costs one simulated push, so 29 parameter settings x 60 trials cost
38 min. The same budget buys about 100 closed-loop episodes. The pilot asks which
knobs are large enough to matter before any closed-loop ablation is spent on them.

## What was actually run
RUN-0001 exactly as DESIGN.md. 1740 trials; every chosen push simulated and banked
(DS-0004, EXP-0037's fingerprint, so the pools were reused).

## Numbers (results/analysis_pilot.json)
| cell | capture | vs planner's 1 s default [95% CI] | evaluations |
|---|---|---|---|
| rank 0.1 / 0.3 / 1 / 3 s | 0.37 / 0.40 / 0.38 / 0.40 | within +-0.03 | 1.6k / 4.8k / 16k / 48k |
| gd 0.1 / 0.3 / 1 / 3 s | 0.52 / 0.88 / 0.96 / 1.07 | -0.43 / -0.08 / 0 / +0.11 | 134 / 299 / 879 / 2512 |
| cem 0.1 / 0.3 / 1 / 3 s | 1.32 / 1.33 / 1.29 / 1.27 | all n.s. | 1.4k / 4.4k / 16k / 48k |
| gd lr 5e-4 / 5e-3 | 0.76 / 1.37 | -0.20 / +0.41 | |
| gd restarts 1 / 32 | 0.63 / 1.27 | -0.33 / +0.32 | |
| gd / cem / rank pool 16 | 0.93 / 0.99 / 0.41 | -0.03 / -0.30 / +0.03 | |
| gd / cem / rank pool 256 | 1.01 / 1.45 / 0.41 | +0.06 / +0.17 / +0.03 | |
| cem pop 256 | 1.43 | +0.14 [-0.01, +0.27] | |
| cem elite 0.06 / 0.25 | 1.11 / 1.36 | -0.18 / +0.08 | |

## Reading
- **The EXP-0039 GD settings are far from GD's optimum.** Larger steps and more
  restarts each add +0.3-0.4. At lr 5e-3, GD at 1 s beats default CEM. The
  closed-loop "CEM > GD" is at least partly a tuning result, and so are model rankings
  measured under GD (C-045): a model's GD score mixes its gradient quality with how
  well the fixed lr suits its objective scale.
- **CEM is budget-insensitive at one step** (0.1 s is as good as 3 s). What helps it
  is a larger population, 256 instead of 64. For the decision-time trade-off (TODO B4),
  CEM can buy many more actions for free; GD needs time (0.1 s: -0.43).
- **Rank is flat from 1.6k to 48k candidates** at about 0.4. It stays far below the
  128-push corpus pool's best, and CEM goes past it. Either the pile-aware candidate
  distribution rarely contains the pushes these goals need, or ranking more candidates
  only selects more over-optimistic predictions. Not separated here.
- Repeat noise per trial is large (sd 0.27-0.39 for GD/CEM). A full design needs
  about 100+ trials per cell for 0.05-level effects.

## Sweep (addendum 1): fresh states 10-19, two models (results/analysis_sweep.json)
| cell (1 s unless noted) | worldframe | nfd_3ch_randlen | worldframe - nfd_3ch [95% CI] |
|---|---|---|---|
| rank default | 0.355 | 0.346 | +0.009 [-0.019, +0.038] |
| GD lr 1.5e-3 / 8 restarts (EXP-0039 default) | 0.997 | 0.715 | +0.282 [+0.095, +0.427] |
| GD lr 1.5e-3 / 32 | 1.261 | 0.902 | +0.359 |
| GD lr 5e-3 / 8 | 1.221 | 1.025 | +0.196 |
| GD lr 5e-3 / 32 | 1.285 | 1.151 | +0.134 [-0.023, +0.292] |
| GD lr 1.5e-2 / 8 | 1.277 | 1.120 | +0.157 |
| GD lr 1.5e-2 / 32 | 1.311 | 1.121 | +0.190 |
| GD lr 5e-2 / 8 | 1.258 | 0.976 | +0.282 |
| GD lr 5e-2 / 32 | 1.303 | 1.066 | +0.237 |
| GD lr 5e-3 / 32 at 0.3 s | 1.334 | 1.190 | +0.144 |
| CEM 64 / elite 0.125 (EXP-0039 default) | 1.106 | 1.065 | +0.041 [-0.091, +0.167] |
| CEM 64 / 0.25 | 1.220 | 0.990 | +0.230 |
| CEM 256 / 0.125 | 1.220 | 1.117 | +0.103 |
| CEM 256 / 0.25 | 1.240 | 1.142 | +0.098 |
| CEM 1024 / 0.125 | 1.245 | 1.147 | +0.098 |
| CEM 1024 / 0.25 | 1.244 | 1.193 | +0.052 [-0.079, +0.195] |
| CEM 256 / 0.25 at 0.1 s | 1.267 | 1.118 | +0.149 |

Reading of the sweep:
- The GD effects replicate on fresh states for both models. Any lr in 5e-3 to 5e-2 with 32
  restarts is on a plateau. The EXP-0039 default is the worst GD cell for both.
- Tuning changes the MODEL GAP, not just the level. The default GD settings hurt nfd_3ch
  more than worldframe (their objectives differ in scale), so the default-GD gap of 0.28
  shrinks to 0.13-0.19 when tuned. EXP-0039's closed-loop GD gaps (C-045) were measured at
  the default and are probably inflated the same way.
- Once each planner is tuned, GD and CEM are about equal at one step. And less time
  (GD 0.3 s, CEM 0.1 s) loses nothing.
- Chosen closed-loop defaults from here on: GD lr 5e-3, 32 restarts; CEM pool = population
  1024, elite 0.25 (the best mean over the two models; chosen after seeing the sweep).

## What would change the verdict
Closed-loop confirmation that the tuned settings also win over 8 pushes. The pilot's
knobs were chosen and read on the same 10 states (hence `selection`); the sweep's
predictions T1-T3 were pre-registered and read on fresh states. Also the closed-loop check that the
one-step winners also win over 8 pushes.

## Threats
One model; one step (not closed loop: `indirectness`); pile-aware candidates only;
lyapunov only; 10 n20 scatter states.

## Unrelated findings
none

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- EXP-0065 / ISS-013: 54 % of DS-0006's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pool numbers were not re-scored on legal-only candidates.
- ISS-013: planner candidates / initialisations in this record came from the pre-fix pile-aware sampler (ISS-010 class, ~half illegal touchdowns in audited banks); executed pushes have not been audited for touchdown legality.
