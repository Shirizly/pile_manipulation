---
id: EXP-0049
title: >
  For n=20 single-layer states and exact 20 mm perpendicular pushes, pile-aware candidates
  beat placement-aware ones on every state for best-of-N (scatter +0.005, clump +0.030
  lyapunov at N=16); placement-aware wastes 22 % (scatter) / 63 % (clump) of pushes on nulls
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  On n=20 single-layer states (8 DS-0006 scatter states, slates 120-127; 5 constructed
  single-layer clump states), TRAINING_PHYSICS, 96 simulated candidates per (state, sampler),
  every executed push exactly 20 mm and perpendicular, scored by soft lyapunov improvement
  averaged over 12 goals (O T S L C X H Z I, quadrants 0/1/3): pile-aware sampling
  (min_swath 3) gives a higher best-of-16 true improvement than placement-aware perpendicular
  sampling on every state, and a lower null-push fraction.
provenance:
  commit: 3bae8cd7
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0049-sampler-study/code/sampler_study.py, code/analyse.py
  data: ["DS-0006 step-0 states of slates 120-127", "artifacts/clump_states.pt (constructed here)"]
  code_path: "generate_action_samples -> 4-D [sx,sy,ex,ey] (start + direction kept, length forced 20 mm, yaw derived perpendicular) -> GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=False); truth from occ_for_scoring"
  seed: "torch.manual_seed(hash((state, sampler))) per unit (python hash, not reproducible across processes); clump generator seed 0"
  split: "not applicable"
  runtime: "~40 min GPU, 2 parallel processes x 32 envs"
  runs: [RUN-0001]
budget:
  declared: "70 min wall clock including code"
  spent: "~60 min"
  outcome: stopped-early
design:
  varied:
    sampler: "S1 pile_aware min_swath 3; S2 pile_aware min_swath 1 + pile_clearance 2.5 mm; S3 placement_aware + perpendicular + 20 mm; S4 blind perpendicular 20 mm; *_fulllen = S1/S2 restricted to pushes the sampler itself made 20 mm (emulates redraw-on-short); MIX = resampled mixtures of S3 with S1"
    state_type: "scatter (DS-0006) / clump (constructed lattice patches, 1-3 clusters, pitch 5.5 mm)"
  held_fixed:
    n_particles: 20
    push: "exactly 20 mm, yaw perpendicular to travel (4-D action)"
    physics: TRAINING_PHYSICS
    candidates: 96 per (state, sampler)
  baselines: "S4 blind uniform perpendicular; improvement 0 = do nothing"
  metric: "lyapunov improvement v0 - v1 (occ_for_scoring, METRICS.md ground-truth scoring), mean over 12 goals; best-of-N by 400 resamples"
noise_floor: "paired over states (sem reported); sim repeat noise not measured here (EXP-0024/0036)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  best-of-16 (mean over 12 goals): scatter S1 0.0147 / S2 0.0141 / S3 0.0097 / S4 0.0110;
  clump S1 0.0453 / S2 0.0415 / S3 0.0150 / S4 0.0245. S1 - S3 = +0.0050 +- 0.0009 (8/8
  scatter), +0.030 +- 0.004 (5/5 clump). Null fraction scatter S1 0.11 / S3 0.22, clump
  S1 0.08 / S3 0.63. Relaxing pile-aware (S2) does not help.
verdict: supported
downgrades: [imprecision, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Every candidate is simulated, so best-of-N is the TRUE ceiling a perfect ranker could reach
from that candidate set, independent of any model. If pile-aware sampling were "too
limiting" its best-of-N would fall below the placement-aware / blind sets on some states;
it does not on any of 13.

## What was actually run
RUN-0001 (runs/RUN-0001/RUN.md), commit 3bae8cd7 with an unrelated dirty tree (skills, docs, Baselines/NFD, adapters.py, goals.py edits from other sessions; this experiment's code is new and self-contained under code/). Stopped by the time budget: 8/10 scatter and 5/10 clump
states have all four samplers (S1 also has scatter8 and clump5; the analysis drops
incomplete states). Pile-aware's own geometry is exactly perpendicular (max error 5e-7 rad
on non-zero pushes) but it shortens pushes that would hit the tray wall: 125/768 (scatter)
and 145/480 (clump) S1 pushes were short, 97 and 75 of them to ~0 mm. Per the brief these
were post-processed to exactly 20 mm keeping start and direction (for ~0 mm pushes the
direction falls back to the blade normal, sign arbitrary), i.e. they push toward the wall.
The `_fulllen` rows drop those pushes instead; their numbers are within noise of the
full set on best-of-N, so the post-processing does not drive the result. All executed pushes
are exactly 20 mm and perpendicular by construction (4-D action; 0 violations).
Clump states: `code/sampler_study.py::make_clump_states`, 32 proposed, 0 stacked; saved to
`artifacts/clump_states.pt` (10 states; the first 5 were used).

## Numbers (results/sampler_study.json; improvement = lyapunov v0 - v1, mean of 12 goals)

| type | sampler | null frac | >=3 cubes | cubes moved | disp sum mm | outcome L1 | mean imp | best-8 | best-16 | best-32 | best-64 | best-96 | sim s/96 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scatter (8) | S1 pile | 0.11 | 0.18 | 1.66 | 16.3 | 0.249 | +0.0019 | 0.0119 | 0.0147 | 0.0172 | 0.0192 | 0.0202 | 69 |
| scatter | S1 fulllen | 0.11 | 0.18 | 1.67 | 16.3 | 0.246 | +0.0029 | 0.0124 | 0.0151 | 0.0175 | 0.0195 | 0.0201* | |
| scatter | S2 pile relaxed | 0.12 | 0.17 | 1.62 | 16.0 | 0.243 | +0.0015 | 0.0112 | 0.0141 | 0.0166 | 0.0187 | 0.0197 | 77 |
| scatter | S3 placement perp | 0.22 | 0.11 | 1.31 | 12.5 | 0.194 | -0.0007 | 0.0072 | 0.0097 | 0.0119 | 0.0138 | 0.0148 | 72 |
| scatter | S4 blind perp | 0.12 | 0.20 | 1.68 | 16.1 | 0.241 | -0.0006 | 0.0084 | 0.0110 | 0.0133 | 0.0154 | 0.0164 | 78 |
| scatter | S3 75 % + S1 25 % | | | | | | | 0.0090 | 0.0118 | 0.0144 | 0.0166 | | |
| scatter | S3 50 % + S1 50 % | | | | | | | 0.0103 | 0.0132 | 0.0158 | 0.0181 | | |
| clump (5) | S1 pile | 0.08 | 0.84 | 7.53 | 102.6 | 0.905 | -0.0105 | 0.0365 | 0.0453 | 0.0524 | 0.0570 | 0.0587 | 117 |
| clump | S1 fulllen | 0.08 | 0.82 | 7.47 | 103.2 | 0.836 | -0.0041 | 0.0390 | 0.0467 | 0.0531 | 0.0574 | 0.0576* | |
| clump | S2 pile relaxed | 0.11 | 0.81 | 7.07 | 99.9 | 0.882 | -0.0133 | 0.0335 | 0.0415 | 0.0475 | 0.0526 | 0.0551 | 119 |
| clump | S3 placement perp | 0.63 | 0.32 | 2.20 | 18.4 | 0.247 | -0.0024 | 0.0099 | 0.0150 | 0.0206 | 0.0262 | 0.0287 | 86 |
| clump | S4 blind perp | 0.43 | 0.47 | 3.40 | 36.7 | 0.404 | -0.0037 | 0.0175 | 0.0245 | 0.0315 | 0.0377 | 0.0410 | 84 |
| clump | S3 75 % + S1 25 % | | | | | | | 0.0217 | 0.0310 | 0.0391 | 0.0470 | | |
| clump | S3 50 % + S1 50 % | | | | | | | 0.0289 | 0.0386 | 0.0460 | 0.0526 | | |

\* fulllen best-96 is over the ~80 surviving pushes. Paired vs S3 (best-of-16): S1 +0.0050 +- 0.0009
scatter (8/8), +0.030 +- 0.004 clump (5/5); S4 +0.0013 (6/8) / +0.0095 (5/5). Sampling cost is
negligible (0.01-0.14 s per 96; placement-aware slowest); simulation dominates (70-120 s per 96 at
32 envs with a second process sharing the GPU). Clump pushes cost ~1.5x scatter pushes.

## Reading
- Planning candidates: pile-aware S1 is best at every N; every S3 fraction mixed in lowers best-of-N
  monotonically. Its best-of-16 already exceeds S3's best-of-96 in both state types.
- Training data: S1 has the fewest nulls (8-11 %, not zero) and touches all cubes over 96 pushes in
  clumps; S3 spends 22 % / 63 % of the simulation budget on pushes that move nothing, and its
  outcomes are the least diverse (L1 0.19 / 0.25 vs 0.25 / 0.91). S1's mean improvement is negative
  in clumps (it breaks clumps up), which is desirable coverage for a dynamics model, not a defect.
- Relaxing pile-aware (S2: min_swath 1, clearance 2.5 mm) is slightly worse than S1 everywhere.

## What would change the verdict
Finishing the remaining 2 scatter + 5 clump states (~25 min GPU, resumes from artifacts/sim); a
closed-loop test that picks from S1 vs S3 candidates with the true simulator; per-goal breakdown
(some goals may need pushes away from the pile, where placement-aware has an edge).

## Threats
- imprecision: 8 scatter + 5 clump states; the clump effect is large (8 sem), the scatter effect ~5 sem.
- incomplete-design: run stopped by the time budget before all 20 states; no repeat of the sim
  (each candidate simulated once); only the lyapunov value fn.
- Pile-aware short pushes were force-extended toward the wall (see What was actually run); the
  fulllen rows bound this effect as negligible for best-of-N, but a collection driver should
  redraw them (as binned_slate_collection.py does) rather than extend.

## Unrelated findings
- `generate_action_samples(pile_aware=True, push_length=0.02)` shortens 15-30 % of pushes to fit the
  tray and ~10-18 % to ~0 mm (a warning is printed, the push is returned anyway) -- on n=20 scatter
  and clump states, where the "pile" spans most of the tray.
