---
id: EXP-0031
title: >
  With matched physics, the corpus collector and rollout_candidates agree closely
  but not exactly: soft-scored dv r 0.988-0.999, disagreement 5-16% of the
  between-action sd (median ~12%) -- EXP-0023's r = 0.959 was mostly a PHYSICS
  mismatch (0.3/1000 vs 0.25/750) plus image-scoring noise
tier: T1
mode: confirmatory              # prediction in DESIGN.md before running
date: 2026-09-24
hypothesis: null
claim: >
  For the same (DS-0006 start state, push), with both simulators given the
  training physics (particle friction 0.7, box 0.5, density 450, settle cap 3000)
  and truth scored with the soft rasteriser, the corpus collector's outcome and
  GenesisOracleEnv.rollout_candidates' outcome agree with sd(dv difference) <=
  0.1 x sd(dv between pushes).
prediction:
  supports: "sd_diff / sd_between <= 0.1 (lyapunov, corner, soft) on the tested slates"
  refutes: "sd_diff / sd_between > 0.3"
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/crosspath.py
  data: ["DS-0006 slates 0-7, 64 corpus pushes each (random, seed 0)"]
  code_path: >
    corpus: binned_slate_collection outcome (states_); rollout: GenesisOracleEnv
    (oracle_config_with_physics + apply_physics) .rollout_candidates full fidelity,
    batched by push length; both scored with occ_for_scoring and (for contrast)
    occ_from_particles, lyapunov(corner)
  seed: "action sample seed 0"
  split: "not applicable"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "~3 min GPU"
budget:
  declared: "gate for TODO G1a"
  spent: "~15 min incl. a device-bug rerun"
  outcome: within
design:
  varied:
    execution_path: "corpus collector vs rollout_candidates"
    scoring: "soft (primary) vs image (contrast)"
  held_fixed:
    physics: "training physics on both sides"
    states_actions: "8 DS-0006 states x 64 corpus pushes (none altered by the legality projection)"
  baselines: "between-push dv sd on the same states (the scale any disagreement is judged against)"
  metric: "sd(dv_rollout - dv_corpus) / sd(dv_corpus); particle displacement between paths"
noise_floor: "within-path repeat noise is ~1e-4 of between-push variance (EXP-0027 RUN-0005, particle-based)"
depends_on: [score-occupancy-subpixel-stable]
establishes: []
result: >
  Between the thresholds: sd_diff/sd_between 0.052-0.159 across 8 slates (soft;
  median ~0.12), r 0.988-0.999; image scoring 0.130-0.285. Particle displacement
  between paths: median 0.01 mm, p95 2.0-3.3 mm, max 21-58 mm (a few particles in a
  few pushes follow different paths).
verdict: inconclusive
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Earlier the two paths differed in physics AND scoring AND code path at once. Here
only the code path differs, so what remains is the code-path effect.

## What was actually run
RUN-0001 as designed. The env was built with `oracle_config_with_physics` and
`apply_physics` (new in simple_mpc/learned_mpc.py; `TRAINING_PHYSICS`).

## Numbers (results/crosspath.json)
| slate | soft r | soft sd_diff/sd_between | image sd_diff/sd_between | disp median / p95 / max (mm) |
|---|---|---|---|---|
| 0 | 0.989 | 0.152 | 0.249 | 0.01 / 2.6 / 57.7 |
| 1 | 0.994 | 0.113 | 0.203 | 0.01 / 2.0 / 43.5 |
| 2 | 0.989 | 0.149 | 0.206 | 0.01 / 2.0 / 47.4 |
| 3 | 0.992 | 0.128 | 0.195 | 0.01 / 2.9 / 45.7 |
| 4 | 0.996 | 0.094 | 0.201 | 0.01 / 2.1 / 20.8 |
| 5 | 0.988 | 0.159 | 0.285 | 0.01 / 2.3 / 47.7 |
| 6 | 0.994 | 0.112 | 0.183 | 0.01 / 2.0 / 46.8 |
| 7 | 0.999 | 0.052 | 0.130 | 0.01 / 3.3 / 31.8 |

## Reading
- Most particles land identically; a minority of pushes contain a particle whose
  path diverges by centimetres -- a bifurcation sensitive to small differences
  between the two paths (plate servo start, settle bookkeeping, the small
  history leaks of EXP-0027 RUN-0005).
- For the gradient benchmark (TODO G1a): evaluate pool AND optimised pushes through
  ONE path (rollout_candidates with TRAINING_PHYSICS), rather than scoring
  optimised pushes against corpus-recorded pool outcomes (~12% of the between-push
  sd of extra noise, and possibly a small bias).
- EXP-0023's r = 0.959 was mostly the physics difference (DS-0001 0.3/1000 vs the
  oracle env's 0.25/750) plus image-scoring noise.

## What would change the verdict
More slates; identifying the divergent pushes and whether they share a feature
(long pushes, wall contact).

## Threats
8 states, corner goal only.

## Unrelated findings
simple_mpc/config/config_oracle.yaml's physics (0.25 / 750 / 0.3, settle cap 100)
matches NO corpus in the repo; every oracle-MPC ceiling number computed with it was
simulated with different material from the models' training data.
