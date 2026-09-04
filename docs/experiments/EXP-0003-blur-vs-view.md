---
id: EXP-0003
title: Blur, not the choice of view, drives operator accuracy
tier: T1
mode: confirmatory
date: 2026-09-03
hypothesis: H-A2
claim: >
  At matched blur the mask and density views give operator errors within ~7
  points of each other on both sand and cubes, while sweeping blur from 0 to
  1.5 moves the error by 30-35 points; so sand_manipulation.md §8-§9's "the
  mask view beats density" is a blur effect, not a view effect.
prediction:
  supports: "|mask - density| stays under ~10 points at every sigma, while the sigma range spans >20 points"
  refutes:  "one view leads the other by more than the sigma effect at any sigma"
  discriminating: true
provenance:
  commit: 17a7d4a7
  script: scripts/probes/view_blur.py
  data: ["Genesis/data/sand/varied/*_data.pt (40 eps)", "Genesis/data/foresight/L040/**/*_data.pt"]
  code_path: sand_to_mask / sand_to_density
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~20 min, CPU"
design:
  varied: {blur: [0.0, 0.5, 1.0, 1.5], view: [mask, density], res: [32, 64], dataset: [sand-varied, cubes-L040]}
  held_fixed: {crop: 1.0, ridge: 1.0, estimator: "ridge toward identity", grid: 64, normalize: mean, split_rule: identical}
  baselines: [persistence, mean-delta, identity-warp]
  metric: "swept-region rms as a percentage of the persistence rms at the SAME blur"
noise_floor: "not measured for this design; view differences of <7 points are treated as not interpretable"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, particle-projection]
result: "sand: mask 63.0/58.7/40.9/33.2%, density 62.5/59.4/33.9/26.7% across sigma 0/0.5/1.0/1.5. Same pattern at res 64 and on cubes."
verdict: supported
downgrades: [indirectness, imprecision, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The §8/§9 comparison varied view and blur together (mask at σ=1, density at
σ=0), so a view effect and a blur effect are observationally identical in it.
Crossing the two factors separates them: if the view matters, the two columns
should separate at fixed σ; if blur matters, the rows should separate at fixed
view. Only one of those happens.

## What was actually run

Full 2×4 cross on sand, plus a 2×2 at res 64 to check the finding is not a
resolution artifact, plus the same cross on scattered cubes. Every cell is
reported below — no cell was selected.

**A contradiction worth flagging.** `sand_manipulation.md` §6 reports blur
*hurting* density (50.9% → 54.5% at crop 1.0 / res 64), measured on the
single-pile `pile20` set with `linear-nonneg`. This measures the opposite, and
by a wide margin. One of the two is wrong; the differences are dataset
(single-pile vs varied), estimator, and crop. Unresolved.

## Numbers

sand varied (40 eps), res 32, crop 1.0 — error as % of the change at that σ:

| view | σ=0 | σ=0.5 | σ=1.0 | σ=1.5 |
|---|---|---|---|---|
| mask | 63.0% | 58.7% | 40.9% | 33.2% |
| density | 62.5% | 59.4% | 33.9% | 26.7% |
| *identity (warp only), mask* | 76.3% | 82.1% | 95.3% | 97.9% |

res 64: mask 69.4% → 42.7%, density 70.0% → 38.1% (σ 0 → 1). Same ordering.

cubes n50 scattered monolayer, res 32, crop 1.0:

| view | σ=0 | σ=0.5 | σ=1.0 | σ=1.5 |
|---|---|---|---|---|
| mask | 89.4% | 83.6% | 67.4% | 56.1% |
| density | 97.0% | 93.4% | 75.4% | 64.3% |

## What would change the verdict

Resolving the §6 contradiction: run the same cross on the `pile20` single-pile
set with `linear-nonneg`, which isolates dataset from estimator. ~1 h.

## Threats

- `indirectness` — **the load-bearing one.** The metric is normalised within
  each σ, but σ changes *what is being predicted*: a blurred field is a
  genuinely easier and genuinely less informative target. So the σ column is
  not a like-for-like accuracy comparison and must not be read as "blur makes
  the model better". The **view** comparison at fixed σ is like-for-like and is
  what the claim is about. H-C1 is the experiment that resolves what σ is worth
  for control.
- `imprecision`: one split; sand capped at 40 episodes.
- `untested-dependency`: `swept-region-metric`, `episode-split` unchecked.
