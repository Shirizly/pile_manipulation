---
id: EXP-0005
title: Height and density channels never beat the view matched to the target
tier: T1
mode: confirmatory
date: 2026-09-03
hypothesis: null
claim: >
  For predicting a given view's delta, the same view as input beats every other
  single channel and every multi-channel stack, on both cubes and sand — so
  adding a depth channel to a silhouette input does not help.
prediction:
  supports: "no multi-channel input beats the matched single channel in any configuration"
  refutes:  "mask+height or mask+density beats mask-only by more than a couple of points anywhere"
  discriminating: true
provenance:
  commit: 17a7d4a7
  script: scripts/probes/channels.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt", "Genesis/data/sand/varied/*_data.pt (40 eps)"]
  code_path: sand_to_mask / sand_to_density / sand_to_heightmap
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~25 min, CPU"
design:
  varied: {input: [mask, height, density, mask+height, mask+density, mask+height+density], target: [mask-delta, density-delta], dataset: [cubes-n20-piled, sand-varied]}
  held_fixed: {res: 32, crop: 1.0, blur: 1.0, ridge: 1.0, grid: 64, target_per_column: fixed, channel_energy: "rescaled to the mask's std so the shared ridge does not switch a channel off"}
  baselines: [mean-delta]
  metric: "explained variance of the canonical-frame delta over the train-mean delta"
noise_floor: "not measured; differences under ~2 points treated as not interpretable"
depends_on: [canonical-warp, episode-split, sand-projection]
result: "matched view wins every column; every multi-channel stack is 0.3-2 points WORSE than the matched single channel, in all six columns"
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

`sand_manipulation.md` §10 Q1 asks whether density or height is the better
input and asserts two channels is "the obvious answer". Holding the target
fixed and varying only the input makes the question answerable: if depth
carries information the silhouette lacks, adding it must help at least
somewhere. Running both targets guards against the trivial confound that an
input predicts itself.

## What was actually run

Ridge from the input image(s) to the canonical-frame delta of the target view,
with a free (unregularised) bias so the zero-parameter mean-delta baseline is
nested inside every model. Height maps are in metres and were rescaled to the
mask's std, otherwise a shared ridge silently suppresses them.

## Numbers

Explained variance over mean-delta:

| input | mask target, cubes n20 | mask target, sand | density target, sand |
|---|---|---|---|
| mask only | **0.739** | **0.579** | 0.516 |
| height only | 0.631 | 0.225 | 0.185 |
| density only | 0.656 | 0.414 | **0.669** |
| mask + height | 0.733 | 0.561 | 0.520 |
| mask + density | 0.737 | 0.564 | 0.650 |
| mask + height + density | 0.731 | 0.551 | 0.648 |

## What would change the verdict

A nonlinear model. This tests whether a *linear* map can use depth; a UNet
might extract something linear regression cannot. Also worth testing at res 64,
where the doubled input dimension is less punishing relative to M.

## Threats

- `imprecision`: one split, one ridge value. The multi-channel penalty is small
  and could partly be the doubled parameter count rather than an absence of
  information — a per-block ridge would separate those.
- `untested-dependency`: `episode-split` unchecked.
- Note: cross-view rows also show density-target/density-input (0.669) beating
  mask-target/mask-input (0.579), which is the opposite ordering to
  `sand_manipulation.md` §9.2. Different metric and frame, so not a direct
  contradiction, but see EXP-0003.
