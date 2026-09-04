---
id: EXP-0005
title: Depth channels do not help predict the cube silhouette
tier: T1
mode: confirmatory
date: 2026-09-03
hypothesis: null
claim: >
  RESTATED 2026-09-05 to cube-only scope (see Amendment). For predicting the
  blurred-mask delta on piled cubes (n=20), a mask input beats a height input,
  a density input, and every multi-channel stack containing them — so adding
  depth information to a silhouette input does not help a linear model.
prediction:
  supports: "no multi-channel input beats the matched single channel in any configuration"
  refutes:  "mask+height or mask+density beats mask-only by more than a couple of points anywhere"
  discriminating: true
provenance:
  commit: 17a7d4a7
  script: scripts/probes/channels.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt"]
  code_path: points_to_mask / points_to_density / points_to_heightmap
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~25 min, CPU"
design:
  varied: {input: [mask, height, density, mask+height, mask+density, mask+height+density], target: [mask-delta]}
  held_fixed: {res: 32, crop: 1.0, blur: 1.0, ridge: 1.0, grid: 64, target_per_column: fixed, channel_energy: "rescaled to the mask's std so the shared ridge does not switch a channel off"}
  baselines: [mean-delta]
  metric: "explained variance of the canonical-frame delta over the train-mean delta"
noise_floor: "not measured; differences under ~2 points treated as not interpretable"
depends_on: [canonical-warp, episode-split, particle-projection]
result: "mask-only 0.739 beats height 0.631, density 0.656, and every stack (0.731-0.737); no stack helps"
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The question was whether depth carries information a silhouette lacks. Holding
the target fixed and varying only the input makes it answerable: if it does,
adding it must help somewhere.

## What was actually run

Ridge from the input image(s) to the canonical-frame delta of the mask view,
with a free (unregularised) bias so mean-delta is nested inside every model.
Height maps are in metres and were rescaled to the mask's std, otherwise a
shared ridge silently suppresses them.

**Amendment, 2026-09-05.** This experiment originally ran three columns: mask
target on cubes, and mask *and* density targets on MPM sand. The sand path was
withdrawn as non-physical (`docs/rejected_mpm_sand.md`), removing two of the
three columns. Consequences:

1. The surviving cube column is unchanged and still shows no multi-channel gain.
2. The broader generalisation the original claim made — "**each** view predicts
   itself best" — rested on having two targets, and only sand had two. It is
   **withdrawn**, not merely narrowed: no cube run used a density target.
3. Because only one target remains, this record can no longer distinguish "depth
   carries no extra information" from "the mask input has an unfair advantage
   because the target is the mask". That confound is now live and is the reason
   the verdict keeps its downgrades.

## Numbers

Explained variance over mean-delta, cubes n20 piled, mask-delta target,
res 32, σ=1, episode split:

| input | dim | explained over mean-delta |
|---|---|---|
| mean-delta (0 params) | 0 | 0.000 |
| **mask only** | 1024 | **0.739** |
| height only | 1024 | 0.631 |
| density only | 1024 | 0.656 |
| mask + height | 2048 | 0.733 |
| mask + density | 2048 | 0.737 |
| mask + height + density | 3072 | 0.731 |

## What would change the verdict

Two things, both cheap and both now necessary rather than optional:

1. **A density-target cube run**, to break the target-matching confound the
   amendment introduces. Without it this record cannot separate "depth is
   uninformative" from "the input matching the target wins".
2. **A nonlinear model.** This tests whether a *linear* map can use depth; a
   UNet might extract what ridge cannot.

## Threats

- `imprecision`: one split, one ridge value. The multi-channel penalty is small
  and could partly be the doubled parameter count rather than an absence of
  information — a per-block ridge would separate those.
- `untested-dependency`: `episode-split` unchecked.
- **Target matching is an unresolved confound** since the amendment — see
  "What would change the verdict". This is the main reason not to lean on this
  record.

## Unrelated findings

none recorded — this record predates the section (added 2026-09-05).
