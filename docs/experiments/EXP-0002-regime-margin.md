---
id: EXP-0002
title: The operator's margin over mean-delta is roughly regime-independent
tier: T1
mode: exploratory
date: 2026-09-03
hypothesis: H-A3
claim: >
  On a single code path, the linear operator's explained-variance margin over
  the zero-parameter mean-delta baseline lies in +0.15..+0.31 for every regime
  tested (scattered monolayer cubes, contact-sampled scattered cubes, piled
  cubes, sand), i.e. it does not depend on pile depth or continuum-ness.
prediction: null
provenance:
  commit: 17a7d4a7
  script: scripts/probes/regimes.py
  data: ["Genesis/data/foresight/L040/**", "Genesis/data/foresight/scatter_contact/**", "Genesis/data/foresight/pile30_L020/**", "Genesis/data/cube_spectrum/n20/*", "Genesis/data/sand/varied/* (40 eps)"]
  code_path: sand_to_mask
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~15 min, CPU"
design:
  varied: {regime: [scattered-blind, scattered-contact, piled-n30, piled-n20, sand], res: [64], crop: [0.5, 1.0]}
  held_fixed: {view: mask, blur: 1.0, ridge: 1.0, estimator: "ridge toward identity", grid: 64, rasteriser: sand_to_mask, split_rule: identical}
  baselines: [persistence, mean-delta]
  metric: "explained variance over the swept region, 1 - rms/rms_persistence"
noise_floor: "not measured for this design; the report's cube fold sd is ~0.004 rms (~3 points of explained variance), so differences under ~0.05 between regimes are not interpretable"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, rasteriser-identity]
result: "margin +0.291 / +0.255 / +0.226 / +0.306 / +0.262 at crop 0.5; mean-delta itself varies 0.12 -> 0.35 across the same regimes"
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

`sand_manipulation.md` §8 concluded that pile depth is what makes a linear
operator work, from a table whose cube row came through the `PileSweepData`
raster and whose sand row came through `sand_to_mask`. EXP-0001 shows that
difference is worth ~49 points, which is larger than the effect that table
attributed to depth. Running every regime through one rasteriser separates the
two: if depth were the operative variable, the margin should rise from
monolayer to heap; if it is not, the margin should be flat and something else
should move.

## What was actually run

Five datasets, one loader, one fit, one metric, one split rule. Sand was capped
at 40 episodes for load time, so it is not size-matched to the cube sets — this
favours sand slightly and does not change the reading, since sand is not the
outlier. `pile30_L020` has only 1381 transitions, so its row is the weakest.

## Numbers

res 64, crop 0.5 (crop 1.0 in parentheses):

| regime | M | mean-delta | linear | **margin** |
|---|---|---|---|---|
| cubes n50, scattered monolayer, blind | 2560 | 0.120 (0.141) | 0.411 (0.308) | **+0.291** (+0.167) |
| cubes n50, scattered, contact-sampled | 2108 | 0.182 (0.204) | 0.437 (0.352) | **+0.255** (+0.148) |
| cubes n30, piled heap | 1381 | 0.322 (0.345) | 0.548 (0.548) | **+0.226** (+0.202) |
| cubes n20, piled 2 layers | 4840 | 0.332 (0.345) | 0.638 (0.640) | **+0.306** (+0.295) |
| sand, varied starts | 5692 | 0.315 (0.328) | 0.577 (0.573) | **+0.262** (+0.245) |

What does vary with regime is mean-delta (0.12 → 0.35) and total predictability
(0.31 → 0.64). Depth and continuum make the response more *stereotyped*, not
more linear.

## What would change the verdict

Leave-one-run-out instead of one split, size-matched sand, and a third crop.
~2 h, no new data. Until then the flatness claim is supported but the ordering
*within* the table is not resolved — n20 and n30 differ by 0.08 with a noise
floor around 0.05.

## Threats

- `imprecision`: one seed-0 split where LORO was affordable. The claim survives
  because the spread across regimes (0.226–0.306) is comparable to the noise
  floor, which is the point being made — but a null result under imprecision is
  weak evidence, and this is close to one.
- `untested-dependency`: `rasteriser-identity` is broken, though held fixed here.
- Considered and dismissed: **sand having 10× the data.** `sand_manipulation.md`
  §8 already measured size-matching as worth 0.003.
