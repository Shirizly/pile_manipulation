---
id: EXP-0006
title: The margin over mean-delta is flat from 20 to 30 piled cubes, at 2-5x EXP-0002's transitions
tier: T1
mode: exploratory
date: 2026-09-03
hypothesis: H-A3
claim: >
  On one code path, purpose-collected two-layer cube heaps at n=20 and n=30 give
  a linear-operator margin over mean-delta within 0.05 of each other and within
  0.05 of size-matched sand, i.e. neither cube count nor contact-island
  connectivity changes the margin.
prediction: null
provenance:
  commit: 17a7d4a7
  script: scripts/cube_spectrum_analysis.sh
  data: ["Genesis/data/cube_spectrum/n20/**", "Genesis/data/cube_spectrum/n30/**", "Genesis/data/sand/varied/** (32 eps, size-matched)"]
  code_path: sand_to_mask
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "collection 0.5 h (n20) + 7.1 h (n30) GPU; analysis ~2 min"
design:
  varied: {n_cubes: [20, 30], material: [cubes, sand]}
  held_fixed: {view: mask, blur: 1.0, res: 32, crop: 0.5, grid: 64, cube_size: 0.005, density: 1000.0, friction: 0.3, push_length: 0.02, pushes_per_episode: 5, spawn: heap, base_frac: 0.6, rasteriser: sand_to_mask, ridge: 1.0, estimator: "ridge toward identity", n_envs: 64, transitions_sand: 5120}
  baselines: [persistence, mean-delta]
  metric: "explained variance over the swept region, 1 - rms/rms_persistence"
noise_floor: "~0.05, inherited from EXP-0002 (cube fold sd ~0.004 rms, ~3 points explained variance); NOT re-measured for this design -- only one split was run"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, settled-state, footprint-splat]
establishes: []
result: "margin +0.300 (n20, M=5120) / +0.295 (n30, M=2560) / +0.263 (sand size-matched, M=5120); spread 0.037, inside the noise floor"
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0002's two weakest rows were the piled ones: `pile30_L020` had only 1381
transitions and its margin (+0.226) sat 0.08 below the n20 row, which is inside
the noise floor but is also the only ordering that would support a depth or
connectivity story. This collects both piled counts properly — 5120 and 2560
transitions, identical spawn, identical material, identical action sampling —
so if connectivity mattered the n30 row should move away from n20.

There is a specific mechanism it could have had. Cost per transition jumps 27x
between these two counts (0.36 -> 9.93 s) because the heap percolates from
several contact islands into one, and Newton's dense per-island Hessian scales
as island_size^2.64. That is a real structural difference between the two
piles, and it produces no difference in the margin.

## What was actually run

Both cube datasets collected with `Genesis/cube_spectrum_collection.py`
(`--spawn-mode heap`), which redraws an irregular two-layer heap per env per
episode. Sand truncated to its first 32 episode files (5120 transitions) to
size-match; the full-size sand row is also reported and differs by 0.003, which
reproduces the size-matching measurement in `sand_manipulation.md` §8.

Physics verified before fitting, on both cube sets: mass 1.0000 -> 1.0000, zero
cubes outside the tray, z within the two expected layer heights, zero
settle-cap warnings, zero failed episodes. This check is not ceremonial — see
"Threats".

## Numbers

Swept region, res 32, crop 0.5, blur 1.0, mask view:

| dataset | M | mean-delta | linear | **margin** | rank-4 / full |
|---|---|---|---|---|---|
| cubes n20 piled | 5120 | 0.345 | 0.645 | **+0.300** | 0.623 / 0.645 (97%) |
| cubes n30 piled | 2560 | 0.348 | 0.643 | **+0.295** | 0.617 / 0.643 (96%) |
| sand, size-matched | 5120 | 0.328 | 0.591 | **+0.263** | 0.535 / 0.591 (91%) |
| sand, full | 42754 | 0.329 | 0.589 | +0.260 | 0.535 / 0.589 (91%) |

Secondary, and not part of the claim: `col-stochastic` costs cubes ~0.000
(0.6442 vs 0.6447) and sand 0.14 (0.4535 vs 0.5890). Cubes do not leave the
canonical crop the way grains do, so mass conservation is a reasonable
constraint on cubes and a wrong one on sand.

## What would change the verdict

Leave-one-run-out over the 16 (n20) and 8 (n30) episode files instead of one
seed-0 split, which would put a real noise floor under the 0.037 spread. Cheap:
no new data, ~20 min. Until then this is a null result reported under
`imprecision`, which is weak evidence by construction.

Collecting n=50 or n=80 would extend the range, but at ~28 s/transition and
worse; the cost curve is in `docs/scaling_to_200_objects.md`.

## Threats

- `imprecision`: one split, and the claim is a null. The n20/n30 spread (0.005)
  is far inside the floor, but the sand gap (0.037) is not comfortably so.
- `untested-dependency`: `episode-split`, `swept-region-metric` and
  `settled-state` are all `unchecked`.
- Considered and dismissed: **sand having more data** — size-matched and
  full-size sand differ by 0.003.
- Considered and NOT dismissed, recorded because it nearly caused a false
  finding: this session first compared these cube numbers against the -0.003
  cube row in `sand_manipulation.md` §8 and concluded pile depth was the
  operative variable. That row comes through the `PileSweepData` raster, whose
  channels are transposed (EXP-0001, >40 points), so the comparison attributed
  a code-path difference to physics. EXP-0002 had already run every regime
  through one path and found the margin flat. The `provenance.code_path` field
  exists for exactly this, and it still had to be caught by reading EXP-0002
  rather than by the field itself — a cross-record comparison is only checked
  when someone looks.
