---
id: EXP-0006
title: The margin over mean-delta is flat from 20 to 30 piled cubes, at 2-5x EXP-0002's transitions
tier: T1
mode: exploratory
date: 2026-09-03
hypothesis: H-A3
claim: >
  On one code path, purpose-collected two-layer cube heaps at n=20 and n=30 give
  a linear-operator margin over mean-delta within 0.05 of each other, i.e.
  neither cube count nor contact-island connectivity changes the margin.
prediction: null
provenance:
  commit: 17a7d4a7
  script: scripts/cube_spectrum_analysis.sh
  data: ["Genesis/data/cube_spectrum/n20/**", "Genesis/data/cube_spectrum/n30/**"]
  code_path: points_to_mask
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "collection 0.5 h (n20) + 7.1 h (n30) GPU; analysis ~2 min"
design:
  varied: {n_cubes: [20, 30]}
  held_fixed: {view: mask, blur: 1.0, res: 32, crop: 0.5, grid: 64, cube_size: 0.005, density: 1000.0, friction: 0.3, push_length: 0.02, pushes_per_episode: 5, spawn: heap, base_frac: 0.6, rasteriser: points_to_mask, ridge: 1.0, estimator: "ridge toward identity", n_envs: 64}
  baselines: [persistence, mean-delta]
  metric: "explained variance over the swept region, 1 - rms/rms_persistence"
noise_floor: "~0.05, inherited from EXP-0002 (cube fold sd ~0.004 rms, ~3 points explained variance); NOT re-measured for this design -- only one split was run"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, settled-state, footprint-splat]
establishes: []
result: "margin +0.300 (n20, M=5120) / +0.295 (n30, M=2560); spread 0.005, far inside the noise floor"
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
episode.

**Amendment, 2026-09-05.** A third row, size-matched MPM sand (+0.263), was
removed when that path was withdrawn as non-physical
(`docs/rejected_mpm_sand.md`). It was the continuum comparison; the n20-vs-n30
claim never depended on it, and the two cube rows now differ by 0.005 rather
than spanning 0.037.

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

Secondary, and not part of the claim: `col-stochastic` (mass conservation
imposed in the canonical window) costs cubes ~0.000 — 0.6442 vs 0.6447. Cubes
largely do not leave the crop, so the constraint is nearly free here.

## What would change the verdict

Leave-one-run-out over the 16 (n20) and 8 (n30) episode files instead of one
seed-0 split, which would put a real noise floor under the 0.005 spread. Cheap:
no new data, ~20 min. Until then this is a null result reported under
`imprecision`, which is weak evidence by construction.

Collecting n=50 or n=80 would extend the range, but at ~28 s/transition and
worse; the cost curve is in `docs/scaling_to_200_objects.md`.

## Threats

- `imprecision`: one split, and the claim is a null. The n20/n30 spread (0.005)
  is far inside the assumed ~0.05 floor, but that floor is inherited from
  EXP-0002 rather than measured here — a null inside an unmeasured floor is
  weak by construction.
- `untested-dependency`: `episode-split`, `swept-region-metric` and
  `settled-state` are all `unchecked`.
- Considered and NOT dismissed, recorded because it nearly caused a false
  finding: this session first compared these cube numbers against a -0.003
  scattered-cube row and concluded pile depth was the operative variable. That row comes through the `PileSweepData` raster, whose
  channels are transposed (EXP-0001, >40 points), so the comparison attributed
  a code-path difference to physics. EXP-0002 had already run every regime
  through one path and found the margin flat. The `provenance.code_path` field
  exists for exactly this, and it still had to be caught by reading EXP-0002
  rather than by the field itself — a cross-record comparison is only checked
  when someone looks.

## Unrelated findings

none recorded — this record predates the section (added 2026-09-05).
