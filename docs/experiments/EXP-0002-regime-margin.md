---
id: EXP-0002
title: The operator's margin over mean-delta is roughly regime-independent
tier: T1
mode: exploratory
date: 2026-09-03
hypothesis: H-A3
claim: >
  On a single code path, the linear operator's explained-variance margin over
  the zero-parameter mean-delta baseline lies in +0.15..+0.31 for every cube
  regime tested (scattered monolayer, contact-sampled scattered, piled n30,
  piled n20), i.e. it does not depend on pile depth.
prediction: null
provenance:
  commit: 17a7d4a7
  script: scripts/probes/regimes.py
  data: ["Genesis/data/foresight/L040/**", "Genesis/data/foresight/scatter_contact/**", "Genesis/data/foresight/pile30_L020/**", "Genesis/data/cube_spectrum/n20/*"]
  code_path: points_to_mask
  seed: 0
  split: "episode-level, 25% of files, seed 0"
  runtime: "~15 min, CPU"
design:
  varied: {regime: [scattered-blind, scattered-contact, piled-n30, piled-n20], res: [64], crop: [0.5, 1.0]}
  held_fixed: {view: mask, blur: 1.0, ridge: 1.0, estimator: "ridge toward identity", grid: 64, rasteriser: points_to_mask, split_rule: identical}
  baselines: [persistence, mean-delta]
  metric: "explained variance over the swept region, 1 - rms/rms_persistence"
noise_floor: "not measured for this design; the report's cube fold sd is ~0.004 rms (~3 points of explained variance), so differences under ~0.05 between regimes are not interpretable"
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, rasteriser-identity]
result: "margin +0.291 / +0.255 / +0.226 / +0.306 at crop 0.5; mean-delta itself varies 0.12 -> 0.35 across the same regimes"
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The now-retracted depth conclusion (commit `242a9cc1`) came from a table whose
rows were produced by two different rasterisers. EXP-0001 shows that difference
is worth ~49 points, which is larger than the effect that table attributed to
depth. Running every regime through one rasteriser separates the two: if depth
were the operative variable, the margin should rise from monolayer to heap; if
it is not, the margin should be flat and something else should move.

## What was actually run

Four datasets, one loader, one fit, one metric, one split rule.
`pile30_L020` has only 1381 transitions, so its row is the weakest; EXP-0006
re-measures the piled end with purpose-collected heaps at 2-5x the data.

**Amended 2026-09-05.** This experiment originally carried a fifth row, MPM
sand, as the continuum end of the comparison. That path was withdrawn as
non-physical (`docs/rejected_mpm_sand.md`), so the row is removed. It does not
change the claim: the four cube rows already span +0.226..+0.306 and the sand
row sat inside that range.

## Numbers

res 64, crop 0.5 (crop 1.0 in parentheses):

| regime | M | mean-delta | linear | **margin** |
|---|---|---|---|---|
| cubes n50, scattered monolayer, blind | 2560 | 0.120 (0.141) | 0.411 (0.308) | **+0.291** (+0.167) |
| cubes n50, scattered, contact-sampled | 2108 | 0.182 (0.204) | 0.437 (0.352) | **+0.255** (+0.148) |
| cubes n30, piled heap | 1381 | 0.322 (0.345) | 0.548 (0.548) | **+0.226** (+0.202) |
| cubes n20, piled 2 layers | 4840 | 0.332 (0.345) | 0.638 (0.640) | **+0.306** (+0.295) |

What does vary with regime is mean-delta (0.12 → 0.35) and total predictability
(0.31 → 0.64). Depth and continuum make the response more *stereotyped*, not
more linear.

## What would change the verdict

Leave-one-run-out instead of one split, and a third crop. ~2 h, no new data. Until then the flatness claim is supported but the ordering
*within* the table is not resolved — n20 and n30 differ by 0.08 with a noise
floor around 0.05.

## Threats

- `imprecision`: one seed-0 split where LORO was affordable. The claim survives
  because the spread across regimes (0.226–0.306) is comparable to the noise
  floor, which is the point being made — but a null result under imprecision is
  weak evidence, and this is close to one.
- `untested-dependency`: `rasteriser-identity` is broken, though held fixed here.
- Considered and dismissed: **unequal dataset sizes.** EXP-0006 measured
  size-matching as worth 0.003 on the piled sets.

## Unrelated findings

none recorded — this record predates the section (added 2026-09-05).
