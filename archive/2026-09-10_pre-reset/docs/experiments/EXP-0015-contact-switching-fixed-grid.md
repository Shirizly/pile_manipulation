---
id: EXP-0015
title: Contact switching helps the pixel operator by 0.000-0.007 explained — consistently positive, consistently inside the noise floor
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-008
claim: >
  On the CORRECTED grid (commit aac084e3), a contact-switched pixel operator --
  one operator per contact-amount bin, the bin chosen at inference from the
  state and action alone -- beats a single operator by more than the ~0.03
  explained-variance noise floor, on scattered 50-cube monolayers at a
  configuration where each bin's fit is well determined (M/D > 2).
prediction:
  supports: "switched beats single by > 0.03 explained in at least two of three crops"
  refutes:  "the gap is under 0.03 explained in every crop, or is negative anywhere"
  discriminating: true
provenance:
  commit: 006004d0
  script: "fit_linear_foresight.py --bins 3 (its own contact_score switching)"
  data: ["configs/dataset/genesis_foresight_L040.yaml (2560 transitions, 50 cubes, 40 mm perpendicular pushes)"]
  code_path: "PileSweepData raster, post-fix"
  seed: 0
  split: "whole runs held out, 2240 train / 320 test, --split-seed 0"
  runtime: "~12 min CPU total for three configs (OMP_NUM_THREADS=3)"
budget:
  declared: "20 min, 40k tokens"
  spent: "~20 min, ~25k tokens"
  outcome: within
design:
  varied: {crop: [0.25, 0.5, 1.0], estimator: [switched-nonneg, single ridge1]}
  held_fixed: {res: 16, blur: 1.0, bins: 3, dataset: L040, device: cpu, split_seed: 0, ridge: 1.0, metric: swept-region}
  baselines: [persistence, identity-warp]
  metric: "explained (vs persistence, swept region) — see docs/experiments/METRICS.md. Originally recorded as: explained = 1 - ||pred-truth|| / ||I_k+1 - I_k||, swept region, held out"
noise_floor: >
  ~0.03 explained variance, converted from reports/linear_foresight_report.md
  §2.2b's measured fold-to-fold sd of ~0.004 rms against a persistence rms of
  0.1626. NOT re-measured on this design -- a fold sweep is the obvious
  follow-up and is why the verdict below is inconclusive rather than refuted.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend, swept-region-metric, episode-split]
establishes: []
result: >
  switched minus single = +0.0001 (crop 0.25), +0.0069 (crop 0.5), +0.0050
  (crop 1.0). Positive in all three, and 4-300x smaller than the noise floor.
verdict: inconclusive
downgrades: [imprecision]
grade: moderate
supersedes: []
superseded_by: [EXP-0016]
invalidated_by: null
---

## Why this test discriminates

`reports/linear_foresight_report.md` §2.7 tested contact switching and found it
no better than a single operator, but on the transposed grid, where the
operator explained essentially nothing — there was no signal for the switch to
sharpen. EXP-0001 showed that artifact is worth ~49 points, larger than any
effect §2.7 was looking for. So the original null was uninformative, and C-008
was marked `invalidated` rather than refuted.

The re-test also fixes §2.7's own stated complaint: it ran at `M/D = 0.73` per
bin, so each bin's fit was underdetermined before specialisation could pay.
Here `D = 256` and `M/D ≈ 2.9` per bin, which is well determined.

## What was actually run

Three crops at res 16, so `D` stays 256 and the per-bin fit stays well
determined while the canonical window changes. `--bins 3` uses
`fit_linear_foresight.py`'s own `contact_score` — pile mass in the blade's
path, computable from `(I_k, u)` alone, so it is a legitimate inference-time
switching variable rather than oracle information.

The crop sweep was not part of the original plan; it was added after crop 0.25
returned an exact tie, on the reasoning that a configuration where the operator
explains only 0.137 is a poor place to ask whether anything sharpens it. Both
the tie and the sweep are reported.

## Numbers

Swept region, held out, explained variance (0 = persistence):

| crop | switched-nonneg | single ridge1 | **switched − single** | identity | operator strength |
|---|---|---|---|---|---|
| 0.25 | 0.1375 | 0.1374 | **+0.0001** | 0.0218 | weak |
| **0.50** | **0.4152** | 0.4083 | **+0.0069** | 0.0531 | **strongest** |
| 1.00 | 0.2325 | 0.2275 | **+0.0050** | −0.0123 | middling |

Per-bin breakdown at crop 0.25 (explained, switched operator):

| bin | n | explained |
|---|---|---|
| barely | 140 | 0.1096 |
| mildly | 117 | 0.1459 |
| significantly | 63 | 0.1568 |

## What this says

**The sign is consistent and the size is not.** Switching helped in all three
crops, which is mildly encouraging and is what the scalar-level result (C-007:
essentially all the nonlinearity is the contact amount) would predict. But the
largest gap, +0.0069, is about a quarter of the noise floor. On this evidence
you cannot distinguish "switching gives a small real benefit" from "switching
gives nothing and three splits happened to fall the same way".

So **C-008's original statement — that contact switching does not transfer to
the pixel operator — is not refuted by the fix.** It is also not confirmed. The
honest status is that the question is now *askable* (the operator explains
0.41 instead of ~0, and the bins are well determined) and has not been
answered.

Incidental and not part of the claim: crop 0.5 is much the best window
(0.415 vs 0.228 at crop 1.0 and 0.137 at crop 0.25). That is a larger effect
than anything switching does, and it is a free tuning result.

## What would change the verdict

Leave-one-run-out over the 8 runs instead of one split, at crop 0.5 only. That
puts a measured floor under a +0.0069 gap and costs ~20 min. If the gap
survives LORO it is real and small; if it does not, C-008 stands as originally
written and the scalar/pixel boundary in C-007 is confirmed rather than
suspected.

## Threats

- `imprecision`: one seed-0 split; the noise floor is inherited from a
  different design rather than measured here. This is the whole reason the
  verdict is inconclusive.
- Considered and dismissed: **that the bins are still underdetermined.**
  `M/D ≈ 2.9` per bin here against §2.7's 0.73, and the switched operator does
  not degrade relative to the single one at any crop, which underdetermination
  would show.
- Considered and NOT dismissed: **res 16 is coarse.** All three configs run at
  `D = 256` to keep the per-bin fit well determined, but a 16x16 canonical
  window may simply be too coarse for contact structure to matter. Testing at
  res 32 needs either more data or fewer bins.

## Unrelated findings

- `fit_linear_foresight.py`'s `--bins` path prints a per-bin breakdown only for
  the switched estimator, so there is no single-operator per-bin column to
  compare it against. Adding one would make the switched-vs-single comparison
  readable per bin rather than only in aggregate. Logged, not acted on.


## Superseded, 2026-09-06

**This record is superseded by EXP-0016.** Its measurements stand as taken; do
not cite its conclusions. Reason: one seed-0 split against a noise floor BORROWED from a different design. EXP-0016 measured the floor by 8-fold LORO (paired sd 0.0027, ~11x smaller) and settled the claim.

Kept rather than deleted because the register's audit trail depends on being
able to see what was believed and why it changed.
