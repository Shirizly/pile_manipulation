---
id: EXP-0016
title: Contact switching beats a single operator by +0.0059 explained, and an 8-fold LORO floor of 0.0027 says that is real
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-008
claim: >
  On the CORRECTED grid, at the single strongest configuration EXP-0015 found
  (res 16, crop 0.5, blur 1.0, bins 3, L040), a contact-switched pixel operator
  beats a single ridge(1.0) operator by more than the fold-to-fold noise,
  where the noise floor is MEASURED on this exact design via leave-one-run-out
  rather than borrowed from a different one.
prediction:
  supports: "mean paired per-fold difference (switched - single, swept-region explained) exceeds ~2x its own sem = sd/sqrt(8) across the 8 LORO folds"
  refutes:  "mean paired diff is within ~2x sem of zero -- indistinguishable from fold noise"
  discriminating: true
provenance:
  commit: 006004d0
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/exp0016_loro.py`
                                  # did not exist at 006004d0; it was written in the same
                                  # session and first committed at 5401349d. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the 5401349d version of that file.
  script_first_committed: 5401349d
  script: "scripts/probes/exp0016_loro.py (reuses fit_linear_foresight.py's canonicalise/contact_score/fit_operator/fit_operator_nonneg/predict_world/metrics/swept_region_mask verbatim, plus exp0009_rerun.py's predict_meandelta)"
  data: ["configs/dataset/genesis_foresight_L040.yaml (2560 transitions, 8 runs x 320, 50 cubes, 40 mm perpendicular pushes)"]
  code_path: "PileSweepData raster, post-fix (same as EXP-0015/EXP-0009)"
  seed: "n/a -- LORO folds are the 8 runs themselves, no random split"
  split: "leave-one-run-out: 8 folds, each holding out exactly one of the 8 runs (2240 train / 320 test transitions per fold, episode_ids IS the run index here, confirmed 8 unique ids x 320 each)"
  runtime: "~3 min CPU total, OMP_NUM_THREADS=4"
budget:
  declared: "45 min, 110k tokens"
  spent: "~30 min, ~55k tokens"
  outcome: within
design:
  varied: {held_out_run: [0, 1, 2, 3, 4, 5, 6, 7]}
  held_fixed: {res: 16, crop: 0.5, blur: 1.0, bins: 3, dataset: L040, device: cpu, single_estimator: "ridge lambda=1.0", switched_estimator: "nonneg per bin", metric: swept-region explained}
  baselines: [persistence, mean-delta]
  metric: "explained (vs persistence, swept region), paired per LORO fold — see docs/experiments/METRICS.md. Originally recorded as: explained = 1 - ||pred-truth|| / ||I_k+1 - I_k||, swept region, held-out run"
noise_floor: >
  0.0027 -- the sd of the paired per-fold difference (switched - single) across
  the 8 LORO folds, MEASURED on this exact design (not borrowed). sem =
  0.0027/sqrt(8) = 0.0010. This replaces EXP-0015's borrowed ~0.03 figure,
  which came from a different design (linear_foresight_report.md sec 2.2b's
  fold sd on rms, converted, not this switched-vs-single paired quantity).
depends_on: [grid-convention, rasteriser-identity, swept-region-metric, episode-split, canonical-warp, warp-blend]
establishes: []
result: >
  All 8 folds positive: diffs (switched-single) = +0.0036, +0.0057, +0.0023,
  +0.0073, +0.0069, +0.0077, +0.0035, +0.0105. Mean +0.0059, sd 0.0027 across
  folds (the measured noise floor), sem 0.0010. mean/sem = 6.1 -- the paired
  effect clears its own measured floor by a wide margin (>>2x). Fold 4
  reproduces EXP-0015's single-split numbers exactly (switched 0.4152, single
  0.4083, diff +0.0069), confirming this probe's code path matches EXP-0015's.
verdict: supported
downgrades: [selection]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0015 found switched beats single by +0.0001/+0.0069/+0.0050 explained
across three crops on ONE seed-0 split, and could not tell "switching gives a
small real benefit" from "three splits happened to fall the same way",
because its noise floor (~0.03) was converted from a different design's rms
fold sd rather than measured on this one. A design that reports a paired
per-fold difference across genuine held-out folds (not repeated re-splits of
the same 8 runs) gives a floor and an effect measured the same way, so the two
numbers are finally comparable. If the paired mean sits inside its own
across-fold sd, the effect is noise; if it clears several multiples of the
sd/sqrt(n), it is not, regardless of what a floor borrowed from elsewhere
would have said.

## What was actually run

`scripts/probes/exp0016_loro.py` loads L040 once (2560 transitions, confirmed
8 runs of exactly 320 each, `episode_ids` literally is the run index for this
dataset). For each of the 8 runs held out in turn: canonicalise the other 7
runs' 2240 transitions at res 16/crop 0.5/blur 1.0, fit a single ridge(1.0)
operator and a 3-bin nonneg switched operator (bin edges from that fold's
TRAIN quantiles only, exactly as `fit_linear_foresight.py --bins` does),
predict the held-out run, and score swept-region explained variance against
persistence and mean-delta (the round-tripped, zero-parameter canonical-frame
baseline from EXP-0009's probe). No cell was dropped after seeing results;
only crop 0.5 was run, since that is what EXP-0015 named as the follow-up and
the strongest cell of its sweep (see Threats: `selection`).

A CLI reproduction of EXP-0015's exact single-split numbers (switched 0.4152,
single 0.4083 at crop 0.5) was run first as the cost/feasibility check and
matched to 4 decimal places -- confirming the probe's reused functions are on
the same code path before spending the LORO budget.

## Numbers

Swept region, held out (one run per fold), explained variance (0 = persistence):

| held-out run | switched-nonneg | single ridge1 | persistence | mean-delta | **switched − single** |
|---|---|---|---|---|---|
| 0 | 0.4264 | 0.4228 | 0.0000 | 0.1818 | +0.0036 |
| 1 | 0.4310 | 0.4253 | 0.0000 | 0.1821 | +0.0057 |
| 2 | 0.4106 | 0.4083 | 0.0000 | 0.1792 | +0.0023 |
| 3 | 0.4282 | 0.4209 | 0.0000 | 0.1723 | +0.0073 |
| 4 | 0.4152 | 0.4083 | 0.0000 | 0.1756 | +0.0069 |
| 5 | 0.4270 | 0.4193 | 0.0000 | 0.1775 | +0.0077 |
| 6 | 0.4600 | 0.4566 | 0.0000 | 0.1777 | +0.0035 |
| 7 | 0.4011 | 0.3905 | 0.0000 | 0.1514 | +0.0105 |

Both models beat persistence (0.40-0.46 explained) and mean-delta (0.15-0.18)
comfortably in every fold -- that part is not in question here.

**Paired per-fold difference (switched − single):**

- mean = **+0.0059**
- sd across 8 folds = **0.0027** ← the measured noise floor
- sem (sd/√8) = 0.0010
- mean / sem = **6.1**

## What this says

**+0.0069 (EXP-0015's headline number, reproduced exactly at fold 4) survives.**
The mean paired difference (+0.0059) is more than 6 standard errors from zero,
and every one of the 8 folds is positive -- the effect never flips sign or
vanishes, it only varies in size (0.0023 to 0.0105). The measured floor
(sd=0.0027) is an order of magnitude smaller than the ~0.03 EXP-0015 had
borrowed, so the earlier "4-300x inside the noise floor" framing used the
wrong floor: switching's actual gap is about 2x the *measured* per-fold sd on
its worst fold and over 20x on its best.

**C-008, as originally stated ("contact-switching does not transfer from
scalar targets to the pixel operator"), is refuted at this configuration.**
It does transfer -- consistently, on every fold -- but the effect is small in
absolute terms: switched improves explained variance by ~0.6 points on an
operator that already explains ~42 points over persistence. This is not the
>0.03 gap EXP-0015's own prediction block asked for as a bar for "supports";
it is a smaller, measured, real effect, which is a legitimately different
outcome from either "no transfer" or "transfer of practical size."

## What would change the verdict

- Repeating LORO at crop 0.25 and crop 1.0 (both untested here; EXP-0015's
  single-split numbers there were +0.0001 and +0.0050) would show whether the
  effect is crop-general or specific to the window that already fits best.
  Cheap: same script, ~3 min more per crop.
- A shuffled contact-score control (bin assignment randomised, not derived
  from `contact_score`) would rule out the residual possibility that any
  3-way data split -- not specifically a contact-informed one -- buys this
  much, since each bin's fit sees less data than the single operator's. Not
  run here; flagged as the sharpest remaining objection.

## Threats

- `selection`: crop 0.5 was chosen because EXP-0015's own sweep flagged it as
  the strongest cell, not because it was the only crop planned before that
  sweep ran. This repo has already been burned by exactly this pattern
  (`linear_foresight_report.md` §2.2b: a crop=0.5 result "beat persistence" on
  one seed and did not survive reseeding). The mitigation here is that this
  record does not reseed the same split -- it holds out all 8 runs in turn --
  so the historical failure mode (one lucky split) is directly addressed even
  though the crop itself was picked post hoc.
- Considered and dismissed: **that 8 folds is too few to trust a sd.** True in
  the abstract, but the effect does not need a precise sd estimate to clear
  its floor -- it is positive in all 8 folds with no near-zero or negative
  entries, which a 0.03-vs-0.006 argument from imprecise fold counting cannot
  explain away.
- Not addressed: whether the effect is large enough to matter for control
  (MPC action selection), as opposed to one-step pixel prediction -- this
  record only speaks to the latter, same scope as EXP-0015.

## Unrelated findings

- `fit_linear_foresight.py`'s `--bins` path has no CLI flag to hold out a
  single NAMED run; only `split_by_episode`'s random `holdout_frac` sampling
  is wired in. LORO required a small standalone probe rather than 8 CLI
  invocations. Adding a `--holdout-run <id>` flag would make future
  leave-one-out designs a one-line change instead of a new script. Logged,
  not acted on.
- `fit_linear_foresight.py`'s per-bin breakdown (noted as a gap in EXP-0015)
  is still switched-only; still not fixed here, same reason (out of scope).
