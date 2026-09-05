---
id: EXP-0018
title: >
  Five attacks on C-001's reversal, all fail cleanly, and the literal
  res=64/crop=0.5 cell finally runs on the fixed pipeline (57.8%, matching
  the borrowed 57.9%)
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: H-A1
claim: >
  C-001 (the linear/ridge pixel operator on scattered cube monolayers beats
  persistence and mean-delta, post grid-convention-fix) survives five
  discriminating attacks: (1) leave-one-run-out fold spread, (2) generality
  across the res x crop configuration space including the literal untested
  res=64/crop=0.5 cell, (3) the swept-region/blend-outside-mask metric's
  dilution mechanism, (4) the margin over mean-delta specifically (C-011),
  and (5) train/test leakage into the fit.
prediction:
  supports: >
    each attack either fails to move the reported margin outside its own
    measured noise floor, or (for #3) confirms a real but crop-only dilution
    mechanism that does not explain the crop=0.5/1.0 margins; the literal
    res=64/crop=0.5 cell, run directly on this session's fixed pipeline,
    lands within a few points of EXP-0001's borrowed 57.9%
  refutes: >
    any attack collapses the margin to within ~2x its own noise floor, OR the
    res=64/crop=0.5 cell run on the current pipeline lands near 100%
    (persistence) rather than near 55-60%, OR the reversal fails to hold for
    a majority of the res x crop grid
  discriminating: true
provenance:
  commit: ebb89333
  dirty: true                     # scripts/check_register.py has an uncommitted
                                  # change from a concurrent session (adding
                                  # sha-predates-script validation); unrelated to
                                  # this record's own script, which IS committed
                                  # (ab2b6d8e, before it was run).
  data_commit: unrecorded          # L040 predates dataset-provenance stamping (2026-09-05)
  script: >
    scripts/probes/exp0018_config_sweep.py (new, committed ab2b6d8e before
    running); scripts/probes/exp0016_loro.py (prior art, unmodified, reused
    verbatim for the LORO attack)
  data: ["configs/dataset/genesis_foresight_L040.yaml", "Genesis/data/foresight/L040/cube/n50/size0.005/**/*_data.pt"]
  code_path: "PileSweepData raster (fixed, aac084e3+), via fit_linear_foresight.py's canonicalise/fit_operator/predict_world/metrics/swept_region_mask, imported directly, identical to EXP-0009/EXP-0016"
  seed: 0
  split: "grid sweep: episode-level, seed 0, holdout_frac=0.2 (2240 train/320 test, one split reused across all 12 res x crop cells). LORO: 8 folds, one of the 8 runs held out each time (exp0016_loro.py, unmodified)."
  runtime: "config sweep: 12 cells, ~12s CPU total (all cheap: ols+ridge via torch.linalg.solve, no nonneg). LORO: ~1 min CPU (8 folds, res=16/crop=0.5, includes the bins=3 switched-nonneg fits exp0016_loro.py always runs)."
budget:
  declared: "60 min, 140k tokens"
  spent: "~55 min, ~100k tokens"
  outcome: within
design:
  varied:
    attack1_loro: "8 leave-one-run-out folds at res=16/crop=0.5 (exp0016_loro.py, unmodified)"
    attack2_grid: "res in {8,16,32,64} x crop in {0.25,0.5,1.0}, single seed-0 split, ols and ridge(1.0 toward identity)"
    attack3_coverage: "fraction of swept-region pixels inside the crop's canonical validity mask, at each grid cell"
    attack4_meandelta: "same LORO run's mean-delta column, paired against single-ridge per fold"
    attack5_leakage: "code audit, no numeric run"
  held_fixed: {blur: 1.0, view: mask, ridge_target: "toward identity", dataset: genesis_foresight_L040, holdout_frac: 0.2 (grid sweep), estimator_family: "ridge toward identity (no nonneg swept -- see Unrelated findings)"}
  baselines: [persistence, mean-delta, "identity (warp only, at res=64 only)"]
  metric: >
    held-out swept-region rms as % of persistence rms (100% = predicts
    nothing moved) and as % of mean-delta rms; equivalently, explained
    variance vs each baseline (1 - rms_model/rms_baseline) -- the "% of the
    change" normalisation used by EXP-0001/0009/0015/0016.
noise_floor: >
  1.9 points of pct_persist (equivalently 0.019 of raw explained-variance),
  measured this record as the sd of pct_persist across the 8 LORO folds at
  res=16/crop=0.5 (fold values 54.3-60.9%, mean 58.1%). For the margin over
  mean-delta specifically: sd of the paired per-fold diff (single ridge -
  mean-delta) = 0.0152 explained-variance units, sem = 0.0054 across the same
  8 folds. Both are measured on THIS design, not borrowed.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend, swept-region-metric, episode-split, pixel-index-origin]
establishes: []
result: >
  All five attacks fail to overturn C-001; one (attack 3) confirms a real,
  previously-undocumented dilution mechanism that explains part of the
  configuration-space shape without touching the headline cell. The literal
  res=64/crop=0.5 ridge cell -- never completed on the fixed pipeline before
  this record -- now runs in ~5s and gives 57.8% pct_persist / 66.0%
  pct_meandelta, matching EXP-0001's borrowed 57.9% to 0.1 point. The
  reversal holds in 11 of 12 res x crop cells tested (fails only at
  res=8/crop=1.0, 104.2% -- WORSE than persistence, a genuine inconsistency
  logged below). LORO at res=16/crop=0.5 gives a tight 1.9-point fold spread
  around a ~42-point margin, and a mean-delta margin that clears its own
  measured floor by ~45x (mean/sem). No leakage path found on code audit.
verdict: supported
downgrades: [incomplete-design, imprecision, inconsistency, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

C-001's grade was `very-low` for concrete, named reasons: the literal
res=64/crop=0.5 cell had never been run on the fixed pipeline (only inherited
from a different script in a different session), no fold-to-fold spread had
been measured for the base comparison (only for C-008's switched-vs-single
diff, a different quantity), and the configuration space beyond two cells was
unswept. Each of the five attacks below names a concrete mechanism that, if
real, would either shrink the margin below its own noise floor or reveal it
as an artifact of one cell. A design that runs the literal missing cell, the
untested fold structure, and the unswept grid either finds the mechanism or
it does not -- either answer is informative, and neither was known before
running.

## What was actually run

1. **Attack 1 (LORO fold spread).** `scripts/probes/exp0016_loro.py`
   (pre-existing, committed at `5401349d`, unmodified) was re-run as-is. It
   was built for C-008 but computes `persistence`/`mean_delta`/`single`
   (ridge=1.0) explained-variance per fold as a side effect -- exactly what
   C-001's base comparison needs, at res=16/crop=0.5, 8 genuine
   leave-one-run-out folds.
2. **Attack 2 (config-space generality) + Attack 3 (dilution mechanism).**
   New script `scripts/probes/exp0018_config_sweep.py`, committed (`ab2b6d8e`)
   before running. Loads L040 once, builds one seed-0 split, then loops a
   4x3 grid of (res, crop), fitting fresh ols and ridge(1.0) operators at
   each cell and recording pct_persist, pct_meandelta, and
   `cov_swept_in_crop` (the fraction of swept-region pixels that fall inside
   `push_frame_validity_mask` for that crop -- the mechanism by which a small
   crop forces every model, including persistence, to agree outside it via
   `blend_push_prediction`). Nonneg was deliberately excluded from the sweep
   (see Unrelated findings) -- ridge/ols is the estimator family the
   headline claim is stated for, and EXP-0009 already showed ols/ridge/nonneg
   agree to 0.1pt at res=8, so this is a disclosed scope narrowing, not a
   silent one.
3. **Attack 4 (mean-delta margin).** Read off the same LORO run's
   `mean_delta` and `single` columns as a paired series across the 8 folds.
4. **Attack 5 (leakage).** Code audit, no run: traced `split_by_episode`,
   `canonicalise`, `fit_operator`, and the mean-delta/bin-edge computations
   in `exp0009_rerun.py`/`exp0016_loro.py`/`exp0018_config_sweep.py` to
   confirm nothing derived from `occ_te`/`occ1_te` (or their labels) reaches
   `Y0`/`Y1`/`bmd`/the fitted `A` before scoring.

## Numbers

**Attack 1+4 -- LORO at res=16/crop=0.5/blur=1.0 (8 folds, swept-region
explained variance vs persistence; `single` = ridge toward identity, 1.0):**

| fold (held-out run) | single ridge1 | mean-delta | single − mean-delta |
|---|---|---|---|
| 0 | 0.4228 | 0.1818 | 0.2410 |
| 1 | 0.4253 | 0.1821 | 0.2432 |
| 2 | 0.4083 | 0.1792 | 0.2291 |
| 3 | 0.4209 | 0.1723 | 0.2486 |
| 4 | 0.4083 | 0.1756 | 0.2327 |
| 5 | 0.4193 | 0.1775 | 0.2418 |
| 6 | 0.4566 | 0.1777 | 0.2789 |
| 7 | 0.3905 | 0.1514 | 0.2391 |
| **mean** | **0.4190** | **0.1747** | **0.2443** |
| **sd** | 0.0189 | 0.0099 | 0.0152 |
| **sem (n=8)** | 0.0067 | 0.0035 | 0.0054 |

As pct_persist (100 x (1-explained)): fold range **54.3% - 60.9%**, mean
**58.1%**, sd **1.9 points** -- a 42-point margin against a 1.9-point fold
spread (mean/sem ≈ 15.4 on the persistence margin; ≈45 on the mean-delta
margin). Every one of 8 folds beats both baselines by a wide margin.

**Attack 2+3 -- res x crop grid, single seed-0 split, pct_persist / pct_meandelta / swept-region coverage:**

| res | crop | D | M/D | pct_persist (ols) | pct_persist (ridge1) | pct_meandelta (ridge1) | swept-region coverage inside crop |
|---|---|---|---|---|---|---|---|
| 8 | 0.25 | 64 | 35.0 | 86.3 | 86.3 | 93.2 | 0.280 |
| 8 | 0.50 | 64 | 35.0 | 70.5 | 70.5 | 83.4 | 0.824 |
| 8 | 1.00 | 64 | 35.0 | **104.2** | **104.2** | 92.9 | 1.000 |
| 16 | 0.25 | 256 | 8.75 | 86.3 | 86.3 | 91.4 | 0.280 |
| 16 | 0.50 | 256 | 8.75 | 59.6 | 59.2 | 71.8 | 0.826 |
| 16 | 1.00 | 256 | 8.75 | 77.3 | 77.3 | 85.3 | 1.000 |
| 32 | 0.25 | 1024 | 2.19 | 86.3 | 86.2 | 90.5 | 0.280 |
| 32 | 0.50 | 1024 | 2.19 | 67.0 | 57.5 | 67.1 | 0.826 |
| 32 | 1.00 | 1024 | 2.19 | 69.4 | 64.8 | 78.0 | 1.000 |
| **64** | **0.25** | 4096 | 0.55 | 117.0 | 86.2 | 90.1 | 0.280 |
| **64** | **0.50** | 4096 | 0.55 | 289.0 | **57.8** | **66.0** | 0.826 |
| 64 | 1.00 | 4096 | 0.55 | 366.4 | 66.7 | 77.8 | 1.000 |

Bold: the literal `--res 64 --crop 0.5` cell the target task asked for and
EXP-0009 could not complete. Ridge, run directly on this session's fixed
pipeline: **57.8% of persistence, matching EXP-0001's borrowed 57.9% to 0.1
point**, and 66.0% of mean-delta. Unregularised OLS at res=64 is
catastrophic (289-366%, far worse than persistence) once M/D drops below 1 --
consistent with `fit_operator`'s own docstring about underdetermined rows
decaying toward "erase the pile" without a toward-identity prior; ridge fixes
this at every res/crop cell.

**Coverage mechanism (attack 3):** `cov_swept_in_crop` depends only on crop,
not res (0.280 / 0.824-0.826 / 1.000 for crop 0.25/0.5/1.0) -- exactly the
geometric fraction of the swept region that falls inside the canonical
window before `blend_push_prediction` forces the rest to equal persistence
for every model. This mechanistically explains why crop=0.25 flatlines near
86% regardless of res (72% of the scored region is forced-identical across
every model, diluting all of them toward 100% together) -- but it does
**not** explain the crop=0.5/1.0 margins, which persist at 82-100% coverage.
The dilution is real; it is not what produces the reversal.

**Attack 5 (leakage):** no path found. `split_by_episode`/manual
episode-masking runs before `canonicalise`; `Y0,Y1` (hence `A` and `bmd`) are
built only from `occ_tr`/`occ1_tr`; contact-score bin quantile edges (used
only for C-008, not this claim) are computed from `c_tr` alone; blur is a
per-sample 2D convolution applied identically before the split, sharing no
information across samples; workspace bounds are fixed config constants, not
fit from data.

## What would change the verdict

- **LORO at the literal res=64/crop=0.5 cell** (not just single-split): cost,
  measured indirectly -- `exp0016_loro.py`'s switched-nonneg step at D=4096
  would hit the same O(D^3)/4000-iter wall EXP-0009 found infeasible, so this
  needs a version with `--nonneg-max-iter` exposed and the switched arm
  skippable. Ridge-only LORO at res=64 (8 folds x ~1 solve each) would cost
  well under a minute and is the single most valuable remaining cell.
- **nonneg at res=64** on this session's pipeline -- still not directly
  measured here (only ridge/ols); EXP-0009's res=8 agreement (ols/ridge/nonneg
  within 0.1pt) is the only evidence it would not change the picture.
- The res=8/crop=1.0 failure cell (104.2%, worse than persistence) is
  unexplained by the coverage mechanism (coverage=1.0 there, same as the
  winning res=64/crop=1.0 cell at 66.7%) -- worth a dedicated look at what
  makes res=8 specifically fail at crop=1.0 while every higher res does not.

## Threats

- `incomplete-design`: LORO was only run at res=16/crop=0.5, not at the
  literal res=64/crop=0.5 headline cell (single split only there); nonneg was
  not swept at all (ridge/ols only, by design, see above).
- `imprecision`: the res=64/crop=0.5 point estimate (57.8%) itself has no
  fold spread measured directly at that resolution -- the 1.9-point spread
  is measured at res=16, an adjacent cell in the same sweep, not the
  headline one.
- `inconsistency`: 1 of 12 grid cells (res=8, crop=1.0) does NOT show the
  reversal -- ridge lands at 104.2% of persistence, i.e. slightly worse. This
  is outside the headline configuration (crop=0.5) but is a genuine failure
  mode within the tested space and is reported rather than dropped.
- `untested-dependency`: `pixel-index-origin` remains broken (~1px,
  EXP-0001); unrelated to this record's effect sizes (tens of points) but
  cited honestly, as in EXP-0001/EXP-0009.
- Considered and dismissed: **the crop=0.5 sweet spot is itself
  selection-after-the-fact.** It is not -- crop=0.5 was the value EXP-0001
  and EXP-0009 already committed to before this record existed; this record
  reports the FULL grid including crop=0.25 and crop=1.0, which lose, rather
  than reporting only the winning column.
- Considered and dismissed: **attack 3's coverage mechanism secretly IS the
  explanation for the reversal.** Ruled out because crop=1.0 (100% coverage,
  no dilution possible) still shows a 33-point margin at res=64 (66.7% vs
  100%) and a 23-point margin at res=16 (77.3%) -- the margin does not go to
  zero as dilution goes to zero, it shrinks somewhat and then persists.

## Unrelated findings

- Machine load was low this session (`uptime` load average 0.85-2.43 on a
  20-core box) compared to EXP-0009's contended run -- the full 12-cell grid
  (including the res=64 cells EXP-0009 estimated at hours) completed in
  under 15 seconds total, confirming EXP-0009's own diagnosis that
  concurrent-session CPU contention, not `--res 64`'s inherent cost, was the
  actual blocker for ols/ridge (nonneg's O(D^3) FISTA is a separate, real
  cost that contention only worsened).
- Unregularised OLS at res=64 (D=4096, M/D=0.55) gives 289-366% of
  persistence -- i.e. actively worse than predicting nothing moved -- across
  all three crops. This is a large, clean illustration of `fit_operator`'s
  own docstring claim about underdetermined rows decaying toward "erase the
  pile" without the toward-identity ridge prior; it was previously stated in
  prose there but not measured end-to-end at the paper-matched resolution.
- `scripts/check_register.py` had an uncommitted 12-line change in the
  working tree throughout this session (visible via
  `utils.git_provenance()`'s `dirty_files`), apparently from a concurrent
  session per the task's own note about another agent editing the
  register/validator. Not touched here.

## Grade note

Grade stays `very-low` (4 downgrade domains still present, and the formula
caps at 3+), but the evidential content changed substantially: the headline
res=64/crop=0.5 number is no longer borrowed from a different script in a
different session -- it is now reproduced directly on this session's fixed
pipeline within 0.1 point, the configuration space is swept rather than
resting on two cells, and a genuine (not estimated) fold-to-fold spread
exists for the base comparison. `provenance` is the one domain from EXP-0009
that this record actually retires for the ridge/ols numbers; `inconsistency`
is a new domain this record adds, honestly, for the one grid cell that
disagrees.
