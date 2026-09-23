---
# ---- identity -------------------------------------------------------------
id: EXP-0013
title: >
  Switched-linear pixel-space visual operator: 64x64 canonical push-frame
  resolution beats 32x32 on held-out image accuracy (fair, same-grid
  comparison); counter-evidence narrowing C-018/slates-binned-uniform-
  difficulty to point-mass/descriptor readouts
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  (1) A switched-linear (6-bin) pixel-space visual operator fit at 64x64
  canonical push-frame resolution beats the identical fit at 32x32 on
  held-out image-space `accuracy` (`overnight_randlen_test`), when both are
  scored on the SAME 32x32 grid (64x64 prediction and ground truth
  2x2-average-pooled down before scoring). (2) On the control metric
  (`slateN`, K=32 fixed reference, `slates_binned` n20-scatter corpus),
  res64 also beats res32 on all 9 (goal x value_fn) cells, but by a margin
  (~2 sigma) that is suggestive rather than decisive. (3) Both resolutions
  score far above zero on this corpus, contradicting the general reading of
  `INVARIANTS.md`'s `slates-binned-uniform-difficulty` tag and narrowing
  `REGISTER.md`'s `C-018` -- the near-zero result that tag/claim record was
  built on applies to point-mass/descriptor readouts (MODEL-0002, the
  EXP-0011 MLPs), not to a full image-space predictor.

prediction:
  supports: >
    res64-downsampled-to-32 accuracy clears res32-native accuracy by more
    than the ridge-to-ridge noise within one resolution; res64 control
    capture is >= res32 on a majority of the 9 cells; the visual operator's
    control capture on `slates_binned` is measurably above the
    persistence/random baselines (contradicting the point-mass-only reading
    of the near-zero prior).
  refutes: >
    the downsampled accuracy comparison is within ridge-to-ridge noise (no
    resolvable resolution effect); res64 loses to res32 on a majority of
    control cells; the visual operator's control capture on `slates_binned`
    is indistinguishable from persistence/random (confirming the corpus is
    hard for every readout, not just point-mass ones).
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0013-res32-vs-64-visual-operator/code/res32vs64__fit_and_compare.py
  data: ["Genesis/data/Sean", "Genesis/data/overnight_randlen_train",
         "Genesis/data/overnight_randlen_test",
         "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"]
  code_path: experiments/EXP-0013-res32-vs-64-visual-operator/code/
  seed: 0
  split: "overnight_randlen_train/test pre-split (per-file, not re-derived); Sean has no held-out split of its own and is used TRAIN-ONLY (never scored); slates_binned used whole as a fixed control pool, per EXP-0012's prior convention"
  runtime: "~2 min total (cache build 76.7s one-time + fit/eval/control ~35s), GPU (RTX, 8GB)"
  runs: [RUN-0001]

budget:
  declared: "70 min wall-clock / ~140k tokens (coordinator-declared, hard)"
  spent: "~55 min wall-clock (includes recovering from a concurrent, unrelated stray process writing to overlapping filenames in the same temp dir -- see Unrelated findings)"
  outcome: within

design:
  varied:
    RUN-0001: {res: [32, 64], ridge: [0.1, 1.0, 10.0]}
  held_fixed: {n_bins: 6, crop: 1.0, native_raster_grid: 64x64, rasteriser: particles_to_occupancy,
               control_ridge: 1.0, K_fixed: 32, n_resample: 150}
  baselines: [persistence, random]
  metric: "accuracy (docs/experiments/METRICS.md); slateN capture, K=32 fixed reference"

noise_floor: >
  Accuracy: ridge-to-ridge spread within one resolution is ~0.002-0.004
  (e.g. res32 native accuracy 0.1431/0.1431/0.1433 across ridge
  0.1/1.0/10.0) -- the res64-vs-res32 downsampled gap (+0.130 to +0.134) is
  ~30-60x this spread, clearly resolved. Control: per-cell sems are
  0.011-0.032 on n_eff=20 slates; the res64-vs-res32 gap per cell is
  0.03-0.08, i.e. roughly 2 sigma on the individual cells checked in detail
  (e.g. ring/lyapunov: 0.469 vs 0.389, sems 0.031/0.027, combined-sem
  ~2 sigma) -- consistent in direction across all 9 cells (never a loss) but
  not each individually decisive; read as suggestive, not as a second
  independently-decisive result on top of the accuracy comparison.

depends_on: [push-frame-warp-roundtrip, randlen-train-test-file-disjoint]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  (1) RESOLVED, above noise: res64 (downsampled to res32's own 32x32 grid)
  beats res32-native accuracy by +0.130 to +0.134 across ridge in
  {0.1,1.0,10.0}, roughly 30-60x the ridge-to-ridge noise floor within a
  resolution. (2) SUGGESTIVE, not decisive: res64 beats res32 on all 9
  control cells (never a tie or loss), by 0.03-0.08 capture per cell against
  per-cell sems of 0.01-0.03 -- consistently signed but only ~2 sigma per
  cell, not pooled into one combined statistic here. Read as pointing the
  same direction as (1), not as an independent confirmation at the same
  strength. (3) Unplanned but load-bearing: the switched-linear VISUAL
  operator scores far above zero on `Genesis/data/slates_binned/
  n20_scatter_s20a1000_L20-70mm` under `ring`/`T`/`random_quadrant` goals
  (e.g. `T`/lyapunov res32 +0.662±0.024, res64 +0.716±0.022, both 20/0/0 vs
  persistence -0.016 and random -0.018), contradicting the general reading
  of `slates-binned-uniform-difficulty` (which was established using only
  point-mass/descriptor-readout models). `frac(dv_true==0)` (0.03-0.29
  depending on value_fn) is the same order as EXP-0012's own numbers, so the
  corpus's `dv_true` distribution is not the differentiator; the amendment
  to EXP-0012 and the narrowed `C-018`/`slates-binned-uniform-difficulty`
  record this as a readout-conditioned property, not a corpus-universal one.
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

Working tree was dirty at run time: this record's own new files under
`experiments/EXP-0013-res32-vs-64-visual-operator/`, edits to
`experiments/{REGISTER,INVARIANTS,TEMP_LOG}.md` and EXP-0012's
`EXPERIMENT.md` (this amendment), and the source working files under
`experiments/temp/res32-vs-64/`. No project source module was modified.

## Why this test discriminates

The fairness condition (score res64's prediction and the ground truth on the
SAME 32x32 grid via 2x2-average-pooling) rules out the trivial "more pixels
always wins" explanation -- a resolution difference that only reflected
finer sampling of the same information would wash out once both are pooled
to the same grid. It does not: res64 retains real predictive information
that a native-32 fit cannot recover, even after matching grids at score time.

## Sean's contribution to the short-push bin

Pooling `Sean` (206,240 rows) with `overnight_randlen_train` (98,304 rows)
directly answers the previous attempt's blocker: `overnight_randlen_train`
alone gives bin 0 (0-13.3mm) only 524 rows (below `MIN_ROWS_PER_BIN=50`'s
comfortable margin but not below the threshold itself); pooled, bin 0 has
10,893 rows. Every one of the 6 (and 7) bins clears 50 rows by 2-4 orders of
magnitude in the pooled set -- see `results/results_res_compare.json`'s
`bin_counts_6`/`bin_counts_7`.

## Unrelated findings

- A concurrent, unrelated process (`/tmp/run_comparison.py`, not launched by
  this record's own agent) was found running against the same
  `experiments/temp/res32-vs-64/` working directory mid-run, writing to
  overlapping filenames (`operators_res{32,64}_ridge*.pt`,
  `results_res_compare.json`) using a DIFFERENT, non-Sean-pooled fit (calling
  `Baselines/LinearForesight/fit_switched.py` via subprocess against
  `overnight_randlen_train` alone through the slow `load_randlen_cell` path).
  It had also overwritten a tracked repo artifact,
  `Baselines/LinearForesight/runs/operators_res64_accuracy.json` (from
  EXP-0003), which was restored via `git checkout` before this record was
  written. This record's own numbers are taken from a checksummed backup
  (`experiments/temp/res32-vs-64/verified/`, md5 confirmed unchanged) made
  before that process could complete and overwrite the shared filenames
  again; the promoted copies here (`results/results_res_compare.json`) are
  the same file, byte-for-byte. No project source module was touched by
  either process.
- A separate agent instance had also written its own `RESULTS.md` into
  `experiments/temp/res32-vs-64/` referencing this record's own
  `results_res_compare.json` (same numbers, different prose, including one
  overclaim -- "2.2x accuracy improvement over persistence" -- which is
  meaningless: persistence's own `accuracy` is exactly 0 by the metric's
  definition, so no ratio against it exists). That file is superseded by
  this EXPERIMENT.md; not edited in place.
- `docs/CODEMAP.md`'s "Goals and control scoring" section does not index
  `experiments/EXP-0012-*/code/binned-calibration__calibrate.py`'s
  `score_pool_kfixed` (the established K-fixed-reference-with-resample
  implementation for `slates_binned`) -- reused directly here, but only
  found by grepping "wins/losses/K=32" across EXP-0010/EXP-0012. Worth
  adding a line so a future K=32 control eval on this corpus does not
  reimplement it informally.
