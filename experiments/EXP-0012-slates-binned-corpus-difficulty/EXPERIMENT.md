---
# ---- identity -------------------------------------------------------------
id: EXP-0012
title: >
  slates_binned's near-zero switched-linear slateN is a genuine corpus-
  difficulty property (uniformly moderate difficulty, no easy slates), not a
  scoring-path defect, a rasteriser mismatch, a pool-size artifact, or
  cross-operator-bin miscalibration
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm (n20,
  scatter-spawn only), weights/MODEL-0002-descriptor-only-D-all-local's
  near-zero pooled corner/lyapunov slateN (-0.02 at K=32/K=128, vs +0.46 on
  slates_multistep) is explained by (a) genuine corpus difficulty -- this
  corpus's per-slate dv_true spread is uniformly small (std 0.041 vs 0.134),
  i.e. it contains no "easy" slates of the kind that drive switched-linear's
  wins on slates_multistep -- and is NOT explained by (b) a defective scoring
  harness, (c) the point-splat vs official rasteriser mismatch, (d) pool-size
  (1000 vs 128), or (e) the operator being switched by push-length bin at the
  POOL level rather than per candidate. `slates_binned` is n20 + scatter-spawn
  only, and its `slateN` is not the same quantity as `slates_multistep`'s
  (different pool sizes, only comparable at a fixed K reference).

prediction:
  supports: >
    the harness reproduces EXP-0006's published pooled multistep number to
    within its own sem; rasteriser choice moves slateN by <0.02 absolute;
    K=32 and K=128 fixed references on slates_binned stay within noise of
    each other; within-single-operator-bin slateN on slates_binned is NOT
    clearly better than the full mixed-bin pool's slateN (ruling out (e)).
  refutes: >
    the harness fails to reproduce EXP-0006 (defect b); rasteriser choice
    moves pooled slateN by a large margin; K=32 vs K=128 diverge sharply on
    slates_binned (pool-size artifact, d); within-bin slateN is clearly and
    substantially better than the mixed-bin pool's (confirming (e) as a real,
    separate contributor requiring the difficulty-only verdict to be
    softened).
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0012-slates-binned-corpus-difficulty/code/binned-calibration__calibrate.py
  data: ["Genesis/data/slates_multistep/n20_L10mm", "Genesis/data/slates_multistep/n20_L20mm",
         "Genesis/data/slates_multistep/n20_L40mm", "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"]
  code_path: experiments/EXP-0012-slates-binned-corpus-difficulty/code/
  seed: 0
  split: "no train/test split -- both corpora used as fixed eval pools, per EXP-0006/EXP-0004's own prior splits; MODEL-0002 reused unmodified from weights/MODEL-0002-descriptor-only-D-all-local"
  runtime: "~40 min total (RUN-0001 calibration + RUN-0002 within-bin follow-up), CPU only"
  runs: [RUN-0001, RUN-0002]

budget:
  declared: "RUN-0001: 40 min/~90k tokens; RUN-0002 (this promotion's own follow-up): ~15 min/~30k tokens"
  spent: "RUN-0001: ~40min (within); RUN-0002: ~10min (within)"
  outcome: within

design:
  varied:
    RUN-0001: {rasteriser: [particles_to_occupancy, official _draw_particle_grid], K_fixed: [32, 128]}
    RUN-0002: {pool_scope: [full-1000-candidate-mixed-bin, single-operator-bin-only]}
  held_fixed: {goal: corner, value_fn: lyapunov, model: MODEL-0002-switched-linear}
  baselines: [persistence]
  metric: "slateN"

noise_floor: >
  Reproduction check: reproduced +0.458 (sem 0.054, n=60) vs published +0.461
  (sem 0.061, n=60) -- within 1 sem of each other, i.e. inside noise, taken as
  a pass. RUN-0002: across-bin K=32 reference -0.020 (sem 0.043, n=20) vs
  pooled within-bin (K=N exact) -0.056 (sem 0.038, n=100) and pooled
  within-bin (K=32-capped) ~-0.061 -- the two are ~0.04 apart against a
  combined sem of ~0.06, i.e. not distinguishable from each other, and both
  indistinguishable from persistence/zero.

depends_on: [push-frame-warp-roundtrip]
establishes: [slates-binned-uniform-difficulty]

# ---- outcome --------------------------------------------------------------
result: >
  Verdict (a): genuine corpus difficulty. Harness reproduction: +0.458 (sem
  .054) vs published +0.461 (sem .061). Rasteriser: ruled out (corr(v_true)
  0.998, slateN 0.822 vs 0.820). Pool size: ruled out (flat -0.02/-0.016 at
  K=32/K=128). Cross-operator-bin mixing (the ONE candidate explanation the
  source RESULTS.md left unresolved, and the one this record's own follow-up
  targeted): ALSO ruled out -- restricting to single-operator-bin pools gives
  pooled slateN -0.056 (sem .038, n=100), NOT better than (nominally slightly
  worse than, well within noise of) the full mixed-bin pool's -0.020 (sem
  .043, n=20). The originally-drafted explanation ("the operator ranks across
  bins simultaneously") was additionally wrong as literally stated --
  `predict_switched` assigns bins per candidate row, not per pool, in both
  `Baselines/LinearForesight/model.py` and `desc-mlp/eval_control.py` (direct
  code read) -- and the corrected, real version of the concern (independently
  fit operators' predictions are differently calibrated when ranked against
  each other) is now also empirically ruled out. `dv_true` spread (std 0.041
  vs 0.134) remains the only supported explanation.
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

Working tree was dirty at both RUN-0001 and RUN-0002's run time and at this
promotion's own commit time: the source scripts and outputs under
`experiments/temp/binned-calibration/`, plus this promotion's own new files
under `experiments/EXP-0012-slates-binned-corpus-difficulty/` and edits to
`experiments/{REGISTER,INVARIANTS,TEMP_LOG}.md`. No project source module
was modified.

## Why this test discriminates

If cross-bin miscalibration were a real, separate contributor to
`slates_binned`'s near-zero score, restricting every slate's candidate pool
to ONE push-length bin (so every candidate in the ranking comes from the SAME
operator, eliminating any cross-operator comparison) should raise slateN
measurably above the full mixed-bin pool's score. If the corpus's difficulty
is the whole story, within-bin and across-bin scores should be statistically
indistinguishable -- which is exactly what would have been true anyway had
the drafted "per-pool switching" explanation been correct.

## What was actually run

**RUN-0001** (`experiments/temp/binned-calibration/calibrate.py`, promoted
unmodified): (1) reproduced EXP-0006's exact published cell (corner goal,
lyapunov value fn, MODEL-0002 switched-linear, pooled n20_L10mm+L20mm+L40mm,
n=60) using `desc-mlp/eval_control.py`'s own scoring functions, unmodified.
(2) Directly compared `particles_to_occupancy` vs the official
`_draw_particle_grid` rasteriser's `v_true` on `n20_L40mm` (correlation,
pooled slateN both ways). (3) Computed fixed-K (32, 128) slateN references on
both corpora (slates_binned has no official alternate rasteriser -- ships
particle states only -- so this check could only run on slates_multistep).
(4) Characterised `dv_true` spread (mean/median/std/min/max per slate) on
both corpora.

**RUN-0002** (`within_bin_test.py`, written for this promotion): follow-up
demanded by the promotion task -- restrict `slates_binned`'s 1000-candidate
pools to a SINGLE MODEL-0002 operator-bin at a time (grouped by
`bin_index_switch(length_m, sw_edges)`, MODEL-0002's own 6-bin scheme over
0-80mm -- NOT the corpus's own collection-time `bin_realized` field, which
uses a different 5-bin scheme over 20-70mm; verified the two schemes differ
before choosing which to group by), score slateN within each bin separately
(exact K=N and a K=32-capped resample), and compare the pooled within-bin
number to the full-pool across-bin reference from RUN-0001.

## Numbers

### RUN-0001: reproduction

| | mean | sem | n |
|---|---:|---:|---:|
| reproduced here | +0.458 | 0.054 | 60 |
| EXP-0006 published | +0.461 | 0.061 | 60 |

### RUN-0001: rasteriser cross-check (n20_L40mm)

| | corr(v_true, official) | pooled slateN |
|---|---:|---:|
| particles_to_occupancy | 0.998 | 0.822 |
| official _draw_particle_grid | (ref) | 0.820 |

### RUN-0001: fixed-K references

| corpus | pool size | K=32 | K=128 |
|---|---:|---:|---:|
| slates_multistep (pooled) | 128 | +0.535 | +0.466 |
| slates_binned n20_scatter | 1000 | -0.020 | -0.016 |

### RUN-0001: dv_true spread per slate

| corpus | mean | median | std | min | max |
|---|---:|---:|---:|---:|---:|
| slates_multistep | 0.185 | 0.145 | 0.134 | 0.026 | 0.415 |
| slates_binned | 0.154 | 0.147 | 0.041 | 0.105 | 0.276 |

### RUN-0002: within-bin vs across-bin (all corner/lyapunov/MODEL-0002)

| scope | K | mean | sem | n |
|---|---|---:|---:|---:|
| full pool (mixed 5 bins) | 32 | -0.0196 | 0.0426 | 20 |
| full pool (mixed 5 bins) | 128 | -0.0165 | 0.0344 | 20 |
| single-bin-only, pooled over 5 bins | exact (K=N) | -0.0561 | 0.0383 | 100 |
| single-bin-only, pooled over 5 bins | 32-capped | ~-0.061 | ~0.03 (per-bin) | 100 |

Per-bin breakdown (candidate counts match the CODEMAP's stated
`[0, 2675, 5325, 5347, 5301, 1352]` exactly, confirming this used
MODEL-0002's own bin assignment):

| MODEL-0002 bin | n candidates | mean pool/slate | slateN (exact) |
|---:|---:|---:|---:|
| 1 (13.3-26.7mm) | 2675 | 133.8 | -0.199 |
| 2 (26.7-40.0mm) | 5325 | 266.2 | -0.263 |
| 3 (40.0-53.3mm) | 5347 | 267.4 | -0.232 |
| 4 (53.3-66.7mm) | 5301 | 265.1 | +0.175 |
| 5 (66.7-80.0mm) | 1352 | 67.6 | +0.238 |

## What would change the verdict

A model actually trained/fit on `slates_binned`-like mixed-length pools
(rather than MODEL-0002, fit on `overnight_randlen`'s own length
distribution) scoring well above zero there would be the cleanest
alternative test of "is this corpus intrinsically hard for ANY model" vs
"is it hard only for models transferred in from elsewhere" -- not run here,
no such model exists yet (~half a day: fit + eval).

## Threats

- `imprecision`: RUN-0002's within-bin cells rest on n=20 slates per bin (100
  pooled across 5 bins) -- individual per-bin sems (0.036-0.067) are large
  relative to the per-bin means; only the POOLED within-bin vs across-bin
  comparison is treated as informative here, not any single bin's sign.
- Considered and dismissed: that `slates_binned` simply has degenerate goals
  (no discriminating signal at all). `frac(dv_true==0)` = 0.00 in every cell
  checked (RUN-0001); the corpus is hard, not degenerate.
- Considered and dismissed: pool-size artifact from comparing K=1000 to
  K=128. Explicitly tested at matched K (32 and 128) in RUN-0001 -- flat on
  slates_binned at both.
- The rasteriser check (RUN-0001) was only run for one (corpus, goal,
  value-fn) cell, not all 9 -- noted as a residual gap in the source
  RESULTS.md and not re-verified here, but the effect size (corr 0.998) makes
  it an unlikely source of a near-zero-vs-strongly-positive discrepancy.

## Unrelated findings

- The originally-drafted "bin-mixing" explanation in the source
  `experiments/temp/binned-calibration/RESULTS.md` was wrong as literally
  stated (claimed the operator "ranks candidates across bins simultaneously"
  as if bin assignment were per-pool) -- `predict_switched` is per-candidate
  in both places it's used. This EXPERIMENT.md corrects the record; the
  temp RESULTS.md is left as-is (superseded by this file, not edited in
  place) with `experiments/TEMP_LOG.md` updated to point here.
- `docs/CODEMAP.md`'s `slates_multistep`/`slates_binned` entries were missing
  the `particles_to_occupancy` vs `_draw_particle_grid` distinction and the
  "slates_binned has no official rasteriser" note -- both added to CODEMAP
  as part of this promotion (see docs/CODEMAP.md's Datasets section,
  "Two rasterisers exist" / slates_binned row).

## Amendment (2026-09-15, EXP-0013)

**The original numbers in this record are unchanged and stand as measured.**
Their scope was over-general: "every model measured on this corpus scores
near zero" was true of every model tested at the time (MODEL-0002 and the
EXP-0011 MLPs), all of which read out a value through a point-mass/COM
approximation of the predicted state, never a full predicted image.

EXP-0013 fit a switched-linear pixel-space VISUAL operator (predicts an
actual occupancy image, the same family as MODEL-0001, just refit on a
larger pooled corpus) and scored it on this exact corpus
(`n20_scatter_s20a1000_L20-70mm`) with the same K=32-fixed-reference,
sampling-without-replacement methodology this record established
(`score_pool_kfixed`, reused directly from `binned-calibration__calibrate.py`).
Result: far from zero on every one of 9 (goal x value_fn) cells, e.g.
`ring`/lyapunov res32 +0.389±0.027 / res64 +0.469±0.031 vs persistence
+0.011±0.009 and random +0.017±0.015 (20/0/0 wins on both). `frac(dv_true==0)`
in EXP-0013's run (0.03-0.29 depending on value_fn) is the same order as this
record's own (0.00 reported for the one cell RUN-0001 checked, corner/lyapunov
specifically -- not identical goal, but the same corpus and the same order of
magnitude), so the `dv_true` distribution itself is not what changed.

**Conclusion**: the governing condition for "near-zero on this corpus" is the
READOUT (point-mass/descriptor vs full image), not the corpus alone. The
`slates-binned-uniform-difficulty` invariant in `INVARIANTS.md` has been
reworded to scope the claim to point-mass/descriptor readouts, and `C-018` in
`REGISTER.md` has been moved to "Open and contested" with status `narrowed`,
citing this amendment. See `experiments/EXP-0013-*` for the full resolution
comparison this counter-evidence was found inside of.
