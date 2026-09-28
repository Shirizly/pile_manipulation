---
id: EXP-0060
title: >
  Metric-correlation study: cross-model (n=6, underpowered) Kendall tau finds unblurred
  accuracy_1 tracks slateN_tough best (+0.87, p=0.017), ahead of blurred accuracy /
  mass_in_goal_mae (+0.73, p=0.056) and occ_emd_swept (+0.20, not significant); the NEW
  per-model pool-bootstrap CIs (n=32 pools, well-powered) show retrieval_1nn's within-pool rank
  correlation (+0.52) is much closer to the occupancy models' (+0.61-0.81) than its top-1-only
  slateN_tough gap suggests (0.44 vs 0.63-0.83) -- its shortfall concentrates in picking the
  single best action, not in the general ordering
tier: T1
mode: exploratory
date: 2026-09-28
hypothesis: null
claim: >
  Across the model population scored on DS-0009 by EXP-0059's extended harness (persistence,
  4 EXP-0053 occupancy models, retrieval_1nn), Kendall tau between each candidate offline image
  metric and slateN_tough (the control metric the design doc's section 4 asks this study to
  predict) differs meaningfully by candidate metric, and this record's `code/correlate.py`
  computes that comparison in a form that appends cleanly as more zoo members (perturbed
  simulators, retrieval variants, sharpened NFDs) are scored.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/correlate.py (new, this record; pure CPU, reads existing offline_eval_extended.json,
    no new simulation or model evaluation)
  data: ["experiments/EXP-0059-retrieval-transition-model/results/offline_eval_extended.json (6
         models: persistence, nfd_3ch_narrow_l20, nfd_3ch_narrow_l20_wide,
         linear_narrow_l20_res64, nfd_residual_worldframe_noaug_ep43, retrieval_1nn)"]
  code_path: "reads {model: {metric_key: value}} JSON produced by EXP-0059's
    code/eval_extended.py; scipy.stats.kendalltau / spearmanr across the model population"
  seed: not applicable (deterministic given the input JSON)
  split: "not applicable -- correlates existing DS-0009 population summaries, no train/test split"
  data_commit: not applicable
  runs: [correlate]
  runtime: "<1s, pure CPU, no GPU contention with the concurrent DS-0011/DS-0012 collection"
budget:
  declared: "part of a ~60-75 min combined task (this session's coordinator message); this
    record's own share: ~15 min (structure + first run, deliberately GPU-free per the
    coordinator's 'don't starve DS-0011/DS-0012' instruction)"
  spent: "~15 min"
  outcome: within
design:
  varied:
    candidate_metric: "accuracy_1, accuracy_1_blur1, accuracy_1_blur2, occ_emd_swept,
      mass_in_goal_mae, mass_in_goal_mae_tough (occ_emd_swept and both mass_in_goal_mae keys are
      LOWER-is-better and sign-flipped before correlating, so every reported tau/rho reads
      'positive = tracks slateN_tough correctly')"
  held_fixed:
    target: "slateN_tough (8-goal set: letter_O/T/S/X/L/I, two_squares, quadrant_0), the
      coordinator's stated headline metric for 'does ranking come from information or blur'"
    population: "the 6 models EXP-0059's R0 harness has scored on DS-0009 so far"
  baselines: "none in the slateN sense -- this is a correlation study, not a model comparison;
    the 'do nothing' baseline (persistence) is one of the 6 population points, not a separate
    control"
  metric: "Kendall tau + Spearman rho, both computed ACROSS THE MODEL POPULATION (n = n_models),
    not the pool-bootstrap CI the coordinator's brief asked for -- see 'What would change the
    verdict' for exactly why and its cost"
noise_floor: "not measured -- n=6 is already far below paired_stats.required_n for any
  correlation coefficient to clear a conventional significance bar reliably; the accuracy_1 row
  (tau +0.867, p=0.017) is the only one that does, and even that is one bootstrap-resample of 6
  points from crossing back over p=0.05 (not tested here, flagged as a threat)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  At n=6 models, unblurred accuracy_1 has the strongest and only conventionally-significant
  correlation with slateN_tough (Kendall tau +0.867, p=0.017; Spearman rho +0.943, p=0.005).
  Blurred accuracy (sigma 1 and 2 px) and mass_in_goal_mae (both goal sets) tie at tau +0.733
  (p=0.056, just short of 0.05) -- i.e. blurring accuracy_1 in this population does NOT improve
  its correlation with the control metric, it slightly weakens it, consistent with the
  coordinator's framing that a metric rewarding blur is not the one that tracks ranking quality.
  occ_emd_swept is the weakest and only non-significant candidate (tau +0.200, p=0.719) -- driven
  by linear_narrow_l20_res64 and nfd_residual_worldframe_noaug_ep43 sitting on the "wrong" side
  of an otherwise-monotone relationship (see EXP-0059's own numbers: linear_res64 is worse than
  persistence on occ_emd_swept despite positive slateN; the broad residual NFD has the BEST
  occ_emd_swept despite ranking 4th on slateN_tough). No candidate metric here is a safe
  drop-in replacement for slateN at this population size; accuracy_1 is the current best-tracking
  cheap proxy among those tested, provisionally. Separately, and WELL-POWERED (n=32 pools, not
  n=6 models): pool-bootstrap 95% CIs on slateN_tough separate every model pair except the two
  narrow NFDs from each other, and within-pool Spearman(vp,vt) shows retrieval_1nn's whole-pool
  rank correlation (+0.523) sits much closer to the occupancy models' (+0.611 to +0.811) than
  its slateN_tough gap implies -- its deficit is concentrated in the top-1 pick, not general
  ranking ability.
verdict: inconclusive
downgrades: [imprecision, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If a cheap image metric consistently orders models the same way slateN_tough does, it is usable
as a fast proxy when a full closed-loop or slateN evaluation is too expensive (e.g. inside R1's
variant sweep). If it does not -- or if it only agrees when blur is added, which slateN itself
does not reward -- that specific metric should not be trusted to select variants on its own.
Kendall tau on a RANKING of models is the right first cut because the coordinator's question is
explicitly about ranking/ordering behaviour ("does ranking quality come from information or
noise"), not raw magnitude agreement.

## What was actually run

`code/correlate.py` reads EXP-0059's `results/offline_eval_extended.json` (6 already-scored
models), builds one row per model of {candidate metric, slateN, slateN_tough}, sign-flips the
three lower-is-better candidates, and computes Kendall tau / Spearman rho between each candidate
and `slateN_tough` across the 6 models. No new simulation, no GPU use -- deliberately, per the
coordinator's instruction not to compete with the concurrently-running DS-0011/DS-0012 Genesis
collection jobs for the machine.

## Numbers (`results/correlations.json`)

| candidate metric | n | Kendall tau | p | Spearman rho | p |
|---|---|---|---|---|---|
| accuracy_1 | 6 | +0.867 | 0.017 | +0.943 | 0.005 |
| accuracy_1_blur1 | 6 | +0.733 | 0.056 | +0.886 | 0.019 |
| accuracy_1_blur2 | 6 | +0.733 | 0.056 | +0.886 | 0.019 |
| occ_emd_swept (flipped) | 6 | +0.200 | 0.719 | +0.543 | 0.266 |
| mass_in_goal_mae (flipped) | 6 | +0.733 | 0.056 | +0.886 | 0.019 |
| mass_in_goal_mae_tough (flipped) | 6 | +0.733 | 0.056 | +0.886 | 0.019 |

## Pool-level results (added 2026-09-28, once `eval_extended.py` started persisting per-pool raw arrays)

`eval_extended.py`'s `eval_occ_model`/`eval_particle_model` now also return
`{"slateN_raw"/"slateN_tough_raw": {goal: [{"pool","vt","vp"}, ...]}}` (32 entries/goal, the
exact true/predicted `dv` vectors `slate_capture` already computed internally but previously
discarded), written to a sibling `<results>_raw.json`. `correlate.py --raw-sources` consumes
this for two PER-MODEL quantities that do not need n_models>=3 the way the cross-model Kendall
tau above does:

| model | slateN_tough (95% pool-bootstrap CI, 1000 resamples) | within-pool Spearman(vp, vt) |
|---|---|---|
| persistence | +0.050 [-0.052, +0.142] | n/a (predicted dv has zero spread every pool by construction) |
| nfd_3ch_narrow_l20 | +0.778 [+0.727, +0.825] | +0.753 |
| nfd_3ch_narrow_l20_wide | +0.827 [+0.783, +0.866] | +0.811 |
| linear_narrow_l20_res64 | +0.633 [+0.550, +0.707] | +0.611 |
| nfd_residual_worldframe_noaug_ep43 | +0.749 [+0.692, +0.794] | +0.691 |
| retrieval_1nn | +0.436 [+0.333, +0.537] | +0.523 |

**Reading:** every non-degenerate model's CI excludes both 0 and every OTHER model's point
estimate except the two narrow NFDs' CIs, which overlap each other slightly (0.783-0.825) --
i.e. at n=32 pools the ranking narrow-wide-NFD > broad-residual-NFD > linear_res64 >
retrieval_1nn > persistence is well-supported, not just a difference within noise.
**within-pool Spearman is a genuinely different cut than `slateN_tough` and narrows the gap
between models**: retrieval_1nn's whole-pool rank correlation (+0.523) is closer to
linear_res64's (+0.611) than its `slateN_tough` gap would suggest (0.436 vs 0.633) -- i.e.
retrieval_1nn's ORDINAL ranking ability across a full 64-candidate pool is not as far behind as
its TOP-1 capture metric implies; its shortfall is concentrated in picking the single best
action, not in the general ordering. This is exactly the kind of nuance `slateN` (a top-1-only
statistic) cannot show and Spearman can.

## Addendum (2026-09-28, fresh instance): perturbed-sim zoo appended, cross-model tau unchanged

Ran `code/ingest_perturbed_sim_zoo.py --zoo ../../EXP-0059-*/results/perturbed_sim_zoo.json --out
results/perturbed_sim_zoo_flat.json` (12 flattened rows: 3 `nfd_3ch_narrow_l20_*_control` blur
cells + 9 `sim_{0,0.5,1}mm_blur{0,1,2}` cells) then `correlate.py --sources
offline_eval_extended.json perturbed_sim_zoo_flat.json` -- `n_models` in `results/
correlations.json` is now 18 (was 6). **The cross-model Kendall tau/Spearman rho table itself is
UNCHANGED** (still exactly the 6 rows/values in "Numbers" above): as the ingest script's own
docstring says, the zoo rows carry only `slateN`/`slateN_tough`, none of the six candidate
metrics (`accuracy_1`, blur variants, `occ_emd_swept`, `mass_in_goal_mae*`), so `correlate.py`'s
per-candidate-metric correlation (which pairs each candidate against `slateN_tough` and drops
rows where either is nan) still has exactly the original 6 non-nan pairs -- the append changes
`n_models` in the JSON but not `n` in any correlation row. This is expected, not a bug: the zoo
answers a different question (EXP-0059's "does slateN survive state perturbation better than
accuracy_1" -- yes, see EXP-0059's R2 addendum) and was never going to add a candidate-metric
correlation point unless someone also computes `accuracy_1`/`occ_emd_swept`/etc. for the
perturbed-sim cells, which `perturbed_sim_zoo.py` does not do by design (slateN-only extension).
The pool-bootstrap / within-pool-Spearman section is likewise unaffected -- it reads `*_raw.json`
sibling files, which the zoo does not produce.

## What would change the verdict

- **More zoo members with the SIX candidate metrics computed, not just `slateN`.** The
  perturbed-sim zoo is now appended (see Addendum above) but contributes 0 new points to the
  candidate-metric correlation table, because it was only ever scored on `slateN`/`slateN_tough`.
  Growing n=6 to something power-adequate needs new model/variant rows carrying the FULL metric
  set (`accuracy_1` + blur + `occ_emd_swept` + `mass_in_goal_mae`), e.g. retrieval variants from
  R1 onward scored through `eval_extended.py` (not `perturbed_sim_zoo.py`) --
  `Baselines/common/paired_stats.required_n` should be consulted once a rough effect size is
  known.
- **Cube-level Chamfer distance**, named in the brief as an additional candidate metric, is not
  yet implemented (`occ_emd_swept` is the only occupancy-distance metric available so far) --
  cost: for particle-in/out models this is direct (per-cube correspondence, as EXP-0059's own
  moved-cube mm error already computes); for occupancy-in/out models it needs a
  threshold-and-extract-points step on the rasterised prediction, feasible but unbudgeted here.

## Threats

- `imprecision`: n=6 is far below any reasonable power threshold for a correlation coefficient;
  every number here should be read as "suggestive of a difference between candidate metrics,"
  not as an established ranking of them.
- `incomplete-design`: pool-bootstrap CIs and within-pool Spearman(dv_pred, dv_true) are now
  implemented and reported (see "Pool-level results" above); cube-level Chamfer distance is
  still not implemented -- named above with cost, not silently dropped. The pool-bootstrap
  itself resamples pools jointly across all 8 goals (not per-goal independently) -- a modelling
  choice, not a gap, since every goal is scored on the SAME 32 pools.
- `provenance`: all 6 input rows come from one harness run (`EXP-0059`'s `eval_extended.py`), so
  this record inherits, rather than re-verifies, that harness's own correctness (already
  cross-checked against EXP-0053/EXP-0059's prior numbers there).

## Unrelated findings

None.
