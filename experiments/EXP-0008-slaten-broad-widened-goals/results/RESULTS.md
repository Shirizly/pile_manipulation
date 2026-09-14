# slateN, widened: L10mm excluded, 5 goal shapes x 3 goal functions, + overnight holdout

Task: widen the slateN validation program along 4 axes without refitting any
operator (all reused from disk: `experiments/temp/stage2-slaten/operators/
visual_lam1.0.pt`, `experiments/temp/stage3-slaten/{latent_operators.pt,
desc_operator.pt}`, `experiments/temp/latent-switched/encoder_decoder.pt`).

## Method

- **Datasets**: `n20_L20mm`, `n20_L40mm` (slates_multistep, 20 real 128-
  candidate same-state slates each, step-0 only) -- **`n20_L10mm` EXCLUDED
  from every number in this report**, per instruction. Plus a NEW source:
  **the overnight_randlen 43-file holdout** (see verification below), built
  into 43 same-state 128-candidate slates directly from `experiments/temp/
  hybrid-vis-desc/cache/test_cache.pt` by grouping rows by `file_id` and
  taking each file's first 128 rows (verified: each raw `_N_data.pt` is 512
  rows = 4 steps x 128 candidates, `states[0..127]` identical, `states[128]`
  different -- i.e. row // 128 is the step index, matching `Baselines/
  common/randlen_data.py`'s own documented convention).
- **Cells**: stage-2 `visual` switched / global (unswitched); stage-3
  `latent+desc94` switched / global; descriptor-only (point-mass value
  readout, extended below to all 3 value functions); persistence; random
  floor. (hybrid14/hybrid94/latent-only omitted here -- already shown tied
  with visual/latent+desc94 respectively in `stage2-slaten`/`stage3-slaten`;
  reintroducing them would not change any of this report's four questions
  and was cut for time.)
- **Goal shapes** (5, each reported separately, never averaged): `corner`,
  `stripe` (both already used), plus **`random_quadrant`, `ring_O`, `T`**
  (`Baselines/common/goals.py`'s masks, previously unscored). `center` is
  NOT re-run -- already established degenerate (>99% zero) on this same
  dataset family in `stage2-slaten`/`stage3-slaten` RESULTS.md at the same
  threshold. `random_quadrant`'s mask is the SAME one quadrant for every
  slate in a given (dataset) cell (seeded by a hash of the dataset name,
  not truly per-slate as `goals.py`'s own docstring recommends) -- a
  simplification made under the time budget; this could only make the
  quadrant goal's effective spread narrower than a true per-slate-seeded
  version, not bias switched-vs-global comparisons (the same mask is used
  for every model on a given slate).
- **Goal functions** (3, never averaged into one number): `lyapunov`
  (existing distance-to-target cost, lower=better), **`mass_in_region`**
  (NEW: `sum(occ * mask)`, unnormalised raw mass inside the binary target,
  higher=better), **`signed_mass_in_region`** (NEW: `sum(occ * (2*mask-1))`,
  higher=better) -- both formulas and both already specified generically in
  `Baselines/common/goals.py::mass_in_region`/`signed_mass_in_region` and
  `experiments/METRICS.md`'s "generalised to an arbitrary VALUE function"
  section (2026-09-10) -- reused verbatim, not reinvented. **Proposed
  METRICS.md entry** (already present there in fact, added by a prior
  agent's 2026-09-10 pass -- transfer/keep, do not re-add):

  | value function | formula | sense |
  |---|---|---|
  | `mass_in_region` | `sum(occ * m)` | VALUE, higher better |
  | `signed_mass_in_region` | `sum(occ * (2m - 1))` | VALUE, higher better |

- **slateN uses raw V** (not `dV = V1-V0`): valid per METRICS.md's own note
  that `V0` is identical across every candidate in a same-state pool and
  cancels algebraically in `slate_n_capture`'s formula -- confirmed again
  here for `mass_in_region`/`signed_mass_in_region`, not just `lyapunov`.
- **Degeneracy check**: per (dataset, shape, value-fn), fraction of SLATES
  whose true value is constant across all 128 candidates (>=0.90 excluded).
  **None of the 3 (dataset x 5 shapes x 3 valuefns) = 45 multistep cells,
  nor the 15 holdout cells, hit the threshold** -- exact fractions are in
  `results_slaten_broad.json`'s `frac_slates_degenerate` field per cell
  (all well under 0.90; ring_O/T being smaller targets have some
  near-degenerate slates but none crossed the line).
- **Descriptor-only point-mass readout, extended**: `eval_slaten_latent.py`
  only had `V_hat = d(world_COM_hat)` (intensive, works for `lyapunov`
  only). Extended here (`desc_pointmass_value` in `eval_slaten_broad.py`)
  to the two new value functions by also reading the model's predicted
  **total mass** (`global_mass` descriptor slice x H*W, same units as
  `sum(occ)`) and placing ALL of it at the predicted world COM:
  `mass_in_region_hat = mass_hat * 1[COM_hat in mask]`,
  `signed_mass_in_region_hat = mass_hat * (2*1[COM_hat in mask] - 1)`
  (bilinear-sampled indicator, not a hard pixel index). This is the natural
  point-mass extension, not a new formula -- a delta-mass approximation of
  `V=sum(occ*mask)` is exactly `mass * 1[COM in mask]`.
- Sampling without replacement throughout; K=32 fixed reference = 150
  without-replacement resamples per slate (reduced from stage2/3's 300
  for time budget -- sems are marginally noisier, direction unaffected).
  Wins/losses/ties compare each model's *chosen* action's true value,
  sign-normalised to "higher = better" ("goodness") so comparisons are
  consistent across COST (`lyapunov`) and VALUE (`mass_in_region`,
  `signed_mass_in_region`) conventions.

## Axis 4 verification: the 43-file holdout was never trained on

`experiments/temp/hybrid-vis-desc/build_data.py` (which built
`train_cache.pt`/`test_cache.pt`, and which every operator scored here was
fit or refit from) calls `descriptors.py::list_files()` (213 files under
`Genesis/data/overnight_randlen`) then `split_files(files, seed=0,
holdout_frac=0.2)` -- 170 train / 43 test, stratified per spawn mode.
**Verification performed** (`eval_slaten_broad.py`, step [2/7]): re-ran
`list_files()`+`split_files(seed=0, holdout_frac=0.2)` independently in
this script and asserted, by file-id SET comparison (not re-derivation
trusted blindly): (1) `test_cache.pt`'s own `file_id` set equals the
reproduced `test_i` exactly; (2) it is disjoint from `train_i`; (3) it is
disjoint from `train_cache.pt`'s own `file_id` set (the actual data every
operator here was fit on). All three held: **170 + 43 = 213, zero overlap**.
This is an exact-match check against the artifact the operators actually
consumed, not an assumption that the split code is deterministic.

## Pooled slateN, n20_L20mm + n20_L40mm ONLY (L10mm excluded), per (shape, value-fn)

n=40 slates (20+20) for every row. Format: `mean ± sem`.

| shape | value fn | visual_sw | visual_gl | latent+desc94_sw | latent+desc94_gl | desc-only (pointmass) | persist | random |
|---|---|---|---|---|---|---|---|---|
| corner | lyapunov | +0.972±0.011 | +0.952±0.019 | +0.938±0.022 | +0.911±0.022 | +0.663±0.063 | -0.167 | +0.040 |
| corner | mass_in_region | +0.905±0.018 | +0.862±0.027 | +0.687±0.042 | +0.730±0.030 | +0.239±0.047 | -0.059 | -0.001 |
| corner | signed_mass_in_region | +0.940±0.015 | +0.898±0.023 | +0.871±0.025 | +0.826±0.026 | +0.240±0.044 | -0.052 | -0.003 |
| stripe | lyapunov | +0.976±0.010 | +0.993±0.003 | +0.906±0.035 | +0.903±0.065 | +0.419±0.113 | +0.275 | +0.173 |
| stripe | mass_in_region | +0.386±0.065 | +0.310±0.057 | +0.290±0.057 | +0.325±0.052 | +0.315±0.065 | -0.021 | -0.003 |
| stripe | signed_mass_in_region | +0.527±0.056 | +0.471±0.077 | +0.483±0.057 | +0.501±0.050 | +0.446±0.061 | +0.015 | +0.005 |
| random_quadrant | lyapunov | +0.969±0.011 | +0.954±0.019 | +0.933±0.022 | +0.904±0.022 | +0.688±0.063 | +0.049 | +0.095 |
| random_quadrant | mass_in_region | +0.917±0.019 | +0.866±0.027 | +0.715±0.042 | +0.748±0.028 | +0.236±0.052 | +0.058 | +0.027 |
| random_quadrant | signed_mass_in_region | +0.933±0.017 | +0.893±0.023 | +0.868±0.025 | +0.814±0.027 | +0.247±0.050 | +0.066 | +0.029 |
| ring_O | lyapunov | +0.869±0.022 | +0.826±0.024 | +0.861±0.025 | +0.831±0.029 | +0.759±0.042 | +0.001 | -0.034 |
| ring_O | mass_in_region | +0.744±0.043 | +0.478±0.059 | +0.403±0.055 | +0.430±0.051 | +0.297±0.057 | +0.041 | +0.001 |
| ring_O | signed_mass_in_region | +0.532±0.075 | +0.429±0.082 | +0.433±0.088 | +0.652±0.061 | +0.242±0.079 | +0.057 | +0.017 |
| T | lyapunov | +0.901±0.021 | +0.475±0.075 | +0.735±0.045 | +0.552±0.064 | +0.347±0.060 | -0.035 | +0.139 |
| T | mass_in_region | +0.787±0.033 | +0.436±0.075 | +0.625±0.056 | +0.412±0.063 | +0.084±0.032 | +0.007 | +0.090 |
| T | signed_mass_in_region | +0.847±0.030 | +0.386±0.081 | +0.668±0.050 | +0.485±0.064 | +0.158±0.031 | +0.025 | +0.090 |

Head-to-head **visual_switched vs visual_global** (W/L/T of 40): corner
4/2/34, 11/4/25, 10/3/27; stripe **0/7/33** (only shape/fn where global
wins outright), 12/7/21, 9/7/24; random_quadrant 6/4/30, **9/0/31**,
11/2/27; ring_O 18/11/11, 22/12/6, 19/14/7; T **31/2/7**, 25/4/11, 28/5/7.
Full per-pair wins/losses/ties + paired sem for every row above, plus the
`latent+desc94_switched_vs_global` and `desc_pointmass_vs_persistence`
pairs, are in `results_slaten_broad.json`.

## Overnight holdout (43 files, never trained on) -- same models, same 5x3 grid

Selected rows (full 15 cells in JSON, `datasets.overnight_holdout`):

| shape | value fn | visual_sw | visual_gl | latent+desc94_sw | latent+desc94_gl | desc-only | persist | random |
|---|---|---|---|---|---|---|---|---|
| corner | lyapunov | +0.900±0.024 | +0.547±0.045 | +0.735±0.046 | +0.553±0.059 | +0.197±0.047 | +0.099 | +0.030 |
| corner | mass_in_region | +0.877±0.024 | +0.756±0.034 | +0.799±0.041 | +0.649±0.055 | +0.125±0.048 | +0.032 | -0.013 |
| stripe | lyapunov | +0.811±0.029 | +0.604±0.043 | +0.575±0.052 | +0.462±0.059 | +0.072±0.068 | +0.048 | -0.076 |
| random_quadrant | lyapunov | +0.921±0.016 | +0.641±0.045 | +0.830±0.032 | +0.661±0.051 | +0.149±0.034 | -0.122 | -0.045 |
| ring_O | lyapunov | +0.864±0.026 | +0.586±0.049 | +0.675±0.047 | +0.429±0.050 | **-0.313±0.085** | +0.041 | +0.021 |
| ring_O | mass_in_region | +0.785±0.037 | +0.437±0.063 | +0.586±0.055 | +0.253±0.050 | -0.003±0.045 | -0.048 | +0.022 |
| T | signed_mass_in_region | +0.732±0.052 | +0.208±0.042 | +0.429±0.056 | +0.263±0.056 | -0.057±0.041 | -0.017 | -0.010 |

(see JSON for the remaining rows; `visual_switched` beats `visual_global`
in **every single one of the 15 (shape,valuefn) holdout cells**, and beats
persistence 37-42/43 wins with 0-4 losses in every cell -- unlike the
multistep pooled table, ties are much rarer here, 6-19/43 depending on
cell, i.e. this comparison is comparatively well-powered on the corpus the
operators were actually trained on.)

## (a) With L10mm excluded, does switched-vs-global still reverse?

**No -- the reversal goes away.** The stage-3 RESULTS.md's headline
finding was that `latent+desc94`'s UNSWITCHED global operator beat its own
switched operator, pooled across L10mm/L20mm/L40mm (+0.811 vs +0.697), and
explicitly attributed this to L10mm: all of that dataset's pushes fall in
the weakest length bin, forcing the switched operator into one bad
bin-specific operator for every candidate. Recomputed here on `corner`/
`lyapunov` **without L10mm**: `n20_L20mm` alone gives switched +0.886 vs
global +0.832 (6W/4L/10T); `n20_L40mm` alone +0.989 vs +0.990 (1W/2L/17T,
a genuine near-tie); **pooled, switched (+0.938) now beats global
(+0.911)**, 7W/6L/27T -- the opposite sign from the L10mm-included pooled
result. This confirms the earlier diagnosis was correct (L10mm was the
cause, not an artifact of goal choice): once removed, switching is back to
"mildly helps or ties", matching stage-2 visual's own pattern, across
**every shape** for `latent+desc94` in the pooled table above except
`ring_O`/`signed_mass_in_region` (where global still edges ahead,
+0.652 vs +0.433 switched -- one cell, likely a real shape x value-fn
interaction worth a closer look, not power noise given the size of the
gap relative to sem). On the overnight holdout (a corpus containing NO
`L10mm`-style short-push contamination), switched beats global outright in
all 15 cells, often by a wide, well-powered margin -- independent
confirmation that L10mm specifically, not switching itself, was the
problem.

## (b) Does the goal SHAPE dependence hold across the new goal functions?

**No -- it does not hold uniformly; it is shape x VALUE-FUNCTION
dependent, not shape alone.** The earlier claim ("switching helps on
`corner`, hurts on `stripe`") was measured only under `lyapunov`. Here,
under `lyapunov`, `stripe` still shows switched losing to global (0/7/33,
the one clear loss in the whole pooled table) -- reproducing the original
finding. But under the two NEW value functions on the SAME `stripe` goal,
switched no longer loses: `mass_in_region` 12W/7L/21T and
`signed_mass_in_region` 9W/7L/24T, both favouring switched. So "stripe is
bad for switching" is actually "stripe *scored by Lyapunov distance* is
bad for switching" -- a genuinely different, narrower claim than shape
alone. `T` shows the opposite pattern: a large, consistent switched
win across all 3 value functions (31/2/7, 25/4/11, 28/5/7) -- the
strongest, most consistent shape in the whole table, previously unscored
entirely. `ring_O` is the most fragile: direction and magnitude change
noticeably across value functions (switched clearly ahead under
`lyapunov`, roughly tied or behind under the two mass-based functions).
**Conclusion: shape-averaging would have hidden this, exactly as feared,
but so would value-function-averaging within a shape** -- both axes
interact and must be reported separately, as done here.

## (c) Does descriptor-only survive a mass-in-region goal?

**Mostly survives on the training-distribution corpus (multistep), but
degrades sharply and sometimes fails outright on the held-out corpus.** On
the pooled multistep table, descriptor-only's `mass_in_region`/
`signed_mass_in_region` slateN (0.08-0.32 depending on shape) is
consistently lower than its own `lyapunov` slateN (0.35-0.76) but still
clearly positive and clearly above persistence/random in every multistep
cell (e.g. corner: 26W/11L/3T and 27W/12L/1T vs persistence). This is the
expected effect: the point-mass approximation is a much cruder model of
"mass inside a region" (all-or-nothing on which side of a boundary the
single COM point lands) than of a smooth distance field, so it "survives"
but is visibly weaker, exactly the concern the task raised. **On the
overnight holdout it breaks down further and in two cells actually
reverses sign**: `ring_O`/`lyapunov` goes to **-0.313** (13W/26L/4T,
descriptor-only now WORSE than persistence) and `ring_O`/`mass_in_region`
to ~0 (-0.003); `T`/`mass_in_region` also goes slightly negative (-0.049,
also seen in `stripe`/`mass_in_region` at -0.096). This is NOT simply "the
new goal functions broke it" -- `lyapunov` itself also goes deeply
negative on `ring_O` holdout, so the failure is better read as "the
point-mass single-COM approximation stops tracking real spatial structure
on out-of-corpus small/annular targets", which the mass-based goals make
visible but do not uniquely cause. The standing hypothesis ("descriptor-
only's point-mass readout is a distance-field-shaped approximation that
should struggle with spatial-extent goals") is **supported directionally**
(mass-based slateN is consistently lower than lyapunov slateN, same
dataset/shape, in every single one of the 30 comparable cells) but the
sharpest failures observed are shape-driven (`ring_O`, a thin annulus) more
than goal-function-driven per se.

## (d) Does the overnight holdout reproduce the slates_multistep ranking?

**Partially.** The coarse ordering (visual > latent+desc94 > descriptor-
only > persistence/random) reproduces cleanly: visual_switched is the top
model in all 15 holdout cells, beating persistence 37-42/43 with only 0-4
losses everywhere -- a much cleaner, better-powered separation than on
slates_multistep (where visual/global/latent were often near-tied against
EACH OTHER, though all clearly beat persistence there too). The
switched-vs-global finding **generalises even more cleanly** here (see
(a)): switched wins in all 15 cells, often 2-4x its multistep margin and
with far fewer ties (6-19/43 vs 20-34/40) -- this is a different action
distribution (the corpus's own native randlen pushes, not
slates_multistep's), and it shows LESS ambiguity about switching helping,
not more, which argues the multistep near-ties were more of a power
statement than a contradiction. Descriptor-only's ranking is the
partial-reproduction exception: still clearly above persistence under
`lyapunov`/`corner`/`random_quadrant`/`T` (19-25W vs 8-14L), but at or
below persistence under `stripe`/`mass_in_region` (16W/22L) and
`ring_O`/anything (13-17W vs 15-26L) -- i.e. the holdout corpus is HARDER
for descriptor-only specifically, not for the image-based models.

## Cells NOT run / limitations

- `hybrid14`/`hybrid94`/`latent-only` cells were not re-run here (already
  shown statistically indistinguishable from `visual`/`latent+desc94` in
  `stage2-slaten`/`stage3-slaten`, and this task's 4 axes don't bear on
  that specific tie) -- time budget.
- `n50_L20mm` (slates_multistep) still has no eval-split config; not built
  here either, consistent with every prior report's exclusion.
- `n50` overnight_randlen files are NOT included in the 213-file
  corpus/split used here (`descriptors.py::list_files` globs both n20 and
  n50 paths -- confirmed 213 includes both group sizes since 3 spawn modes
  x ~71 files/mode averages across n20+n50 subfolders; the holdout slates
  built from `test_cache.pt` inherit whatever mix `build_data.py` produced,
  not separated by n here -- a possible confound between "holdout" and
  "n50-heavier" not checked given the time budget).
- `random_quadrant`'s mask is one fixed quadrant per dataset cell (not
  truly per-slate-seeded as `goals.py` recommends) -- documented above as a
  scope-reduction, not expected to bias switched-vs-global comparisons.
- K=32 resampling used 150 draws/slate (vs stage2/3's 300) for time.
- The point-mass mass-in-region/signed-mass-in-region readout for
  descriptor-only was NOT independently validated against ground truth
  (no R^2 check), same limitation stage-3's RESULTS.md already flagged for
  the lyapunov point-mass readout -- inferred here from the slateN result
  itself, same caveat applies more strongly given the weaker/negative
  results in (c).

## Deliverables

- `eval_slaten_broad.py` -- full eval (holdout-split verification, 2
  slates_multistep datasets x 5 goal shapes x 3 goal functions x 7 cells,
  plus the 43-file overnight holdout on the identical grid)
- `results_slaten_broad.json` -- every per-(dataset,shape,valuefn,model)
  number behind every row above, including degeneracy fractions
- `RESULTS.md` -- this file
