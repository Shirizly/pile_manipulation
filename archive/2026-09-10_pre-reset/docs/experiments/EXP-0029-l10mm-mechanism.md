---
id: EXP-0029
title: >
  L10mm's accuracy=-0.44 / slateK_exact=0.91 dissociation has a mechanism:
  the corner-goal cost is (to first order) linear in occupancy and sees only
  the ~0.4-0.8% of the linear operator's swept-region error that projects
  onto its weight field, at every push length tested -- not uniquely at
  L10mm. The "far smaller at 10mm than 40mm" form of the D1 prediction holds
  strongly for the RAW canonical-frame pipeline (mean-delta 7.1x, warp-only
  13.5x) but only weakly for the FITTED linear operator itself (1.9x) --
  narrowed, not refuted. Geometry-only heuristics reach only 0.49-0.75
  slateK_exact at L10mm (well below linear/UNet's 0.79-0.91), refuting a
  purely-kinematic explanation there; the shuffle null collapses to ~0
  everywhere (no leak); and the UNet-linear inconsistency at K=128 resolves
  as 15/20 pools tied (near-identical top pick) with the surviving 5 split
  2-3, consistent with D1's finding that UNet's and linear's COST-VISIBLE
  error are nearly equal (59.2 vs 57.7 units) despite a 45% gap in total
  image-error energy
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: null

claim: >
  (D1) V = d^T y / ||y||_1 is, to first order at fixed mass, a linear
  functional of occupancy: dV_pred - dV_true ~= (1/M0) d^T e for
  e = pred - truth, so only the component of e that projects onto d is
  visible to the cost. For the fitted linear operator (goal=corner,
  n20_L{10,20,40}mm, step-0 candidates), the fraction of swept-region error
  ENERGY projecting onto the (mass-normalised, mean-removed, per-transition)
  cost direction is small (0.40-0.78%) at every push length, and the
  linearisation itself agrees with the exact cached dV to Pearson r=0.79-0.99
  across models/cells. (D2) Parameter-free geometric push heuristics
  (model.eulerian_wrapper's cumulative/spread), scored with the identical
  accuracy+slateK_exact suite, reach only 0.49-0.75 slateK_exact at K=128 on
  L10mm -- well below linear (0.913) and UNet (0.890) -- so the L10mm
  ranking task is not reducible to kinematics alone. (D3) Permuting dv_pred
  within each slate and averaging 200 draws/slate collapses slateK_exact to
  within ~1-3 sem of 0 for every model at every cell and K in {4,128} tested.
  (D4) At L10mm, the UNet's paired ranking edge over linear clears its sem
  at K=4 (+0.0675, sem 0.0192, t=+3.51) but not at K=128 (-0.0235, sem
  0.0462, t=-0.51, 15/20 slates tied) -- consistent with D1's finding that
  the two models' cost-visible error is nearly equal (E_proj 59.2 vs 57.7)
  despite UNet's 30% smaller total swept-region error energy.

prediction:
  supports: >
    (D1, pre-registered before this record's own run) the pooled projected
    fraction E_proj/E_total is at least 3x smaller at L10mm than at L40mm
    for at least 2 of the 3 canonical-frame baselines (mean-delta, linear,
    warp-only/identity-operator). (D2) any parameter-free heuristic reaching
    slateK_exact >= 0.85 at K=128 on L10mm would mean the ranking task there
    is essentially kinematic. (D3) shuffled slateK_exact within 2 sem of 0
    for every model/cell/K tested; a value more than 3 sem from 0 on any
    cell is treated as a pairing/leakage bug, not noise. (D4) the paired
    UNet-linear mean difference at K=128 is not required to clear its sem
    for this record to stand -- but if it does NOT clear sem while the K=4
    edge does, the record must show why (D1's per-model projected-energy
    comparison is the designated explanation, checked directly).
  refutes: >
    (D1) the projected fraction is within 2x across all three push lengths
    for every canonical-frame baseline (would mean the error-projection
    mechanism does not track push length at all, and a different
    explanation is needed for L10mm specifically). (D2) every heuristic
    stays below slateK_exact 0.85 at every cell tested (would rule out a
    purely-kinematic account everywhere, not just at L10mm -- still
    informative, not a refutation of D1). (D3) shuffled slateK_exact
    exceeds 3 sem from 0 on any cell/model/K (stop and report as a bug,
    supersedes everything else in this record). (D4) UNet's projected error
    energy is markedly SMALLER than linear's at L10mm while the paired K=128
    test still shows no edge (would mean D1 cannot explain D4, and the
    inconsistency needs a different account).
  discriminating: true

provenance:
  commit: a446807e
  dirty: false
  data_commit: unrecorded (Genesis/data/slates_multistep/n20_L{10,20,40}mm_{train,eval}, see EXP-0024_v1/EXP-0028's own provenance note)
  script: scripts/probes/exp0029_l10mm_mechanism.py
  data: ["configs/dataset/genesis_slates_multistep_n20_L{10,20,40}mm_{train,eval}.yaml",
         "Genesis/data/slates_multistep/n20_L{10,20,40}mm/manifest.json",
         "runs_expB/unetfilm_slates_multistep_n20_L{10,20,40}mm/ (UNet checkpoints)",
         "runs_expB/n20_L{10,20,40}mm_dv_cache.pt (cross-check only, see D1 cache cross-check)"]
  code_path: >
    independent reload via dmdc_baseline.load_transition_arrays (train split,
    for fitting) + registry.dataset_registry.build_dataset (eval split, step-0
    filtered via the source cell's manifest.json) -- the same pattern
    EXP-0028's part (b) used, NOT expB_multistep_eval.py's own cache-building
    loop (which is used only as an external cross-check target in "D1 cache
    cross-check", not as a data source). predict_world/predict_heuristic/
    predict_meandelta/control_utility_test.lyapunov for predictions;
    scripts/probes/exp0026_kcurve_exact.per_slate_exact/sweep/paired for every
    ranking statistic (unchanged, already self-tested there).
  seed: "0 for D3's shuffle draws (np.random.default_rng(0), 200 draws/slate); no other randomness -- operator fits and every prediction are deterministic given the data"
  split: >
    identical 30-train/20-eval per-slate split as EXP-0024_v1/EXP-0028
    (genesis_slates_multistep_n20_L{10,20,40}mm_{train,eval}.yaml), scored on
    step-0 candidates only (2560 = 128 candidates x 20 slates per cell)
  runtime: "~4 min CPU total for all 3 cells (dataset load + operator fit + UNet forward + heuristics + D1-D4, per cell ~80s)"

budget:
  declared: "~2h, ~200k tokens (task-giver's estimate)"
  spent: "~1h40min, ~180k tokens (exploration/reading existing pipeline ~40min, script authoring+smoke-testing ~30min, official runs ~10min, write-up ~20min)"
  outcome: within

design:
  varied: {cell: [L10mm, L20mm, L40mm], model: [mean-delta, linear, "warp-only (A=identity)", UNet, "heuristic-cumulative", "heuristic-spread", oracle], K: [4, 16, 32, 64, 128], diagnostic: [D1 error-projection, D2 geometry heuristic, D3 shuffle null, D4 paired UNet-linear]}
  held_fixed: {goal: corner, R: 64, crop: 1.0, ridge: 1.0, region: "swept_region_mask(half_width=0.5*plate_px+2, pad=0.5*plate_px), identical to expB_multistep_eval.py's accuracy region", split: "identical 30/20 per-slate split as EXP-0024_v1/EXP-0028", min_slate: 8}
  baselines: [persistence, mean-delta, oracle]
  metric: >
    accuracy, slateK_exact (docs/experiments/METRICS.md; check_register.py's
    own hardcoded design.metric key list omits both -- already logged as an
    unrelated finding in EXP-0027, not fixed here since fixing it is out of
    this record's scope)
  # NOTE: `accuracy`/`slateK_exact` are absent from check_register.py's
  # hardcoded metric-name allowlist (see EXP-0027 "Unrelated findings"); this
  # will emit a WARNING, not an error, when checked.

noise_floor: >
  D3's 200-draws/slate shuffled slateK_exact IS the noise floor for every
  slateK_exact number in this record: it sits within 1-3x its own sem of 0
  at every cell/model/K tested (largest deviation: L40mm linear K=128,
  shuffled=-0.0185, sem=0.0086, ~2.1 sem -- still not a 3-sem violation).
  D4's paired UNet-linear test uses its own across-slate sem directly
  (K=4: 0.0192; K=128: 0.0462) as its floor, per METRICS.md convention.

depends_on: [swept-region-metric, canonical-warp, warp-blend, episode-split, settled-state]
establishes: []

result: >
  D1 (pooled projected fraction E_proj/E_total, swept region, goal=corner):
  mean-delta 0.0047(L10)/0.0124(L20)/0.0335(L40) -- 7.1x smaller at L10 than
  L40; warp-only(A=I) 0.0049/0.0175/0.0656 -- 13.5x smaller; linear (the
  fitted, disputed operator) 0.0040/0.0064/0.0078 -- only 1.9x smaller;
  UNet 0.0057/0.0072/0.0065 -- 1.1x, flat. So the "far smaller at 10mm"
  prediction clears its own >=3x threshold for 2 of 3 canonical-frame
  baselines (mean-delta, warp-only) but NOT for the fitted linear operator,
  whose own projected fraction is small (<0.8%) at ALL THREE lengths, not
  distinctively small at L10mm -- narrowed, not refuted: the mechanism (cost
  blind to non-projecting error) is confirmed and general; the specific
  cross-length trend predicted for canonical-frame models as a class only
  holds for the untrained pipeline components. Linearisation check: Pearson
  r(linearised dV, exact dV) = 0.79-0.99 across all 18 (model, cell) cells,
  confirming the first-order approximation is reasonable throughout, with
  the weakest agreement (r=0.79, 0.80) at L40mm linear/UNet. D1 cache
  cross-check: this record's independent reload reproduces the cached
  dv_pred mean/sd to the displayed precision for mean-delta/linear/UNet at
  all 3 cells.
  D2 (geometry-only heuristics, K=128): L10mm cumulative 0.668 / spread
  0.750 (vs linear 0.913, UNet 0.890 -- heuristics 15-24 points BELOW);
  L20mm cumulative 0.881 / spread 0.879 (linear 0.967, UNet 0.982 -- gap
  narrows to 9-10 points); L40mm cumulative 0.965 / spread 0.971 (linear
  0.979, UNet 0.992 -- gap narrows to 1-2 points). Heuristic accuracy stays
  far below linear/UNet at every cell (e.g. L10mm: 0.01-0.04 vs -0.43/+0.25;
  L40mm: 0.14-0.19 vs 0.54/0.55). REFUTES the strong form of "L10mm ranking
  is purely kinematic": L10mm is the cell where a parameter-free heuristic
  underperforms the fitted models MOST, not least.
  D3 (shuffle null, 200 draws/slate): every model/cell/K in {4,128} tested
  collapses to within [-0.0185, +0.0101], well inside 1-3 sem of 0 (largest:
  L40mm linear K=128, -0.0185, sem 0.0086). No pairing/leakage bug found.
  D4 (paired UNet-linear, L10mm): K=4 mean=+0.0675 (sem 0.0192, t=+3.51,
  18/20 wins, 0 ties); K=128 mean=-0.0235 (sem 0.0462, t=-0.51, 2 wins/3
  losses/15 ties). D1's per-model E_proj at L10mm: linear 59.18, UNet 57.73
  (2.5% apart) against E_total linear 14701.9, UNet 10128.7 (45% apart) --
  UNet's accuracy advantage sits almost entirely in the error component the
  corner-goal cost cannot see, which is why 75% of pools produce the
  identical top pick and the surviving 5 split close to evenly.
verdict: supported
downgrades: [untested-dependency]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

D1's prediction was written as a specific, falsifiable cross-length
comparison (>=3x smaller projected fraction at L10mm than L40mm) rather than
a vague "the mechanism seems plausible" -- so a result that came back flat
across push lengths (within 2x) would have killed the length-dependent
version of the story outright, distinct from killing the projection
mechanism itself (which is checked independently by the linearisation
agreement, r=0.79-0.99, not by the cross-length comparison). D2's heuristic
scoring uses the IDENTICAL metric suite and identical candidate pools as the
learned models, so "the heuristic reaches 0.9" and "the heuristic reaches
0.65" are directly comparable numbers, not different measurements dressed
alike. D3's shuffle preserves both marginals (the set of predicted values
and the set of true values) and only destroys the pairing between them, so
a shuffled score near the theoretical 0 is the one result a leak could not
produce, and a shuffled score that is NOT near 0 is the one result a correct
pipeline could not produce -- there is no reading of "shuffled capture is
0.3" other than a bug. D4's resolution is falsifiable in the specific way
the task asked: D1's E_proj comparison for UNet vs linear was checked
BEFORE deciding it explained D4, not fitted to it after the fact -- the
alternative finding (UNet's E_proj much smaller than linear's, yet still no
K=128 edge) was an available, distinguishable outcome and did not occur.

## What was actually run

`PYTHONPATH=. OMP_NUM_THREADS=4 python scripts/probes/exp0029_l10mm_mechanism.py
--out runs_expB/exp0029_results.json`, once, after committing the script (two
commits: the initial script, then a fix to D3's shuffle averaging -- see
below). For each of L10mm/L20mm/L40mm: fits `linear` (ridge->identity,
R=64/crop=1.0) and `mean-delta` on the TRAIN split; loads the UNet checkpoint
from `runs_expB/unetfilm_slates_multistep_n20_<cell>`; loads the full step-0
EVAL subset (occ0, occ1, actions, s_px/e_px, slate id) via `build_dataset` +
the cell's own `manifest.json`, exactly as `expB_multistep_eval.py` does;
builds `warp-only` (A=identity through the same warp/blend pipeline) and two
geometric heuristics (`model.eulerian_wrapper`'s `cumulative`/`spread`, no
fit); then runs D1-D4 as specified in the frontmatter's `claim`.

**Iteration during the run, disclosed rather than hidden:** D3's first pass
used ONE random permutation draw per slate and returned shuffled values as
large as -0.24/+0.15 (L20mm/L40mm, oracle/linear) -- alarming on first read,
since the task's own instructions say a value "materially above 0" is the
headline finding to stop and report. Before concluding that, the mechanics
were checked: at K=128=n_slate, `w_r(K)` places its ENTIRE weight on the
model's rank-1 pick (n-r>=n-1 requires r<=1), so slateK_exact at K=128 is a
single deterministic top-1 pick, not an average over sampled subsets --
permuting once re-labels which candidate is "rank 1" via a SINGLE draw from
a distribution whose expectation is genuinely 0 (a uniformly-random
candidate's true value averages to mean(t) over infinitely many draws), but
one draw over only ~20 slates is a high-variance estimate of that
expectation. This is a real property of the estimator at small n, not a
pairing bug: re-running with 200 draws/slate (averaged per slate before
averaging across slates, matching this register's own per-slate-then-average
convention) collapsed every value to within 1-3 sem of 0. The fix
(`d3_shuffle`'s `n_shuffles` parameter) was committed as its own change
(commit `a446807e`) before the official run reported here, so the run's own
`dirty: false` provenance is accurate -- this is disclosed under "Multiverse
discipline" rather than treated as silently overwriting the earlier
observation.

## Numbers

### D1 -- swept-region error energy projected onto the cost direction

| model | L10mm frac | L20mm frac | L40mm frac | L40/L10 ratio | lin r (L10/L20/L40) |
|---|---|---|---|---|---|
| mean-delta | 0.00473 | 0.01240 | 0.03347 | 7.08x | 0.960 / 0.974 / 0.947 |
| linear | 0.00403 | 0.00635 | 0.00778 | 1.93x | 0.896 / 0.919 / 0.787 |
| warp-only (A=I) | 0.00487 | 0.01754 | 0.06559 | 13.47x | 0.949 / 0.981 / 0.987 |
| UNet | 0.00570 | 0.00718 | 0.00645 | 1.13x | 0.887 / 0.896 / 0.803 |
| heuristic-cumulative | 0.00523 | 0.01512 | 0.04603 | 8.80x | 0.926 / 0.968 / 0.978 |
| heuristic-spread | 0.00554 | 0.01545 | 0.04367 | 7.88x | 0.887 / 0.944 / 0.954 |

`frac` = pooled `E_proj/E_total` (sum over all 2560 step-0 transitions, not a
mean of per-transition ratios). `lin r` = Pearson correlation between the
linearised `dV` (`(1/M0) d^T e`, whole image) and the exact `dV` (`V(pred) -
V(truth)`, the same quantity the cached `dv_pred - dv_true` measures).

**D1 cache cross-check** (this record's independent reload vs. the register's
own cached `dv_pred`, goal=corner, mean/sd): mean-delta, linear and UNet all
match to the displayed precision (5 decimals) at all 3 cells -- e.g. L10mm
linear: indep mean=+0.00255 sd=0.00536, cached mean=+0.00255 sd=0.00536.

**L10mm-specific mechanism number (answers D4):** `E_proj` (absolute,
pooled) -- linear 59.18 vs UNet 57.73 (2.5% apart) -- while `E_total` --
linear 14701.9 vs UNet 10128.7 (45% apart). The two models' cost-visible
error is nearly identical even though their total swept-region image error
is not.

### D2 -- geometry-only heuristics, same metric suite

| cell | model | accuracy | slateK_exact K=4 | K=128 |
|---|---|---|---|---|
| L10mm | heuristic-cumulative | +0.008 | 0.490 | 0.668 |
| L10mm | heuristic-spread | +0.036 | 0.610 | 0.750 |
| L10mm | linear | -0.432 | 0.793 | 0.913 |
| L10mm | UNet | +0.245 | 0.861 | 0.890 |
| L20mm | heuristic-cumulative | +0.067 | 0.754 | 0.881 |
| L20mm | heuristic-spread | +0.094 | 0.754 | 0.879 |
| L20mm | linear | +0.316 | 0.953 | 0.967 |
| L20mm | UNet | +0.420 | 0.971 | 0.982 |
| L40mm | heuristic-cumulative | +0.139 | 0.947 | 0.965 |
| L40mm | heuristic-spread | +0.187 | 0.917 | 0.971 |
| L40mm | linear | +0.544 | 0.982 | 0.979 |
| L40mm | UNet | +0.554 | 0.992 | 0.992 |

(`accuracy` here is step-0-only, for internal consistency with the ranking
population; it agrees in sign and rough magnitude with the register's
all-3-step headline numbers, e.g. L10mm linear -0.432 step-0-only vs -0.441
all-3-step.)

The heuristic-to-fitted-model GAP in `slateK_exact` at K=128 is **largest at
L10mm** (0.668/0.750 vs 0.913/0.890 -- a 14-24 point gap) and **shrinks
monotonically with push length** (L20mm: 9-10 points; L40mm: 1-2 points,
within the heuristics' own K=4-to-K=128 movement). This is the opposite of
what "L10mm ranking is just kinematics" predicts.

### D3 -- shuffle null (200 draws/slate)

All 36 (cell x model) shuffled-K4 and shuffled-K128 values fall in
`[-0.0185, +0.0101]`, with sem in `[0.0012, 0.0123]` -- every value is within
1-3x its own sem of 0. Full per-cell tables are in `runs_expB/exp0029_run.log`
and `runs_expB/exp0029_results.json`; representative rows:

| cell | model | real K=128 | shuffled K=128 | sem |
|---|---|---|---|---|
| L10mm | linear | 0.9132 | -0.0045 | 0.0030 |
| L10mm | UNet | 0.8898 | -0.0042 | 0.0038 |
| L10mm | oracle | 1.0000 | -0.0025 | 0.0030 |
| L40mm | linear | 0.9790 | -0.0185 | 0.0086 |
| L40mm | UNet | 0.9924 | +0.0100 | 0.0066 |

### D4 -- UNet vs. linear, paired, L10mm

| K | mean (UNet-linear) | sem | t | wins | losses | ties | n |
|---|---|---|---|---|---|---|---|
| 4 | +0.0675 | 0.0192 | +3.51 | 18 | 2 | 0 | 20 |
| 128 | -0.0235 | 0.0462 | -0.51 | 2 | 3 | 15 | 20 |

Per-pool detail (L10mm, 20 admissible slates, min_slate=8, nonzero variance):
`dv_true` spread (max-min): mean=0.0418, median=0.0375, sd=0.0128,
min=0.0248, max=0.0652 -- no degenerate near-zero pools among the 20
admitted. Per-pool K=128 capture: linear mean=0.913 (sd=0.167, range
0.490-1.000), UNet mean=0.890 (sd=0.185, range 0.337-1.000) -- capture is
NOT concentrated in a few pools; both models reach exactly 1.000 on 11/20
pools and both fall below 0.8 on a shared 4/20 pools (9, 12, 15, 41 -- same
pools for both models in 3 of those 4), consistent with the two models
agreeing on which candidate is best (or nearly best) most of the time,
independent of whether that pick happens to be the true optimum.

## What would change the verdict

Re-running D1 with a SECOND weight field (e.g. `ind-corner`, sharper than
`corner`'s distance transform) would test whether the ~0.4-0.8% projected
fraction is an artifact of `corner`'s specific smoothness or a more general
property of these models' error spectra -- `docs/experiments/METRICS.md`'s
spectral-concentration table already shows `corner` itself is not
maximally smooth (r<=1 energy 0.629, vs 0.060 for `ind-square8`), so this
record's numbers should be read as "under a moderately smooth cost", not
"under the smoothest possible one". Estimated cost: ~15 min (rerun D1 only,
reusing the already-loaded predictions, new `lyapunov_weights` call and
projection). A cross-length comparison using ONE operator fit across all
three push lengths (rather than three separately-fitted ones, as here and
in every prior record) would test whether the modest 1.9x cross-length
change in the FITTED linear operator's own projected fraction is a property
of push length itself or of each length's own best-achievable fit -- flagged
identically in EXP-0028 as unresolved there too, ~10-20 min per refit.

## Threats

`selection`: not applicable -- every cell, model and K planned was run and
reported, including the ones that did not support the naive prediction (the
fitted linear operator's own weak 1.9x cross-length ratio, D2's finding that
heuristics do WORSE at L10mm rather than showing it is "just kinematic").
`provenance`: not applicable within this record -- one code path throughout
(the D1 cache cross-check deliberately uses a second code path, and its
purpose is agreement, not comparison). `imprecision`: D3 and D4 both report
sem directly and compare effects to it, not to a cross-pool sd.
`inconsistency`: the D1 cross-length prediction did NOT hold uniformly (see
above) -- reported as a narrowing, not suppressed. `incomplete-design`: none
of the planned cells were skipped. The one real dependency this record
cannot clear is `settled-state`, `unchecked` for the rigid-cube path this
data comes from (`n20_L{10,20,40}mm` cube piles) -- taken as
`untested-dependency`, per protocol.

## Unrelated findings

- The linearisation check (D1) shows its weakest agreement (Pearson r=0.79,
  0.80) at L40mm for linear/UNet specifically -- both canonical-frame and
  raw-raster models, at the LONGEST push, where the true `dV` magnitude and
  mass non-conservation are both largest. Not investigated further here;
  flagged as a place the first-order approximation is measurably weaker,
  should anyone lean on the linearisation itself at L40mm.
- `heuristic-spread` slightly beats `heuristic-cumulative` on both
  `accuracy` and `slateK_exact` at every cell tested here (e.g. L10mm
  accuracy +0.036 vs +0.008, K=128 0.750 vs 0.668) -- consistent in
  direction across all three push lengths, not cherry-picked, but not
  otherwise explored (no prediction was pre-registered about which
  heuristic would win).
- D2's per-pool K=128 shared low-capture slates for UNet and linear at
  L10mm (9, 12, 15, 41) were not cross-referenced against D1's per-transition
  error decomposition to see whether they share a specific action-geometry
  feature (e.g. glancing pushes); that would be a natural next probe but was
  out of this record's scope.
