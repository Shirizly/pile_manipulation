# Descriptor value-readout: goal-as-configuration follow-up

Follow-up to `experiments/temp/desc-value-regression/RESULTS.md` (solid-mask
goal encoding, catastrophic held-out-goal R²). Corpus:
`Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`, step 0. All ridge
fits: standardised features, λ chosen by 3-fold CV, mandatory
mean/state/goal baselines.

## Stage 0 — upper bound: is value recoverable from the TRUE state descriptor, goal fixed?

Script: `code/stage0_upper_bound.py`. 4 fixed goals (quadrant0/1, letter_T/O)
× 3 value functions × n_fourier ∈ {8,16,24,32}, held-out-state (3 repeats).

**Answer: it depends heavily on the goal shape and value function, and gets
WORSE, not better, as Fourier order increases** — this refutes the prior
stated in the brief.

- `lyapunov` is recoverable for smooth, large-area goals: R²=0.90±0.07
  (quadrant0) and 0.97±0.01 (quadrant1) at n_fourier=8. For the thin
  `letter_O` it is already negative (-1.47) at n_fourier=8.
- `mass_in_region`/`signed_mass_in_region` are NOT recoverable at n_fourier=8
  for any of the 4 fixed goals: R² ranges +0.31 (quadrant0, still noisy,
  ±0.38) down to -8.9 (letter_O). This matches the earlier run's per-goal
  finding.
- **Increasing n_fourier makes every cell worse, often catastrophically**
  (e.g. quadrant0/lyapunov: 0.90 → 0.12 → -0.76 → -0.72 as nf goes 8→16→24→32;
  letter_T/mass_in_region: -6.6 → -73.6 → -78.9 → -1.1). Full table in
  `stage0_results.json`.

Mechanism not fully isolated (untested hypothesis, flagged as such): CV folds
are random row splits within the *training* slates, not slate-aware, so with
D approaching or exceeding the number of independent slates in the training
split, CV likely underestimates the regularisation a truly-held-out slate
needs — high-order DFT bins are the most sample-hungry directions, so they
are hit hardest. This was not confirmed by inspecting chosen λ values (not
logged); report as a plausible but unverified cause.

**Reading for later stages:** value is NOT uniformly unrecoverable from the
true descriptor (the "stop, capacity is pointless" case did not occur) — but
neither is it a clean win, and it is goal-shape-dependent. n_fourier=8 (the
original choice) is at least as good as any higher order tested, so stages
2-4 correctly kept it fixed.

## Stage 1 — goal mask → legal configuration generator

New module: **`Baselines/common/goal_configs.py`** (sibling to
`Baselines/common/goals.py`, which owns masks/value functions — ownership
kept separate since this module's job, packing physical objects, is a
different concern). `docs/CODEMAP.md` updated.

Method: grid-then-jitter with an exact non-penetration bound (`p >=
a*sqrt(2)` given cube edge `a`, footprint half-extent `a*sqrt(2)/2` at ANY
yaw), bisected to land the interior-point count in `[n, 1.2n]`, jittered by
`(p-a*sqrt(2))/2`. Falls back to rejection sampling, then to morphologically
dilating the *placement* mask (not the mask used for the value target), for
masks too thin at 20 objects — this was needed for `letter_T` (216 px raw
area, packing-infeasible at 20×5mm cubes without 1-2 dilation rounds).

**No-penetration assertion: passed on every generated configuration** — 60
configs in the stage-1 check, 1500 in the stage-3 500-goal dataset, all
verified via `assert_no_penetration` (conservative axis-aligned footprint
test), zero failures.

**Cross-check (the actual point of the exercise):** generated-config pixel
mass mean=96.0±5.4 vs. real corpus states mean=94.6±2.1 (500 real states) —
close and same order, confirming generated configurations rasterise
comparably to real states through `particles_to_occupancy`.

## Stage 2 — value regression, 6 fixed goals, config-based `phi(goal)`

`code/stage2_config_goal_regression.py`. Same 6 goals as the original run
(quadrant0-3, letter_T/O), `phi(goal)` = descriptor of the mean occupancy
over K=20 independent configuration samples. Stability: within-sample /
between-goal L2 ratio 0.09-0.14 — `phi(goal)` is reasonably stable, goal
identity dominates sampling noise.

| value fn | split | old (solid mask) diff R² | **new (config) diff R²** |
|---|---|---|---|
| lyapunov | held-out-state | 0.054±0.052 | 0.039±0.170 (same, noisier) |
| lyapunov | held-out-GOAL | **-189.9±263** | **-1.387±2.110** |
| mass_in_region | held-out-state | 0.428±0.017 | 0.417±0.021 (same) |
| mass_in_region | held-out-GOAL | **-5.48±3.49** | **-1.733±1.520** |
| signed_mass_in_region | held-out-state | 0.300±0.052 | **0.425±0.018** (better) |
| signed_mass_in_region | held-out-GOAL | **-366±645** | **-1.549±1.049** |

The configuration-based encoding cuts the held-out-goal catastrophe by 2-3
orders of magnitude — supports the user's subspace-mismatch hypothesis — but
does **not** fix it: all three held-out-goal R² remain negative. With only 6
goals this split is still nearly untestable (large std), which motivated
stage 3.

## Stage 3 — 500 goals

`code/stage3a_generate_goals.py` → `experiments/temp/goal-states/dataset.pt`
(gitignored): 300 rectangles (area = 15% + 35%·Beta(2,5) of workspace, biased
small, random rotation) + 200 letters (T/A/O/S/G × 40, random rotation via
`scipy.ndimage.rotate` since `letter_mask` itself does not rotate) = 500
masks, each with K=3 legal configurations. Generation: 5.5s, all 1500
configs passed non-penetration.

`code/stage3b_regression.py`, S=1000 states, G=500 goals, held-out-goal now a
genuine random 400/100 split (3 repeats):

| value fn | split | mean | state_only | goal_only | **diff** |
|---|---|---|---|---|---|
| lyapunov | state-split | -0.037 | 0.119 | 0.307 | 0.094±0.125 |
| lyapunov | **held-out-GOAL** | -0.014 | 0.116 | 0.311±0.069 | **0.406±0.056** |
| mass_in_region | state-split | -0.007 | -0.013 | 0.381 | -0.122±0.260 |
| mass_in_region | **held-out-GOAL** | -0.009 | 0.005 | 0.346±0.066 | **0.271±0.052** |
| signed_mass_in_region | state-split | -0.002 | 0.003 | 0.381 | -0.163±0.250 |
| signed_mass_in_region | **held-out-GOAL** | -0.009 | 0.018 | 0.342±0.066 | **0.265±0.050** |

**Held-out-GOAL is finally clearly positive and stable** (std 0.05-0.07, not
hundreds) once there are enough goals to make the split meaningful — the
qualitative opposite of stage 2's near-untestable result. `diff` beats
`goal_only` for `lyapunov` (0.406 vs 0.311); for the two mass-based value
functions `diff` is close to but slightly *below* `goal_only` (0.271 vs
0.346; 0.265 vs 0.342) — goal identity still carries most of the signal, an
echo of the original run's "quieter embarrassment," now at least measured
with a real sample size instead of 6 folds.

**A genuine regression, reported plainly:** at state-split (500 goals held
fixed), `diff` is *negative* for both mass-based value functions (-0.122,
-0.163) despite scoring +0.42 in stage 2 with only 6 goals. With 500 diverse
goal shapes in the same training pool, a single linear map on `diff` no
longer separates state-driven variation well — it is dragged around by
however the fixed goal set's `phi(goal)` directions happen to sit. Not fully
diagnosed; flagged as unresolved rather than as a ranking claim.

## Stage 4 — capacity escalation (held-out-GOAL only, single 400/100 split, 60k-row subsample)

`code/stage4_capacity.py`:

| value fn | linear ridge | degree-2 poly ridge | MLP(128,128) |
|---|---|---|---|
| lyapunov | 0.362 | 0.751 | **0.893** |
| mass_in_region | 0.213 | 0.499 | **0.689** |
| signed_mass_in_region | 0.205 | 0.482 | **0.749** |

Monotonic, substantial gains at every step, for all three value functions.
Stage 0 did **not** show an information ceiling near zero (results were
goal-shape-dependent, several cells well above zero), so escalating was
justified and paid off — capacity was the bottleneck for the *goal-varying*
task, not descriptor information content per se. This is a single split (not
stage 3's 3-repeat CV), so treat the exact numbers as indicative; the
ranking (MLP > poly2 > linear) is large relative to stage-3's ~0.05-0.07
goal-split std and unlikely to be noise.

## Stage 5 — decoder question: **recommendation only, not run (budget)**

Not attempted — stages 0-4 consumed the wall-clock budget. Recommendation,
explicitly not a measurement: train the descriptor→occupancy decoder anyway.
Reasoning: (a) stage 4 shows the *readout* (linear→MLP) is a real lever,
which argues for comparing a decoder against an MLP-level direct regressor,
not just the linear baseline this experiment mostly used; (b) a decoder is
fit once and reused by every value function unchanged, which the per-value-
function MLPs here are not; (c) stage 1's cross-check (generated-config pixel
mass matching real states) shows synthetic training rasters are already
validated as realistic, so decoder training data is cheap and trustworthy.
Recommend comparing decoder-then-value-function against stage 4's MLP
specifically (not the linear ridge) since the MLP is now the standing bar.

## Files

- `code/stage0_upper_bound.py` → `stage0_results.json`
- `code/stage1_generator_check.py` (uses `Baselines/common/goal_configs.py`)
- `code/stage2_config_goal_regression.py` → `stage2_results.json`,
  `stage2_features_and_targets.pt`
- `code/stage3a_generate_goals.py` → `../goal-states/dataset.pt`
- `code/stage3b_regression.py` → `stage3_results.json`,
  `stage3_features_and_targets.pt`
- `code/stage4_capacity.py` → `stage4_results.json`
- `code/persist_final_models.py` → `fitted_models.pkl` (stage2 ×3, stage3 ×3
  ridge models, keyed by `(stage, value_fn, feature_mode)`, each with
  `mu,sd,w,b0,lam,feature_mode,n_fourier,goal_source,provenance`),
  `stage4_linear_mass_in_region.joblib`, `stage4_poly2_mass_in_region.joblib`
  (stage-4 MLP intentionally not persisted — budget; its R² is recorded
  above and in `stage4_results.json` only).
