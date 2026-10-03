# Metrics — the `design.metric` key registry

An experiment record's `design.metric` field names a key from this file, not
a prose description. Two standard metrics are reported in every record where
applicable: **`accuracy`** (image prediction) and **`slate4`** (control). Add
any other metric here, with its formula, before citing its key in a record.

## Reset 2026-09-10

The prior metric glossary — built up across the archived work program, with
formulas, denominators, and known ambiguities documented per-metric — was
archived to `archive/2026-09-10_pre-reset/docs/experiments/METRICS.md`. It is
a useful reference for how a metric was defined *there*, but a metric's exact
definition (mean of ratios vs ratio of means, what the denominator is, how it
behaves under preprocessing) must be re-stated here before it's reused, not
assumed to carry over.

## Metrics

## THE TWO STANDARD METRICS

**Report these in every record.**  Where the metric
itself is the object of study (FSS, R², a profile), report these as a reference
row anyway, so results stay comparable across the register.

| key | definition | 0 means | 1 means |
|---|---|---|---|
| **`accuracy`** | `1 − rms(model) / rms(persistence)`, swept region, same norm top and bottom | no better than predicting nothing moved | perfect |
| **`slateN`** | fraction of the oracle's advantage over a random pick that the model captures, choosing from **every candidate the pool holds** (N distinct actions, no replacement) | no better than random | as good as the oracle |

Negative values are meaningful in both: worse than doing nothing, and worse than
choosing at random, respectively.

### `slateN` is the standard, not `slate4` (changed 2026-09-08)

An MPC step chooses among **every** candidate it generated — hundreds or
thousands — not among four of them. So the standard control metric is `slateN`:
the capture fraction at K = the pool's full size, with candidates drawn
**without replacement** wherever any sampling is involved. `slate4` and
`slate16` are retained as legacy keys so pre-2026-09-08 records stay checkable,
and `control_utility_test.rank_metrics` still reproduces them exactly under
`replace=True`; they are not to be quoted as the headline any more.

**Three things `slateN` must be reported with, because it has two real
weaknesses and hiding them would trade one bad default for another:**

1. **Ties, and the effective sample size.** At K = N the metric is
   deterministic per pool, and two decent models very often pick the *same*
   action: measured tie rates are 15/20 pools (10 mm), 9/20 (20 mm), 10/20
   (40 mm), 16/50 (old slates). Effective n collapses to 5–11 pools and a
   paired t that reads 5.5 at K=4 reads 0.75 at K=N. **`slateN` is the most
   faithful point on the curve and the least statistically powerful one**, so
   always report wins / losses / **ties** and the paired sem beside it. A
   "no difference at `slateN`" is usually a statement about power.
2. **It is not comparable across pools of different size.** N is 32 on
   `slates/n20_heap_5mm` and 128 on the `slates_multistep` cells, so "the
   largest available" is a different quantity per dataset. For any
   cross-scenario comparison — which is what a model/scenario map is — also
   report a **fixed-K reference point common to the datasets being compared**
   (K=32 for everything collected so far).
3. **Capture fractions are not comparable across scenarios at all**, whatever
   K. EXP-0028 measured absolute regret at 0.00086 / 0.00096 / 0.00162
   (10/20/40 mm) while the available spread differed by up to 17.9x, so
   between-cell differences in capture are mostly denominator scale. Report
   **`regret_dv` in Lyapunov units** beside `slateN` whenever cells are set
   side by side.

## `slateN` generalised to an arbitrary VALUE function (added 2026-09-10)

`slateN`'s own formula (`(mean(t) − t_chosen) / (mean(t) − t_best)`, this
file's own definition above and `control_utility_test.py`'s closed-form
`slateK_exact` at K=n) only assumes `t` is a per-candidate scalar the model
can rank — it does not have to be Lyapunov `dV`. The cross-corpus GNN/NFD
report (`Baselines/common/eval_report.py`) uses three interchangeable
scalar VALUE functions, all computed on the post-push occupancy grid
(using the raw value V, not V1−V0, is equivalent here: V0 is identical
across every candidate in a same-state pool, so it cancels in every term
of the formula above — confirmed algebraically, not just assumed):

| value function | formula | sense |
|---|---|---|
| **`lyapunov`** | `control_utility_test.lyapunov(occ, d)` — mass-normalised mean distance to a target region, `d` a normalised distance-transform field | COST (lower is better) |
| **`mass_in_region`** | `sum(occ * m)` — raw, unnormalised pile mass inside a binary target mask `m` | VALUE (higher is better) |
| **`signed_mass_in_region`** | `sum(occ * (2m − 1))` — like the above, but mass OUTSIDE the mask subtracts instead of contributing zero | VALUE (higher is better) |

`Baselines/common/goals.py::slate_n_capture(value_pred, value_true,
higher_is_better)` implements the formula once, generically, taking a
`higher_is_better` flag to uniformly express a COST value function's
"pick the model's best" as an argmin instead of an argmax.

**Three goal SHAPES** (also `Baselines/common/goals.py`), each a binary
(H,W) mask feeding any of the three value functions above:

  * **`random_quadrant`** — one of the workspace's 4 quadrants, chosen by
    a rng SEEDED PER SLATE (`seed = slate index`), so the same quadrant is
    assigned to a given slate regardless of which model is being scored
    (never re-randomised per model/per call — that would make cross-model
    comparison on "the same slate" meaningless).
  * **`ring_O`** — the letter 'O' rendered as a font glyph
    (`utils.py::gen_goal_shape`'s own asset,
    `env/target_shapes/helvetica_thin/helvetica_O.npy`, binarised at the
    same `<= 0.5` threshold that function uses internally), a genuine
    ring/annulus (the glyph's hollow centre is not part of the mask).
    Centred, fixed for every slate in a corpus.
  * **`T`** — same, for the letter 'T'. Centred, fixed for every slate.

Reported per (goal, value function), AND averaged over the 3 goals per
value function (never averaged across value functions — a COST and two
VALUE conventions do not share units, and `signed_mass_in_region`'s scale
also differs from `mass_in_region`'s).

**Known ties/degeneracy caveats above (K=N determinism, effective sample
size, cross-scenario incomparability) apply identically here** — they are
properties of the K=N regime and the cross-corpus comparison, not of
Lyapunov `dV` specifically.

**Follow-up idea, not implemented (recorded per 2026-09-10 direction):**
`mass_in_region`/`signed_mass_in_region` respect fine pixel-level detail
(a single misplaced cube changes them) but are not smooth in a way a
gradient-based MPC controller could descend; `lyapunov`'s distance field
is smooth but blurs fine detail. Some combination of the two — sharp where
it matters, differentiable everywhere — might get both properties at
once. Not attempted here.

## `descriptor_accuracy` (added 2026-09-13, transferred from EXP-0004)

Used when the model's prediction target is a vector of analytic push-frame
descriptors (`phi`) rather than an image, so raw-pixel `accuracy` does not
apply directly. Let `phi_t1` be the true next-step descriptor vector for a
sample (the `const` block -- a degenerate always-constant descriptor
sub-block -- dropped before scoring), and `mu`, `sigma` the TRAIN-set
per-dimension mean/std of `phi_t1` (fit once, reused for every split).
Z-score every vector with these fixed train-set stats:

```
z(phi) = (phi - mu) / sigma          # per-dimension, using TRAIN mu/sigma always
```

Then, pooling across all retained dimensions and all samples (rms computed
over the flattened pooled residual, not per-dim then averaged):

```
descriptor_accuracy = 1 - rms(z(phi_pred) - z(phi_true)) / rms(z(phi_persist) - z(phi_true))
```

where `phi_persist = phi_t0` (the persistence/do-nothing prediction, i.e.
"nothing changed"). 0 means no better than persistence in normalised space;
1 means perfect; negative means worse than persistence. This is an explicit
proxy for `accuracy` (it never reconstructs occupancy), not a substitute for
it -- see EXP-0004's `indirectness` downgrade.

## Kill-probe latent loss ratio (added 2026-09-13, transferred from EXP-0007)

Used for a FiLM-conditioned residual latent predictor `P(z, a)` scored
against the mandatory `dz=0` ("predict no latent change") baseline, per the
design doc's Stage-2 loss:

```
L_latent(model)    = mean_i || P(z_i, a_i) + z_i - E(T_{a_i}(X_i)) ||^2
L_latent(baseline) = mean_i || z_i          - E(T_{a_i}(X_i)) ||^2     # dz=0
ratio = L_latent(model) / L_latent(baseline)
```

Lower is better; `ratio < 1` means the predictor beats predicting no
latent change at all (e.g. EXP-0007's ~0.093 train / ~0.095 holdout, a
~10x improvement over `dz=0`). Report `z_std` (per-channel latent std)
alongside this ratio -- a collapsed encoder can trivially drive both
`L_latent(model)` and `L_latent(baseline)` toward 0 together (making the
ratio meaningless), which is exactly what EXP-0007 found training the loss
as literally specified, with no regulariser.

**`mass_in_region`/`signed_mass_in_region` already exist above (2026-09-10
section) and are reused verbatim here and in EXP-0004/EXP-0008 -- not
redefined.**

## GNN accuracy: node-count-bottlenecked comparison (added 2026-09-10)

A particle/node-based model (the GNN baseline) predicts only
`n_particles` positions, decoupled from a cell's true cube count (see
`Baselines/GNN/SPEC.md`'s "CORRECTION" section) — comparing its rasterised
prediction directly against a full-fidelity many-cube ground truth would
conflate "wrong dynamics" with "fewer nodes than cubes exist", two
different failure modes. `Baselines/common/eval_report.py` instead passes
BOTH sides of a GNN's `accuracy` comparison through the identical
node-count bottleneck: the ground truth occupancy (both the pre-push
`occ0`, used as `metrics()`'s persistence baseline, and the post-push
`occ1`, the comparison target) is itself FPS-resampled to `n_particles`
nodes and rasterised back
(`Baselines/GNN/perception.py::resample_occupancy_through_nodes`), using
the exact same `sample_nodes_xy` + fixed-size-cube rasteriser the model's
own prediction goes through. A grid-native model (NFD) has no such
bottleneck and is compared directly against raw ground truth. This makes
`accuracy` measure each model's own dynamics-prediction quality relative
to the ceiling ITS OWN representation could achieve, rather than
penalising a node-based model for a representational choice unrelated to
whether it predicts motion correctly.

## `candidate_throughput_ms` (added 2026-09-13, from EXP-0009)

**`candidate_throughput_ms`** — see below.

The wall-clock cost (median, ms) of a model's `predict_occ`-equivalent call
on ONE current state repeated K times, paired with K real candidate actions
cycled from an eval cell (the "same state, many candidate actions" shape an
MPC inner loop repeats every step). Two variants, both required together
whenever this key is cited:

  - `candidate_throughput_ms.end2end` -- preprocessing (all per-candidate,
    action-dependent work: e.g. `draw_plate_soft`, push-frame warp/
    `canonicalise`, push-frame descriptor computation, image->particle
    conversion) PLUS the forward call, back to back, in one timed region.
  - `candidate_throughput_ms.fwd_only` -- the forward call alone, timed
    separately on the SAME preprocessed input (isolates preprocessing cost
    as its own number rather than folding it silently into "the model").

Excludes the one-time particle->occupancy rasterisation of the CURRENT
STATE (shared across every candidate; in a real deployment the state
arrives as an image from perception, not re-derived per candidate).

Methodology (mandatory, not optional, when reporting this key): >=3
untimed warm-up calls discarded, >=10 timed repeats, MEDIAN + IQR (not
mean), `torch.cuda.synchronize()` immediately before AND after every timed
region on GPU, and the actual device of the tensors fed to the forward
call recorded per model (introspected, not assumed from a `--device` flag
-- see `Baselines/common/benchmark_time.py`'s module docstring for the
device-consistency trap this guards against). See
`Baselines/common/benchmark_time.py::time_predictor_at_k` /
`experiments/EXP-0009-time-budget-bench/code/bench.py::time_model_at_k`
for reference implementations.

Derived quantity, reported alongside: `N_i`, the number of candidates
model `i` can evaluate (`end2end`) in the time the FASTEST model in the
comparison takes to evaluate its own reference batch size (linear
interpolation on the measured (K, end2end_ms) curve) -- this is the actual
quantity an equalised-wall-clock-budget control comparison needs.

## `top1_regret` (added 2026-09-15, from EXP-0014)

**`top1_regret`** — per slate, the realised value cost of the action the model actually picked,
relative to the pool's best:

    top1_regret = value_true[argmin_c value_pred[c]] - min_c value_true[c]

in the value function's own units (Lyapunov units for `lyapunov`), `>= 0`,
0 = picked the oracle's action. Reported as a mean +/- sem over slates.
Implemented in `scripts/probes/pool_survey.py`.

**Lower is better — the opposite sense to `slateN`/`accuracy`.** It is NOT a
substitute for `slateN` and must not be reported alone: it is unnormalised,
so it is incomparable across corpora and across value functions, and it reads
only the top-1 pick while `slateN` reads the model's whole ordering. Its use
is that it stays in interpretable physical units, where `slateN`'s ratio can
hide whether a large captured fraction is a large or trivial absolute gain.

Companion quantity, same script: `|R_K| = |M_K - P_K|`, the K-weighted
expectation of the same regret over random K-candidate subsets, using the
`M_K`/`P_K` defined in this file's `slateK_exact` entry (implemented in
`scripts/probes/pool_common.py::rk_curve`, importing `slateK_exact`'s own
validated `w_r(K)` weights rather than reimplementing them).

## `value_readout_r2` (added 2026-09-15, from EXP-0015)

**`value_readout_r2`** — coefficient of determination of a regression predicting a scalar value
function `value(state, goal)` from features of an analytic descriptor of the
state and of the goal:

    value_readout_r2 = 1 - SS_res / SS_tot

on a held-out split, where `SS_tot` uses the TRAINING-split mean, so the
baseline "predict the training mean" scores ~0 rather than being flattened to
0 by construction. Higher is better; negative means worse than that baseline.

**Two splits, always reported as a pair — they answer different questions:**

  * **held-out state** — unseen states, goals seen in training. Answers "does
    this readout order states under a goal it knows?"
  * **held-out goal** — entire goal shapes withheld. Answers "does it
    generalise to a new target?" This is the one that matters for control,
    and the two can diverge by orders of magnitude.

**Mandatory baselines** (without them the number is uninterpretable, see
EXP-0015): `mean_only`; `state_only` (features from the state alone — if it
matches the full model, the goal term is doing nothing); and `goal_only`
(ignores the state and predicts one constant per goal — if it matches the
full model, the apparent skill is between-goal variance, not state ordering).

**Bounded-metric caveat** (see this file's header guidance): R² saturates at
1, so differences compress near the ceiling. Report the baselines beside it,
not a raw difference against them.

## `repeat_noise_ratio` (added 2026-09-23, from EXP-0024)

**`repeat_noise_ratio`** — characterises the noise floor a resimulated `dv`
ground truth would carry, relative to the signal `slateN` actually ranks on.
For one settled start state and a fixed goal/value function:

    within_var  = mean over actions of [ Var(dv) over R repeats of that SAME
                  action from the SAME state ]
    between_var = Var over actions of [ mean dv per action ]
    repeat_noise_ratio = within_var / between_var

`between_var` is exactly the quantity `slateN` ranks candidates on (dv spread
across distinct actions from one state); `within_var` is the resimulation
noise that would corrupt it if the "true" dv used for scoring were itself a
single noisy draw. Ratio near 0: resimulating the same action would return
(near-)identical dv, so slateN's ground truth is not noisy by this
mechanism. Ratio approaching or exceeding 1: repeat noise is comparable to
or swamps the between-action signal, and slateN has a real, measurable
ceiling from this source alone. Requires R >= 3 repeats per action to be
non-degenerate; reported per state, not pooled, since `between_var` differs
by orders of magnitude state to state (a flat state has little to rank).

## Ground-truth scoring (changed 2026-09-24, EXP-0027 / EXP-0028)

**Every true `dv` is now computed from particle states drawn with a
mass-conserving soft splat, not from a hard-footprint occupancy image.**
Model predictions are unchanged (scored as the images the models output).

*Why.* The hard rasterisers give one particle 4, 5 or 6 pixels depending on
its sub-pixel position (`simple_mpc.adapters.occ_from_particles`: 1.25 px
disk; `PileSweepData._draw_particle_grid`: `int()`-truncated `cv2` boxes),
and `lyapunov` divides by the total occupied pixels. On 83 repeats of one push
whose particles landed within ~1 mm of each other, image-based `dv` spread
0.019 (sd 4.0e-3, ~12% of the 3.2e-2 between-action sd on that state), while
the same distance field sampled at particle centres spread 0.0015 (sd 3e-4).
Fixing only the divisor (a fixed per-state normaliser) left sd 2.5e-3; the soft
splat gives 2e-4 with an unchanged mean (-0.0485 vs -0.0487). Invariant
`occupancy-dv-subpixel-stable` (broken) records the defect;
`score-occupancy-subpixel-stable` (holds) the fix.

*What.* `transforms/functional.py::splat_particles_mass` -- a separable
Gaussian per particle (sigma `SCORE_SPLAT_SIGMA_PX` = 1.0 px), each normalised
to exactly the same mass wherever it falls, overlaps added. Two entry points:
- `simple_mpc.adapters.occ_for_scoring(states)` -- the 64x64 +/-64 mm slate
  grid (DS-0001, gradient benchmarks, the ground-truth bank); same grid and
  axis convention as `occ_from_particles`.
- `Baselines/common/eval_report.py::truth_for_scoring(cell, rows)` -- the
  corpus-image frame, redrawn from each row's `states_`
  (u = pos * to_pxl + ctr - 1.0; the -1.0 is measured: it centres the soft
  truth on the dataset's own image to within 0.2 px on L40mm and randlen_test).

*Harness.* `eval_report.py --truth-scoring soft` is the default;
`--truth-scoring image` reproduces every slateN reported before 2026-09-24,
and each output records `capture.truth_scoring`. **Numbers from the two
settings are not the same quantity; compare within one setting.** The effect
on the reference table is small (EXP-0028: lyapunov rank agreement tau
0.91-0.97, mean shifts ~0.01, paired model-vs-model noise unchanged) -- the
harness's slate-to-slate noise is model x state / pool variation, not truth
aliasing. The fix matters most where true outcomes are compared directly
(repeats, gradient benchmarks, closed-loop progress).

*Not changed.* `mass_in_region` / `signed_mass_in_region` are computed with the
same functions on whichever image they are given, so under soft truth they
also stop aliasing. `accuracy` (image prediction) still compares against the
dataset's hard `occ1` -- it is an image-reconstruction score and the models
were trained on that representation.

## `gradient_gain` and friends (added 2026-09-23, from EXP-0023)

Every metric above scores a model as a **ranker** of a fixed candidate pool.
These score it as a **source of gradients** — an objective an MPC controller
optimises against, which is how a dynamics model is actually used and a
different ability, because optimisation actively seeks out the model's own
errors. Design rationale:
`experiments/EXP-0023-model-as-gradient-source/DESIGN.md`.

All four are defined on the **true, simulated** `dv` of three actions taken
from the same state and the same seed pool: `a_rank` (the model's best pick
in the pool), `a_grad` (gradient descent through the model, started at
`a_rank`), `a_oracle` (CEM with the simulator itself as the model).

| key | definition | sense |
|---|---|---|
| **`gradient_gain`** | `dv_rank − dv_grad` | > 0: optimising against this model beat just picking from the pool. **< 0 is the informative outcome** — a model that ranks well can still be a bad thing to optimise against. |
| **`pool_escape`** | `pool_ceiling − dv_grad`, `pool_ceiling = min_c dv_true[c]` over the seed pool | > 0: optimisation found something better than ANYTHING the pool contained. Separates "a wider pool would do" from "gradients add real value". |
| `capture_vs_oracle` | `dv_grad / dv_oracle` | 1 = matched the oracle. Unstable near `dv_oracle ≈ 0`; report the raw `dv` values beside it and flag degenerate states. |
| `regret_vs_oracle` | `dv_grad − dv_oracle` | absolute headroom left, in the value function's own units. |

**SIGN — read this before using `dv` with any value function other than
`lyapunov`.** `dv = value(after) − value(before)` **does NOT have a fixed
direction across this project's value functions**, because the value functions
themselves do not (see the table above):

| value function | sense | an IMPROVING push gives |
|---|---|---|
| `lyapunov` | COST (lower better) | `dv < 0` |
| `mass_in_region` | VALUE (higher better) | `dv > 0` |
| `signed_mass_in_region` | VALUE (higher better) | `dv > 0` |

So "`dv` is a cost" is true **only for `lyapunov`**, and a reader who carries
that rule over to a mass value function will invert their conclusion. An
earlier version of this note asserted the cost sense "repo-wide"; that was
wrong, and it was wrong in a way no measurement caught, because every place
that currently computes a raw `dv` happens to use `lyapunov` alone
(`scripts/probes/binned_pool_cache.py` records `"value_fn": "lyapunov"` in its
own metadata; EXP-0023 ran `corner`/`lyapunov` only).

**What IS safe to rely on:**
- `slateN`/capture is normalised and always reads **higher = better**
  regardless of the underlying value function —
  `Baselines/common/goals.py::slate_n_capture` takes `higher_is_better` and
  switches argmax/argmin accordingly.
- The three difference metrics above (`gradient_gain`, `pool_escape`,
  `regret_vs_oracle`) read **positive = better**. As first written (EXP-0023's
  `stage3_analyse.py`) their subtraction order assumed the COST sense; use
  `goals.gradient_benchmark_metrics(..., higher_is_better=...)`, which flips it
  exactly as `slate_n_capture` flips argmax to argmin.
- `simple_mpc.adapters.assert_dv_convention` checks the adapter's declared
  `value_fn` / `higher_is_better` against the registry in `goals.py` AND
  against goal geometry (moving mass onto the goal must read as an
  improvement), so it now covers mass value functions as well.

EXP-0023's `DESIGN.md` originally stated the difference metrics with the
opposite subtraction order under an implicit reward convention; that
discrepancy is documented and corrected in both that design doc and its
record.

**Report per state, never only as a mean.** These are per-optimisation
quantities on a handful of states, not pool statistics.

**The power check is PAIRED (changed 2026-09-23, EXP-0026).** The earlier rule
here -- "between-arm sd of means must exceed between-state sd of means" -- is
not a power check: the between-state sd is a property of the state population
and does not shrink with more states, and it is not the noise an arm-vs-arm
comparison sees (every arm runs on the same states, so state difficulty
cancels). Use `Baselines/common/paired_stats.py` instead: per-pair differences
per state, bootstrap CI over states, sign-flip permutation p with Holm over
pairs, Friedman / two-way ANOVA for the global model effect, and
`required_n(sd_diff, delta)` for sizing. Note the floor: an exact sign-flip test
on S states cannot go below p = 2/2^S, so with S = 10 and 15 Holm-corrected
pairs NO pair can reach 0.05 whatever the data -- size S from `required_n`
first.

**`pool_escape` carries no ranking information beyond `dv_grad`.** Its
`pool_ceiling` term is constant per state, so it cancels in every arm-vs-arm
difference (EXP-0026: identical pairwise statistics). Keep it for its absolute
reading ("did GD beat the pool?"), not for ordering arms. `gradient_gain` is the
difference of two per-arm quantities and is the NOISIEST of the four; for
ranking models as MPC objectives, lead with `dv_grad` (EXP-0026).

**Sense-safe implementation:** `Baselines/common/goals.py::gradient_benchmark_metrics`
computes all four with a REQUIRED `higher_is_better` keyword (look it up with
`higher_is_better_for(value_fn)`), so they are correct for mass value
functions too; `simple_mpc.adapters.assert_dv_convention` now checks an
adapter's declared `value_fn`/`higher_is_better` against goal geometry
(tests: `tests/test_value_sense.py`). TODO M4.

**Report bound-hit rates beside them.** An arm whose optimiser is pinned to
the workspace or push-length constraint is being scored on the constraint,
not on its gradients.

## `achieved_fraction` (added 2026-09-25, from EXP-0046)

**`achieved_fraction`** — closed-loop progress normalised by the goal's
reachable ceiling. For one episode on goal g with true soft-lyapunov values
V_0 (start), V_k (after k pushes), and the goal ceiling V*(g):

    achieved_fraction_k = (V_0 - V_k) / (V_0 - V*(g))

V*(g) = the lowest `lyapunov(occ_for_scoring(states), dist_field_from_mask(g))`
over single-layer placements of the episode's 20 non-overlapping axis-aligned
5 mm cubes inside the tray (centres on a 0.5 mm grid; greedy + lattice +
coordinate-descent packing; EXP-0046 `code/vstar.py`, table in
`results/vstar.json`). 1 = reached the best flat arrangement, 0 = no progress.
V* is an achievable (upper-bound) optimum of a heuristic packer, and
stacking is excluded, so values slightly > 1 are possible in principle.
Report per episode, then mean over episodes (and the ratio of means
mean(V_0 - V_k) / mean(V_0 - V*) alongside it). For the 30 `many` goals
V* is 0-0.006 (vs V_0 ~ 0.3), so `achieved_fraction` ~ (V_0 - V_k)/V_0;
it matters mainly for goals whose capacity < 20 cubes (letter_I, J, Y).

## `completion_time` (added 2026-09-25, EXP-0051; the headline control utility)
For one closed-loop episode with per-push planning times t_1..t_K and a simulated execution
time t_act per push: completion push k* = first k >= 1 with mass_in_region(s_k) / total mass
>= theta x the goal's optimum (EXP-0046 `mass_frac_best_placement`; theta = 0.9 unless
stated), or, for the coverage variant, covered_frac(s_k) >= 0.8 x the goal's layout ceiling.
completion_time = sum_{j<=k*} t_j + t_act k*. Episodes that never complete are censored at
K (1 s + t_act) when averaged (`mean_time_tact2_censored`: t_act = 2 s, K = 20 -> 60 s).
Lower is better. Code: EXP-0051 / EXP-0055 `code/analyse.py`.

## `coverage_emd`, `covered_frac` (added 2026-09-25, EXP-0055)
Uniform-coverage metrics for "the material should match the shape", on SCORING images
(`occ_for_scoring`), code `simple_mpc/value_functions.py::coverage_metrics`:
- `coverage_emd` = sliced W1 between the mass-normalised image p and the uniform distribution
  over the goal mask, 64 px grid, 64 equally spaced projection directions over [0, pi),
  distances in 64-px pixel units: mean over directions of sum_i |CDF_p - CDF_t|_i * gap_i
  along the sorted pixel projections. Lower is better. Reported as `emd_ach` =
  (E_0 - E_k) / (E_0 - E*), E* the goal's ceiling.
- `covered_frac` = |{goal pixels x : blur_{sigma=1.25 px}(p)(x) >= 0.5 / |mask|}| / |mask|.
  Higher is better. Reported as `cover_rel` = covered_frac / ceiling.
- Ceiling per goal: best (lowest E) of 8 Lloyd layouts of 20 cube centres over the mask
  pixels (`uniform_layout`, seeds 0-7), rasterised with occ_for_scoring. For large goals
  (quadrants) 20 cubes cannot cover the mask, so the ceiling covered_frac is < 1.

## `oracle_completion_pushes` (added 2026-09-25, EXP-0057)
Number-of-actions-to-solve for the simulator-as-model oracle ablation, per episode: the
completion push k* from `completion_time` above (mass-only variant, theta = 0.9), directly on
the recorded push index (not converted to wall time -- the oracle's planning time is the
quantity under study, not held fixed). Never-solved-by-push-20 episodes are CENSORED AT 21
(steps + 1), distinct from `completion_time`'s K-push convention, so the censored value is
never confused with an episode that solved exactly on the last push. Reported per cell as:
`frac_solved`, `median_pushes_solved_only` (solved episodes only), and `censored_mean_pushes`
(mean over all episodes with unsolved -> 21). Paired differences (same task set, same cell
comparison) use `delta_completion_pushes` = target cell's k* minus the default cell's k*, per
(goal, start), bootstrapped 95% CI. Code: `experiments/EXP-0057-*/code/analyse.py`.

## `node_accuracy`, `slateN_point_goal` (added 2026-10-03, EXP-0064 — FleX particle GNN)

For particle/node models scored in particle space (no occupancy image). Units: FleX /24.

| key | formula | notes |
|---|---|---|
| `node_accuracy` | `1 − sqrt(Σ‖pred − true‖²) / sqrt(Σ‖s0 − true‖²)`, sums over every (state, candidate, tracked node, xyz) of a group; tracked nodes = the model's own ≤30 FPS-sampled particles | Same form as `accuracy` but over **all tracked nodes, not a swept region**, so static nodes enter both terms; not comparable with image `accuracy`. `node_accuracy_moved` restricts the sums to nodes whose true xz displacement > 0.1 FleX units. |
| `slateN_point_goal` | slateN (this file's formula, cost sense) with value `V = mean_i ‖p_i − g‖` over xz positions, goal point `g ~ U(±wkspc_w)²` seeded per state (`RandomState(state_idx)` for goal 0; `RandomState(10000 + 100·state_idx + k)` for extra goals) | **nodes** variant: truth and prediction on the tracked nodes (EXP-0064 as run). **full** variant: truth on ALL true after-particles; prediction carries each particle by its nearest tracked node's predicted displacement. Model ties at its best are broken in expectation (mean truth over the tied set), so `persistence` scores 0 exactly. |
