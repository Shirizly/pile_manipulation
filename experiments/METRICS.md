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
