# Metrics — exact definitions

Every `design.metric` field in a record must name a metric **from this file**,
by its key. Prose like "swept-region rms as % of persistence" is not a
definition: it does not say whether the average is a ratio of means or a mean of
ratios, what the denominator is, or how it moves when the field is preprocessed.
Those three ambiguities have each caused a real misreading in this project.

Source of truth: `fit_linear_foresight.py::metrics`.

---

## `pct_persistence` — "% of the change"

```
rms(model)   = mean over transitions of  sqrt( Σ_region (pred − truth)² / n_region_px )
pct          = 100 × rms(model) / rms(persistence)
```

Persistence predicts `pred = s`, so its error **is** the change `s′ − s`.
So this is **the model's per-pixel RMS error as a percentage of the per-pixel
RMS of the change that actually occurred**, inside the swept region.

- `100%` = error as large as the change ⇒ no better than predicting nothing moved.
- `0%` = perfect.
- `>100%` = worse than doing nothing.

**Three things it is not, and each has bitten:**

1. **It is a ratio of means, not a mean of ratios.** Numerator and denominator
   each average over transitions first. Big-change transitions carry more weight
   than a per-transition ratio would give them.
2. **The denominator depends on preprocessing.** Blur shrinks the change, so
   `pct` at σ=1 and `pct` at σ=0 are percentages *of different quantities* and
   must never be set side by side. EXP-0018 (blurred) and EXP-0021 (sharp)
   appeared comparable and were not; EXP-0022 exists to resolve it.
3. **It depends on the region.** A larger scoring region includes more untouched
   pixels, where persistence is exact, which *lowers* the denominator and makes
   every model look worse. Only compare `pct` across runs with the same region rule.

## `explained`

```
explained = 1 − mean‖pred − truth‖_F / mean‖truth − s‖_F        (Frobenius, per transition)
```

Fraction of the change the model accounts for. **Not the same quantity as
`1 − pct/100`**: `explained` uses the raw Frobenius norm while `rms` divides by
`n_region_px` before averaging. They coincide only when the region size is
constant across transitions — true for fixed-length pushes, false in general.
Every table in this register so far uses fixed-length pushes, which is why
`explained ≈ 1 − pct/100` throughout; with variable-length pushes they diverge
silently.

## `explained_over_meandelta`

`explained` with `mean-delta` in place of persistence as the reference. The
baseline that matters for canonical-frame models (C-011): the warp normalises
the action away, so beating persistence is close to free, while beating a
constant canonical-frame displacement is not.

## `pct_persistence_wholeimage`

`pct_persistence` computed over the **whole grid** rather than the swept region.
Used by EXP-0004 and EXP-0014 (the UNet ablations), because their probe does not
have per-sample push pixel coordinates.

**Read it with care and never against a swept-region number.** ~95% of the grid
is pixels no push could have changed, where persistence is exact by
construction, so the denominator is dominated by agreement about nothing and
every difference is compressed. At n=5 it is ~99.7% untouched. A model scoring
72% whole-image and a model scoring 72% swept-region are not comparable.

## `canonical_delta_profile`

Not an error metric. The mean canonical-frame difference `⟨I_{k+1} − I_k⟩`
profiled along the push axis within a band about it (EXP-0010, C-006). Reported
as a table of column-offset vs mean delta; the structure of interest is
depletion across the swept rectangle and deposition just ahead of the blade.
Judge it against the per-fold sd of each column, not against a baseline model.

## `r2_grouped_cv`

Coefficient of determination under k-fold cross-validation **grouped by run**,
for scalar targets (band displacement, `dV`) rather than images. Used by
`variance_decomposition.py`, `density_stratified.py`, EXP-0019.

Two traps, both hit: R² is a *ratio*, so stratifying a dataset shrinks the
target variance and can move R² without the absolute error moving — always
report RMSE alongside (EXP-0019 found linear/boosted RMSE flip on max-displacement
while shares looked stable). And "linear share" = linear R² / boosted R² can
exceed 100% when the booster overfits a narrowed range; that is a statement
about the booster, not about linearity.

## `world_alignment_cosine`

Cosine between a grid-derived transport direction and the world-frame push
direction (EXP-0020). Tests a convention against physics rather than against
other code. `+1` = perfect agreement, `0` = the no-information value — the
pre-fix convention scored exactly 0.

## `soft_iou`, `l1_per_mass`, `frobenius`

Reported alongside but rarely the headline. `frobenius` sums over N² pixels and
is **not comparable across resolutions**; it exists only to sit beside Suh &
Tedrake's Table 1 at 32×32.

## `dV` ranking metrics — `spearman`, `slate4`, `partial`

From `control_utility_test.py::rank_metrics`. These score **action selection**,
not image accuracy, and can disagree with the above — that disagreement is the
subject of C-030/C-035/C-044.

- `spearman` — rank correlation between predicted and true `dV` over candidates.
- `slate4` — of the oracle's advantage over a random pick within a slate of 4,
  the fraction the model captures. `1.0` = oracle, `0.0` = random, negative =
  worse than random.
- `partial` — the above with the state's own `V₀` and contact score regressed
  out, because cross-state slates confound "good action" with "easy state".
  **Prefer same-state slates** (`Genesis/data/slates/`) where available:
  EXP-0012 measured the cross-state confound as inflating noise-related ranking
  damage about two-fold.

## `FSS(r)` and `FSS_useful`

Fractions Skill Score at neighbourhood radius `r`, from
`scripts/probes/fss_scale_decomp.py`. **Bounded above by 1**, so it saturates,
and a *difference* of two FSS values is forced toward 0 at large `r` regardless
of skill — see the skill's "Bounded metrics have a ceiling". Compare against the
usable-skill threshold `FSS_useful = 0.5 + f₀/2` for base rate `f₀`, not against
another model's FSS.
