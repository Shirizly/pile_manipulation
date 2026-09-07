# Metrics — exact definitions

Every `design.metric` field in a record must name a metric **from this file**,
by its key. Prose like "swept-region rms as % of persistence" is not a
definition: it does not say whether the average is a ratio of means or a mean of
ratios, what the denominator is, or how it moves when the field is preprocessed.
Those three ambiguities have each caused a real misreading in this project.

Source of truth: `fit_linear_foresight.py::metrics`.

---

## THE TWO STANDARD METRICS

**Report these in every record.** Both go UP when better. Where the metric
itself is the object of study (FSS, R², a profile), report these as a reference
row anyway, so results stay comparable across the register.

| key | definition | 0 means | 1 means |
|---|---|---|---|
| **`accuracy`** | `1 − rms(model) / rms(persistence)`, swept region, same norm top and bottom | no better than predicting nothing moved | perfect |
| **`slate4`** | fraction of the oracle's advantage over a random pick that the model captures, within a candidate slate of 4 | no better than random | as good as the oracle |

Negative values are meaningful in both: worse than doing nothing, and worse than
choosing at random, respectively.

**Why two.** Image accuracy and control utility have been measured
dissociating three times in this project — EXP-0008 (noise destroys ranking at
a cost rms barely sees), EXP-0013/EXP-0017 (a rank-1 operator with better rms
and worse ranking), EXP-0022 (the UNet's whole advantage sits in a frequency
band control appears not to consume). A single image number cannot serve an MPC
objective, so the standard is a pair.

**Why `accuracy` and not `explained`.** They are the same idea, but `explained`
was defined with the Frobenius norm while `pct` used the pixel-normalised one,
so the two disagree whenever region size varies. `accuracy` uses one norm
throughout and is exactly `1 − pct_persistence/100`. Prefer it; `explained` is
kept only so old records still parse.

**What `accuracy` still cannot do:** be compared across different
preprocessing. Its denominator is the size of the actual change, which shrinks
under blur and grows with region. `accuracy = 0.42` at σ=0 and `accuracy = 0.42`
at σ=1 are not the same achievement. Compare within a configuration; across
configurations, say so.

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

**`slate4` draws its 4 candidates WITH replacement.** `rank_metrics` builds
slates with `torch.randint`, so a "slate of 4" is 4 draws from the candidate
pool and holds ~3.44 distinct actions on average; `slate16` holds ~13. Measured
cost of the difference (EXP-0026, same 49 slates, same predictions): linear
0.9503 with replacement vs 0.9583 without, UNet 0.9688 vs 0.9743, mean-delta
0.7956 vs 0.8063 — every `slate4` in this register is **0.5–1.1 points low**
relative to a true 4-distinct-candidate slate. No verdict in the register turns
on that, but a `slate4` and a `slateK` value must not be set side by side
without saying which sampler produced them.

## Weight-field functional family — `lyapunov_weights` goal keys

Every `dV`/`slate4`/`slateK`/`slateK_exact` number above is computed from
`V = d^T y / ||y||_1` (`control_utility_test.py::lyapunov(occ, d)`), where `d`
is a WEIGHT FIELD from `lyapunov_weights(grid_res, goal, device)`. Until
EXP-0024_v2/EXP-0026_v2, `goal` was always `center`/`corner`/`stripe` — three
target REGIONS, but always the same FUNCTIONAL FORM (a Euclidean distance
transform, normalised to its own max). A distance transform is low-pass by
construction (smooth everywhere except right at the target boundary), so
every number above measures action selection through a cost that may not be
able to see fine spatial detail — which is exactly the frequency band C-044
says the UNet's advantage over the linear operator lives in. EXP-0024_v2
tests this directly by adding a FUNCTIONAL-SHARPNESS axis, independent of
target size, and EXP-0026_v2 re-scores the register's degradation arms
through it.

**New goal keys** (`center`/`corner`/`stripe` are unchanged, byte-identical):

```
distclip-corner-r{2,4,8}   min(dist_to_corner_halfplane_px, r) / r
                            -- a tolerance knob at a FIXED target (the corner
                            half-plane): r large -> smooth, converging to
                            `corner`; r -> 0 -> an indicator.
ind-corner                  1 outside the corner half-plane, 0 inside
                            -- the r -> 0 limit of distclip-corner-r*, same
                            fixed target as `corner`.
ind-square8                 1 outside an 8x8 px square (offset one
                            side-length in from the corner: rows/cols
                            [H/8 : 2H/8]), 0 inside -- the SHARP, SMALL-target
                            form.
dist-square8                distance transform to the SAME 8x8 px square
                            -- isolates target SIZE from functional
                            sharpness: same functional form as `corner`,
                            much smaller target.
```

**Measured spectral concentration** (`scripts/probes/spectral_concentration.py`:
fraction of `|FFT(field - mean(field))|^2` within radius r, in grid units, of
DC, on a 64x64 grid — demeaning is required, since every field here has a
large, shape-independent DC component that would otherwise swamp the ratio):

| weight field | r<=1 | r<=4 | r<=8 |
|---|---|---|---|
| `corner` (dist-corner, what every pre-2026-09-07 record uses) | 0.629 | 0.883 | 0.941 |
| `distclip-corner-r8` | 0.619 | 0.926 | 0.966 |
| `distclip-corner-r4` | 0.582 | 0.906 | 0.964 |
| `distclip-corner-r2` | 0.557 | 0.886 | 0.952 |
| `ind-corner` | 0.541 | 0.870 | 0.937 |
| `dist-square8` (target-size control) | 0.683 | 0.908 | 0.953 |
| `ind-square8` (sharp; DEGENERATE, see below) | 0.060 | 0.514 | 0.819 |

`distclip-corner-r8` -> `r4` -> `r2` -> `ind-corner` is monotone at a FIXED
target: sharpening the functional lowers r<=1 concentration from 0.619 to
0.541. `dist-square8` sits essentially at `corner`'s own concentration
(0.683 vs 0.629) despite an 8x linear target shrink — **shrinking a
distance-transform target does not sharpen it; switching functional form
does.** This is the load-bearing fact the functional/size split rests on,
and it is measured, not assumed.

**Degeneracy warning, found the hard way**: `ind-square8` is exactly
degenerate (`dv_true` == 0 for every candidate) on both datasets tested
(`Genesis/data/slates/n20_heap_5mm`, `Genesis/data/slates_multistep/n20_L20mm`
step-0) — the target sits far enough from these piles' reachable footprint
that no single push crosses its 8x8 px boundary. This is the small-target
mirror of C-040's centred-target degeneracy: a target can fail either by
being symmetric with a centred pile (`center`, C-040: `dV`≈0 because equal
mass enters and leaves) or by being small and remote enough that `dV`=0 by
construction for every candidate (`ind-square8`). **Screen with
`scripts/probes/functional_degeneracy_screen.py` BEFORE computing any
slateK/slateK_exact number on a new target/functional** — it reports
`dv_true` mean/sd, %helpful, %|dv|<1e-9, #distinct values, and the fraction
of slates clearing the same per-slate variation threshold
`exp0026_kcurve_exact.py` itself uses to skip a slate.

**What EXP-0024_v2/EXP-0026_v2 found using this family**: sharpening the
functional at a fixed target (`corner` -> `ind-corner`) moves the linear
operator's `slateK_exact` down by at most ~5 points (never below 0.92) and
does not widen the paired UNet-linear gap — C-044/C-045 are
functional-independent over the range tested. But the register's
degradation-arm story (C-035/C-039/C-046 — systematic perturbations are
K-invariant, only independent-per-candidate noise degrades with selection
pressure) is NOT functional-independent: under `ind-corner` the systematic/
independent split narrows sharply on `n20_heap_5mm` and partially inverts on
`n20_L20mm` (displacement becomes the largest K-decliner, hf-noise goes
flat). See those two records for the numbers; `slateK_exact` computed under
any goal OTHER than `corner`/`center`/`stripe` should be read as "under this
functional", not as a general control-utility fact, until more of the
register has been re-scored this way.

## `slateK`, `regret_dv`, `pick_pctile` — selection under pressure

From `scripts/probes/exp0026_kcurve.py`. The same candidate-slate idea as
`slate4`, swept over the slate size **K**, with subsets drawn **without
replacement** (`torch.randperm`) so K is a count of distinct candidates. At
`K = n_slate` there is exactly one subset — the deterministic
top-1-of-everything test.

```
per draw:  chosen = dv_true[argmin_j dv_pred_j]     over the K sampled candidates
           oracle = min_j dv_true_j
           rand   = mean_j dv_true_j
slateK      = mean over draws of (rand − chosen) / (rand − oracle)   [bounded, ↑ better]
regret_dv   = mean over draws of (chosen − oracle)                   [Lyapunov units, ↓ better]
pick_pctile = mean over draws of #{j : dv_true_j < chosen} / K       [fraction, ↓ better]
```

`slateK` is a **mean of per-draw ratios**, matching `rank_metrics`' own
convention, so `slateK` at K=4 is directly comparable with a `slate4` from a
without-replacement sampler.

**Always report `regret_dv` beside `slateK`.** `slateK`'s denominator
`(rand − oracle)` grows with K — the best of 31 candidates beats the average by
more than the best of 4 does — so a flat `slateK` curve does **not** mean the
selection problem is unchanged by K. Measured (EXP-0026, linear operator,
n=50 slates): `slateK` 0.958 → 0.958 from K=4 to K=31 while `regret_dv` grows
0.0012 → 0.0032, a 2.7× rise in the value actually left on the table. Both
statements are true and neither alone is the answer.

`pick_pctile` is the scale-free companion: what fraction of the offered
candidates were better than the one chosen (0 = picked the best). It is the
metric to quote when K itself varies, since it does not depend on the spread of
`dv_true` at all.

## `slateK_exact`, `worstK`, `rank_profile` — the ordering, not a sample of it

From `scripts/probes/exp0026_kcurve.py` (closed form) — **prefer these over the
sampled `slateK` above.** Sampling subsets is a Monte-Carlo estimate of a
quantity that has an exact expression, because within a fixed slate the model
picks its rank-`r` candidate exactly when `r` is drawn and every better-ranked
candidate is not:

```
w_r(K) = C(n-r, K-1) / C(n, K)            weight on the model's rank r
E_chosen(K)  = sum_r  t_(r) * w_r(K)      t_(r) = true dV at MODEL rank r
E_oracle(K)  = sum_r  t_[r] * w_r(K)      t_[r] = true dV sorted ascending
                                           (= E[min over a random K-subset])
slateK_exact = (mean(t) - E_chosen(K)) / (mean(t) - E_oracle(K))    per slate
```

Average the per-slate values across slates (not a pooled ratio of sums) so the
model comparison stays pairable. Two consequences of the closed form worth
knowing before reading any curve:

- **The whole K-curve is a family of linear functionals of the model's
  ordering.** Nothing but the ordering and the true values enters, so K is only
  a knob on how sharply weight concentrates at the top. At n=32: K=4 puts
  0.125 on rank 1 and still 0.056 on rank 8; K=31 puts 0.969 on rank 1.
- **It is not the same estimand as the sampled `slateK`**, which is a mean of
  per-draw ratios with a *subset-specific* denominator and therefore weights
  every pool equally regardless of how much was at stake in it. Measured on the
  same 50 slates (EXP-0026 data, linear operator): exact 0.977 vs sampled 0.958
  at K=4, converging to 0.958 vs 0.958 at K=31. Quote which one you used.

```
worstK = max over model-rank r with (n - r) >= K-1  of
             [ t_(r) - min_{r' > r} t_(r') ]        / (mean(t) - min(t))
```

**`worstK` is the adversarial pool**: the largest value-inversion an adversary
can exploit by offering K candidates and letting the model pick its best-ranked.
`0` = never worse than the best on offer; `1` = as bad as an average random
pick. Report it beside `slateK_exact`, because **the average-case number is
where model classes look equivalent and the worst case is where they do not**:
measured (EXP-0026 data, K=4), linear 0.471 vs UNet 0.316 — a paired difference
of +0.155 (sem 0.031, t=5.07, 42/50 slates) against an average-case difference
of 0.007, i.e. **~22x more separation**. It is also nearly K-independent
(0.483 → 0.334 from K=2 to K=16), since it is a property of one inversion in the
ordering rather than of the pool size. A real MPC pool is neither uniform nor
adversarial — CEM concentrates where the model says value is high — so the
operating point sits between the two columns.

```
rank_profile[r] = mean over slates of  #{j : t_j < t_(r)} / n
```

The mean true percentile of the candidate the model ranked `r`-th
(`0` = truly best, `1` = truly worst). This is the diagnostic the summary
metrics cannot give: a model can rank the best action first and a catastrophic
one second, which destroys it in any pool missing the best, and every
average-over-pools metric dilutes that. Report the profile at
r = 1, 2, 3, 4, 8, 16, n and the fraction of slates whose rank-2/3/4 pick falls
in the true bottom quartile. Measured on EXP-0026's data, no model showed this
pathology (linear 0.035 → 0.058 → 0.076 → 0.095 across ranks 1-4; 0% of slates
bottom-quartile at ranks 2-4), which is *why* its K-curve is flat — so a flat
curve is only trustworthy alongside this check.

## `FSS(r)` and `FSS_useful`

Fractions Skill Score at neighbourhood radius `r`, from
`scripts/probes/fss_scale_decomp.py`. **Bounded above by 1**, so it saturates,
and a *difference* of two FSS values is forced toward 0 at large `r` regardless
of skill — see the skill's "Bounded metrics have a ceiling". Compare against the
usable-skill threshold `FSS_useful = 0.5 + f₀/2` for base rate `f₀`, not against
another model's FSS.
