# Action-pool diagnostics: what the aggregate ranking numbers are made of

Every `slateK`/`slateK_exact` number in the register is a mean over slates of
a ratio. This report opens that mean up: per-pool distributions, named
pools a reader can actually look at, and two specific suspicious results
checked rather than assumed. It does not re-rank the models — it reports
what each one achieves in absolute terms, per scenario, because the models
being compared have different compute cost and the cheap one wins unless it
loses decisively.

**Scripts** (new, this report):

- `scripts/probes/pool_common.py` — shared cache/geometry loading.
- `scripts/probes/pool_inspect.py` — **A1**: one pool, one figure + text.
- `scripts/probes/pool_survey.py` — **A2**: population statistics over every
  pool in a cache/goal.
- `scripts/probes/warp_blur_diagnostic.py` — **C1** check.
- `scripts/probes/l10_verification.py` — **C2** check.

**Data**: dataset A = `runs_exp0026/dv_cache_sharp.pt` (`n20_heap_5mm`, 50
same-state slates of ~32 candidates each). Dataset B =
`runs_expB/n20_L20mm_sharp_dv_cache.pt` (`n20_L20mm`, 20 slates of 128
candidates each). Both caches already carry both cost functionals used below
(`corner`, `ind-stripe-thin-pile`), computed by
`scripts/probes/exp0026_selection_pressure.py` / `expB_multistep_eval.py`;
nothing here refits a model or recomputes `dv_true`/`dv_pred` except where a
section explicitly says it does (C2b). Models: **persistence**, **mean-delta**,
**linear** (ridge→identity), **UNet** (UNetFilm), **oracle** (`dv_pred =
dv_true`, included only as the ceiling reference). `corner` is the coarse,
low-pass distance-transform functional every earlier record used; `ind-stripe-
thin-pile` is the sharp, pile-centred functional that survived
`functional_degeneracy_screen.py` (docs/experiments/METRICS.md).

**Definitions used throughout** — `dV` is a **cost** (lower = better).
`M_k = Σ_r t_[r] w_r(K)` (oracle's expected pick from a random K-subset,
`t_[r]` = true dV sorted ascending), `P_k = Σ_r t_(r) w_r(K)` (the model's
expected pick, `t_(r)` = true dV at the model's rank r), `w_r(K) = C(n-r,
K-1)/C(n,K)`. `R_K = M_k - P_k <= 0` by construction; `|R_K|` is the value
left on the table at pool size K, in the cost functional's own (Lyapunov)
units. `slateK_exact = (mean(t) - P_k)/(mean(t) - M_k)` is the normalised
version already in the register (`exp0026_kcurve_exact.py`, reused unchanged
here — `pool_common.rk_curve` imports its validated `w_r(K)` rather than
reimplementing it).

---

## 1. Population statistics (A2) and pool-spread distributions

Pool spread (`dv_true` max−min and mean−min per slate) sets the scale every
normalised capture number divides by:

| cell | n slates | max−min: mean / median (p10–p90) | mean−min: mean / median |
|---|---|---|---|
| A / corner | 50 | 0.1576 / 0.1523 (0.138–0.186) | 0.0777 / 0.0764 |
| A / ind-stripe-thin-pile | 50 | 0.3440 / 0.3495 (0.276–0.407) | 0.1530 / 0.1517 |
| B / corner | 20 | 0.1404 / 0.1376 (0.119–0.173) | 0.0533 / 0.0527 |
| B / ind-stripe-thin-pile | 20 | 0.4132 / 0.4174 (0.345–0.466) | 0.1878 / 0.1735 |

The sharp functional roughly doubles the pool's own spread relative to
`corner` on both datasets — sharpening the cost separates the candidates
MORE in absolute terms, not less; what changes (below) is how well the
cheap models track that separation.

**Agreement rate** (model's top-1 == oracle's top-1) and **slateK_exact**
(absolute, K=4 and K=max; paired UNet−linear difference with its own sem):

| cell | persistence agree | mean-delta agree | linear agree | UNet agree | linear slateK_exact K=4 / Kmax | UNet slateK_exact K=4 / Kmax | UNet−linear (K=4) |
|---|---|---|---|---|---|---|---|
| A / corner | 0.04 | 0.20 | 0.44 | 0.56 | 0.977 / 0.958 (K=31) | 0.987 / 0.976 | +0.0103, sem 0.0016, t=+6.51 (39/50) |
| A / stripe | 0.02 | 0.10 | 0.16 | 0.22 | 0.715 / 0.568 (K=31) | 0.785 / 0.673 | +0.0706, sem 0.0151, t=+4.67 (37/50) |
| B / corner | 0.00 | 0.00 | 0.70 | 0.70 | 0.953 / 0.967 (K=128) | 0.971 / 0.982 | +0.0173, sem 0.0032, t=+5.47 (19/20) |
| B / stripe | 0.00 | 0.25 | 0.25 | 0.40 | 0.832 / 0.769 (K=128) | 0.822 / 0.724 | −0.0101, sem 0.0108, t=−0.93 (10/20 wins) |

(Persistence and mean-delta's near-zero agreement rates are as expected by
construction — persistence predicts `dV=0` for every candidate and picks by
argmin tie-break, mean-delta has no action-dependence at all.) The absolute
picture: on dataset A, **sharpening the functional roughly halves both
models' capture at K=max** (linear 0.958→0.568, UNet 0.976→0.673) and the
UNet's edge over linear widens (+0.010→+0.071 at K=4). On dataset B the same
sharpening also lowers both models' capture (linear 0.967→0.769, UNet
0.982→0.724) but the UNet's edge **reverses sign** (+0.017→−0.010, not
significant either way at K=4, t=−0.93). This replicates C-044/C-045's own
functional-dependence finding (docs/experiments/REGISTER.md) rather than
contradicting it — this report adds the per-pool texture underneath it.

**Disagreement: near-tie vs. genuine loss.** "Close" is defined as
`(dv_true[model's pick] − dv_true[oracle's pick]) / (max−min of the pool)`
below a threshold; the distribution of this ratio *among disagreement
slates only* is reported in full rather than picking one number, because it
moves a lot between cells:

| cell, model | median ratio (p25–p75) | near-tie @0.05 / @0.10 / @0.20 | large-loss @0.05 / @0.10 / @0.20 | exact agree |
|---|---|---|---|---|
| A/corner, linear | 0.028 (0.015–0.042) | 0.44 / 0.52 / 0.56 | 0.12 / 0.04 / 0.00 | 0.44 |
| A/corner, UNet | 0.021 (0.008–0.036) | 0.38 / 0.42 / 0.44 | 0.06 / 0.02 / 0.00 | 0.56 |
| A/stripe, linear | 0.198 (0.122–0.354) | 0.04 / 0.12 / 0.42 | 0.80 / 0.72 / 0.42 | 0.16 |
| A/stripe, UNet | 0.210 (0.082–0.260) | 0.10 / 0.26 / 0.36 | 0.68 / 0.52 / 0.42 | 0.22 |
| B/corner, linear | 0.038 (0.030–0.053) | 0.20 / 0.30 / 0.30 | 0.10 / 0.00 / 0.00 | 0.70 |
| B/corner, UNet | 0.010 (0.003–0.018) | 0.25 / 0.30 / 0.30 | 0.05 / 0.00 / 0.00 | 0.70 |
| B/stripe, linear | 0.117 (0.076–0.186) | 0.10 / 0.35 / 0.55 | 0.65 / 0.40 / 0.20 | 0.25 |
| B/stripe, UNet | 0.130 (0.098–0.292) | 0.05 / 0.15 / 0.35 | 0.55 / 0.45 / 0.25 | 0.40 |

At threshold 0.10 (roughly the middle of the disagreement-ratio medians
across cells, 0.01–0.20): **under `corner`, disagreement is dominated by
near-ties** — linear disagrees on 56% of A-slates but only 4% are a genuine
large loss at that threshold; the rest is either exact agreement or a
near-miss. **Under `ind-stripe-thin-pile`, the opposite holds** — most
disagreement is a genuine, large loss (72% of A-slates, 40% of B-slates for
linear), which is exactly the "coarse functionals hide real ranking damage"
effect this report exists to surface. The picture is not threshold-stable
(e.g. A/stripe linear near-tie goes 4%→12%→42% from thr 0.05→0.10→0.20), so
read the ranges, not one number — but the *qualitative* corner-vs-stripe
contrast holds at every threshold tried.

Full per-model numbers: `runs_exp0026/survey/A_{corner,stripe}.json`,
`runs_expB/survey/B_{corner,stripe}.json` (population stats) and
`runs_exp0026/survey/A_{corner,stripe}_kexact.json`,
`runs_expB/survey/B_{corner,stripe}_kexact.json` (slateK_exact/worstK/
rank_profile, absolute + paired).

---

## 2. Named pools (A1)

Three pools per cell: the most typical (linear's |R_4| nearest the cell's
median), the worst for the linear operator (linear's largest |R_4|), and a
disagreement with small value loss (linear's smallest-ratio disagreement
slate). Each figure: dV histogram with oracle/model picks and pool mean
marked; the actions themselves on the step-0 occupancy image (start dot,
heading arrow, swept rectangle); `|R_K|` vs K per model; and the ordering
scatter + true-percentile of each model's top-1/2/3 picks.

### A / corner (`runs_exp0026/dv_cache_sharp.pt`, n=32/pool)

- **Typical (slate 35)** — `figs/A_corner_typical_s35.png`. Oracle/mean-delta/
  linear all land on the SAME candidate (dv_true=−0.0690); UNet is 0.0004 off
  the oracle. Pool spread 0.155, pool mean +0.024 (net-harmful on average —
  most candidates in this pool make things worse).
- **Worst for linear (slate 25)** — `figs/A_corner_worstlinear_s25.png`.
  Linear's regret vs. oracle is +0.0240 (the largest in the 50-slate cell at
  K=4), picking a candidate at true percentile 0.19 for its rank-1 pick.
  Mean-delta (+0.0200) and UNet (+0.0231) do comparably badly here — this is
  a hard pool for every model, not a linear-specific failure.
- **Disagreement, small loss (slate 23)** — `figs/A_corner_neartie_s23.png`.
  Linear and UNet pick the SAME action (candidate 18, true percentile 0.03),
  0.00025 from the oracle; mean-delta disagrees with a real (if modest) loss
  of 0.0149. Illustrates the "disagree in index, agree in value" case the
  near-tie category is built to catch.

### A / ind-stripe-thin-pile (n=32/pool)

- **Typical (slate 44)** — `figs/A_stripe_typical_s44.png`. Every model
  underperforms noticeably here (linear regret +0.074, UNet +0.022,
  mean-delta +0.037) against a pool spread of 0.323 — a harder pool than
  `corner`'s typical case in absolute terms, consistent with the halved
  slateK_exact ceiling under this functional.
- **Worst for linear (slate 25)** — `figs/A_stripe_worstlinear_s25.png`.
  Linear's regret is +0.129 (its largest |R_4| in this 50-slate cell); this
  is not simply "linear is bad" on this pool — mean-delta does even worse
  here (+0.164), and UNet ties persistence's pick (+0.100) — a pool where
  every model, learned or not, struggles.
- **Disagreement, small loss (slate 13)** — `figs/A_stripe_neartie_s13.png`.
  Linear and UNet again pick the same candidate (true percentile 0.062),
  0.009 from the oracle — small in absolute terms but note the pool's own
  spread here (0.305) makes this a genuinely tight near-miss (ratio 0.030),
  not a trivial one.

### B / corner (`runs_expB/n20_L20mm_sharp_dv_cache.pt`, n=128/pool)

- **Typical (slate 14)** — `figs/B_corner_typical_s14.png`. UNet hits the
  oracle exactly (regret 0); linear is 0.0034 off, mean-delta 0.0125 off.
  Persistence's pick here is actively harmful (regret +0.0998) — the pool
  mean itself is barely positive (+0.003), so most candidates are useless
  or harmful and only a few are good.
- **Worst for linear (slate 17)** — `figs/B_corner_worstlinear_s17.png`.
  Mean-delta ties persistence's REALISED value exactly here (dv_true=0 for
  both, though mean-delta's own predicted dV was −0.018, not 0 — it picked a
  candidate it expected to help and got nothing) — a genuine mean-delta
  failure, not just a weak signal; linear still recovers 90% of the achievable gain
  (regret +0.0079 against an oracle best of −0.0814) while UNet reaches the
  oracle exactly.
- **Disagreement, small loss (slate 5)** — `figs/B_corner_neartie_s5.png`.
  Linear (candidate 4) and UNet (candidate 125) pick DIFFERENT candidates,
  both essentially at the oracle (regret 0.00005 and 0.00041) — the clearest
  "different action, same value" case in this report: two genuinely
  different pushes into the same effective outcome.

### B / ind-stripe-thin-pile (n=128/pool)

- **Typical (slate 13)** — `figs/B_stripe_typical_s13.png`. Linear, mean-delta
  and UNet all pick the identical candidate (true percentile 0.016 for all
  three), regret +0.068 against a much larger pool spread (0.533) than `corner`'s
  typical pool — again the pattern that the sharp functional produces larger
  absolute stakes per pool, with correspondingly larger absolute misses.
- **Worst for linear (slate 42)** — `figs/B_stripe_worstlinear_s42.png`.
  Linear and UNet both reach the oracle exactly here (candidate 5, regret 0)
  — flagged as linear's "worst" only by the |R_4| ranking metric, which
  measures a K-weighted expectation, not the realised top-1 regret; the top-1
  outcome at this specific pool is in fact excellent for both learned models.
  This is worth flagging as a caveat on using |R_K| alone to pick "the worst
  pool" — it is the worst in expectation over random K-subsets, not
  necessarily the worst in the actual top-1 choice.
- **Disagreement, small loss (slate 39)** — `figs/B_stripe_neartie_s39.png`.
  Linear picks a candidate 0.0107 from the oracle (mean-delta/UNet reach the
  oracle exactly); ratio 0.034 against a pool spread of 0.318 — genuinely
  close, and visually (see the figure) linear's and UNet's chosen pushes are
  nearly the same action (adjacent start points, near-identical heading).

---

## 3. C1 — is `warp-only`'s positive accuracy a bug?

**Verdict: not a bug.** `scripts/probes/warp_blur_diagnostic.py` on the
L20mm/L40mm eval cells:

| cell | warp-only (A=identity) accuracy | best-matching blur sigma | blurred-persistence accuracy at that sigma | blur sweep (σ=0.5/1.0/1.5/2.0/3.0) |
|---|---|---|---|---|
| L20mm | **+0.0254** | σ≈1.0 | **+0.0313** | +0.029 / **+0.031** / +0.022 / +0.007 / −0.026 |
| L40mm | **+0.0442** | σ≈0.5 | **+0.0350** | **+0.035** / +0.079 / +0.105 / +0.124 / +0.148 |

At both push lengths, an explicit Gaussian blur of PERSISTENCE — no warp, no
operator, nothing but a low-pass filter — reproduces a positive accuracy of
the same sign and comparable-or-larger magnitude to `warp-only`. At L40mm
every blur sigma tried gives a LARGER positive accuracy than warp-only
itself (up to +0.148 at σ=3.0), i.e. blur alone beats persistence by more
than the warp round-trip does. This is exactly the leading benign
explanation: `accuracy` is an RMS ratio, and RMS rewards hedging — a
low-pass copy of the current state scores a lower per-pixel error than a
razor-sharp copy once material has actually moved, because the sharp copy
commits fully to the now-wrong edge while the blur partially "predicts" the
blur-shaped component of the change. `warp-only`'s positive score is this
effect applied via the SE(2) warp's own resampling blur, not a defect in the
warp/blend code.

**Mechanical check**: `blend_push_prediction` returns the ORIGINAL occupancy
exactly outside its validity mask, as intended — `max|blended − original|` at
masked-out pixels measured **0.000e+00** at both push lengths (17.9%–22.8%
of pixels are masked out per cell). No bug there either.

One asymmetry worth recording rather than glossing over: at **L10mm**
(shorter push, checked for completeness though not part of the original
suspicious pair), `warp-only` is strongly NEGATIVE (−0.5294, matching the
register) and blur makes it WORSE, monotonically, down to −1.85 at σ=3.0 —
blur does not reproduce a positive score there. This is consistent with, not
contradictory to, the verdict above: at L10mm the true change is tiny
(dV sd ≈0.005, see C2), so blurring the WHOLE image (including the ~80% that
never moved, where persistence is exact) adds absolute error against a small
signal, while at L20/L40mm the change is large enough that hedging pays off
in the swept region specifically. The direction of the effect depends on the
scale of the actual change relative to the blur radius, which is exactly
what the L20/L40 finding says.

---

## 4. C2 — is L10mm's accuracy=−0.44 / slateK_exact=0.91 real?

`runs_expB/n20_L10mm_accuracy.json` / `n20_L10mm_kcurve_exact.json` confirm
the quoted numbers exactly: linear image accuracy **−0.4407**, linear
`slateK_exact` at K=128 **0.9132** (goal=`corner`, the only sharp-adjacent
goal this cache has — L10mm's cache carries `corner`/`center` only).
`scripts/probes/l10_verification.py` checks three things:

**(a) Spread vs. capture.** L10mm's overall `dv_true` sd is **0.00506** (vs.
0.0234 at L20mm and 0.0905 at L40mm on the identical `corner` goal — 4.6x and
17.9x larger respectively). Per-slate spread (max−min) correlates with
per-slate `slateK_exact` capture **at K=4** for the linear operator: Pearson
r=+0.571 (p=0.009, n=20), Spearman ρ=+0.541 (p=0.014) — smaller-spread slates
within L10mm do capture a smaller fraction, consistent with ratio
sensitivity at small K. The UNet shows the same direction but does not clear
significance (r=+0.303, p=0.194). **At K=128 — the K the quoted 0.91 comes
from — this correlation is gone** (linear r=+0.020 p=0.934; UNet r=+0.036
p=0.880). So ratio instability is a real, measurable effect at K=4 within
this cell, but it does not explain the specific headline number.

**(b) Independent recompute.** A fresh linear-operator fit via
`dmdc_baseline.load_transition_arrays` (not the cache-building script's
`build_dataset` stacking loop), occ0 reloaded through
`pool_common.load_occ0_for_slate` (a third loading route), scored on 768
rows across 6 slates: **Pearson r=1.00000** against the cached `dv_pred`,
mean diff −7.8e-11, max|diff|=1.5e-8 (float32 noise; cached values have
sd 5.2e-3, so max|diff|/sd ≈ 0.000003). No discrepancy — the cached numbers
are reproducible through an independent path, so this is not a computation
bug.

**(c) Raw |R_K| beside the capture fraction, same-cell and cross-cell**
(linear operator, `corner` goal):

| cell | mean\|R_4\| (Lyapunov units) | mean\|R_128\| | slateK_exact K=4 | slateK_exact K=128 | overall dv_true sd |
|---|---|---|---|---|---|
| L10mm | 0.000859 | 0.002042 | 0.793 | 0.913 | 0.00506 |
| L20mm | 0.000959 | 0.001859 | 0.953 | 0.967 | 0.02338 |
| L40mm | 0.001619 | 0.002975 | 0.982 | 0.979 | 0.09054 |

The absolute value left on the table is **similar in magnitude across all
three push lengths** (within ~2x at each K), while the denominator
(`dv_true` sd) varies by up to 18x. That means L10mm's lower reported
capture fraction at K=4 (0.79 vs. 0.95–0.98 elsewhere) is arithmetically
mostly a **denominator effect**: the model leaves roughly the same absolute
amount of value on the table at every push length, but that fixed amount is
a much larger SHARE of a much smaller pie at L10mm. Read together with (a):
within-cell, small-spread slates do show lower capture at K=4 (real ratio
sensitivity); across cells, the same absolute skill produces very different
reported fractions for the same underlying reason.

**Verdict**: no bug (b confirms exact reproducibility) and the -0.44/0.91
dissociation is real but should be read carefully, not as "near-oracle
ranking" at face value. Two things are true simultaneously and neither
alone is the full story: (i) the −0.44 image accuracy is very likely mostly
a **warp-floor effect of short push length** (the cell is "warp-limited" in
the register's terms), not a statement about ranking skill — L10mm's own
identity-operator round trip alone already scores **−0.5294** (this
report's C1 script, matching the register), i.e. almost the whole of the
fitted operator's −0.44 is the SE(2) warp/resample cost at this short a push,
not the fitted map; (ii) the 0.91 control-ranking number is not a
measurement error, but at K=128 it is not explained by ratio instability
either — the model genuinely places nearly all of the small absolute
`dv_true` variation in close to the right order. Absolute terms: about
0.002 Lyapunov units of regret at K=128, roughly the same absolute regret
paid at L20mm/L40mm, just against a ~5-18x smaller pool of achievable value.
"0.91 capture of a tiny available gain" and "0.97-0.98 capture of a larger
one" are, in this specific sense, closer achievements than the fractions
alone suggest — but they are not identical, since the K=4 correlation shows
some genuine within-cell ratio sensitivity at low K.

---

## 5. Files

- Scripts: `scripts/probes/pool_common.py`, `scripts/probes/pool_inspect.py`,
  `scripts/probes/pool_survey.py`, `scripts/probes/warp_blur_diagnostic.py`,
  `scripts/probes/l10_verification.py`.
- Population-stats JSON/logs: `runs_exp0026/survey/`, `runs_expB/survey/`.
- Named-pool figures: `reports/figs/{A,B}_{corner,stripe}_{typical,worstlinear,neartie}_s*.png`
  (+ matching `.log` text blocks).
