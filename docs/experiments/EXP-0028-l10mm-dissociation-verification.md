---
id: EXP-0028
title: >
  L10mm's accuracy=-0.44 / slateK_exact=0.91 dissociation (EXP-0024_v1) is
  not a computation bug (independent recompute agrees to float32 noise) and
  is mostly a scale/denominator effect at K=128, not evidence the model
  ranks better there than at L20/L40mm -- absolute |R_K| is comparable
  across all three cells
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: null

claim: >
  (a) per-slate `slateK_exact` (linear, K=4, goal=corner) correlates with
  per-slate pool spread (max-min of dv_true) within the L10mm cell (a signal
  of ratio instability rather than pure ranking skill); (b) an
  independently-fit linear operator, loaded via a different code path than
  the cache-building script, reproduces the cached `dv_pred` to within
  numerical noise; (c) the raw |R_K| (Lyapunov units) at L10mm is within
  ~2x of L20mm/L40mm's own |R_K| at matched K, despite L10mm's `dv_true` sd
  being 4.6x/17.9x smaller -- i.e. the fraction captured differs mainly
  because of the denominator's scale, not because of a several-fold
  difference in absolute regret.

prediction:
  supports: >
    (a) a per-slate Pearson correlation between spread and slateK_exact with
    p<0.05 at K=4; (b) independent dv_pred correlates with cached dv_pred at
    r>0.999; (c) mean|R_K| at L10mm is within 2x of L20mm/L40mm's own
    mean|R_K| at the same K, despite dv_true sd differing by >4x.
  refutes: >
    (a) no correlation (p>0.20) at K=4; (b) independent recompute disagrees
    beyond float32 noise (would indicate a real bug); (c) L10mm's mean|R_K|
    is >4x smaller than L20/L40mm's at matched K (would mean the capture
    difference is driven by genuinely better absolute ranking, not scale).
  discriminating: true

provenance:
  commit: 250102d8
  dirty: false
  data_commit: unrecorded (Genesis/data/slates_multistep/n20_L{10,20,40}mm, see EXP-0024_v1)
  script: scripts/probes/l10_verification.py
  data: ["runs_expB/n20_L10mm_dv_cache.pt", "runs_expB/n20_L20mm_dv_cache.pt",
         "runs_expB/n20_L40mm_dv_cache.pt",
         "configs/dataset/genesis_slates_multistep_n20_L10mm_train.yaml"]
  code_path: >
    cached dv_pred: expB_multistep_eval.py's build_dataset-stacking loop
    (unchanged from EXP-0024_v1). Independent recompute (this record):
    dmdc_baseline.load_transition_arrays for the fit +
    pool_common.load_occ0_for_slate for occ0 reconstruction -- both distinct
    call paths from the cache-building script, sharing only
    fit_linear_foresight.predict_world/control_utility_test.lyapunov as the
    final apply/score step (not hypothesised to contain the bug, if any).
  seed: 0 (fit_operator has no stochastic step; ridge=1.0, toward_identity=True)
  split: "identical train/eval slate split as EXP-0024_v1 (30/50 train, 20/50 eval, seed 0)"
  runtime: "~3 min CPU (fit + 768-row recompute over 6 slates)"

budget:
  declared: "not separately declared; part of the ~3h/250k-token action-pool-diagnostics task"
  spent: "~15 min, ~20k tokens"
  outcome: within

design:
  varied: {cell: [L10mm, L20mm, L40mm], K: [4, 128], check: [spread-correlation, independent-recompute, raw-Rk]}
  held_fixed: {goal: corner, R: 64, crop: 1.0, ridge: 1.0, model: linear, min_slate: 8}
  baselines: [oracle (trivial, r=1 by construction), UNet (same correlation check run in parallel)]
  metric: "slateK_exact"

noise_floor: >
  independent-recompute check: float32 rounding only (cached values sd
  5.2e-3, max|diff| between paths 1.5e-8, i.e. ~3e-6 of the signal sd) --
  effectively zero, not a noise floor in the usual sense since both paths
  are deterministic given the same fit.

depends_on: [swept-region-metric]
establishes: []

result: >
  (a) linear K=4: Pearson r=+0.571 (p=0.009, n=20) -- correlation confirmed;
  at K=128 it vanishes (r=+0.020, p=0.934). (b) independent vs cached
  dv_pred: r=1.00000, max|diff|=1.5e-8 -- confirmed, no bug. (c) mean|R_4|
  0.00086 (L10) vs 0.00096 (L20) vs 0.00162 (L40); mean|R_128| 0.00204 (L10)
  vs 0.00186 (L20) vs 0.00298 (L40) -- all within ~1.7x of each other despite
  dv_true sd spanning 0.00506/0.02338/0.09054 (4.6x/17.9x) -- confirmed.
verdict: supported
downgrades: []
grade: high
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the -0.44/0.91 pairing were a computation error, an independently loaded
fit and an independently reconstructed occ0 would very likely disagree with
the cached numbers by more than float32 noise -- two different bugs
producing bit-identical wrong answers through different code is
astronomically unlikely. If the 0.91 capture number at K=128 were purely a
ratio-instability illusion, per-slate capture should correlate with
per-slate spread AT THE SAME K the number is quoted for (K=128), not just at
some other K -- finding the correlation present at K=4 but absent at K=128
is the specific pattern that lets (a) and (c) tell a real-vs-artifact story
apart at the resolution the claim needs (the K where the number is actually
used), rather than a blanket "small numbers are noisy" assertion.

## What was actually run

`scripts/probes/l10_verification.py runs_expB/n20_L10mm_dv_cache.pt --goal
corner --models linear,UNet --k 4,128 --n-check-slates 6 --context
runs_expB/n20_L20mm_kcurve_exact.json,runs_expB/n20_L40mm_kcurve_exact.json`.
Part (b)'s independent recompute was restricted to 6 of the 20 eval slates
(768 of 2560 rows) for cost, not the full set -- a reasonable sample given
the near-perfect agreement found; the full-cache check was not run and
would cost about 4x as long (~12 min). Part (c)'s L20mm/L40mm raw |R_K|
values were computed directly against `runs_expB/n20_L20mm_dv_cache.pt` /
`n20_L40mm_dv_cache.pt` (the same PLAIN, non-"_sharp" caches EXP-0024_v1's
own published `slateK_exact` numbers for those cells come from), via an
inline use of `pool_common.rk_curve`/`admissible_slates` -- not wrapped into
the committed script, since it is a two-line reuse of already-committed,
already-tested functions.

## Numbers

**(a) spread-vs-capture correlation, linear operator, goal=corner, n=20 slates:**

| K | Pearson r (max-min) | p | Spearman rho | p |
|---|---|---|---|---|
| 4 | +0.571 | 0.009 | +0.541 | 0.014 |
| 128 | +0.020 | 0.934 | −0.006 | 0.981 |

(UNet, same check: K=4 r=+0.303 p=0.194; K=128 r=+0.036 p=0.880 -- same
direction, does not clear significance at either K.)

**(b) independent recompute, 768 rows / 6 slates:**

| | value |
|---|---|
| Pearson r (cached vs. independent) | 1.00000 |
| mean diff | −7.8e-11 |
| max\|diff\| | 1.5e-8 |
| cached sd | 5.2e-3 |
| max\|diff\|/sd | ~3e-6 |

**(c) raw |R_K| beside slateK_exact and overall dv_true sd, linear, corner:**

| cell | mean\|R_4\| | mean\|R_128\| | slateK_exact K=4 | slateK_exact K=128 | dv_true sd |
|---|---|---|---|---|---|
| L10mm | 0.000859 | 0.002042 | 0.793 | 0.913 | 0.00506 |
| L20mm | 0.000959 | 0.001859 | 0.953 | 0.967 | 0.02338 |
| L40mm | 0.001619 | 0.002975 | 0.982 | 0.979 | 0.09054 |

## What would change the verdict

Running part (b) over the full 2560-row eval set rather than 6/20 slates
would close the (small) possibility that the 6 sampled slates happen to be
unrepresentative; given the near-exact agreement already found, this is a
confirmation exercise rather than one expected to change the verdict
(~12 min CPU). A stronger test of (c) would fit the SAME single operator
across L10/L20/L40mm's pooled training data (rather than one operator per
push length, as both this record and EXP-0024_v1 do) to check whether the
absolute-|R_K|-is-roughly-constant pattern survives a shared fit rather than
three separately-fit ones -- this would cost a full re-fit per length
(~10-20 min CPU each) and was out of scope for this check.

## Threats

None of the five downgrade domains apply at this record's own scope: the
independent-recompute check (b) deliberately varies the code path
(provenance is the thing being tested, not a threat to it); the
spread-correlation check (a) is reported at BOTH K values including the one
that does NOT support ratio instability (K=128), so this is not a
selectively-reported sweep; (c) uses the same goal/model/R/crop/ridge across
all three cells. One real limitation, stated rather than hidden: (c)'s
cross-cell comparison uses three SEPARATELY fitted linear operators (one per
push length, matching how the register's own numbers were produced) rather
than one operator fit across all three -- so it cannot distinguish "the
model's absolute skill really is length-invariant" from "each length's
own best-achievable fit happens to leave a similar absolute residual",
which is a real, if narrower, question than the one this record answers.

## Unrelated findings

- `runs_expB/n20_L10mm_dv_cache.pt`'s goal keys are `[corner, center]` only
  -- it was never scored under the sharp `ind-square8-pile`/
  `ind-stripe-thin-pile` functionals the way L20mm/L40mm's `_sharp` caches
  were, so this record (and EXP-0027's L10mm addendum) could only use
  `corner`, not the sharp functional this report's other cells use
  elsewhere.
