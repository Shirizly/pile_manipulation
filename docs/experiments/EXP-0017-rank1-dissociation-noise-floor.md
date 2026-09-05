---
id: EXP-0017
title: >
  The rms/control dissociation survives a noise floor only at rank 1 -- ranks
  2..128 beat mean-delta on BOTH metrics, and EXP-0013's center-column mean-delta
  number was mistranscribed
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-030

claim: >
  C-037's dissociation ("every rank-truncated operator 1..128 beats mean-delta
  on swept-region rms and loses to it on slate-4 control utility") survives a
  measured seed-to-seed noise floor, on the goal that carries real dV signal
  (corner), at both res 32 and res 64.

prediction:
  supports: >
    For most/all ranks in {1,2,4,8,16,32,64,128}, the paired difference
    (mean-delta slate4 minus rank_r slate4) exceeds 2x the seed-to-seed sd of
    that paired difference AND the paired rms difference (mean-delta rms minus
    rank_r rms, favoring rank_r) also exceeds 2x its own paired sd, on corner,
    at res 32 -- and the pattern replicates at res 64.
  refutes: >
    The dissociation (both halves significant, same direction as C-037) holds
    at fewer than half the tested ranks, or reverses sign (rank_r better on
    BOTH metrics) for most ranks >= 2.
  discriminating: true

provenance:
  commit: dbf21ba2
  script: scripts/probes/shrinkage_vs_control.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt (4840 full-length pushes, 16 episodes)"]
  code_path: points_to_mask
  seed: "0,1,2,3,4 (res 32); 0,1,2 (res 64)"
  split: "episode-level, 4 of 16 files held out, seeds 0-4"
  runtime: "~5-8s/seed at res 32, ~17-27s/seed at res 64 (OMP_NUM_THREADS=4, --lambdas 10 to skip the ridge sweep -- ranks use a script-hardcoded lambda=10 regardless of --lambdas)"

budget:
  declared: "70 min, 160k tokens"
  spent: "~45 min, ~55k tokens"
  outcome: within

design:
  varied: {seed: [0, 1, 2, 3, 4], res: [32, 64], goals: [center, corner]}
  held_fixed: {dataset: cube_spectrum/n20, view: mask, cube_size: 0.005, min_grains: 1.0, crop: 1.0, blur: 0, grid: 64, ranks: [1,2,4,8,16,32,64,128,256,512], rank_lambda: 10, estimator: "ridge toward identity, rank-truncated", split: episode-level}
  baselines: [persistence, mean-delta, identity-warp, oracle]
  metric: >
    swept-region rms as % of persistence; slate-4 regret AND its
    state/contact partial-correlation variant (control_utility_test.rank_metrics),
    both paired against mean-delta WITHIN each seed, then averaged and sd'd
    across seeds (paired design, not independent-sample sd)

noise_floor: >
  Seed-to-seed sd of the paired (rank_r - mean-delta) difference, per rank, per
  goal, per resolution -- see Numbers. Typical magnitude on corner (the
  informative goal): d_sl4 sd = 0.005-0.017 (raw slate4), 0.004-0.009
  (partial-corr); d_rms sd = 0.05-0.15 points. This is 5-40x smaller than the
  paired means, i.e. most paired differences ARE resolved by this design --
  the question is which direction they resolve in.

depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, particle-projection]
establishes: []

result: >
  REFUTED as broadly stated, SUPPORTED in a narrower form. Only rank=1
  dissociates from mean-delta on corner (rms better by 8.8-12.2 points at a
  paired sd of ~0.1-0.15; slate4 worse by 0.088-0.096 at a paired sd of
  0.007-0.009 -- ratios -6.7 to -23, far past the +-2sd bar). Ranks 2-512 do
  NOT dissociate: they beat mean-delta on BOTH rms and slate4/partial-corr,
  with ratios of +6.6 to +31 sd, on both res 32 and res 64. The center goal is
  weak-signal (68% of transitions have dV_true exactly 0) and its rank-1..128
  numbers in EXP-0013's OWN Markdown table used the wrong mean-delta baseline
  (0.601 = partial-corr, not 0.395 = raw slate4) -- see "What was actually run".
  With the correct baseline, center shows the same pattern as corner: only
  rank=1 dissociates.

verdict: refuted
downgrades: [indirectness]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the dissociation is real (not noise), the paired (rank_r - mean-delta)
difference in slate4 should be reliably negative (rank worse) across seeds at
the SAME ranks where the paired rms difference is reliably positive (rank
better) -- both by more than a couple of seed-sd. If it is a single-split
fluke, the paired differences should be small relative to their own
seed-to-seed sd, or change sign under resplitting. Running 5 (res32) / 3
(res64) independent episode-splits and computing the paired sd directly
answers which.

## What was actually run

A trimmed version of the EXP-0013 sweep: `--lambdas 10` (the rank block uses a
script-hardcoded `lambda=10` for the pre-truncation ridge fit regardless of
`--lambdas`, so the 13-point ridge sweep is irrelevant to this claim and
dropped to save time -- confirmed by re-running seed 0 and reproducing
EXP-0013's table exactly). 5 seeds at res 32, 3 seeds at res 64 (all three
matched the res-32 pattern; stopped at 3 because the pattern was already
unambiguous and budget was better spent on the `center`-degeneracy check
below). `dv_true` diagnostics (mean, sd, fraction helpful/harmful/exactly-zero)
were computed once on the full 4840-transition set, not per split, since
degeneracy is a property of the goal geometry against this dataset, not of the
train/val split.

**Deviation to disclose:** the res-32/seed-0 feasibility timing check (run to
answer "is a 5-seed sweep affordable") used the real `--lambdas 10` config and
therefore returned full outcome numbers, not just a runtime. The prediction
above was written in threshold-relative terms (multiples of the seed-sd)
before that run and does not reference any specific number from it, so the
cell was kept in the final 5-seed analysis rather than discarded. Disclosed
per the skill's pilot rule rather than silently retained.

**A finding that changes the reading of EXP-0013's own table, found while
reproducing it.** `control_utility_test.rank_metrics` returns `(pear, spear,
sign, slate, part)`; the script labels the printed columns `sp_ / sl4_ / pa_`
for spearman / slate4 / partial-corr, in that order, for each goal. Comparing
my reproduced seed-0 res-32 run cell-by-cell against EXP-0013's Markdown
table: **the `oracle`, `ridge *`, `rank *` and `persistence` rows all correctly
show `sl4_cent` in the "slate4 center" column, but the `mean-delta` and
`identity` rows show `pa_cent` instead** -- 0.601 and 0.146 are the
partial-correlation values, not slate4 (true `sl4_cent` = 0.395 and 0.162
respectively; verified against my own script printout, reproduced exactly).
The `corner` column is unaffected for every row (EXP-0013's corner numbers all
check out against `sl4_corn`). This means the specific comparison "rank 1..128
slate4-center (0.338-0.570) vs mean-delta (0.601)" that anchors C-037's
headline compared apples (`sl4_cent` for the operators) to oranges
(`pa_cent` for the baseline) -- the correct comparison is rank 1..128
(0.338-0.605, same values) **vs mean-delta 0.395**, under which ranks 2-512
are already ABOVE mean-delta (only rank 1 is below). This is not corrected in
EXP-0013 itself (out of this record's scope), but it explains why the
1..128-wide dissociation looked stronger on `center` than it turns out to be
on 5 seeds of direct measurement: part of the appearance was a transcription
error, not signal. It does not touch EXP-0013's verdict on C-036 (ridge/rank
optima), which reads columns straight from the data dict, not the hand-copied
table.

## Numbers

### Goal degeneracy check (full dataset, 4840 transitions, before any split)

| goal | dV_true mean | dV_true sd | frac dV<0 (helpful) | frac dV>0 (harmful) | frac dV==0 |
|---|---|---|---|---|---|
| center | +0.0056 | 0.0212 | 0.069 | 0.253 | **0.677** |
| corner | +0.0151 | 0.0676 | 0.423 | 0.575 | 0.002 |

`center` is heavily but not totally degenerate on this cross-state dataset
(68% exact zeros, sd 3.2x smaller than corner) -- weaker than EXP-0012's
same-state 96%-degenerate result but the same direction and mechanism (C-040).
**Conclusion rests on `corner`; `center` is reported for completeness only.**

### Paired (rank_r vs mean-delta) differences across seeds, res 32 (n=5 seeds), goal=corner

Positive `d_rms` = rank has lower (better) rms%. Positive `d_sl4` = rank has
higher (better) slate4/partial-corr than mean-delta. `ratio` = mean/sd of the
paired difference (>2 or <-2 in magnitude = resolved by this noise floor).

| rank | rms% mean (sd) | slate4 mean (sd) | d_rms mean (sd) | d_sl4 mean (sd) | ratio (slate4) | part-corr d mean (sd) | ratio (partial-corr) |
|---|---|---|---|---|---|---|---|
| mean-delta | 69.24 (0.10) | 0.825 (0.014) | -- | -- | -- | 0.873 (0.006) | -- |
| **1** | 60.48 (0.10) | 0.732 (0.017) | +8.76 (0.15) | **-0.092 (0.014)** | **-6.69** | -0.096 (0.007) | **-13.11** |
| 2 | 55.82 (0.12) | 0.881 (0.010) | +13.42 (0.12) | +0.056 (0.006) | +9.75 | +0.033 (0.005) | +6.59 |
| 4 | 54.68 (0.15) | 0.909 (0.009) | +14.56 (0.14) | +0.084 (0.010) | +8.27 | +0.059 (0.006) | +10.37 |
| 8 | 54.28 (0.15) | 0.926 (0.011) | +14.96 (0.14) | +0.101 (0.009) | +11.72 | +0.076 (0.004) | +20.43 |
| 16 | 54.08 (0.12) | 0.943 (0.010) | +15.16 (0.08) | +0.118 (0.009) | +13.68 | +0.087 (0.006) | +15.40 |
| 32 | 53.96 (0.15) | 0.953 (0.007) | +15.28 (0.12) | +0.128 (0.009) | +14.03 | +0.097 (0.005) | +17.94 |
| 64 | 53.96 (0.15) | 0.955 (0.007) | +15.28 (0.12) | +0.130 (0.009) | +14.50 | +0.100 (0.005) | +18.75 |
| 128 | 53.86 (0.15) | 0.957 (0.006) | +15.38 (0.12) | +0.133 (0.009) | +15.15 | +0.102 (0.006) | +18.24 |
| 256 | 53.86 (0.15) | 0.963 (0.006) | +15.38 (0.12) | +0.138 (0.010) | +14.18 | +0.103 (0.005) | +19.31 |
| 512 | 53.86 (0.15) | 0.964 (0.006) | +15.38 (0.12) | +0.139 (0.010) | +14.74 | +0.104 (0.006) | +18.42 |

**Only rank 1 dissociates** (d_rms and d_sl4 opposite in sign, both |ratio|>2).
Every rank >= 2 has both d_rms > 0 AND d_sl4 > 0 -- rank beats mean-delta on
BOTH axes, resolved at 6.6-19.3 sd.

### Same table, res 64 (n=3 seeds), goal=corner

| rank | rms% mean (sd) | slate4 mean (sd) | d_rms mean (sd) | d_sl4 mean (sd) | ratio (slate4) | part-corr d mean (sd) | ratio (partial-corr) |
|---|---|---|---|---|---|---|---|
| mean-delta | 72.60 (0.08) | 0.823 (0.014) | -- | -- | -- | 0.865 (0.008) | -- |
| **1** | 60.37 (0.05) | 0.736 (0.014) | +12.23 (0.09) | **-0.087 (0.004)** | **-23.25** | -0.088 (0.009) | **-10.32** |
| 2 | 55.60 (0.00) | 0.882 (0.009) | +17.00 (0.08) | +0.059 (0.006) | +9.57 | +0.039 (0.005) | +7.49 |
| 4 | 54.47 (0.05) | 0.910 (0.012) | +18.13 (0.05) | +0.087 (0.007) | +11.70 | +0.067 (0.005) | +14.07 |
| 8 | 54.07 (0.09) | 0.930 (0.011) | +18.53 (0.05) | +0.107 (0.008) | +14.30 | +0.081 (0.003) | +30.99 |
| 16 | 53.97 (0.09) | 0.939 (0.010) | +18.63 (0.05) | +0.116 (0.009) | +13.37 | +0.090 (0.006) | +14.11 |
| 32 | 53.90 (0.08) | 0.953 (0.011) | +18.70 (0.00) | +0.130 (0.005) | +24.28 | +0.103 (0.007) | +14.88 |
| 128 | 54.10 (0.08) | 0.953 (0.008) | +18.50 (0.00) | +0.130 (0.007) | +19.60 | +0.106 (0.007) | +16.24 |
| 512 | 54.17 (0.09) | 0.959 (0.007) | +18.43 (0.05) | +0.136 (0.008) | +16.91 | +0.109 (0.007) | +15.62 |

**Same pattern at res 64**: only rank 1 dissociates, ranks >=2 win on both axes,
resolved even more sharply than at res 32 (e.g. rank-1 slate4 ratio -23.3 vs
-6.7). The ordering is NOT a res-32 artifact.

### Goal=center, res 32 (n=5 seeds) -- for completeness, weak-signal caveat applies

| rank | rms% mean (sd) | slate4 mean (sd) | d_rms mean (sd) | d_sl4 mean (sd) | ratio (slate4) |
|---|---|---|---|---|---|
| mean-delta | 69.24 (0.10) | 0.420 (0.062) | -- | -- | -- |
| 1 | 60.48 (0.10) | 0.364 (0.075) | +8.76 (0.15) | -0.056 (0.021) | -2.69 |
| 2 | 55.82 (0.12) | 0.469 (0.050) | +13.42 (0.12) | +0.049 (0.030) | +1.67 |
| 128 | 53.86 (0.15) | 0.570 (0.023) | +15.38 (0.12) | +0.150 (0.044) | +3.42 |
| 512 | 53.86 (0.15) | 0.612 (0.030) | +15.38 (0.12) | +0.192 (0.034) | +5.71 |

Rank 1 again dissociates (barely past the +-2sd bar, ratio -2.69, on a metric
that is 68% exact zeros); ranks 2+ mostly agree with mean-delta's improving
rms, same qualitative story as corner but noisier, as expected from the
degeneracy check above.

## What would change the verdict

More seeds at rank 1 specifically (n=5 is thin for a ratio of -6.7, though it
would need to move a lot to cross zero). A second dataset (`cube_spectrum/n30`)
would test whether rank-1's dissociation is a property of rank-1 truncation in
general or an artifact of this 16-episode collection. Cost: ~1 min for either,
data-permitting.

## Threats

- `indirectness`: slate-4 regret and its partial-corr variant are still
  proxies for realized MPC dV under an executed plan, not realized control
  performance itself -- same threat as EXP-0008/EXP-0013, unresolved by this
  record.
- **Candidate slates are drawn from different states** (standing confound,
  cannot be removed here). The partial-correlation column, which regresses out
  the state's own V0 and contact score, shows the IDENTICAL qualitative
  pattern (rank 1 dissociates, ranks >=2 agree) as the raw slate4 column, at
  similar or larger effect sizes -- so the confound is not manufacturing this
  result. EXP-0012 measured this same confound roughly doubling noise-related
  ranking damage on a same-state dataset (84% -> 42% relative Spearman drop);
  cited here as the calibration, not re-measured.
- `center` goal: 68% of transitions have `dV_true` exactly 0 here (vs
  EXP-0012's 96% on a same-state collection) -- weak-signal, not fully
  degenerate, but its numbers are noisier and its rank-1..128 dissociation in
  EXP-0013's OWN table used a mistranscribed mean-delta baseline (see "What
  was actually run"). Conclusion rests on `corner`.
- Res 64 only has 3 seeds (vs 5 at res 32) -- stopped early because the
  pattern was already unambiguous at n=3 and budget was reserved for the
  degeneracy check; take as `incomplete-design` for the res-64 arm specifically
  if a stricter n=5 match is wanted, though the effect sizes here (ratios of
  -10 to -30) leave little room for 2 more seeds to change the sign.

## Unrelated findings

- `scripts/probes/shrinkage_vs_control.py`'s rank-truncation block uses a
  script-hardcoded `lambda=10` for the pre-truncation ridge fit, independent
  of whatever `--lambdas` is passed. This is intentional (documented in a
  comment) but easy to miss when reading the CLI help, which implies `--ranks`
  and `--lambdas` are independent orthogonal sweeps.
- The gram-matrix hoisting fix noted in EXP-0013's "Unrelated findings" is
  still not applied to the shared `fit_operator` function (only inlined in
  this script) -- still logged there, not re-logged here as a new item.

