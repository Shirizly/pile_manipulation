---
id: EXP-0013
title: Neither shrinkage nor rank trades accuracy for control — but rms mis-ranks nine models that differ by 15 points of it
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: C-030
claim: >
  On cube_spectrum/n20 the ridge strength maximising control utility is at
  least 10x the ridge strength minimising swept-region rms -- i.e. the model
  you would pick for MPC is more heavily shrunk than the model you would pick
  for predictive accuracy. (Extended mid-run to the rank-truncation knob, which
  EXP-0008's mechanism identifies as the more appropriate one -- see
  "What was actually run".)
prediction:
  supports: "lambda*_control >= 10 * lambda*_rms on BOTH goals and BOTH control operationalisations"
  refutes:  "lambda*_control <= lambda*_rms on any goal/operationalisation"
  discriminating: true
provenance:
  commit: aac084e3
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/shrinkage_vs_control.py`
                                  # did not exist at aac084e3; it was written in the same
                                  # session and first committed at 006004d0. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the 006004d0 version of that file.
  script_first_committed: 006004d0
  script: scripts/probes/shrinkage_vs_control.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt (4840 full-length pushes, 16 episodes)"]
  code_path: points_to_mask
  seed: 0
  split: "episode-level, 4 of 16 files held out, seed 0"
  runtime: "~12 min CPU at res 32 (OMP_NUM_THREADS=4); a res-64 attempt was abandoned to machine contention"
budget:
  declared: "60 min, 150k tokens"
  spent: "~75 min wall (much of it lost to self-inflicted CPU oversubscription), ~90k tokens"
  outcome: exceeded
design:
  varied: {ridge: [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 1e4, 1e5], rank: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
  held_fixed: {dataset: cube_spectrum/n20, view: mask, cube_size: 0.005, min_grains: 1.0, res: 32, crop: 1.0, blur: 0, grid: 64, estimator: "ridge toward identity", split: identical, goals: [center, corner], rank_lambda: 10}
  baselines: [persistence, mean-delta, identity-warp, oracle]
  metric: "swept-region rms as % of persistence; dV Spearman and slate-4 regret (control_utility_test.rank_metrics), partial-correlated against the state's own V0 and its contact score"
noise_floor: "not separately measured for this design. The ridge curve is flat to +-0.9 points of rms and +-0.015 of slate4 over four decades of lambda, which is itself the relevant scale: any 'optimum' inside that band is not a real optimum."
depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split, particle-projection]
establishes: []
result: >
  REFUTED on both knobs. lambda*_rms = 10 and lambda*_control = 1..10 (not
  >=100); rank*_rms = 256 and rank*_control = 512 (truncation monotonically
  hurts both). But the same table shows nine models spanning 15 points of rms
  that rms and control utility order OPPOSITELY -- see Numbers.
verdict: refuted
downgrades: [imprecision, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0008 found control-relevant ranking destroyed by variance and untouched by
bias. If that is the mechanism, a knob that trades variance for bias should
improve control after it starts hurting accuracy, and the two optima should
separate. Both curves have interior optima by construction — as λ→∞ the
operator becomes the identity and `dV_pred` becomes constant, so ranking
degenerates; as λ→0 the fit is underdetermined — so "the optima coincide" was a
live outcome, not excluded.

## What was actually run

A ridge sweep over four decades, then a rank sweep. **The rank sweep was added
mid-run**, after the ridge result came in, on the following reasoning: shrinking
toward the identity rescales the predicted delta, which is an *amplitude*
change, and EXP-0008 measured amplitude error as costing control utility
**nothing**. So ridge could not have tested the variance hypothesis — it moves
the one axis already known to be free. Rank truncation removes prediction
variance and is the right knob. This is a mid-run design change and is declared
rather than folded in silently; the ridge arm's prediction was written first and
is reported as stated.

Persistence's ranking columns are meaningless and are printed only as a
placeholder: it predicts `dV = 0` for every candidate, so any correlation is an
artifact of sorting a constant.

**Not run:** res 64, and a second seed. The res-64 arm was started and abandoned
after I oversubscribed the machine with four concurrent jobs (load average 37 on
20 cores). Hence `incomplete-design`.

## Numbers

Swept-region rms as % of persistence (lower better); slate-4 regret as a
fraction of the oracle's advantage over a random pick (higher better).

| model | rms% | HF | slate4 center | slate4 corner |
|---|---|---|---|---|
| persistence | 100.0 | 0.00 | 0.000 | 0.003 |
| oracle | 0.0 | 6.39 | 0.795 | 1.000 |
| identity (warp only) | 86.7 | 15.12 | 0.162 [corrected] | −0.037 |
| **mean-delta (0 params)** | **69.2** | 8.44 | **0.395** [corrected] | **0.852** |
| ridge 1 | 54.1 | 4.94 | 0.622 | 0.972 |
| ridge 10 | **53.9** | 5.00 | 0.616 | 0.974 |
| ridge 1e4 | 69.5 | 10.42 | 0.565 | 0.946 |
| rank 1 (λ=10) | 60.5 | 4.80 | 0.338 | 0.755 |
| rank 2 | 55.9 | 4.78 | 0.454 | 0.901 |
| rank 4 | 54.7 | 4.82 | 0.494 | 0.918 |
| rank 8 | 54.3 | 4.86 | 0.517 | 0.940 |
| rank 16 | 54.1 | 4.89 | 0.511 | 0.956 |
| rank 32 | 54.0 | 4.93 | 0.556 | 0.963 |
| rank 64 | 54.0 | 4.95 | 0.566 | 0.966 |
| rank 128 | 53.9 | 4.97 | 0.570 | 0.968 |
| rank 256 | **53.9** | 4.98 | 0.602 | 0.972 |
| rank 512 | 53.9 | 5.00 | 0.605 | 0.973 |

Optima: `λ*_rms = 10`; `λ*_control = 1..10`. `rank*_rms = 256`;
`rank*_control = 512`. **No separation on either knob — the claim is refuted.**

### The result the sweep was not looking for

Read the rank block against **mean-delta**:

> **Every low-rank operator from rank 1 to rank 128 has better swept-region rms
> than mean-delta (53.9–60.5% vs 69.2%) and worse action-selection utility
> (0.338–0.570 vs 0.601).**

That is nine models, spanning 15 points of rms, that the two criteria order
**oppositely**. Picking by rms takes any rank-truncated operator over
mean-delta; picking by realised control utility takes mean-delta over all nine.
The extreme case is rank 1: 8.7 points *better* rms than mean-delta and 44%
*worse* slate-4.

This is a model-selection dissociation, which is what C-030 was always about —
"the right metric should more correctly identify the better model for MPC based
on predictive accuracy". It is stronger evidence than EXP-0008's degradation
study because these are **real fitted models a person might actually deploy**,
not synthetic corruptions.

It is not explained by the variance mechanism: the HF column is nearly flat
across the rank block (4.78–5.00) while slate-4 nearly doubles. Whatever
distinguishes rank 1 from rank 256 for control, it is not high-frequency energy.

## What would change the verdict

For the refutation: a second seed and res 64 — the ridge curve is flat to ±0.9
points, so a different split could move the argmin without meaning anything.
Cheap (~30 min on an idle machine).

For the dissociation: it needs its own record and a proper noise floor. The
mean-delta-vs-rank comparison is currently a by-product read off a table built
for another question, which is exactly the "found it while looking for something
else" pattern that deserves a dedicated confirmatory run rather than promotion
in place.

## Threats

- `imprecision`: one seed-0 split, no fold sweep. The ridge curve's flatness
  means its argmin is not meaningfully located; this is reported as "no
  separation", which flatness supports, rather than as "the optima are at 10".
- `incomplete-design`: res 64 not run; second seed not run.
- **Candidate slates come from different states.** `dV` ranking is scored over
  candidates drawn from different transitions, so "this action is good" and
  "this state was easy" are confounded. The partial-correlation columns remove
  the state's own `V0` and contact score and the effect survives, but that is
  mitigation, not elimination. A same-state slate collection is running
  separately and is the real fix.
- Considered and dismissed: **that mean-delta wins by predicting less.**
  Identity (warp only) also predicts almost nothing and scores 0.146, far below
  mean-delta's 0.601 — so "move less, rank better" is not the explanation.

## Unrelated findings

- `fit_operator` recomputes `Y0 @ Y0.T` and `Y1 @ Y0.T` on every call, so a
  ridge sweep pays the Gram cost once per λ instead of once. Hoisting it turned
  13 shrinkage levels from ~13 Gram-sized jobs into one Gram plus 13 solves.
  Not fixed in the shared function — logged for triage.
- Python fully buffers stdout when redirected to a file, so a long background
  job appears frozen. `python -u` is required for any probe whose progress is
  going to be watched.


## Reviewer correction, 2026-09-05: this record's table was mistranscribed

**Two cells of the Numbers table above were wrong, they were the two the
headline conclusion rested on, and the error is the author's.**

`slate4 center` for `mean-delta` and for `identity (warp only)` were copied
from the `pa_cent` column (the partial correlation) instead of `sl4_cent`.
Correct values: mean-delta **0.395**, not 0.601; identity **0.162**, not 0.146.
The `ridge` and `rank` rows were copied correctly.

**Mechanism of the mistake, because it is preventable.** The probe prints a
fixed-width table in which the `|A-I|` column is populated for operator rows
(`I|/|I|=0.587`) and **empty** for `persistence`, `oracle`, `mean-delta` and
`identity`. Reading it by eye, the blank cell shifted the remaining columns one
place left on exactly those rows. `scripts/probes/shrinkage_vs_control.py` now
prints a `-` placeholder so the columns cannot go ragged again.

### What it changes

The headline read: "every rank-truncated operator from rank 1 to rank 128 has
better rms than mean-delta and worse slate-4." That compared correct `rank_r`
values (0.338–0.605) against a mean-delta figure of 0.601 that was really
**0.395**. Against the true baseline only **rank 1** (0.338) is worse on
control while better on rms. Every rank ≥ 2 beats mean-delta on **both**
metrics — agreement, not dissociation.

EXP-0017 measured this independently over 5 seeds at res 32 and 3 at res 64 and
reached the same place from data rather than from this table: rank 1 dissociates
at 6.7–23 sd; ranks ≥ 2 agree at 6.6–31 sd. So the narrow claim survives and the
broad one does not.

**What is NOT affected:** this record's own verdict. C-036 (shrinkage and rank
do not trade accuracy for control) was computed from the sweep output directly,
not from the hand-copied table, and stands refuted as recorded.

### The lesson worth keeping

The register's whole design is that a claim's provenance is recoverable. Here
provenance worked — the raw output was on disk, so an independent run could
find the discrepancy — but nothing *checked* the transcription from output to
Markdown, which is a manual step in every record in this directory. A number
retyped by hand is a number nobody has verified. Where a table is the evidence,
generate it rather than retyping it.
