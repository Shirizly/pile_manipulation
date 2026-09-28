# EXP-0026 — how many states does it take to rank models? (Phase A of the benchmark line)

Written 2026-09-23, BEFORE any of the analyses below were run (plan gate).

## Why

EXP-0023's power check compared between-arm sd with between-STATE sd. The
between-state sd is a population property of the states: it does not shrink
with N, so "add states until it passes" (TODO M2) may never pass, and it is
not the noise a paired design is exposed to (every arm sees the same states, so
state difficulty cancels in an arm-vs-arm difference). The relevant noise is
the arm x state interaction (EXP-0023 residual sd 0.0183). Separately, every
slateN number in the cross-model reference table rests on 20-21 slates per
corpus with no interval on any margin.

## Claims (each scoped; each could be false)

C1 (EXP-0023 re-analysis, `corner`/`lyapunov`, 10 DS-0001 states, 6 arms,
    100-action pool, 120 Adam steps). Under a paired sign-flip test with Holm
    correction over the 15 arm pairs, no arm pair is resolved on
    `gradient_gain`; and the states needed to resolve a 0.01 `gradient_gain`
    difference at 80% power (alpha 0.05, paired, per-pair sd) is >= 30.
    MODE: exploratory -- the arm means were already seen in EXP-0023.
C2 (slateN, `Baselines/common/eval_report.py` harness, L20mm / L40mm /
    randlen_test, 3 goals x 3 value fns, goal-averaged per slate). The
    paired-bootstrap 95% CI (resampling slates) of the goal-averaged
    lyapunov slateN difference between `linear_switched_res32` and
    `nfd_randlen` includes 0 on at least 2 of 3 corpora.
C3 (same harness). Across all model pairs on these corpora, pairs whose
    |delta slateN| < 0.03 are unresolved (95% CI includes 0) in >= 80% of
    cases -- i.e. the 20-slate harness cannot order margins below ~0.03.

Predictions: C1 supports if no Holm-significant pair AND N(0.01) >= 30; refutes
otherwise. C2 supports / refutes as stated. C3 supports / refutes on the 80%
threshold. All are `discriminating: true`: each has a plausible opposite outcome
(the arm x state interaction may be small relative to arm gaps; slate-to-slate
agreement between models may be high enough that paired CIs are tight).

## Design

- Unit of replication: the STATE (slate). Goals and value fns on one slate are
  not independent replicates; they are averaged within slate, and their
  cross-goal correlation is reported so the effective gain from goals is known.
- Paired statistics only: per-pair difference per state; bootstrap over states
  (10k), exact / Monte-Carlo sign-flip permutation, Holm over pairs; Friedman
  test for the global arm effect.
- Required N: n = ((z_{1-a/2} + z_{0.8}) * sd_d / delta)^2 with sd_d the
  per-pair sd of paired differences, reported as median over pairs, for
  delta in {0.005, 0.01, 0.02} (dv units) / {0.01, 0.02, 0.05} (slateN units),
  at alpha 0.05 and at Bonferroni alpha for all pairs.
- Do-nothing baseline: `random` (slateN expectation exactly 0), `persistence`
  reported but degenerate (METRICS.md).
- A3 (secondary, DS-0001 cache, 20 slates x 1000 candidates, cached nfd /
  visual-switched / descriptor predictions): subsample K=100 pools, decompose
  the variance of per-pool capture into between-slate and within-slate
  (pool-sampling) parts -> does drawing more pools per state substitute for
  more states?

Fixed: all checkpoints as registered in `eval_report.MODELS` /
`simple_mpc.adapters.OCC_ADAPTERS`; the harness's goal set and slate set.

## Cost and the cheapest invalidating check

~15 GPU-min for eval_report over ~10 models x 3 corpora (a 4-model run took
5 min). Cheapest check: re-produce the stored `cross_corpus_report.json`
goal-averaged numbers for `nfd_randlen` from the new per-slate output (must
match to 1e-6) before any power statistic is computed.

## The result that would most embarrass this

Paired CIs are tight (models agree slate-by-slate on which slates are hard),
so 20 slates were enough all along and the "power problem" was only the wrong
test. The design catches it: that outcome is C2/C3 refuting.

## Addendum (written after A1-A3 and A2's goal-ICC ~ 0, BEFORE A4 ran)

A2 found the per-slate paired difference between two models essentially
uncorrelated across the harness's 3 goals (median ICC 0.00-0.10), i.e. goals
behave as independent replicates. Goals cost no simulation; states do.

C4 (A4, same 3 corpora and 12 models, lyapunov and mass_in_region, 24 random
    goals = random disks/rectangles at random positions). Averaging per-slate
    slateN over G = 24 goals instead of G = 3 shrinks the median per-pair sd of
    the paired difference by >= 2x (independent replicates would give
    sqrt(8) = 2.8x). Refutes if the shrink is < 1.5x (goals then share most of
    their information and "more goals" does not substitute for states).
    Also report the ICC over the 24 goals and required slates at G = 24.
