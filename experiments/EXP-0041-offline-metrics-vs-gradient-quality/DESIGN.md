# EXP-0041 -- do offline metrics predict a model's gradient-descent quality?

Written retrospectively 2026-09-25 from EXPERIMENT.md (frontmatter only; there is no body, code/ or
runs/ directory) and EXP-0039 DESIGN.md Addendum 2, not before the analysis. No pre-run plan exists
in TODO.md; no command-ledger entry was found.

## Why
EXP-0037 measured each model's value as a gradient-descent objective (gain_capture, grad_capture);
EXP-0038 tabulated offline metrics for the same models; EXP-0040 added roughness. Question: within
one model family, which offline metric orders models the way their gradient-descent quality does --
in particular, does image `accuracy` do better than ranking skill `slateN`?

## Design
- No new data or simulation: an inline analysis (script not saved) reading EXP-0037
  results/analysis.json, EXP-0038 results/metric_table.json and EXP-0040 results/smoothness.json.
- Targets: EXP-0037 gain_capture and grad_capture per arm (DS-0006 gradient benchmark).
- Predictors: `accuracy` (METRICS.md; measured on randlen_test from EXP-0028, i.e. different states
  from DS-0006), `slateN`, spearman, optimism, top8 (EXP-0038 table), roughness (EXP-0040).
- Statistic: scipy spearmanr across arms. n = 5 NFD arms for accuracy (the arms with harness
  accuracy); n = 7 arms for the other metrics.
- Baselines: not applicable (a correlation across models; no do-nothing arm).
- Budget: not recorded.

## Predictions
Exploratory: no pre-registered prediction. The result was then pre-registered as hypothesis H6 in
EXP-0039 (DESIGN.md Addendum 2): under GD, Spearman(accuracy) >= Spearman(slateN) against
closed-loop improvement; under the rank planner the reverse.

## Deviations
- No plan was recorded, so deviations cannot be assessed.
- The analysis code was run inline and not saved; the exact computation is recoverable only from
  the three input files named above.
- Predictor sets differ in n (5 vs 7 arms), so the accuracy and slateN correlations are not over the
  same models.
