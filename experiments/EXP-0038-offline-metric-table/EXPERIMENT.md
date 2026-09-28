---
id: EXP-0038
title: >
  Offline ranking metrics are nearly redundant across models (slateN, top-1 regret,
  pool Spearman / Pearson, top-8 capture: Kendall 0.89-1.00), optimism is a separate
  axis (0.39-0.50); the linear model ranks the whole pool as well as NFD (Spearman
  0.79) but picks the top worse (slateN 0.66 vs 0.75-0.81)
tier: T0
mode: exploratory
date: 2026-09-24
claim: >
  (descriptive; input to the closed-loop metric-validity study) Candidate offline
  metrics computed on DS-0006 (160 states x 128 pushes, 30 goals, soft truth,
  lyapunov) for 8 models + the NFD ensembles (ensemble_nfd = 7 nfd* models incl. epoch30 and nfd_3ch_finetuned; ensemble_nfd5 = EXP-0037/0039 five), and their mutual rank agreement.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/metric_table.py
  data: ["DS-0006 via EXP-0030 RUN-0001 predictions / truth", "EXP-0028 soft 30-goal report (accuracy)"]
  code_path: "per (state, goal) pool: argmin pick -> slateN, regret, optimism; spearman/pearson over pool; top-8 capture"
  seed: "none"
  split: "not applicable"
result: >
  slateN / top1_regret / spearman / pearson / top8: ensemble_nfd 0.837 / 0.008 /
  0.902 / 0.925 / 0.665; nfd_residual_warped_flipaug_randlen 0.810 / 0.010 / 0.854 /
  0.887 / 0.643; NFD variants 0.745-0.771 (spearman 0.79-0.84); linear_switched_hard
  0.659 / 0.017 / 0.790 / 0.833 / 0.572; nfd_3ch_finetuned 0.512 / 0.025 / 0.639 /
  0.654 / 0.430. Optimism (pred - true dv at the pick): ensemble -0.000, nfd_3ch_randlen
  +0.001, others -0.010 to -0.030. Kendall across the 9 models: ranking metrics
  0.89-1.00 with each other; optimism 0.39-0.50 with them. Full table
  results/metric_table.json (per-state values kept for resampling).
verdict: supported
downgrades: [indirectness]
grade: moderate
---
