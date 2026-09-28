---
id: EXP-0041
title: >
  Across the 5 NFD models, image accuracy orders them exactly as their gradient-descent
  gain does (Spearman +1.00; grad_capture +0.90) while slateN does not (+0.43 / +0.57)
  -- suggestive (n=5) that accuracy predicts GRADIENT-based control quality even though
  it fails to predict ranking
tier: T0
mode: exploratory
date: 2026-09-24
claim: >
  Among models of one family, whole-image prediction accuracy predicts a model's value
  as a gradient-descent objective (EXP-0037 gain_capture / grad_capture) better than
  its ranking skill (slateN) does.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: "inline (reads EXP-0037 results/analysis.json, EXP-0038 results/metric_table.json, EXP-0040 results/smoothness.json)"
  data: ["EXP-0037 (DS-0006 gradient benchmark)", "EXP-0038 offline metric table (DS-0006)", "EXP-0028 randlen_test accuracy"]
  code_path: "scipy spearmanr across arms"
  seed: "none"
  split: "not applicable"
result: >
  gain_capture vs accuracy (n=5 NFD arms with harness accuracy): Spearman +1.00;
  grad_capture vs accuracy +0.90 (p 0.04). vs slateN (n=7): +0.43 / +0.57; spearman
  +0.18 / +0.39; optimism +0.43 / +0.46; top8 +0.29 / +0.50; roughness -0.43 / -0.14.
  Accuracy order: residual_worldframe 0.484 > residual_warped 0.462 > nfd_3ch_randlen
  0.456 > warped_flipaug 0.409 > warped 0.400 = gain order. n=5, accuracy measured on
  randlen_test (different states from DS-0006); pre-registered as H6 in EXP-0039.
verdict: inconclusive
downgrades: [imprecision, indirectness]
grade: low
---
