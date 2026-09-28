---
id: EXP-0034
title: >
  Blurring prediction and truth before scoring accuracy (removing 2x2-px
  cube-orientation detail) monotonically raises accuracy's agreement with
  slateN (NFD family Kendall +0.07 -> +0.20 -> +0.33 at sigma 0/1/2 px) --
  suggestive, not significant at n=6
tier: T0
mode: exploratory
date: 2026-09-24
claim: >
  Part of image accuracy's disagreement with slateN within a model family comes
  from pixel-scale artifacts (quantised cube orientation / sub-pixel position in
  the corpus rasteriser), so accuracy computed after a matched Gaussian blur of
  prediction, truth and previous frame agrees better with slateN.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/blur_accuracy.py
  data: ["randlen_test (eval_report CORPORA)", "EXP-0028 RUN-0001 soft 30-goal report (slateN)"]
  code_path: "predict_occ -> blur(pred, truth, prev) -> fit_linear_foresight.metrics swept-region accuracy -> kendalltau vs slateN"
  seed: "none"
  split: "not applicable -- held-out randlen_test"
result: >
  Kendall tau(accuracy, lyapunov slateN): all 10 non-GNN models +0.69 / +0.73 /
  +0.78 at sigma 0 / 1 / 2 px (all p < 0.01); NFD family (6) +0.07 / +0.20 /
  +0.33 (p 1.00 / 0.72 / 0.47). Monotone in the predicted direction; the
  within-family test has almost no power at n=6.
verdict: inconclusive
downgrades: [imprecision, indirectness]
grade: low
---
