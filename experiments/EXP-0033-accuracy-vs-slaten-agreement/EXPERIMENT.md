---
id: EXP-0033
title: >
  Image accuracy ranks model FAMILIES like slateN does on randlen_test (Kendall
  +0.79 over 12 models) but not models WITHIN the NFD family (+0.07 to +0.20
  over 6), where the choices are actually made
tier: T0
mode: exploratory
date: 2026-09-24
claim: >
  On the eval_report harness with soft truth scoring (EXP-0028 reports),
  across-model rank agreement between `accuracy` and lyapunov slateN is high
  over all 12 models on randlen_test but near zero within the 6 NFD-family models.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/agreement.py
  data: ["EXP-0028 RUN-0001 reports (L20mm, L40mm, randlen_test; 3- and 30-goal)"]
  code_path: "report.json accuracy and capture.averaged_over_goals.lyapunov -> scipy kendalltau"
  seed: "none"
  split: "not applicable -- held-out eval corpora as in EXP-0028"
result: >
  randlen_test: all 12 models Kendall +0.79 (p<0.001) for both goal sets; NFD
  family (6) +0.20 (3-goal) and +0.07 (30-goal), p >= 0.7. L20mm/L40mm (pilot
  corpora) mixed: all-12 +0.06 to +0.58, NFD +0.20 to +0.73. Caveat: slateN
  differences within the NFD family are themselves only partly resolved (rank
  intervals overlap, EXP-0028), so low agreement is partly slateN noise; and
  slateN is not yet shown to predict closed-loop control (TODO G1b).
verdict: supported
downgrades: [imprecision, indirectness]
grade: low
---
