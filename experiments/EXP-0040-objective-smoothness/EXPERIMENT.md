---
id: EXP-0040
title: >
  The push-frame warp makes a model's predicted objective ~4-5x rougher along
  action-space lines (warped arms: roughness 0.09-0.10, ~8 local minima per line;
  world-frame: 0.019-0.024, ~3), but roughness does not explain which arms optimise
  well (Spearman with gain_capture -0.43, p 0.34, 7 arms)
tier: T0
mode: exploratory
date: 2026-09-24
claim: >
  The residual arms' superior gradient-descent performance (EXP-0037) is explained by
  a smoother predicted objective (lower roughness / fewer local minima along
  action-space lines through good pushes).
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/smoothness.py
  data: ["DS-0006 states 0-9 (EXP-0037 stage-1 rank picks as line centres)"]
  code_path: "OCC_ADAPTERS predict_step along 41-point lines (+/-8 mm, random unit direction in the 4-D action) -> per-goal lyapunov dv -> mean |2nd diff| / range; interior local minima"
  seed: "line directions rng seed 0"
  split: "not applicable"
result: >
  REFUTED as the explanation (null, low power): roughness nfd_3ch_randlen 0.019,
  nfd_residual_worldframe_noaug_ep43 0.024, linear_switched_soft 0.057, ensemble 0.059,
  nfd_residual_warped_flipaug_randlen 0.092, nfd_warped_randlen 0.094,
  nfd_warped_randlen_flipaug 0.101 (local minima per line 2.7 / 3.2 / 4.8 / 6.8 / 7.9 /
  7.6 / 8.1). Roughness separates WARPED from world-frame arms ~4-5x, but the
  residual-warped arm is rough yet the 3rd-best optimiser; Spearman(roughness,
  gain_capture) -0.43 (p 0.34), (roughness, grad_capture) -0.14. Candidate cause of the
  warp's roughness: the push-frame resampling changes non-smoothly with the action.
verdict: refuted
downgrades: [imprecision, indirectness]
grade: low
---

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- EXP-0065 / ISS-013: 54 % of DS-0006's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pool numbers were not re-scored on legal-only candidates.
