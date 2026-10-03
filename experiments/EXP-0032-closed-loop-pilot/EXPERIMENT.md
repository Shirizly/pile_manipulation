---
id: EXP-0032
title: >
  Closed-loop MPC pilot with learned models works end to end; planner choice
  dominates (CEM > GD > rank, 8/8 paired episodes for 2 of 3 models), episode sd is
  small (~0.03-0.05), the offline-weakest model is also weakest in closed loop, but
  between the two good models the closed-loop order depends on the planner and on
  model speed under a wall-clock budget
tier: T0
mode: exploratory               # a pilot; DESIGN.md states no claim is tested
date: 2026-09-24
claim: >
  (descriptive) The learned-MPC harness runs closed-loop episodes with 3 models x 3
  planners at a 1 s budget on DS-0006 starts, and the resulting closed-loop
  improvement separates planners and at least the weakest model.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/pilot.py, code/analyse.py
  data: ["DS-0006 start states 0-3"]
  code_path: "simple_mpc.learned_mpc.run_episode (plan: rank/gd/cem) -> GenesisOracleEnv.step with TRAINING_PHYSICS; progress = lyapunov(occ_for_scoring)"
  seed: "none set (pile-aware candidate sampling and CEM draws unseeded)"
  split: "not applicable"
result: >
  Improvement V0 - V_8 (soft lyapunov; higher = better), mean (sd) over 8 episodes:
  NFD warped+flip rank +0.077 (0.036) / gd +0.156 (0.039) / cem +0.262 (0.044);
  linear_switched_hard +0.075 (0.025) / +0.118 (0.027) / +0.275 (0.039);
  nfd_3ch_finetuned +0.052 (0.028) / +0.137 (0.051) / +0.162 (0.050). CEM > GD >
  rank in 8/8 paired episodes for NFD-warped and linear (finetuned: gd-cem 2-6).
  Offline slateN (DS-0006, EXP-0030): 0.775 / 0.696 / 0.531. CEM evaluations per
  decision: 6987 (NFD) vs 14856 (linear) vs 15182 (finetuned) -- speed doubles the
  search. Predicted-vs-true per-step dv r: ~0.9 for NFD and linear under CEM, 0.43
  for finetuned; all models optimistic (predicted dv more negative than true by
  0.006-0.044). Caveat: this run's rank planner used only 64 candidates (budget
  unused); fixed in the harness afterwards.
verdict: supported
downgrades: [imprecision, incomplete-design]
grade: low
---

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Superseded: EXP-0042 found these planner defaults far from optimum (C-047); EXP-0039 / EXP-0043 / EXP-0044 are the powered closed-loop runs (with tuning, model differences shrink to seed size, C-049).
- ISS-013: planner candidates / initialisations in this record came from the pre-fix pile-aware sampler (ISS-010 class, ~half illegal touchdowns in audited banks); executed pushes have not been audited for touchdown legality.
