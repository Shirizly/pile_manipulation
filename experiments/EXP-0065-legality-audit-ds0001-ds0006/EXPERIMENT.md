---
id: EXP-0065
title: >
  ISS-010's illegal-touchdown defect also affects DS-0001 and DS-0006: 46 % / 54 % of their
  pre-fix pile-aware candidate pushes put the 40x2 mm blade on a cube at touchdown
tier: T0
mode: exploratory
date: 2026-10-03
claim: >
  The pile-aware sampler path audited in ISS-010 (DS-0008..0013) was also used to collect
  DS-0001 (20 slates x 1000) and DS-0006 (160 slates x 128) through
  Genesis/binned_slate_collection.py before the 2026-09-28 fix, and their candidate pushes
  overlap a cube at touchdown at a rate comparable to DS-0008..0013 (~0.4-0.6). On DS-0006
  (EXP-0030's cached soft-truth dv, 8 models + NFD ensemble, 30 goals, lyapunov / mass_in_region),
  restricting each 128-push pool to its legal candidates changes slateN by more than a size-matched
  random subset does, and changes the model ordering.
provenance:
  commit: d72bb304
  dirty: true
  script: "code/audit_ds0001_ds0006.py (RUN-0001), code/rescore_ds0006_legal.py (RUN-0002, reads EXP-0030 artifacts/RUN-0001), code/audit_closed_loop_actions.py (RUN-0003, reads recorded episodes of EXP-0050..0057), code/onpolicy_optimism.py (RUN-0004), code/closed_loop_models_legal_rescore.py (RUN-0005), code/seed_floor_legal.py (RUN-0006)"
  data: [DS-0001, DS-0006]
  code_path: "EXP-0059 code/audit_tool_placement.py::_row_illegal (exact SAT, Baselines/common/cube_overlap.overlaps_rect_pairs, blade 40x2 mm vs 5 mm cubes, pre-push states, p_starts, angles); 0 mm and 1 mm margin"
  seed: "none (deterministic census)"
  split: "not applicable -- every row of step0.pt"
  data_commit: "not recorded (data gitignored; audited as on disk 2026-10-03)"
result: >
  Fraction of candidate pushes whose blade overlaps a cube at touchdown, 0 mm / 1 mm margin:
  DS-0001 0.461 / 0.591 (per slate 0.21-0.63, median 0.48; n 20,000); DS-0006 0.543 / 0.798
  (per slate 0.31-0.76, median 0.55; n 20,480). results/audit_ds0001_ds0006.json.
  DS-0006 re-score (results/rescore_ds0006_legal.json): the TRUE best candidate is an illegal
  touchdown in 0.72 (lyapunov) / 0.54 (mass) of state x goal cells; illegal candidates improve
  lyapunov ~2x more than legal ones on average (0.0120 vs 0.0062). slateN legal-only minus
  size-matched random subset: linear_switched_hard +0.113 [+0.096, +0.129] lyap / +0.084 mass;
  NFDs +0.013..+0.068 lyap, -0.001..+0.036 mass; ensemble +0.043 / +0.021. Kendall(all, legal)
  0.83 on both; linear_switched_hard moves from 8th of 9 (0.659) to 5th (0.812) on lyapunov.
  EXECUTED closed-loop pushes (results/audit_closed_loop_actions.json; cube yaw not recorded, so bounded by
  the cube's inscribed / circumscribed disk, which brackets DS-0006's exact rate 0.467 <= 0.543 <= 0.584):
  illegal-touchdown share >= 0.17 (EXP-0051), 0.36 (EXP-0052), 0.53 (EXP-0054), 0.16-0.24 (EXP-0055),
  0.28 (EXP-0050 sim CEM), 0.23-0.24 (EXP-0057 perfect-model CEM, 0.63 for the truncated 256x1 cell).
  Model-dependent: EXP-0054 worldframe NFD 0.69 vs narrow / nfd_3ch 0.46-0.51; EXP-0051 NFD 0.21-0.24 vs
  linear 0.11-0.14.
  On-policy (results/onpolicy_optimism.json; only EXP-0051 is interpretable -- its planning objective is
  lyapunov, the unit of true_dv; elsewhere pred_dv is the mass-augmented objective): executed illegal
  pushes realise as much or more true lyapunov improvement than legal ones (linear -0.018 / -0.021 vs
  -0.012; NFD -0.011 / -0.016 vs -0.014 / -0.012, GD / CEM), and the models are 1.2-4x more optimistic on
  them (true - pred +0.003..+0.016 vs +0.000..+0.008) -- illegal touchdowns are both effective in this
  simulator and over-predicted.
  Control corpus DS-0007 (Sean, results/audit_ds0007.json): 0.000-0.002 illegal in every shard except
  scattered_n50 0.010 and scattered_n100 0.093 -- the non-pile-aware Sean collection is essentially legal.
  RUN-0005 (results/closed_loop_models_legal_rescore.json): EXP-0039/0044's four closed-loop models on DS-0006,
  lyapunov slateN all-candidates -> legal-only (160 states): nfd_3ch 0.745 -> 0.800, seed1 0.787 -> 0.823,
  linear_switched_soft 0.758 -> 0.826, worldframe 0.761 -> 0.826. Agreement with EXP-0044's 5 Holm-resolved
  tuned closed-loop pairs: slateN 1/5 -> 4/5 (margins ~0.003, inside noise); on the 8 closed-loop start
  states only (40-47, legal) 0/5; optimism at the pick (less optimistic = better) 4/5; accuracy (EXP-0044) 3/4.
  The 'slateN disagrees with closed loop' evidence was largely an illegal-push effect; nothing here
  separates the metrics with power (and the closed-loop reference itself executed illegal pushes, RUN-0003).
  RUN-0006 (results/seed_floor_legal.json): EXP-0036's 4-seed NFD slateN floor on DS-0006 halves on
  legal-only candidates -- sd 0.019 -> 0.010 (lyapunov), 0.025 -> 0.013 (mass_in_region).
verdict: supported
downgrades: [indirectness]   # overlap at touchdown is the geometric proxy ISS-010 used; the physical consequence per row (ejection vs harmless nudge) is not measured
grade: moderate
---

RUN-0002 re-scores only EXP-0030's DS-0006 table; the other exposed DS-0006 / DS-0001 tables are not
re-scored. Exposure (see ISS-013 in `experiments/OPEN_ISSUES.md`): every offline
slateN/ranking number scored on DS-0006 pools (EXP-0030, EXP-0036, EXP-0037, EXP-0038,
EXP-0040, EXP-0041, EXP-0042, EXP-0046 redundancy, EXP-0048, EXP-0049 scatter states) or
DS-0001 pools (EXP-0011..0014, EXP-0016, EXP-0023, EXP-0026 A3, EXP-0029) ranks a pool about
half of whose candidates are illegal touchdowns. The closed-loop records (EXP-0032,
EXP-0039..0057) used DS-0006 only for start STATES (legal), but drew planner candidates from
the same pre-fix sampler; their executed pushes have not been audited.

Dirty tree: unrelated working-tree edits by other sessions; this record only reads data.
