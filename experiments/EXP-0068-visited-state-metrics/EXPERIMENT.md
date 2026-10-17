---
id: EXP-0068
title: >
  PRE-REGISTERED (2026-10-05, before simulation): offline metrics scored on closed-loop VISITED states (DS-0023, 48
  states from a legal CEM re-run of the RUN-0002 zoo) vs the same metrics on DS-0016 SPAWN pools, as predictors of the
  closed-loop model order (EXP-0039 RUN-0002 CEM / rank arms, EXP-0066)
tier: T1
mode: confirmatory              # H1 / H2 and every aggregation fixed in DESIGN.md before any cell ran
date: 2026-10-05
hypothesis: null
claim: >
  For 8 offline metrics (accuracy_1, accuracy_1_blur2, occ_emd_swept, mass_in_goal_mae, slateN, top-1 regret, optimism,
  within-pool Spearman; lyapunov objective, soft truth, 20 mm legal pile-aware candidates, the 8 closed-loop letters)
  scored on the RUN-0002 zoo (8 OCC models), the version scored on closed-loop VISITED states (DS-0023: 48 states after
  4 / 8 / 12 CEM pushes, stratified over letter, push index and model; 64 sampler candidates per state; each state scored
  on the letter it was visited under) has a higher Spearman rho with the closed-loop order than the version scored on
  DS-0016 step-0 SPAWN pools (32 pools x 64, same letters), in both the CEM and the rank arm of EXP-0039 RUN-0002 (H1);
  and on visited states raw hard-raster accuracy_1 is no longer at chance as a predictor of the CEM order (H2).
prediction:
  supports: "H1: rho_visited > rho_spawn for >= 6 of 8 metrics in BOTH arms; H2: rho_visited(accuracy_1, CEM) >= 0.5 and pair agreement (seed pairs excluded) >= 0.70"
  refutes: "H1: <= 4 of 8 metrics in either arm (the embarrassing result: visited-state scoring buys nothing); H2: rho < 0.3 or agreement < 0.60"
  discriminating: true
provenance:
  commit: a5ccb680
  dirty: true                    # learned_mpc.py topk/quats recording, driver --record-quats, EXP-0066's uncommitted fallback/seed code (see body)
  data_commit: "DS-0016: see its DATASET.md (2026-09-28); DS-0006 starts: config provenance; DS-0023: this record"
  script: code/select_states.py, code/collect_pools.py, code/score_pools.py, code/analyse.py (driver experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py)
  data: ["DS-0023 (collected here)", "DS-0016 test_pools_v2", "DS-0006 states 40 and 48 (starts)", "EXP-0039 results/run0002_analysis.json", "EXP-0066 results/*_ps0.json"]
  code_path: >
    closed loop: simple_mpc.learned_mpc.run_episodes_batched (cem, legalize + legal_fallback, planner_seed 0, topk 16,
    quats recorded) -> GenesisOracleEnv with TRAINING_PHYSICS; pools: set_particle_state + generate_action_samples
    (DS-0016 sampler) + legalize_pushes for the planner candidates + execute_action, TRAINING_PHYSICS; scoring: EXP-0059
    eval_extended / EXP-0053 eval_narrow metric functions imported unchanged, soft truth occ_for_scoring
  seed: "planner seed 0; sampler torch seed 68 (pools); state selection deterministic (DESIGN.md)"
  split: "not applicable"
  runtime: "pending"
  runs: [RUN-0001, RUN-0002, RUN-0003]
budget:
  declared: "7 h wall, <= 2.5 GPU-h (hard; partial rather than overrun)"
  spent: "pending"
  outcome: within
design:
  varied:
    state_source: "visited (DS-0023, after 4 / 8 / 12 CEM pushes) vs spawn (DS-0016 step-0 pools)"
    metric: "accuracy_1, accuracy_1_blur2, occ_emd_swept, mass_in_goal_mae, slateN, top1_regret, optimism, spearman"
    arm: "EXP-0039 RUN-0002 CEM (8 models, 128 eps), rank (8 models, 64 eps); EXP-0066 ps0 (8 models, 128 eps)"
  held_fixed:
    candidates: "64 legal pile-aware 20 mm pushes per state (DS-0016 sampler config), soft truth, TRAINING_PHYSICS"
    goals: "the 8 closed-loop letters O T S L X Z C H; visited primary = goal-matched, spawn primary = all 32 pools x 8 letters"
    models: "the RUN-0002 zoo (8 OCC models) + EXP-0066's linear boot1/boot2 and nfd_3ch_randlen_seed1"
    closed_loop_reference: "RUN-0002 scores and Holm pairs unchanged; the re-run here only supplies states"
  baselines: "spawn-state metrics (the existing practice); do-nothing = persistence has accuracy 0 by construction and is the floor of every pool metric; random ranking = slateN 0; within-seed closed-loop pairs as the resolution floor"
  metric: "slateN (capture), accuracy (swept-region, hard raster), top1_regret, optimism, spearman, mass_in_goal_mae, occ_emd_swept; closed loop = mass_in_region over pushes 4-16 / optimum (RUN-0002)"
noise_floor: >
  n = 8 models: a one-sided permutation p < 0.05 needs rho >= ~0.74; state-bootstrap (48 states / 32 pools, 2000 draws)
  95 % CIs on every rho; closed-loop within-seed spread 0.053 (CEM and rank). RUN-0002 spawn values to beat: CEM
  accuracy_1 +0.10, slateN +0.31, regret +0.50, mass_in_goal_mae +0.79; rank accuracy_1 +0.17, regret +0.81.
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x, benchmark-physics-matches-training]
establishes: []
result: "pending (pre-registration; no cell has run)"
verdict: inconclusive
downgrades: [incomplete-design, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If the gap between offline and closed-loop orderings comes from WHERE the models are scored (spawn piles vs the half-
organised piles a planner actually reaches), scoring the same metrics on visited states must raise their rank
correlation with the closed-loop order, and should move accuracy_1 -- which is at chance on spawn pools -- off chance.
If the gap comes from the planner's optimisation itself (exploiting model error along the search path, RUN-0002
reading iv), a model-agnostic corpus of visited states with random legal candidates will not help, and the visited
metrics will be no better than the spawn ones. Both outcomes are informative; the second embarrasses the design.

## What was actually run
Pending. Pre-registration: `DESIGN.md` (cells, state selection rule, aggregations, criteria), written while the GPU was
held by EXP-0067. Code changes made before the run (uncommitted, user rule: do not commit): `simple_mpc/learned_mpc.py`
(`plan(topk=)` running top-k, `quats` recording), the EXP-0043 driver (`--record-quats`), `tests/test_learned_mpc.py`
(one new test). EXP-0066's uncommitted `legalize_fallback` / `planner_seed` code is also in the tree and used.

## Numbers
Pending.

## What would change the verdict
Pending.

## Threats
- incomplete-design: pre-registration state; cells not yet run.
- untested-dependency: `benchmark-physics-matches-training` is `broken` in INVARIANTS.md (DS-0001-era); every simulation here runs TRAINING_PHYSICS explicitly (apply_physics), as DS-0016 and RUN-0002 did.

## Unrelated findings
none

## Paused 2026-10-05 (user) -- status on disk, 14:45 CEST

**Nothing has been simulated.** The GPU was held by EXP-0067 from 09:05 until ~14:30 (5.5 h of this record's 7 h
budget, spent waiting in bounded foreground loops); the queue was launched at 14:36 with the CUT design (1 start, 24
states, DESIGN.md fallback) and stopped by the coordinator 24 s later, before the driver wrote its first checkpoint.
Ledger: `EXP-0068/RUN-0001/closed_loop` start 14:36, end `killed` 14:42.

What exists on disk:
- `results/` -- EMPTY (no `cl_full.json`): **RUN-0001's closed-loop states do NOT exist.** The coordinator's resume
  message assumed they did; the first session had never reached the GPU (every `wait_gpu.sh` call returned "busy" with
  EXP-0067's PID in the lock until the session was cut off at ~11:20), so `cl_full` was being run for the FIRST time,
  not re-run. The only pre-existing closed-loop episodes (EXP-0039 RUN-0002 / RUN-0003, EXP-0043, EXP-0066) store xyz
  only, no quaternions, and are not re-simulable (DESIGN.md, "Why the saved episodes cannot be reused").
- `datasets/DS-0023-visited-state-pools/data/` -- EMPTY (no `starts.pt`, no pools); `DATASET.md` written, `status: pending`.
- Code complete and tested: `code/select_states.py`, `collect_pools.py`, `score_pools.py` (CPU smoke test on DS-0016
  passed), `analyse.py` (dry run on fabricated inputs passed), `queue.sh`, `wait_gpu.sh`, `post_gpu.sh`;
  `simple_mpc/learned_mpc.py` (`plan(topk=)`, `quats` recording) + driver `--record-quats` + 1 new test (12/12 pass).
- Pre-registration unchanged: `DESIGN.md`, this frontmatter. `runs/RUN-000{1,2,3}-*/RUN.md` are drafts marked PENDING.

**To resume (one command, GPU exclusive, ~50 min for the cut design / ~70 min for the full one):**
`bash experiments/EXP-0068-visited-state-metrics/code/queue.sh` (creates `experiments/temp/gpu.lock` with its PID,
resume-safe via `results/*.done`; currently set to the CUT design `--starts 40`, `--n-per-cell 1` = 24 states; restore
`--starts 40 48` / `--n-per-cell 2` in `queue.sh` for the pre-registered 48), then
`bash experiments/EXP-0068-visited-state-metrics/code/post_gpu.sh` (analysis), then fill `## Numbers`, the verdict
(`result`, `verdict`, drop `incomplete-design`), DS-0023's `status: active` + verification, the three RUN.md files and
one REGISTER.md line. Budget already spent: ~5.8 h wall, 0 GPU-h.
