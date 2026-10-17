---
id: EXP-0076
title: >
  Expert demonstrations and pipeline search (n20 / n50 x scattered / inbetween / piled x random disc / square / letter goals, success = in-goal
  mass >= 0.95, objective = physical push time): a beam over TRUE simulator states with model-proposed children solves more and faster than the
  learned planner alone; planner knobs (model, budget, horizon, phase schedule) are on a plateau inside run-to-run noise
tier: T1
mode: exploratory
date: 2026-10-16
hypothesis: null
claim: >
  On EXP-0076 scenarios (results/scenarios.json: Sean TEST-split chain-start states, goals passing the 60 % area rule; Genesis TRAINING_PHYSICS,
  legal pushes; success = soft-raster in-goal mass >= 0.95 within 12 pushes (n20); time = EXP-0070 physical push time), (1) a beam over true
  simulator states (B=4 nodes x 8 children proposed by the ens2bal fetch + legal-scoring CEM planner) solves at least as many n20 scenarios as the
  same planner alone and reaches success faster on jointly solved scenarios; (2) for the feasible (model-only, <= 3 s / decision) pipeline, no
  tested change of model / budget / horizon / phase schedule beats the ens2bal H=1 baseline beyond run-to-run noise.
provenance:
  commit: 13b0886f
  dirty: true
  script: "code/beam.py (runner), tasks.py (scenarios / metric / time), simcore.py, store.py, summarize.py, replay.py, gif.py, mine_recorded.py, structure_analysis.py, queue_*.sh"
  data: ["results/scenarios.json (540 scenarios; Sean TEST split via EXP-0074 sean_data.split_files)", "MODEL-0025 (ens2bal), MODEL-0016 (vanilla64bal)"]
  data_commit: "Sean corpus as registered (DS-0007 doc); EXP-0074 split_files.json"
  code_path: "beam.py: Scenario beam -> propose (wide_planner.plan_seq fl/ot pool, legal scoring, CEM + GD, top-k distinct) -> simcore.Sim.step_from (set_particle_state -> execute_action -> update_material_state) -> true value / mass -> beam selection -> store.offer"
  seed: "torch seed 0 per run (F0_seed1: seed 1); Genesis bank sampling under the same global RNG"
  split: "scenarios use Sean TEST-split files only (models trained on TRAIN)"
  runs: ["results/runs/*.json (tags in LOG.md)"]
  runtime: "2026-10-16 18:30 - 2026-10-17 08:30, one RTX 4070 laptop GPU, one job at a time except short debug probes (noted in LOG.md)"
budget:
  declared: "~14 h unsupervised (user, 2026-10-16)"
  spent: "~13 h GPU (one job at a time except short debug probes), 2026-10-16 18:30 - 2026-10-17 07:30"
  outcome: within
design:
  varied:
    pipeline: "feasible (B1 k1) vs greedy sim-verify (B1 k16) vs beam (B4 k8, B8 k4, mu 0 / 0.05)"
    feasible_planner: "model ens2bal / vanilla64bal; CEM 1,280 / 2,560 / 5,120; H=1 fl vs H=2 OT cumspill; phase schedule H2->H1 at mass 0.75; tail rule; time-aware re-ranking time_mu 0 / 0.02 / 0.05"
  held_fixed:
    physics: "TRAINING_PHYSICS, real settle / clearance budgets, legalised pushes (wide_planner.make_legal_scorer constants)"
    success: "in-goal mass >= 0.95 (occ_for_scoring), max 12 pushes n20 (16 in the n20 expert bulk and n50)"
  baselines: ["feasible ens2bal H=1 fl CEM 1,280 + GD 20 (the EXP-0075 best H=1 planner)", "start state (do nothing: mass0, none solved)"]
  metric: "solved count; physical time to success (task_time_physical without plan time, METRICS.md) on jointly solved scenarios; pushes to success"
noise_floor: "F0_seed1 (baseline config, planner seed 1, 36 n20 scenarios): solved 28 vs 27, time +0.11 s [-0.34, +0.58], 1 / 2 scenarios solved by one run only"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  (1) Pilot (18 n20): beam 18/18 vs planner 14/18 vs greedy sim-verify 16/18; beam - planner time on jointly solved -0.86 s [-1.73, -0.05],
  beam - greedy -0.66 s [-1.02, -0.32]; beam solved 37 of the 68 n20 scenarios the planner left unsolved. (2) Feasible ablations on 36 n20
  (baseline 27/36, seed replicate 28/36): F1-F4, F6-F8 within noise on time, F5 (vanilla64 H=2) worse +0.75 s [+0.07, +1.50]. Demos:
  272 of 540 scenarios solved (242 n20, 30 n50), 233 in < 10 pushes; open-loop replay keeps 158/203 feasible vs 19/55 expert demos >= 0.95; accepting
  expert success only at >= 0.97 gives 7/9 reproducing demos (24 stored as expert_robust; +0.5 s / +0.7 push).
verdict: supported
downgrades: [imprecision, inconsistency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
Every arm runs on the same scenarios (paired), and success / time are measured on TRUE simulator states, so model error cannot flatter any arm.
The feasible baseline and the do-nothing start state bracket the arms. A beam that only re-ranked by model value would match greedy
verification; the pilot separates lookahead through true states from verification.

## What was actually run
LOG.md is authoritative (every run, its tag, its outcome); SUMMARY.md is the state summary. Runs (results/runs/<tag>.json): p1A/p1B/p1C
(pilot), f20_all (feasible over 270 n20), F0_seed1..F8 (feasible ablations, 36 n20), f50s_feas (45 n50 scattered), X1/X2/X4/X5 (beam
ablations, 12 n20), E20_unsolved + E20u_b2k8 (beam over the 68 unsolved n20), R20_* (portfolio restarts), f50d_feas/f50d_feas2 (12 dense n50).
Deviations: E20_unsolved stopped by PID after 9 of 68 for throughput and continued with a cheaper beam (B2k8); the planned E20_slow beam
was replaced by feasible restarts; X3 (beam with H=2 proposals) not run (time). Two GPU debug probes overlapped production runs
(plan times of p1B / f20_all partly contended; outcomes unaffected).
Uncommitted at run time: all of EXP-0076 code (new), plus the pre-existing dirty files listed in git status at session start.

## What would change the verdict
More planner seeds per arm (only the baseline has a seed replicate); n50 and dense inits at scale (cost-limited, see LOG.md).

## Threats
- Restore-based beam chains do not reproduce open loop exactly (replay check: ~half of the divergence is plain run-to-run nondeterminism).
- One model instance; 36-scenario ablation set; per-scenario outcomes are near-threshold and noisy.
- n50 dense starts are barely sampled (simulation cost of pushed piles).

## Unrelated findings
- Genesis push cost on dense n50 piles grows to 2-13 s per sim step (contact islands), not amortised by batching envs.
- A second Genesis build in one process fails (nvrtc); `pkill -f <pattern>` killed its own shell twice (use PIDs).
