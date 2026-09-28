---
id: EXP-0057
title: >
  Oracle-budget ablation queued (pre-registration): at a fixed 256-sim/decision simulator-CEM
  budget (halved from the pre-registered 512 by measured throughput), prediction is that the
  default 64x4 pool/iteration split solves goals in fewer pushes than pure sampling (256x1)
  but is not reliably beaten by more-iteration splits (32x8, 16x16)
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  On the perfect-model (simulator-as-model) CEM ceiling, at a FIXED per-decision simulation
  budget (256 simulated pushes), the split between CEM pool size and iteration count changes
  the number of pushes needed to reach 0.9 x the goal's mass optimum (EXP-0046
  `mass_frac_best_placement`), on the 8-goal EXP-0055 set x DS-0006 starts 40/41, TRAINING_PHYSICS,
  lyapunov as the default planning objective; and switching the objective to a success-aware one
  (lyap - w x mass, w in {1,3}; crowd_floor) changes it further, holding the split fixed at the
  default 64x4.
prediction:
  supports: >
    oracle_A_256x1 (pure sampling, no refinement) has a HIGHER censored_mean_pushes (>=1 push
    worse, 95% bootstrap CI excluding 0) than oracle_A_default (64x4) on the paired 16-task set;
    at least one success-aware objective (B_mass1/B_mass3/B_crowd) has a LOWER censored_mean
    completion push count than lyapunov-default on the same split, driven by the letter goals
    (quadrant_0/two_squares already solve fast under lyapunov per EXP-0050).
  refutes: >
    every (A) split (256x1 through 16x16) reaches the same censored_mean_pushes within its
    bootstrap CI (no split effect at this budget), OR no objective cell improves on lyapunov's
    completion pushes outside its CI (the objective does not matter at a perfect-model ceiling).
  discriminating: true
provenance:
  commit: 3bae8cd7
  dirty: true
  script: "experiments/EXP-0050-oracle-ceiling/code/oracle.py (extended with --value, --n-envs,
    --l-min/--l-max, and cell/sims_used/plan_time_s/signed_frac/obj_values recording)"
  data: ["Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys starts 40/41 (44/45 in the extra queue)",
         "experiments/EXP-0046-goal-ceiling/results/vstar.json (completion threshold)",
         "experiments/EXP-0055-capacity-aware-value/results/analysis_main.json (learned-model reference row)"]
  code_path: "Genesis pile-aware sampler -> project_push -> SandboxManipulation.execute_action
    (16 episodes batched across n_envs=128) -> simulator CEM (iter0 pile-aware, then Gaussian
    refit to true-best elite, elite frac 0.25) -> best-ever-simulated executed"
  seed: "--seed-base 0: torch/numpy seeded per (goal, start, push, iteration) via crc32 before the pile-aware draw and each CEM refit -> identical random streams per task at every cell (added 22:35 before any push completed; verified: two runs pick identical actions, outcomes differ ~0.002 in-goal mass from GPU physics nondeterminism)"
  split: "not applicable (control episodes, not a train/test split)"
  data_commit: not applicable
  runs: [RUN-0001]
  runtime: "throughput test ~8 min; main queue ~11.4h estimated; extra queue up to ~37-46h
    estimated (see DESIGN.md); both launched detached, run past this record's own session"
budget:
  declared: "~2h wall clock, ~250k tokens (setup only; queues run detached 12h + up to 48h after)"
  spent: "~1.9h wall clock, setup + throughput test + code changes + smoke test + launch"
  outcome: within
design:
  varied:
    "(A) split": ["default 64x4", "256x1", "128x2", "32x8", "16x16"]
    "(B) objective (at default split)": ["lyapunov (shared with A)", "lyap-1xmass", "lyap-3xmass", "crowd_floor"]
  held_fixed:
    budget: "256 simulated pushes/decision (halved from the pre-registered 512; see DESIGN.md
      Step 2 -- even 256 does not fit all 5 (A) cells in 12h, so 3 run in MAIN and 2 in EXTRA,
      see Deviations)"
    tasks: "8 goals x starts 40/41 (16 tasks), same set at every cell"
    n_envs: "128 (throughput test: best headroom/throughput trade-off of 32/64/128/256)"
    physics: TRAINING_PHYSICS (same sim used for candidates and execution, so no train/eval
      physics mismatch by construction)
    elite_fraction: 0.25
  baselines: ["EXP-0055 learned-model `lyap` cell (worldframe NFD GD, 1s budget) on the same
    goals/starts, read-only reference, not re-run", "oracle_A_256x1 doubles as the 'do nothing
    beyond sampling' baseline for the split axis (no CEM refinement)"]
  metric: "oracle_completion_pushes (experiments/METRICS.md, added by this record)"
noise_floor: "not yet measured for this exact cell; EXP-0043 repeat sd on related lyapunov
  curves was 0.01-0.04 (different quantity, cited as an order-of-magnitude prior only)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  QUEUED, NOT YET RUN: this record captures the pre-registration, throughput test, and launch;
  the queues run detached (12h main + up to 48h extra) past this session's budget. No cell has
  a completed result at the time of writing. See TODO.md "Oracle budget ablation (EXP-0057)"
  for how to check progress and where results land once cells finish.
verdict: inconclusive
downgrades: [incomplete-design]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates
If the split between pool size and refinement iterations does not matter at a fixed budget,
every (A) cell should reach the goal in statistically indistinguishable numbers of pushes; if
pure sampling (256x1) is measurably worse, refinement is doing real work even with a perfect
model. Separately, EXP-0050 already showed a perfect-model planner optimising pure lyapunov
leaves letters 43-65% filled at a much smaller budget (96 sims/push) -- if a success-aware
objective at 256 sims/push still does not close that gap, the objective (not search budget) is
the bottleneck for letters; if it does close the gap, extra budget was the missing ingredient.

## What was actually run
Step 1 (throughput): `code/throughput_test.py --n-envs {32,64,128,256}`, 3 batched settle+push
steps each, TRAINING_PHYSICS, GPU otherwise idle (`nvidia-smi` checked empty first). Results in
`results/throughput.json`. Chose n_envs=128 (see DESIGN.md).

Step 2/3: extended `experiments/EXP-0050-oracle-ceiling/code/oracle.py` (shared with EXP-0050,
which stays valid -- only new CLI flags added, no default behaviour changed) with `--value
{lyap,crowd_floor}`, `--n-envs`, `--l-min/--l-max`, `--out-dir`, and per-push recording of
`sims_used`, `plan_time_s`, `signed_frac`, `obj_values`, and a `cell` tag per episode. Wrote
`code/run_queue.py` (resumable job queue over `scripts/run_probe.py`, skips complete cells,
copies `runs/COMMANDS.jsonl` lines into `experiments/COMMANDS.jsonl`) and `code/analyse.py`
(completion push k*, paired bootstrap vs `oracle_A_default`, per-goal table, EXP-0055 reference
load).

**Smoke test** (2 goals x 1 start x 2 pushes x 2 cells, cem-pop=4/iters=1): ran to completion,
killed mid-push-1 and restarted to confirm checkpoint/resume (resumed at push 1, finished push
2 correctly), ran a second cell with `--value crowd_floor`, then ran `run_queue.py`'s actual
`run_queue()` function end-to-end through `scripts/run_probe.py` for two tiny cells (confirmed
ledger start/end events, `COMMANDS.jsonl` copy -- found and fixed a duplicate-line bug from a
timestamp-based copy window, switched to a line-count offset), then ran `code/analyse.py` on
the smoke outputs (completion table, paired deltas, EXP-0055 reference all populated
correctly). Smoke outputs deleted after verification; the one ledger duplicate line it produced
was also removed from `experiments/COMMANDS.jsonl`.

**Deviations from the pre-registered brief** (both forced by the throughput measurement, DESIGN.md
"Deviations"): (1) budget 512 -> 256 (the pre-agreed fallback); (2) even at 256, all 5 (A) cells
do not fit the 12h main-queue target (19.0h estimated) -- 3 splits (default, 128x2, 32x8) run in
MAIN (~11.4h), the other 2 (256x1, 16x16) run at the FRONT of EXTRA so the full (A) sweep still
completes, just after ~12h rather than within it. 16 tasks (the brief's own pre-agreed floor
trim) was never reduced further.

**Not implemented / threats to the pre-registered design, stated up front:**
- **Seeding (fixed 2026-09-25 22:35, before any push of the main queue completed).** The first
  launch (22:16) was unseeded; it was stopped after 13 min with no push completed (archived in
  runs/aborted_unseeded_2216/) and relaunched with `--seed-base 0`: identical random streams per
  (task, push, iteration) at every cell. Residual: Genesis GPU physics is not bitwise
  deterministic (same actions -> in-goal mass differs by ~0.002).
- **Early stopping (added at the same relaunch, `--stop-solved`).** An episode stops being simulated
  at its first push with in-goal mass >= 0.9 x optimum (`solved_at`); later entries carry the state
  forward (action None, sims_used 0), so per-push arrays keep length 21 and completion is unchanged.
  Saves wall time only; the extra queue gets correspondingly more done.
- Paired by task set AND by RNG stream (seeded); physics residual ~0.002 in-goal mass.
- One physics regime (TRAINING_PHYSICS), one n_envs choice (128), one elite fraction (0.25) --
  all held fixed, not swept.
- 16 tasks per cell is a small paired sample for letter-level per-goal claims; the aggregate
  paired bootstrap is the more trustworthy read.
- Detached, unattended run: a GPU OOM or crash mid-cell is recoverable (oracle.py checkpoints
  every push) but a systemic failure (e.g. a code bug hit only at full scale) would not be
  caught until someone reads the logs -- see TODO.md for where to look.

## Unrelated findings
`docs/CODEMAP.md` did not have an entry for `code/oracle.py`'s per-episode simulator CEM
batching pattern (flatten all episodes' candidates, chunk by n_envs) as a reusable idea for
future perfect-model planners -- added a one-line pointer in this record's TODO.md entry
instead of CODEMAP, since the code itself is experiment-local (EXP-0050's), not a new project
module.
