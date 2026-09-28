---
# ---- identity -------------------------------------------------------------
id: EXP-0027
title: >
  A 30-goal slateN harness roughly doubles resolved model pairs on L40mm and
  settles the top-1 on L20mm/L40mm, but reorders models vs the 3-goal harness;
  the gradient benchmark stalled at stage 2 on non-repeatable ground truth,
  which RUN-0005 traced to the IMAGE-BASED SCORE amplifying ~1 mm physical
  differences ~13x (the simulator leaks only a little state between calls);
  gradient-descent endpoints are hypersensitive to 1e-5 forward perturbations
tier: T1
mode: confirmatory              # C1-C6 pre-registered in DESIGN.md before any run; the two
                                # Genesis findings and the optimiser-sensitivity finding are exploratory
date: 2026-09-24
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  (C1) the "many" goal set (26 helvetica_thin letters + 4 quadrants) is
  difficulty-matched to the eval_report harness's 3 goals (every model's
  goal-averaged lyapunov slateN within 0.05); (C2) it raises the number of
  CI-resolved model pairs >= 1.5x on L20mm AND L40mm; (C4) on DS-0001 (20
  states x 12 goals, 6 EXP-0023 arms) goals replicate states for
  gradient-descent action quality (sd shrink >= 2x from 1 to 12 goals);
  (C5) >= 5/15 arm pairs Holm-significant on grad_capture; (C6) this
  pipeline reproduces EXP-0023's stage-2 dv to 1e-4.

prediction:
  supports: >
    C1: all |shift| <= 0.05. C2: >= 1.5x on both. C4: shrink >= 2x. C5: >= 5/15.
    C6: max |ddv| <= 1e-4.
  refutes: >
    C1: any |shift| > 0.05. C2: < 1.2x on both. C4: < 1.5x. C5: <= 1/15. C6: > 1e-4.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 3bae8cd7
  dirty: true                   # this session's uncommitted code: eval_report --goal-set,
                                # gt_bank CPU pinning, plus EXP-0026's uncommitted changes and the
                                # user's pre-existing edits (see body)
  script: >
    Baselines/common/eval_report.py --goal-set many (RUN-0001);
    code/item1_compare.py (RUN-0004); code/stage1_grad_many.py (RUN-0002);
    code/stage2_genesis_bank.py (smoke only, rows deleted); code/check_c6.py;
    code/probe_batch_dependence.py, code/probe_reset_fix.py (RUN-0003)
  data: >
    eval_report CORPORA L20mm, L40mm, randlen_test (same slates as EXP-0026
    RUN-0001); DS-0001 (20 states, EXP-0023's pool generator);
    EXP-0023 artifacts (stage1_actions.pt, stage2_genesis.pt)
  code_path: >
    item 1: eval_report._load_cell -> predict_occ -> _capture_report(goal_set)
    -> paired_stats. item 2: simple_mpc.adapters OCC_ADAPTERS predict_step ->
    row-wise lyapunov over 12 goal fields -> EXP-0023's projected Adam ->
    GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=False)
  seed: "pool: torch.randperm, generator seed 0 (EXP-0023's); bootstrap seed 0"
  split: "not applicable -- no fitting"
  data_commit: not applicable
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005]
  runtime: "RUN-0001 ~30 min CPU; RUN-0002 4.4 min GPU; RUN-0003 probes ~4 x 1.5 min GPU"

budget:
  declared: "not re-declared by the user for this step; working to EXP-0026's ~3 h"
  spent: "~2 h wall-clock; ~35 CPU-min, ~12 GPU-min"
  outcome: stopped-early     # item 2 stage 2 stopped on the C6 failure; C4/C5 not run

design:
  varied:
    goal_set: "harness default (3) vs many (30); item 2: 12 goals"
    models: "12 eval_report models (item 1); 6 OCC_ADAPTERS arms (item 2)"
    corpus: "L20mm, L40mm, randlen_test (item 1); DS-0001 (item 2)"
  held_fixed:
    slates: "eval_report's own step-0 pools; DS-0001's 20 states, EXP-0023 pools for 0-9"
    optimiser: "EXP-0023's projected Adam, 120 steps, lr 1.5e-3, 20-70 mm, 4 mm margin"
    value_fn_item2: lyapunov only
  baselines: >
    random pick (slateN / capture 0 by construction) and persistence
    (degenerate ranker) for item 1; pool mean and pool best (capture 0 / 1) for item 2.
  metric: slateN (item 1); grad_capture / rank_capture (item 2, defined in DESIGN.md; not yet in METRICS.md because never computed)

noise_floor: >
  Item 1: per-pair sd of the paired per-slate lyapunov slateN difference,
  30-goal: 0.064 / 0.055 / 0.141 (L20mm / L40mm / randlen_test). Item 2:
  NOT established -- the ground-truth repeat noise found here (up to 7e-3 dv
  across calls; 4.8e-2 regime offset) is itself the missing floor. No
  training-seed floor anywhere (TODO H1).

depends_on: [goal-mask-axis-convention-row-y-col-x, randlen-step0-pool-size-128]
establishes: [genesis-snapshot-restore-repeat-determinism]

# ---- outcome --------------------------------------------------------------
result: >
  C1 REFUTED: the 30-goal set is HARDER, not matched -- goal-averaged
  lyapunov slateN drops by up to 0.15 (L20mm) / 0.19 (L40mm) / 0.04
  (randlen_test). C2 BETWEEN THRESHOLDS: CI-resolved pairs 38 -> 46 (x1.21)
  on L20mm, 22 -> 51 (x2.32) on L40mm, 47 -> 47 on randlen_test; Holm-
  significant pairs 19 -> 31, 9 -> 39, 42 -> 44. C4, C5 NOT RUN. C6 FAILED
  (1.55e-2). RUN-0005: over 83 copies of one push from one restored state,
  particles land within 0.9 mm median / 4.4 mm max (particle-based dv sd
  3e-4 vs 3.2e-2 between actions) but image-based dv spreads 1.9e-2 (sd
  4e-3): the footprint rasteriser, not the physics, makes most of the
  difference (new tag occupancy-dv-subpixel-stable, broken). The brief
  invalidation of EXP-0023/0024 was withdrawn. Stage 1 also showed GD endpoints move up to 18 mm between two
  runs whose forwards differ by ~1e-5.
verdict: inconclusive
downgrades: [incomplete-design, inconsistency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

EXP-0026 found goals near-independent replicates of a slate, using random
disks / rectangles that were easier than the harness's goals. Item 1 widens
the harness's OWN goal family so the gain could be measured at the harness's
difficulty; if letters (all centred, overlapping) share most of their
information, resolved pairs barely move and C2 refutes. Item 2 asks the same
question for gradient-descent action quality, where each goal needs its own
optimisation and simulation, and so could behave differently.

## What was actually run

- RUN-0001: `eval_report.py --goal-set many` (new option; the default 3-goal
  path was regression-checked byte-identical against EXP-0026 RUN-0001 on
  L40mm for 2 models before the run), 12 models x 3 corpora.
- RUN-0004: `code/item1_compare.py`, default vs many, paired per slate.
- RUN-0002: `code/stage1_grad_many.py`, 20 states x 12 goals x 6 arms. Pools
  for slates 0-9 asserted identical to EXP-0023's; corner a_rank identical to
  EXP-0023's on 10/10 slates for all 6 arms.
- RUN-0003 (stopped): stage 2 smoke test on slate 0 (175 unique actions
  simulated and banked), `check_c6.py` FAILED, then two probes on the same
  slate (`probe_batch_dependence.py`, `probe_reset_fix.py`). The 175 banked
  rows were DELETED from DS-0004 (history-contaminated draws). The full
  stage 2 and stage 3 (`code/stage3_analyse.py`, written, never run) were not run.
- A first stage-2 attempt died on CUDA OOM (item 1's run held 7 GB), a second
  on a device bug in `gt_bank.py` (Genesis sets torch's default device to
  CUDA); fixed with explicit CPU pinning + a test.

Dirty tree: this session's `eval_report.py` `--goal-set` option and
`gt_bank.py` CPU pinning, on top of EXP-0026's uncommitted changes and the
user's pre-existing edits.

## Numbers

### Item 1 — 3-goal vs 30-goal harness (results/item1_compare.json)

| corpus / vf | resolved pairs 3 -> 30 goals | Holm < .05 | median sd_diff | slates for d=0.02 | goal ICC (30) | max mean shift |
|---|---|---|---|---|---|---|
| L20mm lyapunov | 38 -> 46 (x1.21) | 19 -> 31 | 0.136 -> 0.064 | 368 -> 82 | 0.019 | 0.153 |
| L40mm lyapunov | 22 -> 51 (x2.32) | 9 -> 39 | 0.070 -> 0.055 | 100 -> 61 | 0.022 | 0.193 |
| randlen_test lyapunov | 47 -> 47 | 42 -> 44 | 0.185 -> 0.141 | 676 -> 394 | 0.128 | 0.044 |
| L20mm mass / signed | 35 -> 46 / 28 -> 41 | 26 -> 32 / 25 -> 31 | | | 0.027 / 0.024 | |
| L40mm mass / signed | 22 -> 47 / 22 -> 51 | 4 -> 28 / 1 -> 38 | | | 0.023 / 0.020 | |
| randlen mass / signed | 48 -> 54 / 49 -> 55 | 37 -> 48 / 36 -> 48 | | | 0.054 / 0.107 | |

On L20mm the sd halves (2.15x) but resolved pairs rise only 1.21x: many
remaining pairs are genuinely close. randlen_test again shows a slate-level
component (ICC 0.13) that goals cannot average out.

**Ranking, lyapunov, 30 goals** (mean [95% bootstrap CI over slates], rank interval, P(first)):

- L20mm: nfd_residual_worldframe_noaug_ep43 0.861 [0.837, 0.885] rank 1-2 P1=0.91;
  gnn_l20l40 0.844 (1-6); linear_switched_res32 0.835 (2-7); the NFD variants
  0.816-0.831 (2-8); linear_switched_res64 0.786 (9); gnn_randlen_n30 0.682 (10);
  linear_single_res32 0.505 (11); linear_single_res64 0.411 (12).
- L40mm: gnn_l20l40 0.902 [0.887, 0.914] rank 1 P1=0.99; linear_switched_res32
  0.881 (2-3); nfd_randlen 0.870 (2-5); other NFD variants 0.810-0.850 (3-10);
  linear_switched_res64 0.842 (4-8); gnn_randlen_n30 0.782; linear_single_res32 0.773;
  linear_single_res64 0.695 (12).
- randlen_test: the six NFD-family models 0.909-0.932, ranks overlapping 1-6
  (top P1=0.56, nfd_residual_warped_flipaug_randlen); linear_switched_res32
  0.853 (7, resolved); gnn_randlen_n30 / linear_switched_res64 /
  linear_single_res32 0.63-0.69 (8-10); gnn_l20l40 / linear_single_res64 0.47-0.48 (11-12).

**The goal set changes the ORDER, not just the precision.** E.g. L40mm
`linear_single_res32` is 0.915 (rank 3-11) under 3 goals and 0.773 (rank
10-11) under 30; `nfd_warped_randlen_flipaug` falls from 0.911 to 0.810.
Two readings, not separable here: letters are a harder task that exposes
weaker models, or different goals simply reward different models. Either way
a model ranking is conditional on the goal distribution, which must be chosen
to match the MPC tasks it is meant to predict.

### Item 2 — stage 1 (artifacts/RUN-0002-grad-stage1/stage1.pt)

Mean bound-hit fraction (box / length): nfd_3ch_randlen 0.04/0.17,
nfd_3ch_finetuned 0.03/0.25, nfd_warped_randlen 0.01/0.14, nfd_residual_warped
0.01/0.04, linear_switched_soft 0.02/0.23, linear_switched_hard 0.00/0.06.

**Optimiser sensitivity (exploratory).** On the corner goal, slates 0-9, the
rank-only pick is identical to EXP-0023's and today's code reproduces
EXP-0023's recorded predictions exactly (0 difference on its a_grad; <= 5e-5
on a_rank, from pool batch size), and single-vs-batched forwards agree to
<= 8e-6 -- yet the gradient-descent endpoint differs from EXP-0023's by a
per-arm median of 0.2-0.5 mm (nfd_3ch_randlen, linear_switched_soft/hard),
~0.7 mm (nfd_3ch_finetuned), ~3.4 mm (nfd_residual_warped) and ~13 mm
(nfd_warped_randlen: 2-18 mm on 10/10 slates), max 17.7 mm,
and the best predicted dv differs by up to 0.083. 120 projected Adam steps
amplify ~1e-5 numerical differences into a different local optimum. One GD
run per (model, state, goal) is therefore a noisy sample of the model's
optimisation outcome, not a property of the model.

### Item 2 — stage 2: C6 failure and the ground-truth finding (RUN-0003)

C6: re-simulating EXP-0023's own slate-0 actions through this pipeline gives
dv up to 1.55e-2 from EXP-0023's recorded stage-2 values. In EXP-0023's own
batch, linear_switched_soft and linear_switched_hard have the identical a_rank
yet recorded dv -0.0642 vs -0.0637.

Probes, DS-0001 slate 0, 32 envs, full fidelity, lyapunov(corner):

| test | max |ddv| |
|---|---|
| identical 32-action batch, run twice in a row | 7.3e-3 (positions 17 mm) |
| same 8 actions, other slots + other companions | 5.2e-3 |
| 32 copies of one action, first call | 0 |
| 32 copies of one action, after other calls | 1.9e-2 spread |
| with scene.reset() before each call: same batch twice | 5.8e-3 |
| with reset: same actions, other companions | 9.3e-3 |
| with reset: 32 copies of one action | 0 |
| un-reset batch vs reset batch, same actions | 4.8e-2 |

So: within one call identical actions agree (what EXP-0024 tested), but
across calls an action's outcome depends on its batch companions and on the
env's history, and the reset / un-reset regimes differ systematically by as
much as the between-action spread (pool sd ~0.02-0.03 on DS-0001). The
mechanism is not established (candidates: GPU-nondeterministic contact
accumulation that only matters once envs diverge, and solver / plate state
that set_particle_state does not reset).

### RUN-0005 — the non-repeatability, made visible (artifacts/RUN-0005-nondeterminism-video/)

Same slate, same call sequence as the probes, every physics step recorded
(all slots' particles + plate pose; env-0 camera). Test push 0 ran 83 times.

- Deterministic given the call history: a full rerun reproduced every value.
- Particle start states restored exactly (error 0) in every call.
- A batch runs as many steps as its longest push: call C (all copies of one
  short push) ran 88 recorded steps, mixed batches 106 -- so companions change
  how long a push's material keeps being simulated.
- Between the two most different copies (same call, same step count), plate
  POSITION agrees to 3e-5 mm throughout but its YAW differs by a constant
  0.51 deg from the moment it is placed; before placement the two plates sat
  wherever the previous call left them (37 mm / 159 deg apart). Particles first
  separate (> 0.1 mm) at sweep step 12; final max separation 2.2 mm, 3 of 20
  particles moved > 1 mm (`before_after.png`, `pair_topdown.mp4`).

| 83 copies of one push | spread | sd |
|---|---|---|
| particle-based lyapunov dv (same distance field, sampled at particle centres) | 1.5e-3 | 3e-4 |
| image-based lyapunov dv (the score every benchmark uses) | 1.9e-2 | 4.0e-3 |
| between 1000 different pushes on this state (image-based) | | 3.2e-2 |

The extreme pair's images hold 95 vs 90 occupied pixels (a particle covers 4,
5 or 6 pixels depending on its sub-pixel position; overlaps are clipped at 1)
while the distance-weighted sums are 27.19 vs 27.50: V is normalised by the
pixel count, so ~1 mm of motion swings dv by up to 0.019. So the large numbers
in the probe table above are image-scoring noise on top of a small physical
leak (plate-yaw residual, step-count dependence). Physically the simulator is
repeatable to ~1e-4 of the between-action variance; the SCORE is repeatable
only to ~1.6% of it (sd ratio 0.12).

## What would change the verdict

- A characterised or fixed simulator: either a reset protocol that makes
  repeats identical (then re-run stage 2 as designed, ~35 min), or a
  measured repeat-noise sd per action so ground truth can be averaged over
  replicates (~2-3x stage-2 cost). Either unblocks C4/C5.
- Knowing which regime (reset / un-reset) matches the corpus collection path,
  which decides whether DS-0001 / slates_multistep / randlen ground truth
  carries the same noise.
- For item 1's reordering: a goal set drawn from the actual MPC task
  distribution (closed-loop tasks), which is the ranking that matters.

## Threats

- The Genesis probes use ONE state and ~40 distinct actions; the magnitudes
  are one state's, the existence of the effect is not in doubt (it
  reproduces in a strict-xfail test with plain pool actions).
- Item 1's letter goals are all centred; a goal family with off-centre
  targets might replicate states better or worse.

## Unrelated findings

- `simple_mpc/gt_bank.py` broke under Genesis because Genesis calls
  `torch.set_default_device("cuda")`; fixed (explicit CPU tensors) with a test.
  Any other Genesis-adjacent helper that creates tensors without a device has
  the same latent bug.
- A background "wait for run_probe exit" watcher never fired: `run_probe`
  prints its exit line to its own stdout, not to the job log.
