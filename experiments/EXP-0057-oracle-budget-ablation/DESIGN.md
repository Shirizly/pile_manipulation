# EXP-0057 — DESIGN: oracle (simulator-as-model) MPC budget ablation

Written 2026-09-25, BEFORE launching the main queue. Builds on EXP-0050 (`code/oracle.py`,
which already implements simulator-as-model CEM: iteration 0 = pile-aware samples, then
Gaussian refits to the true-best elite, projected legal, best-ever-simulated executed) and
EXP-0055's analysis conventions (mass/coverage metrics on `occ_for_scoring`, paired bootstrap).

## Question
For a PERFECT model (the simulator itself), at a FIXED per-decision simulation budget standing
in for a fast learned model's time budget, how many pushes does closed-loop MPC need to solve
each goal, and how does that depend on (A) how the budget splits between CEM pool size and
iterations, and (B) the planning objective? Task success is always judged by in-goal mass
(EXP-0046 `mass_frac_best_placement`), regardless of what the optimiser minimises.

## Physics / candidate generation (checked, not assumed)
`code/oracle.py` builds ONE `GenesisOracleEnv`/`SandboxManipulation` and uses it for BOTH
candidate simulation and the executed push — same object, same `_settle_steps`/
`_clearance_ctrl_steps` (copied from `env._real_settle_steps`/`_real_clearance_steps`), same
`TRAINING_PHYSICS` (`simple_mpc.learned_mpc.apply_physics`: friction 0.7, density 450, box
friction 0.5, settle 3000) applied via `set_material_properties` after construction. So
candidate rollouts and the executed push are physically IDENTICAL by construction — there is
no train/eval physics mismatch to check here (unlike a learned model). Candidates come from
`sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)`, the same pile-aware
sampler EXP-0050/EXP-0045 use, projected legal via `project_push` (20-70 mm default; ablated in
the extra queue).

## Step 1 — throughput test (results/throughput.json)
Measured: 20-cube TRAINING_PHYSICS settle+push, batched at n_envs = 32/64/128/256, 3 batches
each, GPU otherwise idle (`nvidia-smi` confirmed empty before starting):

| n_envs | sims/s | nvidia-smi peak MiB (of 8188) |
|---|---|---|
| 32 | 3.29 | not polled |
| 64 | 4.81 | 1915 |
| 128 | 5.99 | 3537 |
| 256 | 7.06 | 7413 (90% of the card) |

Diminishing returns (1.46x/1.24x/1.18x throughput per doubling) and 256 leaves only ~10%
headroom on a SHARED 8 GB card for a job that must run 12-60 h unattended — one memory spike
(fragmentation, a larger candidate batch, another process) risks an OOM crash. **Chosen:
n_envs = 128** (43% memory, comfortable margin, still 82% of the 256-env throughput).

`code/oracle.py`'s existing batching ALREADY does what the brief asked "restructure if
needed": `sample()`/`simulate()` flatten every episode's candidates into one list and chunk it
by `n_envs`, so raising `--n-envs` alone fills envs efficiently regardless of episode count or
split — no restructuring was needed, only exposing `K` as `--n-envs` (was hardcoded 32).

## Step 2 — sizing and the budget fallback (APPLIED)
With 16 episodes (8 goals x 2 starts) run together in lockstep (oracle.py advances every
episode's push k in the same batched sim calls), one push's total simulated actions = 16 x
budget, and total wall time for one cell = `steps x (16 x budget) / throughput`, INDEPENDENT of
the P/I split (P x I = budget is fixed; the sim only cares about the total action count per
push, not how it's divided into iterations). Measured throughput at n_envs=128 = 5.99 sims/s.

**Per cell (16 tasks, 20 pushes, batched together) = 20 x 16 x budget / 5.99 s.**
- budget 512: 20 x 16 x 512 = 163,840 simulated actions; /5.99 = 27,353 s = **7.6 h/cell** x 5
  (A) cells = **38.0 h** for priority 1 alone — far over the 12 h target.
- budget 256 (halved, per the pre-agreed fallback): 20 x 16 x 256 = 81,920; /5.99 = 13,677 s =
  **3.80 h/cell** x 5 (A) cells = **19.0 h** — still over 12 h.

**Fallback APPLIED: budget = 256 sims/push (the pre-agreed halving).** It still does not fit
all 5 (A) cells in 12 h, so a SECOND deviation was needed (see "Deviations" below): the main
queue runs 3 of the 5 (A) splits (~11.4 h), and the other 2 (A) splits move to the front of the
EXTRA queue, so the full (A) ablation still completes, just not inside the 12 h window. 16 tasks
(the pre-agreed lower trim) is kept throughout — never trimmed further, per the brief's "trim
tasks, never the pairing."

Task set: letter_O, letter_T, letter_S, letter_X, letter_L, letter_I, two_squares, quadrant_0
(EXP-0055's 8-goal set) x DS-0006 starts 40, 41 (16 tasks). Extra queue adds starts 44, 45
(paired extension) once time allows.

## Objective / value functions (code/oracle.py `--value`, `--mass-weight`)
- `lyap` (default): pure lyapunov, `simple_mpc.learned_mpc.lyap`.
- `lyap - w x mass`: `--mass-weight w` (already existed, EXP-0052's objective); w in {1, 3}.
- `crowd_floor`: lyapunov + 0.1 x `simple_mpc.value_functions.CrowdingPenalty(mask,
  floor_single=True)` -- added for this experiment, same formula `simple_mpc/learned_mpc.py`'s
  `ModelObjective` uses for `value="crowd_floor"`.
`values[]` in the results JSON is ALWAYS pure lyapunov (for cross-cell comparability); the
raw selection objective is `obj_values[]`. Task success is ALWAYS judged by `mass_frac[]`
(in-goal mass fraction on `occ_for_scoring`), never by the planning objective.

## Design cells
(A) split ablation, objective = lyapunov, budget = 256 sims/push, elite fraction 0.25 (fixed):
| cell | P (cem-pop) | I (cem-iters) | elite |
|---|---|---|---|
| default | 64 | 4 | 16 |
| 256x1 | 256 | 1 | 64 |
| 128x2 | 128 | 2 | 32 |
| 32x8 | 32 | 8 | 8 |
| 16x16 | 16 | 16 | 4 |

(B) objective ablation, default split (64x4):
| cell | objective |
|---|---|
| B_mass1 | lyap - 1 x mass_frac |
| B_mass3 | lyap - 3 x mass_frac |
| B_crowd | crowd_floor |

Extra queue also runs: remaining starts (44, 45) for all 5 (A) splits; push-length-range
ablation at the default cell (20-70 mm default already covered by `oracle_A_default`; 20-40 mm;
fixed ~20 mm via `--l-min 0.020 --l-max 0.0201`).

## Completion / censoring
First push k with in-goal mass fraction >= 0.9 x `mass_frac_best_placement` (EXP-0046
`results/vstar.json`), on `occ_for_scoring`; since the 22:35 relaunch episodes stop being
simulated once solved (`--stop-solved`; state carried forward, so `mass_frac[k]` for k > k* stays
at the solved value). Seeded per (task, push, iteration) with `--seed-base 0`. Unsolved-by-20 -> censored at 21 in the
`censored_mean_pushes` summary (`experiments/METRICS.md` `oracle_completion_pushes`); solved
episodes also get `median_pushes_solved_only`. Paired differences are per (goal, start) vs
`oracle_A_default`, bootstrap 95% CI (4000 resamples), in `code/analyse.py`.

## Deviations from the pre-registered brief (both forced by measured throughput)
1. **Budget 512 -> 256** (pre-agreed fallback, applied because even the halved budget alone
   does not fit 5 cells x 16 tasks in 12 h — see Step 2).
2. **Priority-1 split 3 (main) + 2 (extra-queue front)**, instead of all 5 in the main queue.
   The brief's only explicit next fallback after halving was "trim tasks, never the pairing",
   but 16 tasks is already the pre-agreed floor trim; splitting the (A) sweep across the two
   queues (main finishes in ~12 h with the default cell centred by one split on each side;
   extra queue's front two complete the sweep before priority 2 or any extension work) keeps
   every cell fully paired and does not trim below 16 tasks. Both deviations are stated here,
   before the queues launch, per T1's plan-gate discipline.

## Cost / risk
Cheapest check that could have invalidated the plan: the throughput test itself (Step 1, ~8
min GPU). It did — the pre-registered 512 budget was infeasible, caught before any ablation
cell ran. The one result that would most embarrass this design: if `oracle_A_256x1` (pure
sampling, no CEM refinement) does about as well as the default split — that would say the
budget split doesn't matter and the whole (A) axis is degenerate; the design catches this
because 256x1 is one of the two extremes and is scheduled early (front of the extra queue,
right after the main queue).
