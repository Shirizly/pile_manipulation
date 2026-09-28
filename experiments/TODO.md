# TODO — prioritised task list

Written so an **orchestrator agent can work straight down it** without asking
the user anything. Each task states: what to do, why it matters, what would
count as done, and the known traps. Priorities are HIGH / MID / LOW.

**Standing rules for every task here** (they have each been violated at least
once in this project):
- Lead with `slateN`; `accuracy` is a diagnostic, not a verdict
  (`experiments/METRICS.md`). Use `random` as the ranking floor, never
  `persistence` — it predicts `dv=0` for every candidate and is degenerate.
- `accuracy` is a **ratio of population means**, not a mean of per-row ratios.
- **Reproduce a known number before trusting a new pipeline.**
- Match training arms by **gradient steps**, measured, not by epoch count. The
  loader batch is `batch_size // augmentation_factor`, so an "epoch" means
  different things under different augmentation settings.
- One GPU, 8 GB, shared machine: run training **sequentially**, never parallel.
- Persist every fitted object with its resolved config. Record runs through
  `experiments/COMMANDS.jsonl` and the `experiment-log` structure.
- Never commit unless the user asks.
- **Every job checkpoints as it goes** (results/manifest rewritten atomically after
  every unit of work) -- `project-overview`, "Every job must survive being cut off".
- **Score ground truth with the soft, mass-conserving rasteriser** (default since
  2026-09-24; METRICS.md "Ground-truth scoring"). Never mix soft- and image-scored
  numbers in one comparison.
- **L20mm / L40mm are for pilots only** -- too narrow for conclusions. Real results
  use broad training data and diverse test sets.
- **Never simulate > 50 objects; schedule any sizeable 30+ collection carefully.**
  Piled spawns are deprioritised (artificial here; another engine will supply
  realistic piles, not directly comparable).
- **Null results are kept and written up**, not dropped.

---

## The goals this list serves (user, 2026-09-24)

1. **Which offline metrics predict closed-loop MPC performance** (gradient descent,
   CEM/MPPI)? slateN is the working assumption; trusted gradient data is missing.
2. **Which metrics are consistent and informative** (reliable across goals, pools,
   seeds) -- the broader, easier version of 1.
3. **Where do models significantly beat each other, and what about the STATE
   decides it** -- a short descriptor is best, a simple (linear) map second, a
   trained selector last -- and a switched pipeline that beats any single model.
4. Toward a **unified benchmark dataset**: same-state pools (states x pools x
   actions, binned push lengths, full particle states stored, soft truth scoring).
   DS-0001 and DS-0005 have this shape; new collections should too.

---

## NOW 2026-09-25 ~08:50-11:50 (3 h cap, user): capacity-aware value functions + goal-aware sampling
User diagnosis: thin shapes fail because the value function (and the sampler) is blind to what is already
in the goal; also needed when the goal is uniform COVERAGE, not just "mass somewhere inside".
- EXP-0055 (value functions: V1 sliced EMD to the uniform target, V2 linearised EMD field, V3 over-capacity
  repulsion) and EXP-0056 (samplers: S1 misplaced mass, S2 deposit-aware carry model, S3 OT mix) -- one
  shared run, EXP-0055 RUN-0001, 7 cells x 32 episodes (8 goals x starts 40-43), GD 1 s, 20 pushes. DONE:
  only V3 (crowding) helps, modestly (+0.03 in-goal mass, +0.075 coverage, completion time unchanged); V1 EMD
  trades in-goal mass for EMD; V2 fails (overshoot); all samplers -0.03..-0.04 (C-058, C-059). RUN-0002
  (exploratory) crowd_floor fixes most of V3's quadrant failure. Repeat noise: per-episode sd 0.09 in-goal mass.
  Report per cell: the optimised value + lyapunov + mass_in_region/signed + coverage + completion time.
- crossed cell not run (no sampler won). NEXT candidates: CEM with crowd_floor and with the samplers
  (~16 min/cell); crowd_floor lam/sigma sweep on more starts (needs ~4x episodes to resolve +0.03);
  lyapunov churn (~1.8 cubes knocked out per push late) suggests placement prediction, not the value, limits.
- Visualisations DONE: scripts/probes/transition_panel.py (2x3 transition figure; hard-gate linear =
  linear_switched_hard), GIFs with the orientation fix (EXP-0051 + EXP-0055 artifacts/demos/).
  OPEN (user decision): letter goals read correctly only with world y DOWN; seen from above they are
  mirrored -- fixing letter_mask would change every goal.
- Cut for the 3 h cap (resume later): CEM for the samplers (the pool matters most there); crowd_lam sweep
  (0.1 set a priori); success-objective reference cell (EXP-0052 has it, unpaired); clump starts.

## OVERNIGHT 2026-09-25 01:35-09:30 (8 h budget, user-approved plan; run without stopping)
Status markers: [ ] todo, [~] running, [x] done, [>] deferred to TODO below.
Rules: 1 GPU -> GPU stages strictly sequential; CPU stages in parallel. Every job checkpoints atomically.
Conservative on time; deeper noise control later. New sims: PERFECTLY perpendicular pushes only,
push length exactly 20 mm, n = 20, single layer (scatter AND single-layer clumps), training physics
unless stage C says otherwise. Existing 18-22 mm / near-perpendicular rows: extra TRAINING data
only, never test.
- [x] A DONE EXP-0046: V*<=0.006 all goals (I/Y/J capacity <20); closed loop reaches 0.87-0.92 of achievable
      after 8 pushes, 0.94-0.96 after 24 (b>=0.1 s); lyapunov optimum puts only 82-97% of mass inside letters
      (soft-splat spill); two_squares goal added (goal-set many_plus); 12-goal reduced set O T F S X M W J L I q0 q1.
      (was: A (CPU agent, 45 min): goal ceiling V*(g) for every goal (lyapunov-optimal placement of 20 cube
      footprints, scored with the soft scorer); check the lyapunov-optimal state also fully
      satisfies mass_in_region and signed_mass; add a two-disconnected-squares goal; goal
      redundancy analysis (shape + per-goal performance similarity) -> droppable goals;
      normalised re-score (V0-V)/(V0-V*) of EXP-0039/0044/0045.
- [x] B DONE EXP-0047: no clean exact-20mm perp chained n20 single-layer data at training physics -> collect;
      ~6.2k matched-physics 18-22 mm single-layer rows (overnight_randlen + Sean) usable as extra training;
      single-layer clumps exist only in Sean piled/inbetween + overnight mixed. (was: B (CPU agent, 30 min): data inventory -- per corpus: physics, n, layers (single-layer incl.
      clumps), push length, perpendicularity (+-2 deg), chains; counts for the narrow domain.
- [x] C DONE EXP-0048 (5/12 states; resumable at slate 105): other physics sets change 20 mm push outcomes
      ~ a 0.5 mm action jitter (2x repeat noise), dv diff 0.05 between-action sd, ranking Spearman 0.991
      -> POOL P_binned / P_oracle 20 mm data; longer / multi-step other-physics data pretraining-only until
      tested. Soft-scored input changes 14% already at 0.25 mm jitter. (was: C (GPU agent, 45 min): physics-sensitivity pilot -- same states + actions under each
      dataset physics set vs imperceptible state/action perturbations under one physics.
      Decides: mix / pretrain-only / drop the other-physics data.
- [x] D DONE EXP-0049 (8/10 scatter + 5/10 clump states; resumable ~25 min): pile-aware min_swath 3 is the
      best 20 mm perpendicular candidate source (best-of-16 beats placement-aware best-of-96; placement
      22%/63% null pushes); relaxing or mixing does not help -> no sampler-improvement agent (user rule).
      Pile-aware SHORTENS wall-limited pushes (some to ~0 mm): a collector must redraw them. Clump
      generator: EXP-0049 code/sampler_study.py make_clump_states. (was: D (GPU agent, 60 min): sampler study (pile-aware strict / relaxed, placement-aware
      perpendicular 20 mm, uniform) on scatter + single-layer-clump states: null-push fraction,
      particles moved, best-of-N true value per goal.
- [~] E EXP-0050: greedy16 DONE (0.080 at 8 pushes; pure sampling weak -- NOT evidence of a sampler problem, user
      correction); simulator-CEM lyapunov (8 goals x start 40, 12 pushes) RUNNING (queue_E4.sh, ~15 min/push);
      simulator-CEM with success objective (letters O T S L + two_squares) QUEUED after EXP-0052 (queue_E5.sh).
      (was: E RUNNING EXP-0050 (reduced: greedy16 8x8, lookahead 4x6, greedy64 8x12; queue_E2.sh): ceilings -- oracle greedy (true best of N simulated candidates per push)
      closed loop on the EXP-0045 cells + a 2-push-lookahead oracle check on a few episodes.
- [x] F DONE 05:27: DS-0008 6,144 train rows (all valid, all single layer), DS-0009 2,048-push pools + 1,024 chain
      rows, DS-0010 5,777 extra rows. (restarted 04:13 since EXP-0052 showed the objective only partly explains letter failure):
      DS-0009 test pools + chains, DS-0008 train (24 chunks) via queue_F.sh; DS-0010 extra 5,777 rows extracted.
      (was: F on hold) 20 mm perpendicular collection (train + clean test), sampler from D,
      states scatter + single-layer clumps, only what B says is missing.
- [x] G DONE 06:31 EXP-0053 (C-056): narrow NFD acc 0.506 / slateN 0.774 vs broad same-arch 0.453 / 0.695; wide NFD
      0.486 / 0.803 (overfits after epoch 15); linear narrow res64 0.452 / 0.582. configs ready -- Baselines/NFD/configs/nfd_3ch_narrow_l20{,_wide}.yaml (60 epochs), linear single operator
      via Baselines/LinearForesight/fit_switched.py --n-bins 1 --train-cfg configs/dataset/genesis_narrow_l20_train_plus_extra.yaml
      --test-cfg ..._test_chains.yaml --res 32/64. (was: G (GPU, ~90 min): linear single-operator 20 mm res32 + res64; NFD res64 on the narrow domain;
      one deeper/wider NFD if time.
- [x] H DONE 07:34 EXP-0054 (C-057): under 20 mm pushes the narrow NFDs complete letters WORSE (0.45/0.52 vs 0.56/0.57
      in-goal mass) despite better offline metrics; fixed 20 mm pushes are far slower than free 20-70 mm (0.74). EXP-0053 offline eval (broad baselines done: acc1 0.40-0.47, slateN 0.70-0.72)
      then EXP-0054 closed loop (20 mm, GD w3, 4 models x 7 goals x 4 starts). (was: H (GPU, ~45 min): evaluation -- accuracy, slateN on the clean test set, one-step planner,
      closed-loop (1 s budget) normalised by V*, vs the oracle ceiling.
- [x] S DONE EXP-0051 (C-052): letters NOT solved (in-goal mass 0.62-0.68 of 0.92-0.97, 0/48 complete) although
      lyapunov says 96%; quadrants done in ~5 pushes; two_squares 75%. -> success-objective test EXP-0052 RUNNING.
      (was: S (GPU, ~75 min) EXP-0051: task success + completion time -- 72 episodes x 24 pushes, states
      recorded; mass-in-region / signed-mass fractions vs optimum; pushes/time to 80/90/95% (user
      2026-09-25: TASK COMPLETION TIME incl. action time is the benchmark's one true utility; always
      report mass_in_region / signed_mass beside lyapunov).
- REPLAN 2026-09-25 02:00 (user): models already reach 87-96% of achievable lyapunov (EXP-0046), so
      F/G (20 mm collection + training) are ON HOLD / lower priority; if EXP-0051 shows tasks are
      done, pivot the benchmark to harder settings (two_squares, mass-based success, fewer pushes,
      single-layer clumps, n30-50 carefully) scored by completion time vs the simulator ceiling.
      Looser pile-aware sampling only if D/E show the sampler limits TASK SUCCESS.
- NEXT (proposed, not run): CEM under the 20 mm restriction (gradient-free) to test the landscape hypothesis;
      mixed action space (long gathering pushes + 20 mm placement pushes); multi-step lookahead with the narrow
      model (rollout accuracy is best there); resume EXP-0048 (7 more states) and EXP-0049 (7 more states).
- [>] NFD res128 in/out (MID priority, user 2026-09-25); own-output (multi-step) training + multi-step planner (after E's lookahead
      check); more seeds -- deferred, user to budget.

## LATER (user 2026-09-25): expert-demonstration videos -- first version DONE: push-by-push GIFs from recorded
states (EXP-0051 code/demo_gifs.py -> artifacts/demos/, best episode per goal). Still to do: real Genesis
re-simulation videos (plate motion) of the best sequences.
Record videos of particularly good action sequences -- episodes that satisfy goals well in few
pushes (from EXP-0051's recorded states/actions, or the simulator-CEM ceiling runs): re-simulate
the recorded actions from the start state with the on_phase video hooks (utils.write_video_frame,
docs/UTILITIES.md), goal mask overlaid. Pick by completion time.

## NOW: strengthen the closed-loop benchmark (user 2026-09-24; before growing the model set)

### B0. Cheap checks (done / running)
- EXP-0039 re-analysis (free): for a MODEL DIFFERENCE, goal x model and start x model
  variance are each about as large as the residual. So add goals AND starts:
  12 goals x 8 starts resolves about 0.019 (CEM) / 0.029 (GD), vs 0.031 / 0.051 at
  4 x 4 (rough: estimated from 4 x 4). Rank saturates after about 4 pushes;
  GD/CEM still gain about 0.01 per push at push 8.
- Batched execution (temp/batched-exec-timing): 32 envs execute their own pushes
  in ~11.5 s vs ~4.5 s for 1 env (12x throughput); pile-aware sampling costs ms.
  -> a batched closed-loop runner makes episodes ~5x cheaper; planning becomes the
  bottleneck.
- EXP-0042 DONE (pilot + sweep on fresh states, 2 models; C-047): the EXP-0039
  planner defaults were far from optimal. GD lr 5e-3 / 32 restarts +0.29-0.44 one-step
  capture; CEM 1024 / elite 0.25 +0.13-0.14; tuned GD ~ tuned CEM; GD 0.3 s and CEM 0.1 s
  lose nothing; tuning shrinks the worldframe GD lead 0.28 -> 0.13.
  NEW DEFAULTS: GD lr 5e-3, 32 restarts; CEM pool = pop 1024, elite 0.25.
- EXP-0043 DONE (C-048): batched runner matches sequential (7/8 cells); repeat sd
  CEM 0.01-0.02, GD 0.03-0.04.

### B1. Batched closed-loop runner -- DONE (learned_mpc.run_episodes_batched; EXP-0043 code/batched_closed_loop.py)
`learned_mpc.run_episodes_batched`: K episodes in K envs, per-env start states,
sequential planning per env (timed with Genesis idle), one batched execute per step.
Needs a Genesis-free test and an equivalence check against run_episode (the same
distribution of improvement on a few EXP-0039 cells; bitwise identity is impossible).

### B2. Goal breadth -- DONE (EXP-0044, C-049): with TUNED planners the 4 models are within 0.016
(seed-level); planner tuning moves GD 0.03-0.12; tuned GD = tuned CEM; 24x16 episodes per cell
needed to resolve 0.01. NEXT for the benchmark: find a setting with headroom -- oracle ceiling
(Genesis-as-model CEM) on the same cells; tight budgets (B4); n50 states; longer horizons.
12 goal shapes (letters of different topology + quadrants) x 8 starts x the 4 EXP-0039
models x {gd, cem}. Answers: do model rankings hold across goal shapes
(model x goal interaction), and the power table for the final benchmark size.

### B3. MPC-parameter ablation in closed loop (EXP-0042 says: GD lr/restarts, CEM pool/pop/elite)
Only the knobs whose one-step effect exceeds the repeat noise floor, at 2 levels,
2 models (does a knob change the model ranking?).

### B4. Decision time vs number of actions -- DONE (EXP-0045, C-050): knee at ~0.1 s CEM / 0.3 s GD;
saturation ~0.31 after 12-16 pushes; tight budgets separate models by speed.
NEXT (benchmark design): score early pushes (k=1-4) and/or tight budgets; oracle ceiling to
see if 0.31 is the task's limit; harder goals / n50 so the plateau is not reached; fix the
pile-aware candidate distribution (pure ranking stalls at ~0.06).
Long episodes (~24 pushes) at planning budgets {0.1, 0.3, 1, 3} s, value after every
push recorded. Post hoc, for total time T and per-push execution time t_act (a
parameter: real robot seconds per push), compare value after floor(T / (b + t_act))
pushes across b. One set of episodes gives every (T, t_act). ~2 models x {gd, cem}.
One-step hint (EXP-0042): tuned GD at 0.3 s and CEM at 0.1 s lose nothing vs 1 s, so
include 0.03 / 0.1 s budgets as well.

## Oracle budget ablation (EXP-0057) — launched 2026-09-25, check progress / resume / cancel
PAUSED 2026-09-26 12:54 by the user (unexpected result: at 256 sims/decision the perfect-model oracle is
no better than the learned NFD; letters unsolved in 20 pushes; 32x8 > 64x4 > 128x2). Main queue done;
extra queue stopped in oracle_A_256x1 at push 5/20. Resume command in EXP-0057 runs/RUN-0001/RUN.md.
Interim numbers: `CUDA_VISIBLE_DEVICES="" python experiments/EXP-0057-oracle-budget-ablation/code/analyse.py`.

**What's queued.** `experiments/EXP-0057-oracle-budget-ablation/code/run_queue.py main` was
launched detached (nohup) at ~22:15 on 2026-09-25 and covers BOTH queues in one process (the
main queue's `main()` calls the extra queue automatically on completion):
- MAIN (~11.4h estimated): `oracle_A_default` (64x4 split) -> `oracle_A_128x2` -> `oracle_A_32x8`.
- EXTRA (~37-46h estimated, starts automatically): `oracle_A_256x1`, `oracle_A_16x16` (completes
  the 5-way split ablation), `oracle_B_mass1`, `oracle_B_mass3`, `oracle_B_crowd` (objective
  ablation at the default split), 5x `_starts4445` (pairs the split ablation onto 2 more starts),
  `oracle_len_20_40`, `oracle_len_fixed20` (push-length-range ablation).
Every cell = 8 goals (letter_O/T/S/X/L/I, two_squares, quadrant_0) x 2 starts, 20 pushes, 256
simulated pushes/decision, TRAINING_PHYSICS, n_envs=128. Full design and the deviations from the
pre-registered 512-budget / 5-cells-in-12h plan: `experiments/EXP-0057-*/DESIGN.md`.

**Check progress:**
```bash
ps aux | grep run_queue.py                      # is the queue driver alive
tail -f experiments/EXP-0057-oracle-budget-ablation/runs/queue_main.log   # queue-level log
tail -f experiments/EXP-0057-oracle-budget-ablation/runs/oracle_A_default.log  # current cell
cat experiments/EXP-0057-oracle-budget-ablation/results/oracle_A_default.json | python3 -c \
  "import json,sys; d=json.load(sys.stdin); print('push', d['step'], 'of 20')"
```
**Resume a killed job:** just re-run `code/run_queue.py main` (or `extra`) — `oracle.py` itself
checkpoints every push and resumes from the last one; `run_queue.py` skips any cell whose
results JSON already has `step >= 20` for all 16 (or 32, for `_starts4445`) episodes.

**Cancel:** `kill <PID from runs/queue_main.pid or ps>`, then (if a cell's own oracle.py process
is still running under it) `kill $(cat experiments/EXP-0057-oracle-budget-ablation/runs/<tag>.pid)`.
Killing only the queue driver leaves the CURRENT cell's `oracle.py` running to completion; kill
both PIDs to stop immediately. A killed cell resumes cleanly from checkpoint (see above).

**Once cells finish:** `python experiments/EXP-0057-oracle-budget-ablation/code/analyse.py`
(reads every `results/oracle_*.json`) prints the completion-pushes table and paired deltas vs
`oracle_A_default`, and writes `results/analysis.json`.

**Follow-up not yet queued:** mass-weight-1x at the BEST (A) split (design's "if time" item) --
needs the (A) analysis first to name the winner; add a `run_queue.cell(...)` call once known.
`EXPERIMENT.md`'s pre-registered prediction should be checked against `analyse.py`'s paired
deltas once `oracle_A_256x1`/`oracle_A_default` both have >= 16 completed episodes, and the
record's `result`/`verdict` updated (currently `inconclusive`/`incomplete-design` because the
run was still queued at write time).

## HIGH

### G3a. State-level model superiority -- DS-0006 done (EXP-0030), DS-0007 done (EXP-0035)
Results so far: the NFD ensemble beats the best single model on all three corpora
(+0.014 to +0.059); per-state advantage repeats only weakly across pools (r ~0.15
at 64 pushes); per-state switching loses; no short descriptor survives correction.
Next: a selector from model DISAGREEMENT (not geometry); ensemble of seeds (H1);
does the ensemble also win in closed loop (G1b)? -> ANSWERED offline instead (user 2026-09-24):
at a MATCHED time budget the ensemble LOSES by 0.14-0.53 slateN (EXP-0030 A5, C-043);
its win is an equal-count result. Dropped from EXP-0039. Open: a 1 s-scale budget
(thousands of candidates) needs deeper pools than DS-0006's 128. Original notes:
DS-0006 = 160 n20 scatter states x 128 actions, TRAINING-MATCHED physics (DS-0005 was stopped: collector defaults mismatched the training physics), (two disjoint 64-action pools per
state by design). Re-run the EXP-0029 split test at 160 states: split-half
reliability of per-state pairwise advantage, cross-fitted switching gain
(choose on pool 1, score on pool 2), with the 12 eval models where possible
(needs a DS-0005 path into eval_report or the OCC adapters). Then, for pairs
with reliable per-state advantage, regress the per-state advantage on SHORT
STATE DESCRIPTORS (particle spread / radius of gyration, nearest-neighbour
distance / clumpiness, distance to walls, n clusters, contact count) --
one descriptor at a time first, then a linear model, cross-validated BY STATE.
**Ensemble control is mandatory -- and it is now the bar**: EXP-0035 (~800 Sean
states per shard) found the average of seven NFD models' predictions (incl. the weak nfd_3ch_finetuned) beats the best
single model by +0.014 to +0.058, while per-state choice from small pools LOSES
0.02-0.05 and no short descriptor predicted the per-state winner. A switched
pipeline must beat the ensemble. Report null results.
**Done when:** either a descriptor / linear selector beats the best single model
on held-out states with a CI excluding 0, or the null is recorded with power.

### G1a. Trusted gradient benchmark -- IN PROGRESS as EXP-0037
Stage 1 done (40 DS-0006 states x 6 goals x 7 arms incl. the NFD ensemble, 3 GD
restarts each); stage 2 (Genesis, training physics, one path, ~2.3 h) runs after
the closed-loop pilot; stage 3 script written. Original notes:
Unblocked: ground truth is now scored with the soft rasteriser (physical repeat
noise ~1e-4 of between-action variance). Remaining design changes: (1) several
GD restarts per (model, state, goal) -- one run is a noisy sample (EXP-0027:
endpoints up to 18 mm apart under 1e-5 perturbations); (2) states from DS-0005
(160) rather than DS-0001 (20); (3) batch rollouts by push length (a batch runs
as long as its longest push). Score with grad_capture / rank_capture
(EXP-0027 DESIGN.md) and the paired analysis. Code exists:
EXP-0027 `code/stage1_grad_many.py`, `stage2_genesis_bank.py`, `stage3_analyse.py`.
**Done when:** a pre-registered arm comparison on dv_grad / grad_capture has paired
CIs narrow enough to rank the models, and gradient_gain is reported with them.

### G1b. Closed-loop MPC harness + the metric-validity study -- EXP-0039 DONE (4 models)
2026-09-24: accuracy 4/4 and grad_capture 3/3 agree with the resolved closed-loop pairs, slateN 4/6,
spearman 1/6; world-frame residual best under GD/CEM; seeds tie in closed loop; CEM > GD > rank
(C-045, C-046). Next if wanted: grow the population to 6-8 (seeds 2-3 + one warped, ~45 min each)
so accuracy-vs-slateN disagreements not involving worldframe exist; a second budget level.
`simple_mpc/learned_mpc.py` (rank / gd / cem / mppi, wall-clock budget, soft truth,
TRAINING_PHYSICS executor); pilot EXP-0032 (3 models x 3 planners x 2 goals x 4
starts) DONE: CEM > GD > rank; episode sd ~0.03-0.05 (so ~15-20 episodes per cell
resolve ~0.03 gaps); the offline-weakest model is weakest closed-loop; between NFD
and linear the order depends on planner AND speed (linear does 2x the CEM evals in
1 s). Real study QUEUED as EXP-0039, cut by the user 2026-09-24 to 4 models (nfd_3ch_randlen,
seed1, linear_switched_soft, residual_worldframe), 192 episodes, ~3 h; was: more models (the 12-model population incl. the ensemble and
seeds), rank planner now budget-matched, 2-3 budgets, ~16 episodes per cell, then
correlate each offline metric (slateN, accuracy, grad_capture from EXP-0037,
predicted-vs-true calibration, speed) with closed-loop improvement per planner.
Original notes:
Wire `simple_mpc/adapters.py::OCC_ADAPTERS` into the closed-loop MPC loop
(`simple_mpc/mpc.py` supports only `make_adapter`'s Eulerian/PropNet models) and
into a sampling planner (CEM/MPPI with the learned model as the objective).
Tasks: DS-0005-style n20 scatter starts, a fixed goal set, episodes of H pushes,
FIXED COMPUTE BUDGET per decision (wall-clock, a few levels). Pilot: 3 models from
different tiers, 1 task. Then a model population (checkpoints along training,
seeds, families, deliberately degraded models; ~20-30) scored both offline
(slateN, accuracy, grad metrics, optimism gap) and closed-loop, and the rank
correlation per metric -- within and across model families. "accuracy predicts
closed-loop poorly" is a result worth reporting either way.
**Done when:** each offline metric has a measured correlation (with CI) to
closed-loop performance per planner/budget.

### H1. Seed-level noise floor -- DONE (EXP-0036, C-044): slateN seed sd 0.01-0.04, accuracy 0.003
**The single largest hole in everything measured so far.** Every arm in
EXP-0022/0025 is a single training run, so margins of 0.01-0.05 cannot be
ordered, and several verdicts rest on "the direction is consistent across three
corpora" rather than on any margin. Train **3 seeds** of the world-frame NFD
baseline config (`Baselines/NFD/configs/nfd_train_3ch_randlen.yaml`) and 3 of
one warped arm, score all six through `Baselines/common/eval_report.py`, and
report the per-cell standard deviation of `slateN` and `accuracy`.
Also train/score an ENSEMBLE OF SEEDS of one config, to separate "averaging
diverse architectures" from "any averaging" (EXP-0035 ensemble gain).
**Done when:** a seed-spread number exists that later records can cite, and
EXP-0022's `noise_floor` field can be filled in with a measurement instead of a
disclaimer. **Cost:** ~6 training runs, the largest item on this list — but it
determines whether any of the close calls mean anything.

### G2a. ~~Re-issue the reference table under soft scoring~~ DONE (EXP-0028)
Rankings barely move under soft truth (tau 0.91-0.97). Original notes:
12 models x 3 corpora x {3-goal, 30-goal}, soft truth; paired CIs and rank
intervals (`Baselines/common/paired_stats.py`), randlen_test leading, L20mm/L40mm
as pilot-only context. Record how much each ranking moved from image scoring.

---

## MID

### G4a. Bigger, broader training data
Train the main model families on overnight_randlen train + Sean (n20/n50 only --
drop n100 rows or keep them out of training: >50 objects is out of scope for new
simulation but existing data may be used if it helps; decide per family) + corl,
with a held-out Sean split. Needs H1's seed plan. Largest compute item on the list.

### G4b. More diverse test sets from existing data
Build same-state pool test sets from (a) held-out Sean files (~6000 pools of ~8
actions, n20/n50/n100, three spawn modes) and (b) foresight (~120 pools of
32-64, n30/n50). Small pools are fine for state-level questions (power comes
from states). Record each as a DS-#### with the unified shape.

### G2b. Model representation: quantised cube orientation (EXP-0029 finding)
The corpus rasteriser draws each 5 mm cube as a 2x2 px square rotated by its
integer-degree yaw with snapped corners; zeroing yaw changes 56% of occupied
pixels. Models are trained on (and `accuracy` scores) these artifacts.
Hypothesis: part of why accuracy predicts control poorly. Test: re-score
accuracy against soft-splat truth (orientation-free) and see whether the
accuracy-vs-slateN ranking disagreement shrinks. Longer term: 128x128, an
anti-aliased square rasteriser, or an explicit orientation channel for training.

### G1c. Small simulator state leaks between rollout calls (EXP-0027 RUN-0005)
Physically small (particles ~1 mm) but real: the plate is re-placed with a
history-dependent yaw residual (0.5 deg) and a batch runs as long as its longest
push. Find the plate-yaw reset gap; batch by push length meanwhile.
`tests/test_genesis_repeat_determinism.py` (opt-in, strict xfail) tracks it.

---

## LOW

### G4c. Fill gaps so the test corpora form one cohesive set
When mixed corpora are used for testing, record which (particle count x spawn x
push length) cells are missing and queue small collections to fill them --
n20/n30 scatter first; n30+ collections scheduled carefully; no piles unless
needed; never > 50 objects.

### G2c. GNN node sampling is seeded by batch position (LOW: user 2026-09-24 -- GNNs are much slower and not beating faster models on any real metric; nothing GNN-related is prioritised)
`eval_report --device cuda` now works (~5x faster, identical results for NFD/linear).
But `GNNPredictor.predict_occ` seeds node sampling with the row's batch index, so
candidates of one state are predicted from different graphs (ranking noise in every
GNN slateN) and re-batching changes results (accuracy 0.290 vs 0.313 on L40mm).
Invariant `gnn-node-sampling-consistent-within-state` (broken). Fix: seed from the
occ0 content; then re-score the GNN rows. Needs a decision since it changes GNN
reference numbers (probably upward).

### Backlog: model-development items (from before the goals were set)
#### H2. Decide the fate of the flow/advection line
`experiments/EXP-0025-flow-warp-nfd-pilot/` measured a **model-free ceiling**:
warping `occ0` by the GROUND-TRUTH particle displacement field scores
`accuracy` 0.167 and `slateN` 0.813/0.770/0.512 on the L20mm eval cell.
**Both are below what existing baselines already achieve on that same corpus**
(NFD: 0.407 accuracy, 0.853 lyapunov; LinearForesight switched res32: 0.262,
0.902). A parameterisation whose *perfect* version loses to a *trained*
baseline is a dead end as formulated.
Either **drop the line**, or **modify it to raise the ceiling and re-measure the
ceiling FIRST** (it is model-free and cheap, so never train before checking it).
Ceiling limitations identified, in likely order of impact: (a) the target field
is rigid translation only, no rotation; (b) bilinear `grid_sample` blurs a
near-binary occupancy field even at ground truth — worst in the near-static
regime where persistence error is already tiny; (c) stacked cubes collide in the
2D projection. A hybrid "warp **plus** a small additive residual correction"
breaks the pure-advection constraint and should lift the ceiling most.
**Done when:** either the line is closed out in the record, or a modified
parameterisation has a measured ceiling above the baselines above.

#### H3. World-frame NFD + residual — finish and interpret
RUN-0022 (no augmentation, 240 epochs) and RUN-0023 (flip-only, 120 epochs) are
**already running**; epochs are matched to the baseline's 668,100 gradient
steps. Score both against the world-frame baseline and against the warped
residual arm (RUN-0019).
Expect a **smaller** gain than the warped arm's, because
`UNetModels_modular.UNet(residual=True)` already adds raw `occ0` into the
pre-sigmoid logit and every stock NFD config enables it — part of the effect is
already banked. `residual: false` is set in these configs to avoid
double-counting.
**Blocked on M1 for a clean read:** the baseline used FULL x8 augmentation, so
neither new arm has an augmentation-matched control yet.

#### M1. Augmentation-matched world-frame NFD controls
Train a plain (non-residual) world-frame NFD on `overnight_randlen` at
`augmentation: false` (240 epochs) and `augmentation: flip` (120 epochs),
matching H3's arms. Without these, H3's residual arms can only be compared to a
full-x8-augmented baseline, which confounds the residual effect with an
augmentation effect. **This is what unblocks a clean read of H3.**

#### M3. Refit and diff `operators_res64.pt`
`Baselines/LinearForesight/runs/operators_res64.pt` was **missing from disk**
(never git-tracked) and was restored from an untracked stray file in
`experiments/temp/` to unblock scoring. Its provenance is unverified, and the
`linear_*_res64` rows in EXP-0022's frame-scoring table rest on it. Re-run
`Baselines/LinearForesight/fit_switched.py` at res=64 on the same corpus and
diff against the restored file. See `experiments/OPEN_ISSUES.md`.

#### M5. Residual head on LinearForesight / other model families
The residual parameterisation is the only effect that has transferred across
scales in this project (L20mm pilot -> randlen, warped). LinearForesight's ridge
is already regularised **toward identity**, which is the linear analogue, so it
is likely a no-op there — but that has not been checked, and the GNN/latent
families have not been checked either. Cheap reasoning task before any training.

#### L1. The unexplained push-length dependence
The warped arm's `accuracy` deficit versus the world-frame baseline grows
sharply at short pushes (8-11x across length bins) and is **unexplained after
four candidate mechanisms** were tested: metric artifact (accounts for 12-30%),
canonical action-channel overlap (refuted — channels are fully separated above
~15mm yet the deficit still shrinks 3.3x above that), input-side resampling
(real but magnitude wrong by 1.6-5.7x and confounded by its own control), and
the x8-augmentation handicap (refuted — the profile got *steeper*, not flatter,
under flip-only). See `experiments/EXP-0022-*/results/flipaug_length_profile.md`.
Only worth resuming if the push-frame line is revived.

#### L2. `canon_res=96` warped arm
Supersampling the canonical frame removes only ~28% of the round-trip
degradation, and the warped arms score well below their ceiling anyway, so
resampling is not what binds. Deprioritised on measurement, not on principle.

#### L3. Wall-channel arm
Gated OUT by `experiments/EXP-0022-*/results/wall_proximity.md`: on
`randlen_test` the deficit is non-monotonic in wall distance and goes flat once
push length is held fixed. It does replicate on L20mm/L40mm, but those are
single-push-length corpora. Revisit only if a length-independent wall-linked
residual appears on randlen.

#### L4. Coarse-to-fine flow
Only if H2 keeps the flow line alive. The `flow_coarse16` cell pooled the
**flow field** but left the **image** at 64x64, so it never addressed the
capture-radius problem at all — gradient descent on a photometric loss can only
align features within about one feature width (~2-3 px for a 5 mm cube), while
measured displacements reach 20 px. The standard fix downsamples the IMAGE, so
a large displacement becomes a small one, solves there, and refines upward.

### Done
- ~~M4 sign-safe difference metrics~~ (2026-09-23, EXP-0026)
- ~~M6 many-goal harness~~: `eval_report --goal-set many` (EXP-0027); GPU part moved to G2c
- ~~H0 ground-truth scoring noise~~: soft, mass-conserving truth scoring (2026-09-24,
  METRICS.md "Ground-truth scoring"); leftover simulator leak moved to G1c
- ~~M2 redo EXP-0023~~: superseded by G1a
