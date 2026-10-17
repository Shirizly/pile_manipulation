# EXP-0068 -- do offline metrics scored on closed-loop VISITED states predict the closed-loop model order better than the same metrics on SPAWN states?

Written 2026-10-05 09:20 CEST, BEFORE any simulation (the GPU was held by EXP-0067 while this was written).
Hard budget: 7 h wall, <= 2.5 GPU-h; report partial rather than overrun.

## Question
Every offline metric validated against closed loop so far (EXP-0039 RUN-0002 `results/run0002/offline_ds0016_scores.json`,
EXP-0059 `test_v2.json`, EXP-0060) is scored on DS-0016 step-0 SPAWN pools (settled drop / clump spawns, random legal
pushes). The planner acts on half-organised piles with optimised actions. In RUN-0002's extended analysis the ON-POLICY
prediction quality of the executed pushes was the strongest predictor of the closed-loop order (rank arm on-policy corr
rho +0.88, CEM on-policy MAE +0.79) -- but that quantity is circular (it needs the closed loop run with that model). The
usable, model-agnostic version is a corpus of VISITED states with same-state candidate pools and simulated soft truth.

## Why the saved episodes cannot be reused (checked 2026-10-05 09:05)
`run_episodes_batched(record_states=True)` stores `states` as xyz only (`parts[k, :, :3]`); RUN-0002, RUN-0003, EXP-0043 and
EXP-0066 all used it. Cube yaw is lost, and `set_particle_state` needs quaternions: re-settling a yaw-0 reconstruction of a
clumped state would blow contacting cubes apart. No other re-simulable store exists (DS-0004 holds oracle-MPC outcomes,
not learned-model closed-loop states). So the design's fallback applies: a SMALL closed-loop re-run with full-state
recording. Two backward-compatible code additions were made for it (tests in `tests/test_learned_mpc.py`):
`run_episodes_batched` records `quats` when the executor returns 7-column states, and `plan(topk=k)` records the k
best-predicted candidates the planner evaluated (`topk_actions`); driver flag `--record-quats`, cell key `"topk": 16`.

## Design (all fixed before running)
**RUN-0001 (closed loop, GPU):** EXP-0039 RUN-0002's CEM arm exactly (CEM pop 1024 / elite 0.25 / 0.5 s, n_cand 1024, bank
16384, 20 mm pushes, `--legalize`, TRAINING_PHYSICS, lyapunov objective) plus EXP-0066's `--legalize-fallback` (no illegal
push is executed) and `--planner-seed 0`, `--record-states --record-quats`, `topk 16`; 8 models (the RUN-0002 zoo:
nfd_3ch_narrow_l20_v2 / _seed1 / _seed2 / _soft_s2 / _epoch10, linear_narrow_l20_v2_res64, linear_switched_soft,
nfd_3ch_randlen) x 8 letters (O T S L X Z C H) x DS-0006 scatter starts {40, 48} x 13 pushes = 128 episodes, 4 chunks
(13, not 12: the planner's top-16 at state k is recorded while planning push k+1, so k = 12 needs a 13th decision).
The closed-loop ORDER is NOT re-measured here (2 starts); RUN-0002's 128-episode order is the reference.

**State selection (`code/select_states.py`, CPU, deterministic):** 48 states = 24 cells (8 letters x k in {4, 8, 12}) x 2.
Cell c = 3*goal_index + k_index (0..23); its two states come from the episodes of models (c mod 8) and ((c + 4) mod 8) in
the zoo order above, at starts 40 (c even) / 48 (c odd) and 48 / 40 respectively. Every model contributes exactly 6
states, every start 24, no episode is used twice, and within a letter the 6 states come from 6 different models. No
exclusion by layering or escaped cubes (counts reported). The state is the recorded post-settle state after push k
(`states[k]`, `quats[k]`); the planner's `topk_actions[k]` (what it considered when choosing push k+1) travel with it.

**DS-0023 (`code/collect_pools.py`, GPU, one atomic file per state, resumable):** for each state, broadcast to 32 envs
(`set_particle_state` + `update_material_state`, as `chain_collection.py --mode pools`), draw 64 candidates with DS-0016's
sampler (`pile_aware, min_swath_particles 3, push_length 0.02, start_gap_range [0.005, 0.005]`, exact-length +
perpendicular check, redraw until 64 valid) PLUS the 16 planner candidates passed through `legalize_pushes` (the same
call the closed loop makes before executing; `legal_ok` / `legal_shift_m` recorded, un-legalisable ones kept, flagged
and excluded from scoring), simulate all 80 from the identical state in sub-batches of 32, store the unified benchmark
shape (`states, states_, p_starts, p_stops, angles, pool_idx`) + `source` (sampler / planner), `valid`, `legal_ok`,
`legal_shift_m`, `gap_out_of_window`, and per pool `model, goal, start, k`. ~3840 pushes; DS-0016 throughput was 2048
pushes in ~16 min, so ~30 min.

**Scoring (`code/score_pools.py`, GPU):** the 8 zoo models + EXP-0066's 3 extra instances (linear boot1 / boot2,
nfd_3ch_randlen_seed1), each on (a) DS-0023 and (b) DS-0016 test pools (32 spawn pools x 64), with the metric code of
EXP-0059 `eval_extended.py` reused by import: hard-raster swept-region `accuracy_1` (`eval_narrow.acc`), `accuracy_1_blur2`,
`occ_emd_swept` (goal-free, over every sampler row), and per pool x letter (all 8 closed-loop letters, lyapunov
objective, soft truth `occ_for_scoring`): slateN capture, within-pool Spearman(vp, vt), optimism at the pick, top-1
regret, mass_in_goal_mae. Per-row predicted occupancies persisted (float16) under `artifacts/`.
Aggregations, fixed now: VISITED primary = sampler candidates only, each pool on the letter it was visited under
("goal-matched"); secondary = sampler + legal planner candidates, and all 8 letters per pool. SPAWN primary = all 32
DS-0016 pools x 8 letters (what the existing offline practice does, restricted to the closed-loop letters); secondary =
the 16 scatter pools only. For the EXP-0066 arm: the 24 states visited under O/T/S/X vs DS-0016 x O/T/S/X.

**Closed-loop references:** CEM and rank arms = RUN-0002 (`results/run0002_analysis.json`: per-model scores, Holm-resolved
pairs); EXP-0066 arm = the 8 planner-seed-0 cells (NFD seeds 0/1/2, linear full/boot1/boot2, broad seeds 0/1), episode
scores recomputed from `results/<cell>.json` with EXP-0066 `analyse.py`'s formula and Holm over the 28 pairs.

## Pre-registered hypotheses and criteria
Per metric m, arm a: rho_vis(m, a) = Spearman between the VISITED-primary metric and the arm's closed-loop score over the
arm's 8 models (metrics whose lower is better flipped), rho_spawn(m, a) likewise on SPAWN-primary; exact one-sided
permutation p (8! relabelings) as in `analyse_run0002_extended.py`; pair agreement over Holm-resolved pairs with the
within-training-seed pairs (narrow v2 seeds 0/1/2; EXP-0066: also linear boot pairs, broad seed pair) excluded.
Metrics (8): accuracy_1, accuracy_1_blur2, occ_emd_swept, mass_in_goal_mae, slateN, top1_regret, optimism, spearman.
- **H1 (visited beats spawn):** SUPPORTED if rho_vis > rho_spawn for >= 6 of the 8 metrics in BOTH the CEM and the rank arm;
  REFUTED if <= 4 of 8 in either arm; otherwise inconclusive. Secondary: the same count on pair agreement.
- **H2 (accuracy_1 leaves chance on visited states):** SUPPORTED if rho_vis(accuracy_1, CEM) >= 0.5 AND its pair agreement
  (seed pairs excluded) >= 0.70; REFUTED if rho_vis < 0.3 or agreement < 0.60 (spawn: +0.10 / 0.55).
- **Generalisation gap (exploratory):** per model gap = accuracy_1(spawn) - accuracy_1(visited); its rho with each arm.
- The embarrassing result: visited-state metrics are no better than spawn-state ones (H1 refuted) -- to be reported plainly.
Noise floor: 48 states and 8 models. State-bootstrap (resample the 48 states / 32 pools, 2000 draws) 95 % CI on every rho;
n = 8 means a one-sided permutation p < 0.05 needs rho >= ~0.74, so rho differences inside the bootstrap CIs are
"unresolved", not ranked. The within-seed closed-loop spread (0.053) bounds what any metric can be asked to order.

## Scope caveat (user decision 2026-10-05)
Every model here is a narrow-corpus pilot model or the two broad EXP-0044-era models; narrow data is a pilot instrument,
so the claim is pilot-grade until replicated on broad models.

## Cost (estimate; a cost pilot of 1 chunk x 2 pushes runs first and reads no outcome)
RUN-0001 ~4 chunks x 12 steps x ~26 s = ~21 min + 2 min start; DS-0023 ~30 min; scoring ~10 min: ~1.1 GPU-h of 2.5.
If the GPU is not free in time for the full design, the fallback is 24 states (one per cell, the (c mod 8) pick).
