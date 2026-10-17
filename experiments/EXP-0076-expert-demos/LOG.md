# EXP-0076 log — everything tried, with direct results (newest at bottom)
Brief: BRIEF.md (re-read each round). State of the project: SUMMARY.md. Demos: datasets/DS-0024-expert-demos/.

- 2026-10-16 18:30 setup. Read EXP-0075 LOG (best learned planner: ens2bal + fetch + legal scoring, CEM 1,280, H=1;
  OT pool helps at H>=2; GD +0.03 at H=1 with the base pool), bottleneck notes (proposal > model error), EXP-0057/0050
  oracle (sim-as-model CEM never completed letters at 256 sims/decision), EXP-0070 time model.
  Reference measured from recorded EXP-0075 cells (in-goal mass >= 0.95 within 10 pushes, n20 DS-0006 starts, letters
  O/T/S/X): g5_ens_fl w20 22/32 (median 6 pushes), g5_v64_fl w20 24/32 (6), g6_w15_ens_fl 18/32 (8), g6_w10_ens_fl 0/32.
- 18:35 user answers (BRIEF.md): mass >= 0.95, physical time objective, n20/n50 x scattered/inbetween/piled,
  random rule-checked goals (disc/square/letter O,T,S,X w20/w15), feasible cap 3 s/decision.
- 18:40 code: tasks.py (scenario generator, 60 % rule, Sean TEST-split chain-start states, success metric, physical time),
  simcore.py (E-env Genesis with per-env states), beam.py (model-proposed, sim-verified beam search; B=1,k=1 = plain
  feasible planner), store.py (best solved trajectory per scenario x pipeline, attempts.jsonl).
  results/scenarios.json: 540 scenarios = 30 per (n, init, goal kind); rho 0.14-0.60; none solved at start (mean mass0 0.23).
  Figure figures/scenarios_000.png (one per stratum) checked by eye: masks/cubes/orientation OK.
  Goal areas: disc_n20 (repo default gather goal) has rho 0.70 -> violates the 60 % rule; not used.
- 18:45 cost pilot (code/probe_speed.py; no outcomes): batched push = execute (sweep) dominated, settle ~0.2-1 s.
  n20: E32 10.5-13 s/batch (2.4-3.0 sims/s), E128 20-24 s (5.3-6.2 sims/s); n50: E64 46-70 s (0.9-1.4 sims/s), E128 OOM.
  Bank 4096/env: 0.4 s (E32), 2.4 s (E128), 5.4 s (n50 E64). => sim throughput is the budget; n50 is ~5x dearer than n20.
  A second Genesis build in one process fails (nvrtc) -> one Sim per process.
- 18:48 user: also mine already-recorded full trajectories for qualifying ones (mark separately) -> Haiku subagent,
  CPU only, writes datasets/DS-0024-expert-demos/data/mined/ (code/mine_recorded.py).
- 18:49 PILOT ROUND 1 launched (code/queue_p1.sh): n20, 18 scenarios (first 2 of each init x kind), ens2bal, fl pool
  (fetch + legal scoring), CEM 256+4x256 + GD 20, H=1; arms A feasible B1k1 (E32), B greedy+sim top-16 (B1k16, E128),
  C beam B4k8 (E128). Max 12 pushes. Score: solved (mass >= 0.95), pushes, physical time to success.
- 18:55 PILOT A (feasible B1k1) done: 14/18 solved within 12 pushes (12 in < 10): piled/inbetween 2-6 pushes, scattered hardest
  (3/6 unsolved: final mass 0.88-0.94); 5 min wall for 18 scenarios (1.25 s planning / decision).
- 19:02 MINED (Haiku subagent, code/mine_recorded.py; CPU): 866 JSON files scanned, 11,584 recorded episodes, 1,558 qualify
  (n20, DS-0006 scatter starts, legal by record + 1,518 re-checked with legalize_pushes on stored quats: 0 failures; mass >= 0.95
  truncated at first success; 60 % rule) = 173 unique (goal, start) tasks: letters O/T/S/X w20/w15 (+3 w10 episodes), quadrant_0/3,
  two_squares (1). Median 7 pushes / 9.4 s. Rejected: 9,415 unsolved, 544 rule (disc_n20 rho 0.70), 67 legality-unverified (kept in
  mined/unverified/, not counted). 22 episodes sit at float32 0.95 exactly (counted, same float32 metric as ours). Spot check of 5
  files: recomputed mass and time match. Stored separately: data/mined/ (+ MANIFEST.json best_per_task = 173 fastest).
- 19:05 BUG (fixed): scenarios admitted to beam.py AFTER the Genesis build got CUDA tensors (Genesis sets torch's default device), crash at
  the first refill (p1B at tick 5; 1 scenario finished and kept). Fixed: device="cpu" on every torch.tensor in beam.py / replay.py.
  Also: `pkill -f queue_p1.sh` matched its own shell (known trap) and killed p1C mid-run; processes are now stopped by PID only.
  Requeued (code/queue_p3.sh): p1B resume, p1C, p2B after p2A (n50 feasible, started with the fixed code).
- 19:17 p2A_feas50 (n50 feasible) killed by PID after 12 min without finishing tick 1 (CPU-bound 100 %); debugging n50 path separately.
- 19:42 PILOT B (greedy + sim-verify model top-16, B1k16, E128) done (results/arms_p1A_feas_p1B_greedy16.md): solved 16/18 vs A 14/18;
  on the 13 solved by both: pushes -0.62, time -0.48 s [-1.11, +0.06] (n.s.); 104 sims / scenario vs 7; wall 493 vs 202 s / scenario
  (GPU contended by the n50 debug for part of it). B fixes the scattered stragglers (3 of A's 4 failures) but loses inbetween-letter-000
  (A 6 pushes, B unsolved at 0.93 -- greedy on the true value is not greedy on time). Model regret is small (as EXP-0075 found); the
  lever must be proposals / lookahead -> beam (C) running.
- n50 cost: debug_n50 (2 scattered n50 states, E4) one push batch 44 s; debug_n50b (18 mixed-init n50 states, E18) > 20 min for the FIRST
  batch (contended). Start states checked: no interpenetration (min centre distance 4.95-5.0 mm), up to 8 second-layer cubes (piled).
- 20:06 n50 SIM COST by init (code/debug_n50d.py, 6 envs, first legal bank push, contended GPU): scattered 17 s / batch (169 steps,
  ~0.1 s/step); inbetween 289 s / batch (step time 0.15 -> 3.2 s once blades reach dense clusters); piled > 9 s/step at step 40 (aborted).
  18 mixed envs: 13 s/step (~45 min / batch). One piled n50 state with 2 envs: 0.11 s/step -> cost is per-env contact-island driven and
  roughly additive over envs, so batching does not amortise it. Solver knobs are off-limits (docs/scaling_to_200_objects.md 1.4:
  more iterations slower, islands off 1100x slower, box-box off crashes) and changing dt/substeps would change 'reality'.
  DECISION: n50 scattered gets the full treatment; n50 inbetween / piled get the feasible pipeline (1 sim / decision) in small batches
  late in the run, as many as budget allows; reported as a limitation.
- 20:11 PILOT C (beam B4k8, E128) done; 3-arm table results/arms_p1A_feas_p1B_greedy16_p1C_beam4x8.md (18 n20 scenarios):
  solved A 14/18, B 16/18, C 18/18; mean pushes / time on solved 5.71/6.67 s, 5.81/7.14, 5.78/6.89; sims per scenario 7 / 104 / 185.
  Paired time on scenarios solved by both: C - A -0.86 s [-1.73, -0.05] (n 14, pushes -0.79), C - B -0.66 s [-1.02, -0.32] (n 16, -0.44),
  B - A -0.48 [-1.11, +0.06]. => beam over true states is the EXPERT pipeline (lookahead through true states beats verify-and-greedy).
  Piled n20 scenarios solve in 2-4 pushes in every arm (easy); scattered 6-10.
  Round 2 started: f20_all = feasible planner over all 270 n20 scenarios (E64) -> baseline demo for every n20 scenario.
- 20:31 REPLAY CHECK (code/replay.py, 18 n20 expert demos, open loop in one continuous env each, idle pushes for finished envs):
  3 replays: still >= 0.95 in 8 / 10 / 10 of 18; recorded-vs-replay final cube position diff 2.38 / 2.26 / 2.27 mm mean (max 44 mm);
  replay-vs-replay (a vs b) 1.31 mm mean (max 14.5). => about half of the divergence is plain run-to-run simulator nondeterminism
  (C-033-type call-history dependence), the rest is the restore-between-pushes chain / batch composition. Every recorded transition
  is a genuine simulator transition, but demos that end just above 0.95 (many end at 0.950-0.955) do not reliably reproduce open loop.
  Plan: replay-verify every stored demo at the end (flag replay_solved, 2 replays); consider a success margin for 'robust' demos.
- 20:51 f20_all interim (118 of 270): scattered disc/square solved 31/60 by the feasible planner; strongly rho-dependent:
  rho < 0.3 13/16, 0.3-0.45 14/19, > 0.45 6/25 (final mass 0.88). Feasibility check (EXP-0067 vstar_width.solve_goal, N=20):
  the 4 tightest square goals (rho 0.59-0.60) have placement optimum 0.994-0.995 and lattice capacity 24-28 -> possible in
  principle, but a packing problem.
- 21:11 ROUND 2 DONE f20_all (feasible ens2bal fl H1 CEM 1,280 + GD 20, all 270 n20 scenarios, E64, 56 min): solved 196/270 (73 %)
  within 12 pushes; per stratum solved / median pushes (median phys. time): scattered disc 15/30 8 (10.2 s), square 18/30 9 (11.4), letter
  25/30 8 (10.4); inbetween disc 20/30 6 (7.5), square 22/30 6 (7.2), letter 24/30 6 (7.5); piled disc 26/30 3 (2.8), square 27/30 3 (3.3),
  letter 19/30 2 (2.1). Plan time 1.33 s / decision (p95 1.79) -> inside the 3 s cap. Piled-letter failures are w15 letters (10 of 11,
  final 0.88-0.95); scattered failures are tight goals (rho > 0.45). Round 3 (feasible ablations F1-F5 on 36 scenarios) started.
- 21:36 ROUND 3 partial (36 n20 scenarios = indices 2-5 per stratum; baseline = f20_all on the same scenarios; results/arms_f20_all_F1_v64big_F2_ensH2ot.md):
  F1 vanilla64bal CEM 5,120 + GD 20 (0.38 s / decision): solved 24/36 vs 27/36, time on jointly solved +0.35 s [-0.20, +0.85];
  F2 ens2bal H=2 OT pool cumspill (2.39 s / decision): 26/36, time -0.10 s [-0.93, +0.83]; per-scenario disagreement is large
  (only-baseline 5, only-F2 4) -> added F0_seed1 (baseline config, planner seed 1) to measure the run-to-run noise floor.
- 21:47 F3 phase schedule (H=2 OT until mass 0.75, then H=1 fl; 1.79 s / decision): 26/36, time -0.35 s [-0.82, +0.11] (n.s.).
  PORTFOLIO observation (expert-only trick): union over the 4 feasible arms (f20_all, F1, F2, F3) solves 32/36 (baseline 27/36) and
  best-of-4 time on baseline-solved scenarios is 6.35 vs 7.12 s (-0.77 s) at 4 x ~7 = 28 sims / scenario -- similar gain to the beam's
  -0.86 s (pilot, 185 sims). Per-scenario outcomes of single feasible runs are noisy (borderline final masses 0.93-0.95), so random /
  config restarts + selecting the fastest in hindsight are a cheap expert method. To be compared with the beam head-to-head.
- 22:05 F4 ens2bal 2x budget (CEM 512+4x512 + GD 20, 2.18 s / decision): 26/36, time +0.07 s [-0.47, +0.63] -> budget saturated.
  F5 vanilla64bal H=2 OT pool, CEM 5,120 (0.64 s): 22/36, time +0.75 s [+0.07, +1.50] -> WORSE (the cheap model's 2-step predictions).
  Added (code, default off): beam.py --time-mu (re-rank the model's top distinct proposals by cost + time_mu x physical push time);
  queued F6 / F7 (time_mu 0.02 / 0.05) after round 4; round 6 (expert beam on 68 unsolved + 78 slow n20 scenarios) after that.
- 22:09 NOISE FLOOR F0_seed1 (baseline config, planner seed 1): 28/36 vs 27/36, time +0.11 s [-0.34, +0.58], scenarios solved by one run only:
  1 / 2. => F1-F4 are inside run-to-run noise (+-1-2 solved of 36, CI half-width ~0.45 s); only F5 is resolved (worse). The feasible
  baseline (ens2bal H=1 fl CEM 1,280 + GD 20) is on a plateau w.r.t. model / budget / horizon / phase schedule on this n20 set.
- 22:22 f50s_feas (n50 scattered, 45 scenarios, E45) sim time per tick 87 -> 147 -> 297 s at ticks 1-3 (plan 57 s): cost grows as cubes
  cluster in the goal (contact islands), ~6.6 s per env-push and rising. Implication: an n50 beam (32 sims / decision) would cost ~1 h per
  episode -> n50 demos come from the feasible pipeline (1 sim / decision) only; dense n50 (inbetween / piled) only a handful.
- 22:32 STRUCTURE ANALYSIS (subagent, CPU; results/structure_analysis.md, code/structure_analysis.py; f20_all + pooled runs, n20, descriptive):
  mass 0.8-0.95 tail = 40 % of solved time (gain / push 0.250 / 0.099 / 0.037 / 0.015 by mass bin 0-0.5 / 0.5-0.8 / 0.8-0.9 / 0.9-0.95, a tail
  push still costs 0.94 s); failures are stalls (gain < 0.02 pushes: 9 % of pushes in solved, 50 % in unsolved; 64 % of unsolved end >= 0.90);
  slow solved episodes carry 2.57 extra out->out pushes vs 0.89 extra out->in; transit = 45 % of time (mean transit 54 mm > sweep 39 mm);
  dynamics-ignoring reorder bound 9 % of time; 1.25 s per push slope, fastest solution never had more pushes than slowest (47 scenarios).
  -> F8 tail rule queued (mass < 0.8: CEM 640; >= 0.8: CEM 2,560 + 256 fetch proposals + time_mu 0.02). beam.py --n-extra added.
- 22:43 queues r4-r6 replaced by code/queue_main.sh (after f50s_feas: X1, X2, F6, F7, F8, E20_unsolved, E20_slow); X3 (beam with H=2 OT proposals) dropped for time (F2: H=2 ~ H=1 for the feasible planner).
- 00:13 user asked for good / surprising results so far; answered from LOG (beam > greedy-verify > feasible; portfolio best-of-4 ~ beam at 1/6 the sims; feasible plateau; rho drives difficulty; tail 40 % of time; transit 45 %; replay fragility). f50s: 19/45 solved at tick 13.
- 00:33 f50s_feas DONE (n50 scattered, 45 scenarios = 15 per goal kind, feasible ens2bal fl H1, max 16 pushes, E45; 2 h 24 min wall, sim
  87 -> 870 s per tick): solved 22/45 -- disc 11/15 (7-10 pushes), square 9/15 (9-16), letter 2/15 (13, 16; unsolved letters end at 0.82);
  by rho: 0.3-0.4 11/13, 0.4-0.5 10/19, > 0.5 1/13. Cost per env-push grew with clustering (~7-19 s), roughly additive in active envs.
- 00:59 X1 beam B8k4 vs p1C B4k8 (12 n20 scattered/inbetween pilot scenarios, equal sims 235 / scenario): 12/12 both, time +0.14 s [-0.55, +0.74], pushes +0.00 -> beam shape does not matter at this budget.
- 01:18 X2 beam B4k8 mu 0.05 (beam score = true value + 0.05 x elapsed s): 12/12, time -0.23 s [-1.06, +0.48] vs p1C, -0.38 [-0.93, +0.12] vs X1; pushes -0.25 (n.s.; best point estimate, no downside) -> E20 bulk config set to --mu 0.05 (code/r6_beamcfg.txt).
- 01:26 F6 time-aware re-ranking (time_mu 0.02, top-16 distinct): 23/36 (baseline 27, seed-1 28), time on jointly solved -0.03 s [-0.62, +0.65]; 5 scenarios solved by baseline only, 1 by F6 only -> HURTS success, no time gain (cheaper pushes cost progress; cf. EXP-0070 C-079).
- 01:35 F7 time_mu 0.05: 25/36, time +0.10 s [-0.32, +0.51] -> time-aware proposal re-ranking does not help at either weight.
- 01:44 F8 tail rule (mass < 0.8: CEM 640; >= 0.8: CEM 2,560 + 256 fetch + time_mu 0.02): 24/36, time -0.04 s [-0.51, +0.41] -> no gain. Feasible conclusion: every tested change (model, budget, H, phase schedule, time-aware ranking, tail rule) is inside noise or worse; the EXP-0075 H=1 fl baseline stays the best feasible pipeline. E20_unsolved (beam B4k8 mu 0.05, max 16) started.
- 01:55 E20_unsolved is slow (hard scenarios run 12-16 deep; ~45 s / tick for 4 concurrent; ETA ~04:40). Replaced the queued E20_slow beam by 2 cheap feasible restarts (portfolio, seeds 2/3) on the 78 slow scenarios, then 6 dense n50 attempts (code/queue_tail.sh).
- 02:15 E20_unsolved stopped by PID after 9 of 68 (4 solved: e.g. scattered-disc-012 in 6 pushes, which the feasible planner never solved; 5 reached 16 pushes unsolved, 0.93-0.95). Remaining 59 -> E20u_b2k8 (beam B2k8, mu 0.05, max 14; 8 concurrent, ~2x throughput), then restarts, then dense n50 (code/queue_tail2.sh).
- 03:20 EXPERT BEAM on the 68 n20 scenarios the feasible run left unsolved (E20_unsolved B4k8 9 scenarios + E20u_b2k8 B2k8 59; mu 0.05,
  max 14-16): 37/68 solved (25 in < 10 pushes); by stratum: inbetween disc 7/9, letter 4/5, square 4/8; piled disc 3/4, letter 4/11, square 1/3;
  scattered disc 4/13, letter 4/4, square 6/11; by rho < 0.3 9/18, 0.3-0.45 12/21, > 0.45 16/29. Most unsolved end at 0.90-0.94 (near misses).
  Two piled failures end at 0.25 / 0.10: start state has cubes JAMMED IN A TRAY CORNER (figure scratch pd007) -- the blade cannot get behind them
  legally; both planners stall. results/ids_corner_jammed.json: 9 of 540 scenarios have a cube within 2 mm of both walls (n20: 3, none solved
  by any pipeline) -> flagged as likely tool-infeasible; success rates reported with and without them.
- 03:38 R20_slow_s2 (feasible restart, seed 2) on the 78 slow n20 scenarios (all solved by f20_all in >= 7 pushes): 62/78 re-solved, faster than the original in 38; best-of-2 time -0.78 s on average (portfolio selection; stored demos updated where faster).
- 03:58 R20_slow_s3: 61/78 re-solved. Portfolio on the 78 slow n20 scenarios: mean best time original 10.80 s -> best-of-2 10.18 -> best-of-3 9.86 (pushes 8.91 -> 8.10); diminishing (-0.62, -0.32 s). f50d_feas (6 dense n50: inbetween/piled x disc/square/letter, feasible, max 14, E6) started 03:58; cap ~06:15.
- 04:41 dense n50 (f50d_feas) after 4 pushes: piled square solved (2 pushes), inbetween disc solved (4); piled disc / letter at 0.94; ~6-9 min per tick. Queued code/queue_final.sh: open-loop replay of all n20 demos (expert, feasible; one pass) after f50d.
- 04:56 f50d_feas DONE (6 dense n50, 58 min): 4/6 solved -- piled square 2 pushes (2.4 s), inbetween disc 4 (5.2 s), piled disc 5 (7.1 s), inbetween square 10 (11.6 s); letters unsolved (inbetween 0.86, piled 0.945). Dense n50 sim cost was heavy only for the first 1-2 pushes (857 / 748 s per tick), then 50-540 s. Queued: replay pass, R20_unsolved_s4 (29 still-unsolved n20), f50d_feas2 (12 more dense n50).
- 05:09 FINAL REPLAY PASS (open loop, one continuous env per demo, results/replay_{expert,feasible}_n20_final.json):
  expert 19/55 still >= 0.95 (mean final cube offset 2.68 mm); feasible 158/203 (1.97 mm). By recorded final mass: expert 0.95-0.96 8/30,
  0.96-0.98 8/21, >= 0.98 3/4; feasible 25/47, 55/75, 78/81. => beyond the thin margin, expert (sim-selected) demos reproduce far less than
  feasible ones at the same recorded mass: selecting the best of 8-32 SIMULATED children picks favourable simulator noise (winner's curse
  on the simulator), which an open-loop replay does not repeat. Each transition is still a genuine sim transition; replay flags are in INDEX.json.
- 05:11 DETERMINISM probe (code/probe_determinism.py): one batch of 16 envs with the identical state and push -> identical outcomes (0.000 mm). Replays a vs b (identical batches, separate processes) differ 1.3 mm -> the variation is per-env-slot simulation HISTORY (solver warm start / contact state carried across calls, C-033), not per-env randomness; beam selection picks children whose slot history happened to be favourable. Testing a 're-check' beam (selected children re-simulated in shuffled env slots; select / accept on both).
- 05:23 R20_unsolved_s4 (feasible restart seed 4, max 16) on the 29 n20 scenarios no pipeline had solved: 1/29. X4 (re-check beam) started; then replays of X2 / X4 solved records; then 6 more dense n50 (indices 1).
- 05:57 X4 re-check beam (X2 config + re-simulate top 2B children in shuffled slots, accept only if both >= 0.95): 12/12, time -0.01 s
  [-0.58, +0.51] vs X2, +24 % sims; one-step re-simulation differs little (mean |dmass| 0.0025, p90 0.0086, max 0.030). Replay (open loop):
  X2 3/12, X4 5/12 still >= 0.95 (n.s. at n=12); final cube offset 2.4-2.5 mm both. => divergence ACCUMULATES over the open-loop sequence;
  per-step selection noise is small. The lever for reproducible demos is MARGIN (feasible demos ending >= 0.98 replay 78/81). Added beam.py
  --success (acceptance threshold) and --pipeline-name; margin test (accept >= 0.97, store as expert_robust) queued after dense n50.
- 06:47 f50d_feas2 DONE (6 more dense n50, 50 min): 4/6 solved; dense n50 total 8/12 (discs and squares 8/8, letters 0/4: final 0.79-0.945). X5 margin test (accept >= 0.97) started.
- 07:11 X5 MARGIN TEST (X2 config, accept only mass >= 0.97 in the simulated chain; stored as expert_robust): 9/12 solved (X2 at 0.95:
  12/12); on jointly solved +0.50 pushes, +0.32 s [-0.11, +0.77]. Open-loop replay: 7/9 still >= 0.95 (X2: 3/12, X4: 5/12) -> a 0.97
  acceptance margin makes expert demos reproducible at a cost of ~0.5 push and ~1/4 of the solves. Last GPU slot (07:12-07:55): X5 config on
  18 more n20 scenarios (indices 2-3 of all 9 strata) to grow expert_robust.
- 07:36 X6 robust expert (X5 config, accept >= 0.97) on 18 n20 scenarios (indices 2-3 of every stratum): 15/18. expert_robust now holds
  24 scenarios; versus the fastest 0.95 demo of the same scenarios: 6.82 -> 7.29 s, 5.58 -> 6.25 pushes (+0.5 s, +0.7 push for the margin).
- 07:36 END OF RUN. Final: 272 / 540 generated scenarios with a solved demo (242 n20, 30 n50; 233 in < 10 pushes); 24 robust (>= 0.97)
  expert demos; 1,558 mined episodes (173 tasks) kept separate. INDEX.json (code/finalize.py) lists the best demo per scenario with replay flags
  (240 checked, 168 reproduce >= 0.95 open loop). Records: EXPERIMENT.md, REGISTER C-088 / C-089, DS-0024 DATASET.md, CODEMAP entry.
- 09:03 USER QUESTION (phase-wise model switching). Phase split of recorded runs (code/phase_split.py; 36 n20; bulk = start -> 0.8 mass,
  tail = 0.8 -> 0.95): only resolved difference = v64 SLOWER in the BULK phase, +0.45 s [+0.07, +0.84]; tail differences have CIs ~+-0.5 s on
  ~22 pairs (unresolvable); tail completion 22-28 of 33-36 arrivals. Model switching by phase had NOT been tested (F3/F8 switched horizon /
  budget with the same model). TAIL-SWITCH experiment launched (code/queue_tail_switch.sh, results/tail_scenarios.json): 96 recorded 0.8-crossing
  states (64 from ens runs f20_all / F0_seed1, 32 from the v64 run F1) each continued by ens (F0 config) and v64 (F1 config), 2 planner seeds,
  max 8 pushes, transit of the first tail push from the recorded plate pose; analysis code/tail_switch_analyse.py.
- 09:34 TAIL-SWITCH RESULT (results/tail_switch.md; 96 paired tail starts, entry mass 0.848, <= 8 pushes): seed-averaged v64 - ens tail:
  censored time +0.61 s [+0.28, +0.94], solved -0.057 [-0.115, +0.000]; per seed +0.58 [+0.17, +1.00] / +0.64 [+0.32, +0.96]; same-model seed
  noise floor ens -0.09 [-0.34, +0.16], v64 -0.03 [-0.30, +0.24]. By bulk model: ens-bulk states +0.76 [+0.36, +1.17] (n 64), v64-bulk states
  +0.31 [-0.25, +0.86] (n 32; interaction not resolved). => the accurate model DOES matter in the tail (~0.6 s of a ~2.8 s tail, ~6 pp solves);
  the whole-episode ablation could not see it (CI +-0.5 s on ~23 pairs). Next: bulk-phase power -- F1 config (v64 4x budget) over all 270 n20
  scenarios, paired with f20_all (ens) on time to 0.8 (code/phase_split.py logic).
- 10:00 BULK-PHASE POWER (F1_v64_all = F1 config over all 270 n20, paired with f20_all; results/bulk_compare.md, code/bulk_compare.py):
  time to mass 0.8 v64 - ens censored +0.58 s [+0.40, +0.77] (reached 255 vs 266; both-reached n 254: +0.44 [+0.28, +0.60], +0.26 pushes);
  whole episode: solved 166 vs 196, censored time to 0.95 +1.10 s [+0.81, +1.39]. Per stratum to 0.8: scattered disc +1.35, letter +1.41 (CIs
  exclude 0), inbetween square +0.57, piled +0.11..+0.32. => CORRECTION: F1's 'inside noise' (36 scenarios) was an underpowered null; the cheap
  model costs time in BOTH phases (bulk ~0.6 s, tail ~0.6 s), most on scattered starts, least on piled. No phase where the cheap model is
  equivalent at equal physical time; the ens is within the 3 s cap everywhere, so ens throughout remains the feasible choice. C-089 updated.
- 12:36 USER: switching decisions must use TOTAL time (planning + physical). Recomputed (code/total_time.py, results/total_time.md; plan time =
  recorded per-decision proposal wall clock, robustness variant = run median: ens 1.21-1.29 s, v64 0.38 s; Genesis bank sampling excluded, equal):
  bulk to 0.8 (270 paired): v64 - ens physical +0.58 s -> TOTAL -2.64 [-2.99, -2.31] s (median-plan; recorded -3.27); whole episode to 0.95:
  physical +1.10 -> total -5.03 [-5.65, -4.43] (censored at 12 pushes; on 152 both-solved -3.46 [-3.99, -2.95]); tail switch (96 x 2 seeds):
  physical +0.61 -> total -2.90 [-3.60, -2.18] (both-solved -2.02 / -1.94). Composite (bulk 270 + tail starts): ens/ens 19.5 s, ens/v64 16.7,
  v64/ens 16.1, v64/v64 13.0. => on total time the cheap model wins in BOTH phases (planning 0.8-0.9 s / decision cheaper vs ~0.15-0.4 s
  physical loss per push); switching to the ens anywhere costs time. Remaining cost of v64: fewer solves within the push cap (166 vs 196 of
  270 in 12 pushes; tail 63.5 vs 69 of 96 in 8) -- censoring favours the model that fails more, hence the both-solved rows.
