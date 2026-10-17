# EXP-0066 — closed-loop POWER study: narrow v2 NFD (3 training seeds) vs linear narrow v2 res64 (3 instances), with planner replication, 32 starts, quantiles

Written 2026-10-04 20:40 CEST, BEFORE any scored cell ran (cost pilot only: `results/pilot/pilot_cost.json`,
1 chunk x 4 pushes, timing + fallback check, no outcome read). Hard budget: 9 h wall, <= 7.5 GPU-h.

## Question
EXP-0039 RUN-0002 (8 models x 8 letters x 16 starts, one planner run each) left the cross-family pair "narrow v2 NFD vs
linear narrow v2 res64" marginal: linear res64 was resolved below NFD seeds 0 and 2 but not seed 1, and the three NFD
seeds spanned 0.233-0.286 -- a seed floor as large as many cross-family gaps. Is that gap resolvable at a feasible
closed-loop budget once training seeds, planner noise and more starts are all in the design?

## Cells (one driver call = one cell = 4 goals x 32 starts = 128 episodes = 4 chunks of 32 envs)
Priority order (the queue runs them in this order; later cells are the first to be dropped on budget):
| # | tag | model | planner seed |
|---|---|---|---|
| 1 | nfd_s0_ps0 | nfd_3ch_narrow_l20_v2 (seed 0) | 0 |
| 2 | lin_orig_ps0 | linear_narrow_l20_v2_res64 (full-data fit) | 0 |
| 3 | nfd_s1_ps0 | nfd_3ch_narrow_l20_v2_seed1 | 0 |
| 4 | lin_b1_ps0 | linear_narrow_l20_v2_res64_boot1 (data-bootstrap refit, seed 1) | 0 |
| 5 | nfd_s2_ps0 | nfd_3ch_narrow_l20_v2_seed2 | 0 |
| 6 | lin_b2_ps0 | linear_narrow_l20_v2_res64_boot2 (data-bootstrap refit, seed 2) | 0 |
| 7 | nfd_s0_ps1 | nfd_3ch_narrow_l20_v2 | 1 |
| 8 | nfd_s0_ps2 | nfd_3ch_narrow_l20_v2 | 2 |
| 9 | lin_orig_ps1 | linear_narrow_l20_v2_res64 | 1 |
| 10 | lin_orig_ps2 | linear_narrow_l20_v2_res64 | 2 |
| 11 (optional) | broad_s0_ps0 | nfd_3ch_randlen | 0 |
| 12 (optional) | broad_s1_ps0 | nfd_3ch_randlen_seed1 | 0 |
Cells 11-12 are secondary context only (the broad NFD led RUN-0002); they start only if the queue reaches them before
02:45 CEST, and are not part of H1/H2.

## Settings, identical to EXP-0039 RUN-0002's CEM arm except where marked NEW
- Driver `experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py`: CEM, pop 1024, elite 0.25, 0.5 s per
  decision, n_cand 1024, bank 16384 per env, 16 pushes of 20 mm (`push_len 0.02`), `--legalize`, `--record-states`,
  TRAINING_PHYSICS executor (`GenesisOracleEnv` + `apply_physics`), DS-0006 corpus.
- Goals: letter_O, letter_T, letter_S, letter_X -- fixed before seeing anything (4 of RUN-0002's 8 thin letters).
- Starts: DS-0006 states 40-71 (32 scatter starts; 40-55 are RUN-0002's, 56-71 new). Checked before running: all 32
  pass `Baselines.common.cube_overlap.state_is_legal` (tol 0) and are single-layer (every cube z = 0.0125 m).
- NEW `--legalize-fallback`: a planned push `legalize_pushes` cannot make legal is replaced by the best legal
  candidate of that env's bank under the model's own objective (`simple_mpc.learned_mpc.legal_fallback`). RUN-0002
  executed such pushes unchanged (`legal_ok` False, 15-121 of 2048 steps per model). Executed-illegal must be 0 here.
- NEW `--planner-seed`: torch global RNG seeded per (seed, step, env) before every candidate draw and every `plan`
  call; CEM noise and the Genesis pile-aware sampler both draw from it. Physics is deterministic given the state
  (`genesis-snapshot-restore-repeat-determinism`). Unseeded before (learned_mpc.py:228).
- Linear instances: `Baselines/LinearForesight/fit_switched.py --bootstrap-seed k` (NEW flag) resamples the 11659
  DS-0015 train rows with replacement (7337 / 7362 distinct rows) and refits the same ridge recipe; 41 s per refit.
  **Caveat known before running:** the bootstrap replicates' held-out swept-region accuracy is 0.280 / 0.279 vs the
  full-data fit's 0.332 -- resampling with duplicates weakens the ridge fit, so the linear "instance spread" is a
  data-resampling spread with a downward bias, not a pure seed spread. The NFD seeds are retrains of one recipe.
- Score per episode (EXP-0039 RUN-0002 / EXP-0051 formula): mean over pushes 4..16 of in-goal mass fraction
  (`mass_in_region(occ_for_scoring(states))` / total mass) divided by the goal's `mass_frac_best_placement`
  (EXP-0046 `results/vstar.json`). Higher is better; 1 = optimum held from push 4 on.

## Cost (measured in the cost pilot, 2026-10-04 20:34)
26 s per step for a chunk of 32 envs (RUN-0002: 24 s) -> 16 pushes ~ 420 s per chunk, ~28 min + ~1 min start-up per
cell. 10 primary cells ~ 4.9 h GPU; 12 cells ~ 5.9 h GPU (<= 7.5 GPU-h). Queue starts ~20:50; primary done ~01:45.
The episode count is therefore FIXED at n = 128 per cell (4 goals x 32 starts), 1280 primary episodes.

## Hypotheses (pre-registered)
Family gap G = mean over the 3 NFD cells (planner seed 0) of the cell mean score  -  mean over the 3 linear cells (ps 0).
- **H1 (seeds fixed):** the paired bootstrap (resample the 128 (goal, start) episodes; per episode the NFD-3-mean
  minus the linear-3-mean) 95 % CI of G excludes 0. RUN-0002 on these 4 goals gave G = +0.041 (3 seeds vs 1 linear).
- **H1h (seeds random):** a hierarchical bootstrap that also resamples the 3 NFD seeds and the 3 linear instances with
  replacement; its 95 % CI is reported beside H1. The design's embarrassing result: H1h's CI includes 0 (the family gap
  is inside the seed spread even at 32 starts x 3 seeds). If so, the record says plainly that closed-loop cross-family
  model comparisons are not resolvable at feasible budgets with seeds treated as random, and H1 (seeds fixed) is
  reported as "these three checkpoints vs these three operators", not as a family claim.
- **H2:** training-seed spread = sd (ddof 1) of the 3 NFD cell means at ps 0; planner-seed spread = sd of the 3
  nfd_s0 cell means over ps 0/1/2 (and the 3 lin_orig cell means over ps 0/1/2); linear instance spread = sd of the 3
  linear cell means at ps 0. Each reported as sd and as sd / |G|. Prediction: training-seed spread > planner-seed
  spread (planner noise averages over 128 x 16 decisions; a training seed is one draw), and both < |G|.
- **Minimal detectable gap at this n** (80 % power, two-sided 5 %): MDG = 2.8 x SE(G). Pre-run estimate from RUN-0002
  on these 4 goals: per-model episode sd 0.10-0.12 (the brief's ~0.09 was for 8 goals); paired sd of (NFD-3-mean minus
  linear) 0.118 at n = 64 -> seeds fixed: SE ~ 0.10 / sqrt(128) = 0.009, MDG ~ 0.025. Seeds random: NFD seed sd
  0.033 (RUN-0002 OTSX means 0.293 / 0.227 / 0.251) -> SE(family mean) 0.019 per side -> SE(G) ~ 0.028, MDG ~ 0.08.
  The post-run MDG uses the measured paired sd and measured seed / instance sds, same formula.
- **Legality:** executed-illegal count (exact SAT on the recorded cube poses, `legal_ok`) = 0 in every cell;
  fallback-fire rate reported per model (pilot: 10 / 128 steps for nfd seed 0).

## Analysis (`code/analyse.py`, written before the run)
Per cell: median and 5 / 95 % quantiles of the episode score, overall and per goal; mean with bootstrap CI. H1, H1h,
H2, MDG as above. Variance decomposition on the per-episode scores of the 10 primary cells: sequential
sums of squares for goal, start-within-goal, family, instance-within-family (ps 0 cells), planner seed (the 6
replicate cells), residual. All 10 pairwise cell differences among the 6 ps-0 cells, paired bootstrap p, Holm at
0.05. Executed-illegal count and fallback rate per cell.

## Budget rule
Queue skips any cell not started by 03:50 CEST; analysis runs on whatever cells completed; cells not run are listed.
