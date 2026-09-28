# EXP-0030 — state-dependent model superiority on DS-0005 (160 states, two pools each)

Written 2026-09-24 BEFORE DS-0005 finished collecting (no outcome seen).

## Question (user goal 3)
Where do models significantly beat each other, and what about the STATE decides
it? Preferred answer: a short descriptor; then a linear map; a trained selector last.

## Data / models / scoring
- DS-0005: 160 n20 scatter states x 128 actions, binned 20-70 mm, one step.
  Two disjoint 64-action pools per state; primary split = a fixed random
  permutation per state (seed 0), plus 100 random re-splits for stability.
- Models (OCC_ADAPTERS): nfd_3ch_randlen, nfd_warped_randlen,
  nfd_warped_randlen_flipaug, nfd_warped_randlen_flipaug_epoch30,
  nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43,
  nfd_3ch_finetuned, linear_switched_hard (8; the L20mm pilot is excluded).
- Goals: 26 letters + 4 quadrants; lyapunov (primary), mass_in_region.
- Truth: soft ground-truth scoring (`occ_for_scoring`); predictions: model images.
- Per (model, state, pool): goal-averaged slateN capture.

## Analyses and predictions
A1 reliability: per model pair, Pearson r across 160 states of the per-state
   advantage on pool 1 vs pool 2.
   PREDICTION: median r over NFD-family pairs >= 0.2 (EXP-0029 at K=128 gave
   0.16 on 20 states; at K=64 pools are noisier, so this is a real test).
A2 cross-fitted switching gain over the NFD family (choose per state on pool 1,
   score on pool 2, minus best single chosen on pool 1).
   PREDICTION: gain interval (state bootstrap) includes 0 -- i.e. per-state
   choice from one small pool does not beat the best model (EXP-0029).
A3 ensemble control: average the models' predicted dv, pick by the average.
   PREDICTION: the ensemble beats the best single model (capture, paired over
   states, CI excluding 0). Refuted if not.
A4 descriptors: per state -- radius of gyration, mean nearest-neighbour
   distance, mean distance to the nearest wall, n clusters (7 mm linkage),
   centroid offset from the tray centre, convex-hull area. For every pair with
   A1 r > 0.2: Spearman of per-state advantage (pools averaged) with each
   descriptor (state-permutation p, Holm over descriptors x pairs); then a
   linear selector (all descriptors, ridge, leave-10-states-out CV) choosing
   between the pair, scored on held-out states vs always choosing the better
   model. PREDICTION: at least one descriptor is Holm-significant for at least
   one pair; the linear selector's gain CI includes 0 (exploratory -- no strong
   prior).
Null results are reported in full.

## Baselines
Best single model (the bar a switch must clear); ensemble; random = 0.

## Cost
Predictions: 8 models x 20k actions, ~1 min GPU; analyses CPU, minutes.
Checkpointed per model and per analysis.

## Addendum (before any DS-0005 outcome was seen)
A smoke run on DS-0001 showed `n_clusters` (7 mm linkage) is constant for n20
scatter states (no two cubes within 7 mm), so it carries no information. It is
replaced by `frac_with_neighbour_12mm` (fraction of cubes with another cube
within 12 mm). No other change.

## Addendum 2 (2026-09-24, before any outcome): corpus changed to DS-0006
DS-0005 was stopped partway because its physics did not match the models' training
corpus (invariant `benchmark-physics-matches-training`). The analysis runs on
DS-0006 instead: identical design (160 n20 scatter states x 128 actions, seed 1),
collected with particle friction 0.7, box friction 0.5, density 450. Nothing else
in the plan changes; `--corpus` points at
Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys.

## Addendum 3 (2026-09-24): does the ensemble still win when time is matched? (A5)
Added at the user's request, AFTER A1-A4 were seen, but before any time-matched
number exists. A3 gave every model the same 64 candidates, so the 7-member ensemble
spent ~7x the compute of a single model. A5 matches wall-clock instead.

- Cost: for each model (and each ensemble, run member by member as
  `learned_mpc.ModelObjective` does), GPU wall time to predict and score n candidates
  of one state, t_m(n), n in {1,2,4,...,128}; median of repeated timings.
- Budget B = t_ref(m) for the reference nfd_3ch_randlen scoring m in {8,16,32,64,128}
  candidates. Each model gets n_m(B) = the largest n with t_m(n) <= B (at least 1,
  at most the 128-push pool; linear interpolation between timed sizes).
- Pool = all 128 pushes of each DS-0006 state. For each state, 200 random
  permutations of the pool, shared by all models (nested subsets): a model with
  budget n picks the push with the best PREDICTED value among the first n.
  Capture is normalised by the FULL pool (true mean and true best of all 128), so a
  smaller subset is penalised. Averaged over permutations, then over the 30 goals.
- Arms: the 8 cached single models, ensemble_nfd (7 members, as A3) and
  ensemble_nfd5 (EXP-0037/0039's 5 members), both value functions.
- Comparison: ensemble minus the value function's full-pool best single model from
  A3 (lyapunov: nfd_residual_warped_flipaug_randlen; mass_in_region:
  nfd_residual_worldframe_noaug_ep43), per budget, state-bootstrap 95% CI (5000).
  Also reported: ensemble minus the best single AT that budget (chosen post hoc).
- PREDICTION (A5): time-matched, the 7-member ensemble LOSES to the best single model
  (CI entirely below 0) at every budget, because at m=128 it sees only ~1/7 of the
  pool and the A3 gain (+0.03-0.06) is smaller than the loss from seeing fewer
  candidates. Refuted if the CI includes or exceeds 0 at any budget.
- Limit: budgets above t_ref(128) cannot be tested (singles would need more than the
  128 pushes a pool holds), so this covers the regime where candidate count binds.
- Cost: timing ~2 min GPU; analysis CPU, minutes. Code: code/budget_matched.py.

## Addendum 4 (2026-09-24, after a degenerate first run): cost = throughput
The first A5 run (RUN-0002, first attempt; outputs kept as
artifacts/RUN-0002/timings_percall_run1.json and results/budget_matched_run1_degenerate.json)
timed ONE call per budget. For n <= 128 a call costs ~4-7 ms almost independently of n
(launch overhead), so every model slower per call than the reference got n = 1 and
the comparison said nothing about the ensemble. That regime is not the one a
planner is in: at EXP-0039's 1 s budget a model makes hundreds of calls, and what
limits it is throughput. Changed, before any non-degenerate A5 number existed:
- cost per candidate c_m = t_m(2048) / 2048, scoring 2048 candidates in chunks of 128
  exactly as `learned_mpc.ModelObjective` does (median of 7 timed repeats); an
  ensemble's c is the sum of its members';
- budget: the reference nfd_3ch_randlen scores m in {16, 32, 64, 128} pool pushes;
  every arm gets n_m = floor(m * c_ref / c_m), clamped to [1, 128]. The 128-push
  pool thus stands in for "everything the reference can score in the time";
- prediction, comparison and everything else as addendum 3.
