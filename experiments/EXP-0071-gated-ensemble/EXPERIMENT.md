---
id: EXP-0071
title: >
  Gated ensemble of narrow-v2 occupancy models: offline accuracy_1 0.607 and slateN_tough ~0.83 vs 0.57 / 0.71-0.73 for
  the best single NFD, and +0.030 [+0.020, +0.041] over its mean NFD member in rank-128 closed loop -- but no resolvable gain
  over the best seed or over a plain equal blend, and under CEM at equal wall-clock the gate is the WORST ensemble
  (0.208 vs equal blend 0.325, NFD mean 0.257)
tier: T1
mode: confirmatory              # H1/H2/H3 registered in DESIGN.md before any scored closed-loop cell; offline part exploratory (seen before)
date: 2026-10-06
hypothesis: null
claim: >
  In legal closed loop (letters O T S L X Z C H, DS-0006 starts 40-47 rank / 40-55 CEM, 16 pushes of 20 mm, TRAINING_PHYSICS, --legalize,
  EXP-0039 RUN-0002 episode score), a per-pixel learned convex gate over 5 narrow-v2 occupancy members (3 NFD seeds, sharp_w03, linear res64;
  MODEL-0015, fit on DS-0017 only) scores above the mean of its 3 NFD members' RUN-0002 cells under the rank-128 planner (H1),
  and adds over an equal-weight blend of the 3 NFD members (H2); under CEM 0.5 s it does not (H3, exploratory).
prediction:
  supports: "H1: paired bootstrap CI of gate - NFD3-mean (64 paired episodes, rank) excludes 0 with diff >= +0.02. H2: CI of gate - equal blend excludes 0 (predicted NOT to). H3: CEM gate - NFD3-mean <= +0.01"
  refutes: "H1: CI includes 0 or gate below its mean member"
  discriminating: true
provenance:
  commit: a5ccb680
  dirty: true                    # simple_mpc/adapters.py (GatedEnsembleAdapter, MeanBlendAdapter, registry), MODEL-0015, this record: all uncommitted (no commit requested)
  data_commit: "DS-0017 val_pools_v2 (gate fit), DS-0016 test sets (offline), DS-0006 starts 40-55 (closed loop)"
  script: "code/fit_gate.py, code/parity.py, code/queue.sh (driver EXP-0043 batched_closed_loop.py), code/analyse.py; offline scripts in experiments/temp/ensemble-proto (collect, collect_pools, ens, gate, pools)"
  data: ["DS-0016 test_chains_v2 (flagged rows dropped, 986 rows), test_pools_v2 (32 pools)", "DS-0017 val_pools_v2 (2048 rows)", "MODEL-0015", "EXP-0039 RUN-0002 cells hm_{rank,cem}_nfd_3ch_narrow_l20_v2{,_seed1,_seed2}, linear_narrow_l20_v2_res64"]
  code_path: "simple_mpc.adapters.GatedEnsembleAdapter / MeanBlendAdapter -> learned_mpc.plan (rank / cem) -> run_episodes_batched, legalize=Genesis.action_sampling.legalize_pushes (no fallback, unseeded planner, as RUN-0002)"
  seed: "gate seed 0 (not selected; 3 seeds gave offline slateN_tough 0.83-0.84); planner unseeded; members' training seeds 0/1/2"
  split: "gate fit on DS-0017; scored on DS-0016 (offline) and Genesis closed loop"
  runtime: "rank cells ~5 min each, CEM cells ~25-30 min each (RTX 4070, shared machine load ~10)"
  runs: [RUN-0001]
budget:
  declared: "~3 h wall incl. building the adapters"
  spent: "~3 h"
  outcome: within
design:
  varied:
    model: "gated_ens_narrow_v2 (gate), nfd3_meanblend_narrow_v2 (equal-weight pixel mean of the 3 NFD seeds)"
    planner: "rank-128 (equal candidates) and CEM pop 1024 / elite 0.25 / 0.5 s (equal wall-clock)"
  held_fixed:
    members: "the 5 narrow-v2 members, fixed; gate hyperparameters (hid 24, 60 epochs, lr 2e-3, swept-region-weighted MSE) not tuned"
    episode: "letters x starts as RUN-0002; push_len 0.02; legalize on"
  baselines: "do-nothing = persistence (offline accuracy 0, slateN 0.01; closed-loop push-0 level); RUN-0002 NFD seeds 0/1/2, linear res64; equal-weight blend (control for seed averaging)"
  metric: "accuracy; slateN_tough; mass_in_goal_mae_tough; closed-loop in-goal mass AUC over pushes 4-16 / optimum (EXP-0039 RUN-0002 formula)"
noise_floor: "paired episode-difference sd ~0.11 (EXP-0066) -> SE ~0.014 at n=64, MDG ~0.04; NFD training-seed spread 0.036 (EXP-0066) -- the NFD seed cells here span 0.145-0.198 (rank), 0.233-0.286 (CEM)"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: "Rank arm (64 episodes): gate 0.203, equal blend 0.194, NFD seeds 0.198/0.145/0.175, linear 0.153; gate - NFD3 mean +0.030 [+0.020, +0.041] (H1 supported); gate - best seed (seed 0, post hoc) +0.005 [-0.008, +0.017]; gate - equal blend +0.009 [-0.003, +0.020] (H2 not resolved, as predicted). CEM arm (128 episodes, equal 0.5 s): equal blend 0.325 (+0.068 [+0.053, +0.083] over NFD mean, +0.039 [+0.022, +0.056] over best seed), gate 0.208 (-0.049 [-0.064, -0.035] vs NFD mean; = linear 0.205), n_evals/step 1024 gate vs 4037 blend."
verdict: supported
downgrades: [inconsistency, indirectness, imprecision]
grade: very-low
---

## Why this test discriminates
Offline slateN/accuracy gains of an ensemble can be pure hedging or seed averaging and have been seen to vanish at equal wall-clock (EXP-0035 C-043). The rank arm holds candidates equal, the equal-blend control separates "ensembling" from "learned gating", and the CEM arm tests whether either survives an optimiser that can exploit the model.

## What was actually run
`code/queue.sh`: 4 cells (rank and CEM x gate, equal blend), flags as RUN-0002, 2026-10-06 05:00-06:20 CEST, every episode complete. Baseline cells are RUN-0002's (2026-10-04) -- different session, same driver and flags; planner unseeded so candidate banks differ per run. Uncommitted at run time: `simple_mpc/adapters.py`, MODEL-0015, this experiment dir. Offline numbers (accuracy_1, slateN_tough, mass MAE; 3 gate seeds; retrieval-including variants: accuracy_1 0.632, slateN_tough 0.84-0.88) come from `experiments/temp/ensemble-proto/` caches and were seen before the closed-loop design; they are exploratory, as is anything with the hedged retrieval member (a scoring-only hedge by the project's own rule, never used in closed loop; retrieval needs particles, 26-30 ms/candidate).

## Numbers
Offline DS-0016 (retrieval-free 5-member gate, adapter parity-checked): accuracy_1 0.607 vs NFD seeds 0.557-0.570, linear 0.520; slateN_tough 0.83-0.84 vs NFD 0.70-0.73, NFD-mean blend 0.82; mass_in_goal_mae_tough 1.36 vs 1.64-1.82. Closed loop: `results/analysis.{json,md}` (tables above in `result`).

## Reading
Ensembling narrow NFD seeds helps in both planners: rank +0.021 [+0.012, +0.031] over the member mean, CEM +0.068. The learned gate adds nothing over the plain blend under rank and is badly worse under CEM. Two readings, not separated here: (a) planner exploitation -- the gate picks a member per pixel, so it can predict optimistic outcomes for off-distribution candidates that CEM then finds (EXP-0039 RUN-0002 saw narrow models' on-policy predictions decorrelate under CEM); (b) cost -- the gate evaluates ~4x fewer candidates in the same 0.5 s (1024 vs 4037 per decision). The simple equal blend, which is also free of training data, was the better ensemble in closed loop. Pilot-grade (narrow-trained members, narrow task; policy 2026-10-05).

## What would change the verdict
Seeded planner replicates and a 4th/5th NFD seed (the seed spread equals the H1 effect); CEM at equal candidate count (gate with n_evals matched) to split (a) from (b); a gate trained with exploitation in mind (on planner-visited states) or regularised toward equal weights; closed-loop on broad-trained members.

## Unrelated findings
none
