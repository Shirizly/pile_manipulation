# EXP-0053 -- models trained on the narrow 20 mm domain, offline evaluation (overnight stage G)

Written retrospectively 2026-09-25 from EXPERIMENT.md, code/eval_narrow.py (docstring, argparse
defaults), the runs/ logs, the command ledger and the TODO.md overnight plan (stages G and H), not
before the run. The pre-run plan is quoted where TODO.md records it.

## Why
EXP-0047 found no clean exact-20 mm perpendicular n20 single-layer data, so DS-0008/0009 were
collected (stage F). Does training on that narrow domain give better models on it than the broad
randlen-trained ones? Pre-run plan (TODO.md stage G, "GPU, ~90 min"): "linear single-operator
20 mm res32 + res64; NFD res64 on the narrow domain; one deeper/wider NFD if time." Stage H's plan
adds the offline part: "accuracy, slateN on the clean test set".

## Design
- Train: DS-0008 (6,144 exact-20 mm rows, scatter + clumps) + DS-0010 (5,777 existing 18-22 mm
  matched-physics rows, training only). Test: DS-0009 (1,024 chain rows + 32 same-state 64-push
  pools), disjoint collection seeds.
- Narrow models (seed 0, one seed each): nfd_3ch_narrow_l20 (features [4,8,16]) and
  nfd_3ch_narrow_l20_wide ([16,32,64]), recipe nfd_train_3ch_randlen.yaml except features/data,
  60 epochs, best-val checkpoint; linear single operator (fit_switched.py --n-bins 1) at res32 and
  res64.
- Broad baselines: nfd_3ch_randlen (same architecture as the small narrow NFD),
  nfd_residual_worldframe_noaug_ep43, linear_switched_soft. Persistence (accuracy 0) and random
  pick (slateN 0) are the do-nothing / chance baselines.
- Metrics (eval_narrow.py, 64 px occ_from_particles input): `accuracy` (METRICS.md, swept region),
  split scatter / clump; rollout accuracy k = 1..4 on the chains (model fed its own prediction);
  `slateN` on the 64-push pools with soft truth (occ_for_scoring), lyapunov, EXP-0046's 12-goal set
  + two_squares.
- Budget: declared ~90 min GPU; spent ~65 min (linear fits ~1 min each, NFDs ~30 and ~32 min,
  eval 94 s for the narrow models).

## Predictions
Exploratory: no pre-registered prediction.

## Deviations
- The plan said "NFD res64"; the record does not state the NFDs' training resolution (they are
  evaluated on the 64 px grid), so whether this matches the plan is not recorded.
- The "deeper/wider NFD if time" was run as the 4x wider variant; no deeper variant.
- The wide NFD's best validation epoch was 15 of 60 (overfits afterwards).
- The ledger records one eval invocation (06:32, log lists only the four narrow models); TODO.md
  says the broad-baseline rows were computed earlier. That earlier invocation is not in the ledger.
- One training seed per variant; seed floor taken from EXP-0036.
