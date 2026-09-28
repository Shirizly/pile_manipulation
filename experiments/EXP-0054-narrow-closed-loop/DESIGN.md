# EXP-0054 — do narrow-domain models complete letter goals better? (overnight stage H)

Written 2026-09-25 05:30, BEFORE the narrow models finished training.

## Design
- Batched runner, states recorded, 24 pushes, 1 s per push, actions restricted to exactly 20 mm
  perpendicular pushes (plan(push_len=0.02)), i.e. the narrow domain the new models were
  trained on.
- Planner: GD tuned (lr 5e-3, 32 restarts) on the best success objective found so far
  (EXP-0052): lyapunov - 3 x in-goal mass fraction.
- Models: nfd_3ch_narrow_l20 and nfd_3ch_narrow_l20_wide (EXP-0053, trained on DS-0008 +
  DS-0010), vs the broad nfd_3ch_randlen (same architecture as the narrow small one) and
  nfd_residual_worldframe_noaug_ep43.
- Goals: letter O T S L X Z + two_squares; starts DS-0006 40-43 (112 episodes).
- Scored with EXP-0051's analysis: in-goal mass fraction vs optimum at k = 8, 24; completion at
  0.8 / 0.9 x optimum and completion time.

## Predictions
N1: the better narrow NFD beats the broad nfd_3ch_randlen on letter in-goal mass at k = 24 by
    >= 0.03 (narrow-domain accuracy helps precision tasks).
N2: the ordering of models by in-goal mass follows their EXP-0053 accuracy_1 on DS-0009.
Both could fail: EXP-0050 found model accuracy is not the binding constraint at these budgets.
