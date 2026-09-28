# EXP-0029 — is a model's per-state advantage a property of the STATE or of the action POOL?

Written 2026-09-24 BEFORE running (plan gate).

## Why
experiments/temp/state-specialisation found, on randlen_test (21 states, one
128-action pool each, 30 goals), that which model wins depends on the state
(64/66 pairs) and that choosing the model per state (cross-fitted over goals)
beats the best single model by +0.020 [+0.006, +0.034]. Goals replicate the
value function but not the actions, so "per state" could be "per pool". DS-0001
has 1000 simulated actions per state, so each state can be split into
independent pools with no new simulation.

## Design
- Data: DS-0001 (20 n20 scatter states x 1000 actions). TRUE outcomes scored
  with the soft ground-truth scoring (`occ_for_scoring`, METRICS.md
  "Ground-truth scoring"); predictions are each model's own images.
- Models: the OCC_ADAPTERS arms nfd_3ch_randlen, nfd_3ch_finetuned,
  nfd_warped_randlen, nfd_residual_warped (L20mm pilot checkpoint),
  linear_switched_hard (5 models; linear_switched_soft dropped as a near-copy).
- Goals: the eval_report "many" set (26 letters + 4 quadrants) on the 64x64
  slate grid; value fn lyapunov (primary), mass_in_region (secondary).
- Per state: R = 200 random splits of its 1000 actions into two disjoint pools
  (K = 500 each; and K = 128, two disjoint 128-subsets, to match randlen).
  Per (model, state, pool, goal): slateN capture.
- Quantities (per split, then averaged over splits):
  S1 split-half reliability: per model pair, Pearson r across the 20 states of
     the goal-averaged advantage on pool 1 vs pool 2.
  S2 cross-fitted switching gain: choose the best model per state on pool 1
     (goal-averaged), score the choice on pool 2; minus the best single model
     chosen on pool 1 and scored on pool 2.

## Predictions
C1 (state, K=128): S2 gain > 0 with the 95% interval over splits excluding 0,
   AND median S1 r over pairs > 0.3. Refutes ("pool, not state") if the S2
   interval includes 0 AND median S1 r < 0.1.
C2 (K=500): same criteria; larger pools should give larger r (less pool noise).
discriminating: true -- the randlen result could be entirely pool-specific.

## Caveats written in advance
n20 scatter only; 20 states; the 5 models are a gradient-benchmark subset, not
the 12 of the randlen analysis. A pass here supports "state-specific" for this
state family only. Intervals over splits reflect split variability, not
state-sampling uncertainty; a state bootstrap is reported beside them.

## Cost
Predictions: 5 models x 20k actions on GPU, minutes; scoring: CPU, minutes.
Checkpointed per model (predictions saved) and results rewritten per stage.
