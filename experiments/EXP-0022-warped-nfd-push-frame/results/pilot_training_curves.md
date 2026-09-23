# Pilot training curves -- unwarped vs. warped vs. warped+walls (L20mm)

Three arms, same architecture family (NFD UNet, features [4,8,16]), same dataset
(`slates_multistep/n20_L20mm_train`, identical 5%/5% val/test split, `resolution_scale`
0.5, `min_push_length_m` 0.0198), same recipe (lr 1e-4, StepLR, augmentation on, batch 32,
20 epochs, `eulerian_combined` loss with mse=1.0 and every other term 0). Full run records:
`experiments/EXP-0022-warped-nfd-push-frame/runs/RUN-000{1,2,3}-*/RUN.md`.

| Arm | RUN | final train | final val | best val | best epoch | s/epoch | exit |
|---|---|---|---|---|---|---|---|
| Unwarped control (world frame) | RUN-0001 | 0.005218 | 0.005143 | 0.005102 | 19 | ~32.2 (39->29, warm-up decay) | 0 |
| Warped, no wall channel | RUN-0002 | 0.005491 | 0.005565 | 0.005565 | 20 | ~37.3 (flat) | 0 |
| Warped + wall channel | RUN-0003 | 0.005388 | 0.005508 | 0.005446 | 16 | ~37.9 (flat) | 0 |

All three exited cleanly (status 0), no NaNs, no divergence, no restarts, no GPU
contention -- each run had the 8 GB GPU to itself, confirmed idle immediately before
each start.

## What this does and does not show

All three losses are **world-frame MSE against the same target representation**
(`eulerian_combined`, mse=1.0, all other loss terms 0) -- the warped arms predict in the
canonical push frame internally but are scored back in world frame before these numbers
are computed, so the three rows above are legitimately comparable to each other as the
same quantity. They are consistent with each other: unwarped control lowest (~0.0051),
warped no wall next (~0.0056), warped+walls in between (~0.0055) -- differences of a few
x1e-4 on a metric with epoch-to-epoch noise of the same order (RUN-0002's own val loss
moved by 0.0002 between epochs 18-20 with no other change). That gap is not obviously
outside the run-to-run noise visible within a single arm's own late-training epochs, so
**this table alone does not establish that the unwarped control out-trains the warped
arms**, only that all three converge to a similar, low world-frame MSE on this split.

Two things this table explicitly is NOT:

1. **Not a generalisation measure.** The "val" split here is a 5% slice carved out of the
   TRAIN files by `PileSweepData`'s own file-granularity hash split -- it is held out at
   the level of individual files within the same corpus and collection run, not a
   separate corpus, condition, or time period. It says these models fit that split about
   as well as they fit the 95% they trained on; it says nothing about behavior outside
   `slates_multistep/n20_L20mm_train`.
2. **Not the metric that decides anything about warping.** These are pixel-wise
   world-frame MSE numbers on the full frame. The metrics that actually matter for this
   comparison -- `slateN` and swept-region `accuracy` -- are scored separately and are not
   in this file. A model can have a slightly lower or higher whole-frame MSE than another
   while being better or worse where the pile actually moves; do not use this table to
   rank the three arms.

## The one anomaly worth flagging

RUN-0003 (warped + walls) had a hard_iou of 0.22 at epoch 1 and 0.33 at epoch 2, far
below RUN-0001's epoch-1 IoU (0.63) and RUN-0002's (0.54), before jumping to 0.77 by
epoch 3 and behaving normally (0.78-0.79) for the rest of training. Train/val loss
themselves fell smoothly through those same epochs with no discontinuity, so this reads
as a slow-to-threshold IoU metric during the first two epochs of this arm specifically
(plausibly the extra wall channel perturbing the very early optimization trajectory),
not a training failure -- but it is the one number across all three arms that looks
different in kind, not just degree, and is recorded here rather than smoothed over. See
RUN-0003/RUN.md for the full epoch-by-epoch numbers.

No arm diverged, flat-lined, or needed a restart.
