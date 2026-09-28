# RUN-0022 — WORLD-FRAME NFD + residual head, no augmentation (INTERRUPTED)

**Status: interrupted at epoch 43 of a planned 240** (the session hosting the
process was restarted). It is nonetheless the **best-scoring model on
`accuracy` produced anywhere in EXP-0022**, which is why it gets a result file
despite being an unplanned stopping point.

## Numbers

`slateN` goal-averaged (leads) / swept-region `accuracy` (suspect metric).

| corpus | model | accuracy | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|---|
| L20mm | world-frame baseline | 0.4071 | 0.8527 | **0.8202** | **0.7527** |
| | warped+flip+residual (RUN-0019) | 0.3839 | 0.8459 | 0.7712 | 0.7436 |
| | **worldframe residual, ep30** | 0.4260 | **0.8842** | 0.7881 | 0.7426 |
| | **worldframe residual, ep43** | **0.4322** | 0.8802 | 0.8153 | 0.6725 |
| L40mm | world-frame baseline | 0.5088 | **0.9322** | 0.7832 | 0.8411 |
| | warped+flip+residual | 0.4690 | 0.9208 | 0.8179 | **0.8677** |
| | **worldframe residual, ep30** | **0.5256** | 0.9113 | **0.8424** | 0.8578 |
| | **worldframe residual, ep43** | 0.5247 | 0.9044 | 0.8016 | 0.8552 |
| randlen_test | world-frame baseline | 0.4564 | 0.9417 | **0.8886** | 0.8655 |
| | warped+flip+residual | 0.4622 | 0.9379 | 0.8761 | 0.8528 |
| | **worldframe residual, ep30** | 0.4715 | 0.9252 | 0.8778 | **0.8906** |
| | **worldframe residual, ep43** | **0.4844** | **0.9529** | 0.8803 | 0.8779 |

**It beats the world-frame baseline on `accuracy` in all three corpora**
(+0.025 / +0.016 / +0.028), and on `randlen_test` it is also top on
`slateN`/lyapunov (0.9529 vs 0.9417). The epoch-30 and epoch-43 checkpoints are
close, as expected 13 epochs apart late in a flattening curve.

## Why "18% of the budget" is the WRONG way to read this

The run reached ~120k of the intended 668,100 gradient steps. But gradient
steps and data exposure diverge sharply under different augmentation, and on
the axis that arguably matters more this run has seen **more** data than the
baseline, not less:

| | dataset passes | gradient steps |
|---|---|---|
| baseline (x8 augmentation, 30 epochs) | 30 | 668,100 |
| RUN-0022 (no augmentation, 43 epochs) | **43** | 119,669 |
| ratio | **1.43x** | **0.18x** |

With `augmentation: false` the loader batch is the full 32 and every sample in
a batch is distinct, so one epoch is one pass in 2,783 steps. With the x8 group
the loader batch is 4 and the other 28 slots are synthesised views, so one pass
costs 22,270 steps. **So this model is under-OPTIMISED (5.6x fewer updates) and
over-EXPOSED (1.43x more passes) relative to the baseline, simultaneously.**

## What it does and does not license

**Does:** it is strong evidence that an explicit tanh-residual head helps a
WORLD-FRAME NFD, which was the open question TODO H3 was opened to answer, and
which had never been tested at randlen scale. It also suggests x8 augmentation
may be a poor use of compute for this architecture, since a model with 5.6x
fewer updates and no augmentation at all beat it on the metric the baseline is
strongest on.

**Does not:** it is **not a clean comparison**. It differs from the baseline in
**two** ways at once -- the residual head AND the augmentation setting -- so the
gain cannot be attributed to the residual alone. There is also no seed-level
noise floor anywhere in this project, and the accuracy margins here (0.016 to
0.028) are the size that a noise floor might or might not cover.

## What to run next

This makes **TODO M1 (augmentation-matched world-frame controls) materially
more important than it was**, because it is now the thing standing between this
result and a real claim. The clean 2x2 is
`{plain, residual} x {no augmentation, x8}`, all step-matched, and three of the
four cells do not exist.

`RUN-0023` (world-frame residual, flip-only augmentation, 120 epochs) was
configured and never started -- the sequential driver died with the session.
Its config is `Baselines/NFD/configs/nfd_train_residual_unwarped_flipaug_randlen.yaml`.
