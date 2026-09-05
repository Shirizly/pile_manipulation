---
id: EXP-0004
title: The trained UNet barely uses its action channel
tier: T0
mode: exploratory
date: 2026-09-03
hypothesis: H-A1
claim: >
  On unetfilm_corl_limited_100e, destroying the action channel at inference
  costs less than half of the model's advantage over persistence, i.e. most of
  what it learned is smoothing the occupancy rather than responding to the push.
prediction: null
provenance:
  commit: 17a7d4a7
  dirty: true                     # BACKFILLED 2026-09-05: `scripts/probes/unet_action_ablation.py`
                                  # did not exist at 17a7d4a7; it was written in the same
                                  # session and first committed at aac084e3. So this sha
                                  # bounds the run from below only -- the analysis code
                                  # that actually ran is the aac084e3 version of that file.
  script_first_committed: aac084e3
  script: scripts/probes/unet_action_ablation.py
  data: ["runs_cubes/unetfilm_corl_limited_100e/unet_best.pth", "corl_limited/cubes val split, 200 samples"]
  code_path: "PileSweepData raster (the model's own training path)"
  seed: "n/a (deterministic forward pass; batch permutation for the shuffle arm is unseeded)"
  split: "the run's own val split"
  runtime: "~2 min, CPU"
design:
  varied: {action_channel: [true, zeroed, shuffled], pile_channel: [true, transposed]}
  held_fixed: {checkpoint: unet_best.pth, samples: 200, physics_vector: true, sigmoid: applied, metric: whole-image rms}
  baselines: [persistence]
  metric: "whole-image rms against the target occupancy, as a percentage of the persistence rms"
noise_floor: "not measured; the 3.1-point action effect is not separated from sampling noise over 200 samples"
depends_on: [grid-convention, deploy-train-raster]
result: "persistence 100%, true action 91.8%, shuffled 94.9%, zeroed 96.5% — the action is worth 3.1 of the model's 8.2 points"
verdict: inconclusive
downgrades: [imprecision, indirectness, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the pile and plate channels are mutually transposed (EXP-0001), a network
can only condition on the action by learning the reflection between them. A
network that failed to do so would predict almost the same thing with and
without a valid action channel. Shuffling the channel across the batch keeps
its marginal statistics and destroys only its correspondence to the pile, which
is the specific thing the transpose breaks.

## What was actually run

Inference only, on the checkpoint as trained. The raw output is a logit; a
sigmoid is applied before scoring (without it the rms is ~6.4 and meaningless).
Whole-image rms, not swept-region — so this number is not comparable to any
other record here.

## Numbers

| variant | rms | vs persistence |
|---|---|---|
| persistence (copy the input) | 0.12055 | 100.0% |
| UNet, true action | 0.11063 | **91.8%** |
| UNet, action shuffled across batch | 0.11442 | 94.9% |
| UNet, action zeroed | 0.11633 | 96.5% |
| UNet, pile channel transposed | 0.38568 | 319.9% |

## What would change the verdict

The decisive test is a retrain, not an ablation: same config on `corl_limited`,
once as-is and once with `_draw_particle_grid` corrected. ~1 h GPU. Prediction:
the shuffle gap widens from 3.1 points to most of the model's advantage. Until
that runs, this is consistent with the transpose hypothesis but does not
establish it — a model could underuse its action channel for other reasons
(too little data, 100 epochs, a target dominated by unchanged pixels).

## Threats

- `imprecision`: 200 samples, one checkpoint, unseeded shuffle, no repeats.
- `indirectness`: whole-image rms is ~95% pixels nothing could change, which
  compresses every difference. The swept-region metric would be sharper.
- `untested-dependency`: both cited tags are broken.
- Considered and **not** dismissed: this is `corl_limited`, a deliberately small
  dataset. A model this weak may simply be undertrained.
