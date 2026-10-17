---
id: EXP-0072
title: >
  Zoom-window NFD on the narrow pilot (n20, 20 mm pushes, 1 seed): a push-aligned 64 mm window at 1 mm/px (zoom64) or
  0.5 mm/px (zoom128) predicts better than the 64x64 NFD and a no-crop 128x128 control in the pasted world frame
  (acc1 0.622 / 0.640 vs 0.559 / 0.591) and degrades far less in rollouts (8-push 0.41-0.50 vs 0.12)
tier: T1
mode: exploratory
date: 2026-10-06
hypothesis: null
claim: >
  Pilot-grade (narrow corpus: DS-0015 train_v2, DS-0016 test_chains_v2_clean 896 rows / test_pools_v2 32 pools / 111
  8-push chains; one training seed per model; UNet [4,8,16]): an NFD that predicts inside a push-aligned 64 mm window
  (64 or 128 px) and pastes the predicted change back into the world raster has higher pasted-frame one-step accuracy and
  slateN than the standard 64x64 NFD and than a no-crop 128x128 world model with the same UNet, recipe and pixel pitch;
  its rollout accuracy decays much more slowly over 8 own-prediction pushes, and unrolled training improves rollouts further.
provenance:
  commit: ded30444
  dirty: true
  script: "code/train_zoom.py, score_zoom.py, make_cache*.py, ms_train.py, ms_train_world128.py, ms_eval.py, time_zoom*.py, hr_*.py, ceiling_*.py; window libs model/zoom_nfd/"
  data: ["DS-0015 train_v2 (10,653 rows; 1,353 8-push chains for unrolled training)", "DS-0016 test_chains_v2_clean, test_pools_v2"]
  data_commit: "DS-0015 / DS-0016 as registered"
  code_path: "particles -> window raster (model/zoom_nfd) or 300x300 canvas -> grid_sample window -> UNet -> predicted change pasted into the 64x64 world raster -> eval_narrow-style acc1 / slateN / rollout accuracy"
  seed: "one training seed per model; init EXP-0022 RUN-0010 UNet for ft* runs, scratch for scratch100 / world128"
  split: "DS-0015 / DS-0016 registered splits"
  runs: ["runs/ft100", "runs/ft300", "runs/scratch100", "runs/world128", "runs/world128_300", "runs/zoom128", "unrolled T4/T8 runs (results/ms_eval.json)"]
  runtime: "2026-10-06, single day, shared RTX 4070 laptop"
budget:
  declared: "not recorded (exploration session)"
  spent: "~1 day"
  outcome: within
design:
  varied:
    model: "zoom64 (ft100 / ft300 / scratch100), zoom128, world128 (no crop, same UNet / pitch), standard nfd_3ch_narrow_l20_v2, linear_narrow_l20_v2_res64"
    training: "single-step vs unrolled T=4 / T=8"
    input: "direct window raster vs HR source raster (128-400 px, hard / soft / antialiased thresholds)"
  held_fixed:
    unet: "[4,8,16] (wider nets overfit on this data: NOTES architecture line)"
    window: "64 mm side, plate 40 mm + 12 mm each side"
  baselines: "persistence (rollout baseline = start state); standard 64x64 NFD; no-crop world128 control; linear foresight"
  metric: "accuracy (acc1, pasted 64x64 frame and window frame), slateN, rollout accuracy_k"
noise_floor: "NFD slateN seed sd 0.01-0.04 and accuracy sd 0.003 (EXP-0036, cited in NOTES); pasted-frame caps 0.651 (zoom64) / 0.681 (zoom128) / 0.633 (world128) acc1 (results/ceiling/results.md, ISS-015)"
depends_on: [score-occupancy-subpixel-stable]
establishes: []
result: >
  Pasted acc1 / slateN (results/NOTES.md): zoom64 ft300 0.622/0.765, zoom128 0.640/0.809, world128 (300 ep) 0.591/0.728,
  standard 64x64 NFD 0.559/0.710. Rollout accuracy k = 1/4/8: NFD64 .578/.329/.115, linear .521/.381/.322, world128
  .614/.325/.087, zoom64 .627/.481/.409, zoom128 .650/.507/.442; unrolled: world128 T4 .625/.492/.439, zoom64 T8
  .629/.507/.454, zoom128 T4 .657/.532/.485, zoom128 T8 .654/.535/.495. Timing at K=1024 candidates: zoom64 101 ms
  (world-out), zoom128 301 ms, world128 170 ms, NFD64 38 ms. The zoom models sit within ~0.02 of the pasted-frame caps, so
  pasted numbers cannot resolve further gains among them.
verdict: supported
downgrades: [imprecision, incomplete-design]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The world128 control has the same UNet, recipe and pixel pitch as zoom64 but no crop, so a zoom gain over world128 is not
just resolution; the rollout test checks whether a one-step gain compounds or erodes.

## What was actually run
Exploration session; `results/NOTES.md` is the running record (scores, timing, narrative), `results/ceiling/results.md` the
accuracy-ceiling / simulator-noise study (raised ISS-015). The README said no EXPERIMENT.md would be written until later;
this record was written 2026-10-15 from those files (RAL run plan T0d). No new runs.

## What would change the verdict
More seeds (one per model); the wide-domain replication EXP-0074 found the zoom edge over a capacity-matched vanilla128
shrinks to ~+0.01 there (the edge is a property of short pushes / scattered scenes, where the zoom factor is largest).

## Threats
Narrow pilot domain only (pilot-grade per project policy); pasted frame capped (ISS-015); window-frame numbers are not
comparable across window resolutions; world128 trained with D4 aug vs flip for zoom.

## Unrelated findings
The training-style direct window raster inflates cube area by x1.14-1.28 vs true footprints (NOTES, source-raster fidelity);
models were trained on slightly inflated cubes.
