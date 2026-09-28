---
id: EXP-0053
title: >
  Narrow-domain training (n20 single layer, 20 mm perpendicular) beats the broad models on a
  clean 20 mm test set: same-architecture NFD accuracy 0.506 vs 0.453 and slateN 0.774 vs 0.695;
  a 4x wider NFD ranks best (slateN 0.803) but is not more accurate; narrow linear operators are
  accurate at 64 px (0.45) yet rank poorly (slateN 0.58)
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null
claim: >
  On DS-0009 (clean narrow-domain test: 1,024 chain rows, 32 x 64-push same-state pools, scatter
  and single-layer clumps, exact 20 mm perpendicular pushes, training physics), models trained on
  the narrow domain (DS-0008 + DS-0010) predict and rank better than the broad randlen-trained
  models of the same architecture.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: Baselines/NFD/train_nfd.py (configs nfd_3ch_narrow_l20{,_wide}.yaml), Baselines/LinearForesight/fit_switched.py --n-bins 1, code/eval_narrow.py
  data: ["DS-0008 (train, 6,144)", "DS-0010 (extra 18-22 mm, 5,777, training only)", "DS-0009 (test)"]
  code_path: "OCC_ADAPTERS predict_step on occ_from_particles (64 px slate grid) -> accuracy (METRICS.md, swept region) / rollout / slateN vs soft truth"
  seed: "training seed 0 (one seed per variant)"
  split: "train = DS-0008 + DS-0010; test = DS-0009 (disjoint collection seeds 101/102 vs 1)"
  data_commit: not applicable
  runs: [exp0053_linear_res32, exp0053_linear_res64, exp0053_nfd_3ch_narrow_l20, exp0053_nfd_3ch_narrow_l20_wide, exp0053_offline_eval]
  runtime: "training 2 x ~30 min + 2 x 1 min; eval ~5 min"
budget:
  declared: "~90 min GPU (overnight stage G)"
  spent: "~65 min GPU"
  outcome: within
design:
  varied:
    model: "nfd_3ch_narrow_l20 [4,8,16], nfd_3ch_narrow_l20_wide [16,32,64], linear single operator res32 / res64 (narrow); broad nfd_3ch_randlen, nfd_residual_worldframe_noaug_ep43, linear_switched_soft"
  held_fixed:
    recipe: "nfd_train_3ch_randlen.yaml except features / data / 60 epochs; unet_best.pth (best val: epoch 53 small, epoch 15 wide)"
    goals: "12-goal EXP-0046 set + two_squares, lyapunov, soft truth"
  baselines: "persistence (accuracy 0); broad models; random pick (slateN 0)"
  metric: "accuracy (METRICS.md) 1-step and 1-4-step rollout; slateN (whole 64-push pools)"
noise_floor: "training-seed sd (EXP-0036): accuracy ~0.003, slateN ~0.02-0.04; 32 test pools"
depends_on: [score-occupancy-subpixel-stable, goal-mask-axis-convention-row-y-col-x]
establishes: []
result: >
  accuracy_1 / rollout_4 / slateN: narrow NFD 0.506 / 0.348 / 0.774; narrow wide NFD 0.486 / 0.336 /
  0.803; narrow linear res64 0.452 / 0.324 / 0.582, res32 0.404 / 0.291 / 0.648; broad nfd_3ch_randlen
  0.453 / 0.304 / 0.695; broad worldframe residual 0.466 / 0.335 / 0.721; broad linear_switched_soft
  0.399 / 0.288 / 0.699. Clump states are predicted better than scatter by every model (e.g.
  narrow NFD 0.523 vs 0.441).
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

> **Caveat (2026-09-28, ISS-010):** DS-0008 (train) and DS-0009 (test) were ~45% illegal: the
> pile-aware stop clamp placed the tool ON a cube at touchdown in 44-56% of the narrow-domain
> transitions, and a further ~8-12% moved nothing (`experiments/OPEN_ISSUES.md` ISS-010). DS-0010
> (extra training rows) had ~0% exact overlaps but ~33% within a 1 mm margin. Every number in this record was trained and scored on that data. The
> clean rerun of the narrow NFD and linear models is EXP-0059's "Clean-data v2 rung" (DS-0015
> train, DS-0016 test; narrow NFD `slateN_tough` 0.700-0.731 over 3 seeds, linear64 0.626). The
> narrow-vs-broad comparison claimed here (C-056) has not been re-tested on clean data.


## Why this test discriminates
Same architecture, same recipe and the same clean test set; only the training distribution
differs (narrow 20 mm perpendicular single-layer vs broad randlen). The accuracy seed floor
(0.003) is small next to the observed gaps.

## What was actually run
- Data: DS-0008 (new, exact 20 mm, scatter + clumps, 6,144 rows) + DS-0010 (existing
  18-22 mm matched-physics rows, 5,777) for training. DS-0009 for testing only.
- Models: two NFDs (60 epochs, ~30 min each) and two linear single operators.
- The wide NFD's best validation loss came at epoch 15 (it overfits after that); the small
  NFD's at epoch 53.

## Numbers (results/offline_eval.json)
| model | accuracy_1 (scatter / clump) | rollout 1 / 2 / 3 / 4 | slateN |
|---|---|---|---|
| nfd_3ch_narrow_l20 | 0.506 (0.441 / 0.523) | 0.467 / 0.416 / 0.369 / 0.348 | 0.774 |
| nfd_3ch_narrow_l20_wide | 0.486 (0.413 / 0.505) | 0.444 / 0.397 / 0.351 / 0.336 | 0.803 |
| linear_narrow_l20_res64 | 0.452 (0.292 / 0.496) | 0.421 / 0.375 / 0.337 / 0.324 | 0.582 |
| linear_narrow_l20_res32 | 0.404 (0.193 / 0.464) | 0.375 / 0.344 / 0.304 / 0.291 | 0.648 |
| nfd_3ch_randlen (broad) | 0.453 (0.384 / 0.471) | 0.410 / 0.364 / 0.316 / 0.304 | 0.695 |
| nfd_residual_worldframe_noaug_ep43 (broad) | 0.466 (0.408 / 0.480) | 0.418 / 0.396 / 0.353 / 0.335 | 0.721 |
| linear_switched_soft (broad) | 0.399 (0.201 / 0.455) | 0.361 / 0.336 / 0.299 / 0.288 | 0.699 |

(rollout_1 differs from accuracy_1 because the rollout uses only the first push of each chain.)

## Reading
- Narrowing the training domain helps: +0.05 accuracy and +0.08 slateN for the same small NFD,
  well above the seed floor (accuracy). It also holds for multi-step rollout, where the
  narrow NFD stays ahead at every horizon.
- Width is not accuracy here: the 4x wider NFD overfits early and is less accurate, but it
  ranks pushes best (slateN 0.803). Accuracy and slateN disagree again (as for the linear
  operators: res64 more accurate than res32, but ranks worse).
- Single-layer clumps are easier to predict than scatter for every model (most of a scatter
  image is unmoved cubes, so its swept-region error is dominated by few contact events).
- Whether any of this reaches task completion is EXP-0054's question.

## What would change the verdict
More training seeds (slateN gaps of 0.03 between variants are within the seed range); a
larger narrow training set; NFD res128 (deferred).

## Threats
- One seed per variant.
- The narrow data sampler (pile-aware, chains) differs from the broad corpora's.
- DS-0009 is small (32 pools).

## Unrelated findings
none
