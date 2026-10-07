# Post-hoc mass fix, selected on VAL only (EXP-0074)

Fix: per step, on the pasted-frame delta: zero |delta|<thr, then rescale positive and negative parts to equal mean mass (balance). Internal model state untouched.
Selection rule: per model TYPE, maximise mean over the type's two models (zoom: z128,z64; vanilla: w128,w64) of val MEAN(acc1)+val MEAN(slateN). Columns: slateN / acc1 / roll_1 / roll_2 / roll_3 / roll_4.

## Selected

- zoom: balance thr=0.0 (score 1.543; grid none@0.0=1.450, balance@0.0=1.543, thr+balance@0.05=1.542, thr+balance@0.1=1.541, thr+balance@0.2=1.525)
- vanilla: balance thr=0.0 (score 1.532; grid none@0.0=1.463, balance@0.0=1.532, thr+balance@0.05=1.532, thr+balance@0.1=1.530, thr+balance@0.2=1.520)

## VAL grid (mean over 8 shards; slateN acc1 roll1-4)

| model | setting | slateN | acc1 | roll_1 | roll_2 | roll_3 | roll_4 |
|---|---|---|---|---|---|---|---|
| z128 | none@0.0 | 0.896 | 0.581 | 0.581 | 0.519 | 0.483 | 0.466 |
| z128 | balance@0.0 | 0.937 | 0.613 | 0.620 | 0.545 | 0.504 | 0.485 |
| z128 | thr+balance@0.05 | 0.938 | 0.608 | 0.616 | 0.543 | 0.502 | 0.483 |
| z128 | thr+balance@0.1 | 0.940 | 0.603 | 0.612 | 0.539 | 0.497 | 0.479 |
| z128 | thr+balance@0.2 | 0.934 | 0.590 | 0.600 | 0.527 | 0.485 | 0.465 |
| w128 | none@0.0 | 0.922 | 0.571 | 0.573 | 0.508 | 0.472 | 0.456 |
| w128 | balance@0.0 | 0.954 | 0.594 | 0.599 | 0.534 | 0.497 | 0.479 |
| w128 | thr+balance@0.05 | 0.957 | 0.591 | 0.596 | 0.532 | 0.495 | 0.478 |
| w128 | thr+balance@0.1 | 0.957 | 0.590 | 0.594 | 0.529 | 0.492 | 0.475 |
| w128 | thr+balance@0.2 | 0.959 | 0.581 | 0.584 | 0.519 | 0.482 | 0.465 |
| z64 | none@0.0 | 0.891 | 0.532 | 0.530 | 0.479 | 0.445 | 0.431 |
| z64 | balance@0.0 | 0.935 | 0.601 | 0.608 | 0.532 | 0.487 | 0.467 |
| z64 | thr+balance@0.05 | 0.938 | 0.600 | 0.606 | 0.531 | 0.486 | 0.466 |
| z64 | thr+balance@0.1 | 0.941 | 0.597 | 0.604 | 0.529 | 0.485 | 0.465 |
| z64 | thr+balance@0.2 | 0.940 | 0.587 | 0.594 | 0.523 | 0.479 | 0.459 |
| w64 | none@0.0 | 0.901 | 0.532 | 0.532 | 0.459 | 0.419 | 0.401 |
| w64 | balance@0.0 | 0.943 | 0.572 | 0.574 | 0.505 | 0.464 | 0.443 |
| w64 | thr+balance@0.05 | 0.945 | 0.570 | 0.572 | 0.504 | 0.464 | 0.444 |
| w64 | thr+balance@0.1 | 0.945 | 0.568 | 0.569 | 0.503 | 0.463 | 0.444 |
| w64 | thr+balance@0.2 | 0.942 | 0.558 | 0.560 | 0.494 | 0.456 | 0.437 |

## TEST (selected setting vs none vs balance)

| model | setting | slateN | acc1 | roll_1 | roll_2 | roll_3 | roll_4 |
|---|---|---|---|---|---|---|---|
| z128 | none@0.0 | 0.899 | 0.582 | 0.575 | 0.515 | 0.479 | 0.464 |
| z128 | balance@0.0 (selected) | 0.940 | 0.613 | 0.611 | 0.540 | 0.499 | 0.482 |
| w128 | none@0.0 | 0.924 | 0.572 | 0.563 | 0.503 | 0.467 | 0.453 |
| w128 | balance@0.0 (selected) | 0.957 | 0.595 | 0.588 | 0.528 | 0.491 | 0.475 |
| z64 | none@0.0 | 0.896 | 0.534 | 0.524 | 0.475 | 0.441 | 0.428 |
| z64 | balance@0.0 (selected) | 0.941 | 0.602 | 0.598 | 0.527 | 0.483 | 0.464 |
| w64 | none@0.0 | 0.907 | 0.534 | 0.525 | 0.457 | 0.418 | 0.401 |
| w64 | balance@0.0 (selected) | 0.942 | 0.573 | 0.565 | 0.501 | 0.461 | 0.442 |

## Zoom minus vanilla gaps on TEST (same resolution)

| pair | fix | slateN | acc1 | roll_1 | roll_2 | roll_3 | roll_4 |
|---|---|---|---|---|---|---|---|
| z128-w128 | none | -0.025 | +0.010 | +0.011 | +0.012 | +0.011 | +0.011 |
| z128-w128 | selected | -0.016 | +0.018 | +0.023 | +0.013 | +0.008 | +0.006 |
| z128-w128 | balance | -0.016 | +0.018 | +0.023 | +0.013 | +0.008 | +0.006 |
| z64-w64 | none | -0.011 | +0.000 | -0.001 | +0.018 | +0.024 | +0.027 |
| z64-w64 | selected | -0.001 | +0.028 | +0.033 | +0.025 | +0.021 | +0.022 |
| z64-w64 | balance | -0.001 | +0.028 | +0.033 | +0.025 | +0.021 | +0.022 |

## Test per scattered shard slateN (none -> selected)

- z128: scattered_n20 0.798->0.879, scattered_n50 0.872->0.938, scattered_n100 0.914->0.970
- w128: scattered_n20 0.852->0.917, scattered_n50 0.928->0.971, scattered_n100 0.932->0.972
- z64: scattered_n20 0.832->0.890, scattered_n50 0.859->0.933, scattered_n100 0.877->0.945
- w64: scattered_n20 0.820->0.880, scattered_n50 0.908->0.954, scattered_n100 0.915->0.966
