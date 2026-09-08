# MPC candidate-pool timing

Measures the wall-clock cost of predicting a batch of K candidate
actions from one current state -- the operation an MPC inner loop
repeats every step (K=128 in these slates; hundreds-to-thousands in a
real deployment). See `Baselines/common/benchmark_time.py`'s module
docstring for exact methodology (warm-up, CUDA sync, median/IQR,
device provenance).


## Re-run on an idle GPU

```
conda activate pme && PYTHONPATH=. python Baselines/common/benchmark_time.py --force
```

(Drop `--force` first to confirm the contention gate reports
`contended: false` before trusting the numbers -- it refuses to write
any output at all otherwise.)


GPU was idle (contention gate passed) when these numbers were recorded.


Recorded: 2026-09-08T04:00:14.758692+00:00 UTC. Commit: `c53ad473` (dirty).


## Model size

| model | device (requested) | device (actual) | param count | checkpoint bytes | note |
|---|---|---|---|---|---|
| mean-delta | cuda | cpu | 4,096 | n/a (fit in-memory) | |
| linear | cuda | cpu | 16,777,216 | n/a (fit in-memory) | |
| gnn | cuda | cuda:0 | 38,403 | 161,629 | |
| nfd_unet3ch | cuda | cuda:0 | 30,541 | 155,923 | |
| schenck_singlenet | cuda | cuda:0 | 139,937 | 571,261 | |

## Per-candidate cost vs. pool size K

Median [IQR], over 15 repeats (5 warm-up discarded).

| model | K=1 (us/candidate) | K=32 (us/candidate) | K=128 (us/candidate) | K=1024 (us/candidate) |
|---|---|---|---|---|
| mean-delta | 860.9 [856.4, 867.9] | 92.9 [92.4, 160.9] | 69.2 [69.1, 70.0] | 56.6 [56.6, 56.9] |
| linear | 5004.3 [4986.6, 5011.8] | 285.4 [283.3, 286.8] | 178.0 [177.7, 178.8] | 160.1 [159.7, 160.6] |
| gnn | 2296.1 [2266.3, 2334.7] | 850.9 [834.4, 889.9] | 685.3 [656.8, 710.1] | 671.9 [656.5, 697.6] |
| nfd_unet3ch | 2230.9 [2212.1, 2265.7] | 79.3 [79.2, 80.2] | 38.7 [38.5, 38.9] | 37.0 [36.6, 37.2] |
| schenck_singlenet | 1434.0 [1415.4, 1462.5] | 208.4 [199.1, 211.1] | 380.5 [378.7, 388.8] | 374.0 [368.1, 388.0] |

## Total batch latency vs. pool size K

| model | K=1 (ms) | K=32 (ms) | K=128 (ms) | K=1024 (ms) |
|---|---|---|---|---|
| mean-delta | 0.86 [0.86, 0.87] | 2.97 [2.96, 5.15] | 8.85 [8.84, 8.96] | 57.96 [57.91, 58.30] |
| linear | 5.00 [4.99, 5.01] | 9.13 [9.07, 9.18] | 22.78 [22.75, 22.88] | 163.96 [163.57, 164.45] |
| gnn | 2.30 [2.27, 2.33] | 27.23 [26.70, 28.48] | 87.71 [84.07, 90.89] | 688.06 [672.25, 714.38] |
| nfd_unet3ch | 2.23 [2.21, 2.27] | 2.54 [2.54, 2.56] | 4.95 [4.93, 4.98] | 37.91 [37.47, 38.08] |
| schenck_singlenet | 1.43 [1.42, 1.46] | 6.67 [6.37, 6.76] | 48.71 [48.47, 49.77] | 382.95 [376.98, 397.29] |

## Reading this table
- **Per-candidate us** is the number that matters for choosing a pool
  size K under a wall-clock budget -- it is what one extra candidate
  costs.
- **Total batch latency** is what matters for a single MPC step's
  deadline -- fixed per-call overhead (Python/CUDA launch, any
  per-row Python loop such as the GNN's particle rasterisation) does
  not shrink with K, and can dominate at small K even for a model
  that is cheap per-candidate at large K.
- `mean-delta`/`linear` are timed on CPU throughout (matching how
  they are actually invoked in this repo -- see the module
  docstring); a GPU port was not attempted since the task is fitting
  a small dense operator, not a neural forward pass.
- The GNN predictor's `predict_occ` loops in Python over the batch
  to call `rasterize_particles` once per row (OpenCV, not batched;
  see `Baselines/common/data.py`'s docstring) -- expect its
  per-candidate cost to fall much less steeply with K than a model
  whose entire forward pass is one batched tensor op.

