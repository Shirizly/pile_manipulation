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


Recorded: 2026-09-08T03:41:20.248668+00:00 UTC. Commit: `63f298fb`.


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
| mean-delta | 872.4 [869.1, 880.5] | 171.6 [132.4, 172.1] | 70.5 [70.1, 72.0] | 57.0 [56.8, 57.3] |
| linear | 5043.3 [5017.9, 5069.6] | 296.4 [294.8, 297.3] | 180.1 [179.2, 181.3] | 162.6 [162.1, 163.0] |
| gnn | 2951.5 [2947.9, 2964.0] | 866.6 [860.9, 875.8] | 832.4 [829.9, 833.9] | 820.2 [818.7, 824.0] |
| nfd_unet3ch | 2166.3 [2143.0, 2174.4] | 80.6 [80.0, 81.7] | 38.9 [38.8, 39.0] | 37.2 [36.9, 37.3] |
| schenck_singlenet | 1418.8 [1401.5, 1440.3] | 203.5 [201.6, 205.8] | 384.2 [381.4, 385.0] | 382.5 [371.4, 387.9] |

## Total batch latency vs. pool size K

| model | K=1 (ms) | K=32 (ms) | K=128 (ms) | K=1024 (ms) |
|---|---|---|---|---|
| mean-delta | 0.87 [0.87, 0.88] | 5.49 [4.24, 5.51] | 9.02 [8.97, 9.21] | 58.35 [58.15, 58.70] |
| linear | 5.04 [5.02, 5.07] | 9.48 [9.43, 9.51] | 23.06 [22.94, 23.21] | 166.52 [165.98, 166.94] |
| gnn | 2.95 [2.95, 2.96] | 27.73 [27.55, 28.03] | 106.54 [106.23, 106.74] | 839.90 [838.36, 843.77] |
| nfd_unet3ch | 2.17 [2.14, 2.17] | 2.58 [2.56, 2.61] | 4.98 [4.96, 4.99] | 38.06 [37.83, 38.22] |
| schenck_singlenet | 1.42 [1.40, 1.44] | 6.51 [6.45, 6.58] | 49.18 [48.82, 49.28] | 391.69 [380.30, 397.25] |

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

