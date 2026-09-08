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


> **PROVISIONAL -- taken while the GPU was contended.** NFD and Schenck were both training concurrently on this card when these numbers were recorded (see `contention_details` in `timing_results.json`). Absolute latencies below are almost certainly inflated versus an idle card, and the inflation is not uniform across models (queueing depends on the other jobs' own kernel sizes) -- **do not use these to rank models by cost/candidate**, only to sanity-check the harness's methodology and units. Re-run with the command above once training finishes.


Recorded: 2026-09-08T00:27:41.879195+00:00 UTC. Commit: `d456b24a` (dirty).


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
| mean-delta | 1158.3 [1152.4, 1176.2] | 175.1 [174.3, 176.2] | 74.0 [72.8, 75.2] | 59.5 [59.1, 60.1] |
| linear | 5170.0 [5142.7, 5242.5] | 301.1 [298.7, 303.5] | 183.9 [183.2, 185.6] | 162.8 [158.8, 164.7] |
| gnn | 20245.4 [19680.7, 21432.8] | 3350.0 [3274.9, 3376.4] | 2954.5 [2945.8, 2982.1] | 2932.0 [2909.0, 2937.1] |
| nfd_unet3ch | 5262.3 [5214.9, 7285.6] | 199.6 [193.7, 248.9] | 88.1 [84.0, 100.8] | 94.9 [93.4, 101.5] |
| schenck_singlenet | 5474.1 [5323.8, 6857.1] | 678.1 [627.2, 723.7] | 920.2 [907.9, 935.0] | 892.7 [889.2, 906.1] |

## Total batch latency vs. pool size K

| model | K=1 (ms) | K=32 (ms) | K=128 (ms) | K=1024 (ms) |
|---|---|---|---|---|
| mean-delta | 1.16 [1.15, 1.18] | 5.60 [5.58, 5.64] | 9.47 [9.32, 9.63] | 60.96 [60.55, 61.54] |
| linear | 5.17 [5.14, 5.24] | 9.63 [9.56, 9.71] | 23.54 [23.45, 23.76] | 166.66 [162.66, 168.70] |
| gnn | 20.25 [19.68, 21.43] | 107.20 [104.80, 108.05] | 378.17 [377.06, 381.71] | 3002.35 [2978.81, 3007.63] |
| nfd_unet3ch | 5.26 [5.21, 7.29] | 6.39 [6.20, 7.97] | 11.28 [10.75, 12.91] | 97.21 [95.68, 103.95] |
| schenck_singlenet | 5.47 [5.32, 6.86] | 21.70 [20.07, 23.16] | 117.79 [116.21, 119.68] | 914.16 [910.54, 927.81] |

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

