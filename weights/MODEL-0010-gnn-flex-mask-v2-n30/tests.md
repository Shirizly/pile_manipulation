# MODEL-0010 test history

- DS-0019 (100 slates, 16,583 rows, binary image-mask truth) slateN + accuracy, DS-0020 v2 val accuracy, paired vs
  MODEL-0008 / MODEL-0009 / original checkpoint (N 200 and N 30), rendering cap, strata -> EXP-0062 / RUN-0003 ->
  `experiments/EXP-0062-flex-v2-train-rerun/results/gnn_ds0019.{json,md}`
- inference timing on DS-0019 incl. perception and perception-cached, CPU and (after the GPU reset) CUDA: 77.2 / 43.6 ms per
  slate on CUDA, 259 / 207 on CPU -> EXP-0062 / RUN-0003 ->
  `experiments/EXP-0062-flex-v2-train-rerun/results/timing_gnn.json`
- three-family comparison on DS-0019 (NFD > LF switched > GNN, paired slateN; CUDA/CPU ms per slate side by side)
  -> EXP-0062 (record) -> `experiments/EXP-0062-flex-v2-train-rerun/results/combined_v2.{md,json}`, `figures/combined_v2.png`
- NOTE: scored on CPU only (GPU failed during RUN-0003); no GPU re-score yet.
