# MODEL-0008 test history

- DS-0019 (100 same-state slates, 16,583 rows, binary image-mask truth, default goals x 3 value functions)
  + DS-0020 v2 val accuracy + paired comparison vs MODEL-0005 + strata -> EXP-0062 / RUN-0001 ->
  `experiments/EXP-0062-flex-v2-train-rerun/results/nfd_seed0_ds0019.{json,md}` (eval_report CLI row:
  `results/ds0019_eval_report_nfd_v2_s0.json`)
- inference timing on DS-0019 (per-slate pool batches, batch 128, PNG->mask) -> EXP-0062 / RUN-0001 ->
  `experiments/EXP-0062-flex-v2-train-rerun/results/timing_nfd.json`
- three-family comparison on DS-0019 (NFD > LF switched > GNN, paired slateN; CUDA/CPU ms per slate side by side)
  -> EXP-0062 (record) -> `experiments/EXP-0062-flex-v2-train-rerun/results/combined_v2.{md,json}`, `figures/combined_v2.png`
- DS-0022 (EXP-0064 count-group carrot piles; out of its training domain), per count group, image-mask truth: slateN K=50 0.894-0.948, accuracy 0.456-0.541 -> EXP-0064 / RUN-0015 -> `experiments/EXP-0064-*/results/nfd_lf_image_metrics.{json,md}`
