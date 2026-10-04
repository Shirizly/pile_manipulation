# MODEL-0009 test history

- DS-0020 v2 val (1,665 rows, image-mask truth, swept-region `accuracy`): switched 0.390 [0.382, 0.398],
  single 0.285 [0.277, 0.293] (trajectory-cluster CI) -- lambda + bin scheme SELECTED on this split
  -> EXP-0062 / RUN-0002 -> `weights/MODEL-0009-*/fit.json`, `experiments/EXP-0062-*/results/lf_ds0019.json`
- DS-0019 test (slateN 3 goals x 3 vf, accuracy, strata, paired vs MODEL-0004 and MODEL-0008, binary image-mask
  truth) -> EXP-0062 / RUN-0002 -> `experiments/EXP-0062-*/results/lf_ds0019.{json,md}`, `ds0019_eval_report_lf_v2.json`
- inference timing on DS-0019 slates (CUDA + CPU, warp/operator/unwarp breakdown) -> EXP-0062 / RUN-0002 -> `results/timing_lf.json`
- three-family comparison on DS-0019 (NFD > LF switched > GNN, paired slateN; CUDA/CPU ms per slate side by side)
  -> EXP-0062 (record) -> `experiments/EXP-0062-flex-v2-train-rerun/results/combined_v2.{md,json}`, `figures/combined_v2.png`
- DS-0022 (EXP-0064 count-group carrot piles; out of its training domain), per count group, image-mask truth: switched slateN K=50 0.800-0.899, accuracy 0.309-0.411; paired vs in-domain MODEL-0013 -> EXP-0064 / RUN-0015 -> `experiments/EXP-0064-*/results/nfd_lf_image_metrics.{json,md}`
