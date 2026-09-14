# RUN-0001-timing-sweep

Full timing sweep, all 8 model configurations x 11 batch sizes
(K in {128,96,64,48,32,24,16,8,4,2,1}), >=5 warm-up (discarded), 15 timed
repeats each, `torch.cuda.synchronize()` before/after every timed region.

- Command: see `COMMAND.txt`.
- Commit: `0ddab20f` (dirty -- pre-existing docs/skills reorg unrelated to
  this run; no file this run reads or writes was among the dirty files).
- Device: cuda:0 (RTX 4070 Laptop, 8GB). GPU idle at start (15 MiB used,
  0% util) -- `check_gpu_contention()` passed, not `--force`d.
- Dataset: `configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`,
  loaded via `Baselines.common.data.load_cell`, split "train" (7680
  transitions available; only row 0's state + K cycled real actions used).
- Models loaded from: `Baselines/NFD/runs/nfd_3ch/unet_best.pth`,
  `Baselines/GNN/runs/ckpt_best.pth`, `Baselines/SchenckCNN/runs/schenck.pth`,
  `weights/MODEL-0001-stage2-visual-switched/checkpoint.pt` (switched + a
  second global/unswitched reading of the same file),
  `weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt`,
  `experiments/temp/stage2-slaten/operators/{hybrid14,hybrid94}_lam1.0.pt`.
- Start/end: single foreground run, ~35s wall clock total (see `run.log`
  timestamps are not printed but the process completed within the 120s
  default Bash timeout with margin).
- Status: completed, no errors, no `--force` needed.
- Outputs: `artifacts/RUN-0001-timing-sweep/{results_timing.json,run.log}`,
  `results/{RESULTS.md,budgets.json}` (derived via `code/make_budgets.py`
  reading this run's `results_timing.json`).
