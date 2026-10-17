---
id: MODEL-0020
status: active
---
# MODEL-0020 -- vanilla128, seed 2, unrolled T=4, wide Sean domain

EXP-0074 clean run (pinned split, after the 2026-10-07 04:00 split-leak fix). Two stages: single-step cut at epoch 20 (best-val ckpt) (`train_wide.py`, run `w128_f8_s2`),
then unrolled T=4 fine-tune (14 ep, `ms_wide.py`, run `w128_f8_s2_ms4`). Single-model, single test set (8 Sean shards).

## Read this before reusing
- Inputs/outputs are occupancy rasters of the Sean domain (cubes, 128 mm tray); world frame.
- The post-hoc mass balance (EXP-0074, val-selected) is applied by the predictor (`experiments/EXP-0075-*/code/wide_planner.py::WideEns(balance=True)`), not stored here.
- Training command is RECONSTRUCTED from `code/chain.sh` + LOG.md (04:20 entry); no run_config was saved by the training scripts.

## Provenance
- commit a7ff5cf3 era (dirty; EXP-0074 overnight run 2026-10-06/07), RTX 4070 Laptop shared GPU
- reconstructed: `python -u experiments/EXP-0074-wide-domain-zoom-nfd/code/ms_wide.py --kind world --res 128 --features 8,16,32 --init experiments/EXP-0074-wide-domain-zoom-nfd/runs/w128_f8_s2/unet_ss_best.pth --T 4 --epochs 14 --out experiments/EXP-0074-wide-domain-zoom-nfd/runs/w128_f8_s2_ms4` (seed 2)
- source checkpoint `experiments/EXP-0074-wide-domain-zoom-nfd/runs/w128_f8_s2_ms4/unet_best.pth` (copied; sha256 in `checkpoint.sha256`); per-epoch val in `runs/w128_f8_s2_ms4/log.json` or `runs/w128_f8_s2_ms4.log`

## Load
`wide_planner.Member("world", 128, "weights/MODEL-0020-nfd-wide-vanilla128-ms4-seed2/checkpoint.pth")` (EXP-0075 code), or `experiments/EXP-0074-wide-domain-zoom-nfd/code/eval_wide.py --world 128 <ckpt>` with `EVAL_FEAT=8,16,32`.

## Test history
See `tests.md`.
