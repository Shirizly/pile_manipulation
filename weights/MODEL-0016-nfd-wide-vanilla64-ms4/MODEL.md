---
id: MODEL-0016
status: active
---
# MODEL-0016 -- vanilla64 (the normal 64x64 NFD, [4,8,16]), unrolled T=4, wide Sean domain

EXP-0074 clean run (pinned split, after the 2026-10-07 04:00 split-leak fix). Two stages: 40 ep single-step from the narrow-pilot init (`train_wide.py`, run `w64_s0`),
then unrolled T=4 fine-tune (20 ep, `ms_wide.py`, run `w64_s0_ms4`). Single-model, single test set (8 Sean shards).

## Read this before reusing
- Inputs/outputs are occupancy rasters of the Sean domain (cubes, 128 mm tray); world frame.
- The post-hoc mass balance (EXP-0074, val-selected) is applied by the predictor (`experiments/EXP-0075-*/code/wide_planner.py::WideEns(balance=True)`), not stored here.
- Training command is RECONSTRUCTED from `code/chain.sh` + LOG.md (04:20 entry); no run_config was saved by the training scripts.

## Provenance
- commit a7ff5cf3 era (dirty; EXP-0074 overnight run 2026-10-06/07), RTX 4070 Laptop shared GPU
- reconstructed: `python -u experiments/EXP-0074-wide-domain-zoom-nfd/code/ms_wide.py --kind world --res 64 --features 4,8,16 --init experiments/EXP-0074-wide-domain-zoom-nfd/runs/w64_s0/unet_ss_best.pth --T 4 --epochs 20 --out experiments/EXP-0074-wide-domain-zoom-nfd/runs/w64_s0_ms4` (seed 0)
- source checkpoint `experiments/EXP-0074-wide-domain-zoom-nfd/runs/w64_s0_ms4/unet_best.pth` (copied; sha256 in `checkpoint.sha256`); per-epoch val in `runs/w64_s0_ms4/log.json` or `runs/w64_s0_ms4.log`

## Load
`wide_planner.Member("world", 64, "weights/MODEL-0016-nfd-wide-vanilla64-ms4/checkpoint.pth", feats=(4, 8, 16))` (EXP-0075 code), or `experiments/EXP-0074-wide-domain-zoom-nfd/code/eval_wide.py --world 64 <ckpt>` with `EVAL_FEAT=4,8,16`.

## Test history
See `tests.md`.
