---
id: MODEL-0017
status: active
---
# MODEL-0017 -- zoom64 ([8,16,32], push-aligned zoom window, 64 px), unrolled T=4, wide Sean domain

EXP-0074 clean run (pinned split, after the 2026-10-07 04:00 split-leak fix). Two stages: 40 ep single-step from scratch (`train_wide.py`, run `z64_f8`),
then unrolled T=4 fine-tune (20 ep, `ms_wide.py`, run `z64_f8_ms4`). Single-model, single test set (8 Sean shards).

## Read this before reusing
- Inputs/outputs are occupancy rasters of the Sean domain (cubes, 128 mm tray); zoom predictions must be pasted into the 64x64 world frame for world-frame scoring.
- The post-hoc mass balance (EXP-0074, val-selected) is applied by the predictor (`experiments/EXP-0075-*/code/wide_planner.py::WideEns(balance=True)`), not stored here.
- Training command is RECONSTRUCTED from `code/chain.sh` + LOG.md (04:20 entry); no run_config was saved by the training scripts.

## Provenance
- commit a7ff5cf3 era (dirty; EXP-0074 overnight run 2026-10-06/07), RTX 4070 Laptop shared GPU
- reconstructed: `python -u experiments/EXP-0074-wide-domain-zoom-nfd/code/ms_wide.py --kind zoom --res 64 --features 8,16,32 --init experiments/EXP-0074-wide-domain-zoom-nfd/runs/z64_f8/unet_ss_best.pth --T 4 --epochs 20 --out experiments/EXP-0074-wide-domain-zoom-nfd/runs/z64_f8_ms4` (seed 0)
- source checkpoint `experiments/EXP-0074-wide-domain-zoom-nfd/runs/z64_f8_ms4/unet_best.pth` (copied; sha256 in `checkpoint.sha256`); per-epoch val in `runs/z64_f8_ms4/log.json` or `runs/z64_f8_ms4.log`

## Load
`wide_planner.Member("zoom", 64, "weights/MODEL-0017-nfd-wide-zoom64-ms4/checkpoint.pth")` (EXP-0075 code), or `experiments/EXP-0074-wide-domain-zoom-nfd/code/eval_wide.py --zoom 64 <ckpt>` with `EVAL_FEAT=8,16,32`.

## Test history
See `tests.md`.
