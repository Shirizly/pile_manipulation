---
id: MODEL-0024
status: active
---
# MODEL-0024 -- switched-linear visual foresight, wide Sean domain (64 px, 6 push-length bins)

Suh & Tedrake switched-linear foresight (`Baselines/LinearForesight`), fit on 80k Sean rows by EXP-0074 (`code/fit_lf_wide.py`).
6 per-bin ridge operators (4096x4096) toward identity, lambda 100 selected on val.

## Read this before reusing
- `checkpoint.pt` is a HARD LINK to `experiments/EXP-0074-wide-domain-zoom-nfd/runs/lf_wide/operators.pt` (400 MB): deleting one path keeps the other.
- Its train/val files are NOT the NFD models' (it was fit before split_files.json existed; test files recovered by fingerprint and identical); leakage-free w.r.t. the cached test set (LOG 03:30, 04:00).
- Load: `experiments/EXP-0074-*/code/eval_lf_wide.py::LF` or EXP-0075 `code/bs_lib.py` LF planner wrapper.

## Provenance
- EXP-0074, 2026-10-07 ~03:25; reconstructed command `python -u experiments/EXP-0074-wide-domain-zoom-nfd/code/fit_lf_wide.py` (fit log `runs/lf_wide/fit.log`, `split.log`)
- sha256 in `checkpoint.sha256`; full lambda sweep in `fit.json` (copy of hparams.json)

## Test history
See `tests.md`.
