# MODEL-0009 -- switched-linear visual foresight on FleX binary image masks, fit on DS-0020 **v2** (64x64, 6 push-length bins)

**Status:** active

## What this is

Suh & Tedrake 2020 switched-linear visual foresight (`docs/linear_visual_foresight_baseline.md`),
the EXP-0062 rerun of MODEL-0004 (EXP-0061) on the in-domain DS-0020 v2 training data. Six
per-push-length-bin ridge operators (toward identity) on the 64x64 canonical push-frame
occupancy, plus one single (unswitched) operator in the same bundle. Input and target are the
**binary top-down image masks** (`occ_source="image_mask"`), grid +-7.2 FleX units. Push
lengths / bin edges in **FleX units**.

## Provenance

- commit d72bb304 (dirty tree), 2026-10-02, EXP-0062 / RUN-0002
  (`experiments/EXP-0062-flex-v2-train-rerun/runs/RUN-0002-lf/RUN.md`)
- `python -u experiments/EXP-0062-flex-v2-train-rerun/code/fit_lf_v2.py --out weights/MODEL-0009-linear-foresight-flex-mask-v2/checkpoint.pt`
  (sweep via Gram matrices; final per-bin operators re-fit with `Baselines/LinearForesight/fit_switched.py::fit_bins`,
  max |diff| vs the Gram solve 9e-6)
- fit: DS-0020 v2 `train` (15,093 kept rows, 1,669 trajectories); (bin scheme, lambda) selected on
  DS-0020 v2 `val` (1,665 rows, 188 whole trajectories held out; trajectories 0-99 excluded)
- bins: equal width over [0, 9.41] (EXP-0003 scheme), train rows 1722 / 3973 / 4530 / 3435 / 1323 / **110**
  (last bin M/D = 0.027; above MIN_ROWS_PER_BIN = 50). The collection's own edges (0.96 ... 9.62) were the
  alternative: val 0.3833 vs 0.3903 (equal - collection +0.0070 [+0.0043, +0.0097], trajectory bootstrap);
  that bundle is kept in `alt_collection_bins/` (not the registered model).
- lambda: switched 300, single 1000 (grid 1-10000)

## Contents

`checkpoint.pt` (470 MB): `operators` (6 x [4096,4096]), `bin_edges`, `counts`, `single_operator`,
`mean_delta`, `res`=64, `crop`=1.0, `constraint`="ridge", `ridge`=300, `ridge_single`=1000, `n_bins`=6,
`bin_scheme`="equal", `train_cfg`, `units`, `provenance`. `fit.json`: the full sweep (both schemes, per-bin val).
Load: `eval_report.py --ckpt lf_flex_switched=weights/MODEL-0009-*/checkpoint.pt` (and/or
`--ckpt lf_flex_single=...`) -- the registered `lf_flex_*` specs default to MODEL-0004.

## Regeneration

Command above (closed form, deterministic given the data); `--scheme collection --out .../alt_collection_bins/checkpoint.pt` for the alternative.

## Test history

See `tests.md`.
