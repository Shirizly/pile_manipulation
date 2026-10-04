# MODEL-0013 -- switched-linear visual foresight on FleX binary image masks, fit on DS-0021 (count-group carrots; 64x64, 6 push-length bins)

**Status:** active

## What this is

The EXP-0062 recipe (`fit_lf_v2.py`, MODEL-0009) re-fit on **DS-0021** -- the EXP-0064 count-group
FleX carrot piles (groups 10-30 / 50-70 / 100-150 / 400-500 carrots, one centred blob each).
Six per-push-length-bin ridge operators (toward identity) on the 64x64 canonical push-frame occupancy,
plus one single (unswitched) operator in the same bundle. Input and target are the **binary top-down
image masks** (`occ_source="image_mask"`, `FlexData/image_mask.py`), grid +-7.2 FleX units, table frame
X = x_flex, Y = -z_flex (stored actions are (x, -z): invariant `flex-action-frame-neg-z`). Push lengths
/ bin edges in **FleX units**. Rows flagged by FlexData (escaped |coord| > 10, out-of-grid, null, nan)
are excluded from fit and selection.

## Provenance

- commit 3373e65b (dirty tree: this session's `FlexData/build_cache.py` / `image_mask.py` DS-0021/22
  entries and DS configs, plus other sessions' edits), 2026-10-04, EXP-0064 / RUN-0013
  (`experiments/EXP-0064-obj-count-effect-study/runs/RUN-0013-lf-fit/RUN.md`), CPU only
- `python -u experiments/EXP-0064-obj-count-effect-study/code/fit_lf_ds0021.py --device cpu --out weights/MODEL-0013-linear-foresight-flex-mask-countgroups/checkpoint.pt`
  (= EXP-0062 `fit_lf_v2.py` with only the corpus paths swapped; final per-bin operators via
  `Baselines/LinearForesight/fit_switched.py::fit_bins`, max |diff| vs the Gram solve 1.9e-6)
- fit: DS-0021 `train` (15,193 kept rows of 18,000; 1,745 trajectories with >= 1 kept row of 1,800);
  (bin scheme, lambda) selected on DS-0021 `val` (1,741 kept rows, 198 trajectories) ONLY. Split =
  EXP-0064 GroupedParticleDataset rule (per count group, sorted states, first 90 % train), so val = the
  GNN's (MODEL-0011/0012) val states.
- bins: equal width over [0, 9.478], train rows 1649 / 4081 / 4669 / 3461 / 1244 / **89**; val accuracy
  equal 0.3596 vs the collection's edges 0.3557 (both at lambda 300) -> equal chosen.
- lambda: switched 300, single 300 (grid 1-10000); val accuracy switched 0.3596, single 0.2660.

## Contents

`checkpoint.pt` (470 MB, `Baselines/LinearForesight/predictor.py` bundle: `operators`, `bin_edges`, `counts`,
`single_operator`, `mean_delta`, `res`=64, `crop`=1.0, `ridge`=300, `ridge_single`=300, `bin_scheme`="equal",
`train_cfg`, `provenance`); `fit.json` (full sweep, both schemes, per-bin val); `COMMAND.txt`.
Load: `eval_report.py --ckpt lf_flex_switched=weights/MODEL-0013-*/checkpoint.pt` (or `lf_flex_single=`).

## Read this before reusing

- Meaningful only on the FleX grid / image-mask input (DS-0019..0022 instance configs).
- Escaped-particle rows are excluded here, unlike the GNN MODEL-0012 (trained with them; EXP-0064 issues I-2).

## Regeneration

Command above (closed form, deterministic given the caches; ~6 min on CPU). Caches:
`EXP-0064/code/build_flex_caches_ds0021_ds0022.sh`.
