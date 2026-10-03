# MODEL-0004 -- switched-linear visual foresight on FleX binary image masks (64x64, 6 push-length bins)

**Status:** active

> **Trained on DS-0020 v1 (the OLD payload), not the current DS-0020.** On 2026-10-02 the user
> replaced DS-0020's payload with a new blob/spread collection (v2, `datasets/DS-0020-*/DATASET.md`).
> This instance was fit on the v1 data (2000 trajectories from ONE uniform-spread start state), now archived at
> `datasets/DS-0020-training-data-flex-N864/old_data/` (raw) + `old_data/_ported_v1/` (config.yaml, splits.json,
> cache/ incl. image_masks.npz, DATASET.md). Every DS-0020 path, row count and split below refers to v1; load it
> with `old_data/_ported_v1/config.yaml` (its `config.yaml` was repointed there). Its numbers do not transfer to v2.

## What this is

Suh & Tedrake 2020 switched-linear visual foresight (`docs/linear_visual_foresight_baseline.md`),
fit for EXP-0061 (rerun of EXP-0001 on FleX carrot piles). Six per-push-length-bin ridge
operators (toward identity) on the 64x64 canonical push-frame occupancy, plus one single
(unswitched) operator in the same bundle. Input and target are the **binary top-down image
masks** (`FlexData/image_mask.py`, `occ_source="image_mask"`), grid +-7.2 FleX units,
row = X = x_flex, col = Y = -z_flex. Push lengths / bin edges are in **FleX units**.

## Provenance

- commit d72bb304 (dirty tree), 2026-10-01
- script: `experiments/EXP-0061-flex-cross-corpus-rerun/code/fit_lf_flex.py --out <dir>/operators.pt`
  (payload then renamed `checkpoint.pt` so `.gitignore`'s `weights/*/checkpoint.*` covers it; the
  sweep log `operators_fit.json` renamed `fit.json`)
- fit: DS-0020 `train` (17,837 kept rows), lambda selected on DS-0020 `val` (1,991 rows, whole
  trajectories held out -- splits.json); config in `config.yaml`
- bins (EXP-0003 scheme, equal width over [0, 13.21]): edges 0 / 2.20 / 4.40 / 6.61 / 8.81 /
  11.01 / 13.21, train rows 2124 / 4867 / 5325 / 3961 / 1424 / **136** (last bin under-filled:
  M/D = 0.03; still above MIN_ROWS_PER_BIN = 50, so fitted, not identity)
- lambda: switched 300, single 1000 (grid 0.1-3000)

## Contents

`checkpoint.pt` (torch dict, 470 MB): `operators` (6 x [4096,4096]), `bin_edges`, `counts`,
`single_operator` ([4096,4096]), `mean_delta`, `res`=64, `crop`=1.0, `constraint`="ridge",
`ridge`=300 (switched), `ridge_single`=1000, `n_bins`=6, `train_cfg`, `units`, `provenance`.
Read by `Baselines/LinearForesight/predictor.py` (`build_predictor` = switched,
`build_predictor_single` = single); registered in `Baselines/common/eval_report.py` as
`lf_flex_switched` / `lf_flex_single`.

## Regeneration

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/fit_lf_flex.py \
        --out weights/MODEL-0004-linear-foresight-flex-mask/operators.pt
    # then mv operators.pt checkpoint.pt; mv operators_fit.json fit.json

Closed form (deterministic given the data).

## Test history

See `tests.md`.
