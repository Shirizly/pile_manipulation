# EXP-0061 / RUN-0001 -- final DS-0019 test evaluation (2026-10-01, 11:00-11:20 CEST)

> **Data-location note (2026-10-02).** Every "DS-0020" in this record is DS-0020 **v1** (2000
> trajectories from ONE uniform-spread start state). The user then replaced DS-0020's payload with a new
> blob/spread collection (v2). v1 was archived, not deleted: raw `datasets/DS-0020-training-data-flex-N864/old_data/<traj>/`,
> config/splits/cache (incl. `image_masks.npz`, `image_paths.json`) `old_data/_ported_v1/`; read it with
> `old_data/_ported_v1/config.yaml` (the code/ scripts and `configs/dataset/flex_ds0020_train_ds0019_test.yaml`
> were repointed there; logs and result files are left as written). The numbers here are unaffected.
> v2 data check: `figures/data_check_v2/`, `code/data_check_v2.py`, `code/leakage_scan_v2.py`.

- commit d72bb304, **dirty** (file list in EXPERIMENT.md "What was actually run"); python
  `/home/alon/anaconda3/envs/pme/bin/python -u`, single 8 GB GPU (cuda), OMP_NUM_THREADS=4.
- data: DS-0019 (all 100 slates, 16,583 kept rows), DS-0020 val (1,991 kept rows); corpus
  `flex_ds0019_mask`, truth = binary image mask (`--truth-scoring image`).
- models: MODEL-0005/0006/0007 (NFD seeds 0-2, `Baselines/NFD/runs/nfd_3ch_flex_mask_seed{s}/unet_best.pth`),
  MODEL-0004 (`lf_flex_switched`, `lf_flex_single`), `gnn_flex_drp` (checkpoint
  `Baselines/GNN/data/gnn_dyn_model/2023-01-28-10-42-05-114323/net_epoch_0_iter_1000.pth`, N 200,
  plane 0.24; FPS seed offsets 0/1/2; N 50 sensitivity).
- exact argv: `COMMAND_eval_report.txt`, `COMMAND_final_eval.txt` (this dir) and
  `experiments/COMMANDS.jsonl` run ids `EXP-0061/RUN-0001/*` (start + end events, incl. the failed attempts).
- logs: `../../logs/eval_report_ds0019_gnn_lf.log`, `final_eval_ds0019.log`, `final_eval_ds0020_val.log`.
- outputs: `../../results/ds0019_gnn_lf.json` (eval_report CLI), `../../results/final_eval/`
  (`ds0019.json`, `ds0019_rows.npz`, `ds0020_val.json`, `ds0020_val_rows.npz`, `ds0019_gnn_nvox.json`,
  `summary.{json,md}`, `headline.json`), `../../figures/final_eval/slateN_and_accuracy.png`.
- status: complete. Code change during the run: `Baselines/GNN/flex_predictor.py::perceive`
  small-pile fallback (see EXPERIMENT.md).
