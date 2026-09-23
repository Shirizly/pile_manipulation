# RUN-0001 — diverse-state value readout (EXP-0018)

Three stages, one CPU process each, `OMP_NUM_THREADS=4`, `CUDA_VISIBLE_DEVICES=""`
(the GPU was in use by another agent), python
`/home/alon/anaconda3/envs/pme/bin/python -u`.

| stage | script | wall | output |
|---|---|---|---|
| 1 | `code/stage1_generate_goals_v2.py` | 84 s | `experiments/temp/goal-states/dataset_v2.pt` (2000 goals, 6000 legal configurations, all passed `assert_no_penetration`) |
| 2 | `code/stage2_build_cache.py` | 29 s | `experiments/temp/exp0018-value-readout/cache.pt` (23520 states x 87, 2000 goals x 87, 3 value matrices, corpus/group/push-length labels, state index) |
| 3 | `code/stage3_fit.py` | 735 s | 51 `readout__<family>__<name>__<capacity>__<value>.joblib` + `results/metrics.json` |

Commit `6ea03278`, tree DIRTY (see EXPERIMENT.md). Seed 0 throughout; split seeds
100-102 (goal), 200-201 (state); corpus folds are deterministic.

Logs: `experiments/temp/exp0018-value-readout/stage{2,3}.log`.

Reload: `ValueReadout.load(path)` from
`experiments/EXP-0017-value-readout-instrument/code/value_readout.py`.
The fit subsample is `experiments/temp/exp0018-value-readout/fit_state_subsample_idx.npy`
(indices into `cache.pt`'s state arrays) — a later agent must reuse it, not re-derive it.

Status: completed, all 51 planned cells.
