# RUN-0001 — EXP-0067 stroke-width sweep (learned model) + oracle anchors

- Experiment: EXP-0067. Commit a5ccb680, dirty (uncommitted: EXP-0066's `simple_mpc/learned_mpc.py`,
  `simple_mpc/adapters.py`, EXP-0043 driver flags; this record's `Baselines/common/goals.py` (widened_letter_mask,
  parse_letter_goal, letter_goal_mask, LETTER_WIDTH_DILATE_PX), EXP-0043 driver and EXP-0050 `oracle.py` goal_mask ->
  letter_goal_mask, `oracle.py --vstar-files`).
- Data: DS-0006 corpus `Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys`, states 40-71 (starts).
  Model `nfd_3ch_randlen` (adapter registry). Goal masks from `env/target_shapes/helvetica_thin/helvetica_{O,T,S,X}.npy`.
- Pre-run (CPU): `code/calibrate_width.py` -> `results/width_calibration.json`; `code/vstar_width.py` ->
  `results/vstar_width.json` (ledger `EXP-0067/RUN-0001/vstar_width`). Cost pilot `results/pilot/pilot_cost.json`
  (letter_O_w10, 32 envs, 2 pushes: 25 s per push per chunk; fallback fired 4 / 64 steps, illegal 0).
- Queue: `code/queue.sh` (exact argv per cell inside; ledger `EXP-0067/RUN-0001/<tag>`). Logs + pid files in `logs/`.
  First launch 09:05 CEST failed in every cell within 30 s (`local tag=$1 w=$2 G="...$w"` expanded `$w` before it was
  assigned -> goal names `letter_O_w`; ledger lines w10/w15/w20/oracle_w15/oracle_w10 with status fail at 09:06-09:08
  are that; no outcome produced). Fixed and relaunched 09:07:47.
- Cells / status: see the table below (filled as cells finish).

| cell | started | ended | status | episodes | notes |
|---|---|---|---|---|---|
| w10 | 09:07:47 | 09:36:31 | ok | 128/128 | 4 chunks, ~7 min each |
| w15 | 09:36:31 | 10:04:35 | ok | 128/128 | |
| w20 | 10:04:35 | 10:33:19 | ok | 128/128 | |
| oracle_w15 | 10:33:19 | 12:11:08 | ok | 8/8 tasks x 16 pushes | 282-429 s per push |
| oracle_w10 | 12:11:08 | 14:03:02 | ok | 8/8 tasks x 16 pushes | ~418 s per push (fewer early solves) |

- Outputs: `results/<tag>.json` (driver checkpoint per push, includes recorded states), `results/analysis.{json,md}`
  from `code/analyse.py`.
- Queue total: 4.92 h GPU (09:07:47-14:03:02), one job at a time; `experiments/temp/gpu.lock` held the running
  driver's PID throughout and was removed after the last cell. The agent's session was cut off by a rate limit during
  oracle_w15; the queue ran unattended to completion (every cell checkpointed per push).
- Harmless log line after the last cell: `queue.sh: line 46: 0: command not found` -- the final
  `echo "... gpu_h $(gpu_h)"` line; `gpu_h` pipes through `bc` and the `0:` is bash's expansion of its output in that
  echo. `results/queue.done` still records `gpu_h 4.92`. No cell was affected (each cell's exit code is in the ledger).
- Analysis: `code/analyse.py` (ledger `EXP-0067/RUN-0001/analyse`, 14:37) -> `results/analysis.{json,md}`; per-letter
  rel slopes k = 8 -> 16 in `results/slopes_k8_k16.json` (inline script, recorded in the EXPERIMENT.md numbers).
