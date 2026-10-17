# RUN-0008 — why does minimising J pick short, do-little pushes? (offline diagnosis)

- `python scripts/run_probe.py --tag exp0070_diag_small --exp EXP-0070 -- python experiments/EXP-0070-time-optimal-mpc/code/diag_small.py` -> `results/diag_small.json`, `results/diag_small_rows.json` (270 states: pilot_v0 quadrant states of lyap/time/time_lin, pilot_v0b lyap states of 7 goals, k in {1,2,4}; 8192-push bank + CEM on J + CEM on dV, 0.5 s each, nfd_3ch_randlen, etc_map_v0)
- calibration rows (model dV vs recorded true dV for 5056 executed pushes of v0/v0b/v1/b01): inline re-run of `diag_small.part2` -> `results/diag_small_calibration_rows.json`; by-V split -> `results/diag_small_byV.txt` (inline script, recorded in its own output)
- no Genesis; GPU ~6 min; commit a5ccb680 dirty
