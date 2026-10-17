# RUN-0015 — closed loop: constant 0.5 s vs constant 0.06 s vs rule-A meta schedule

- `time_mpc.py --tag meta_cl --cells "blendnt1.0@0.5,blendnt1.0@0.06,blendnt1.0@meta" --starts 40..45 --map2 results/etc_map_m1.json` (tag exp0070_meta_cl), 2026-10-06 07:07-07:35
- all cells: CEM on model lyap dV, mass re-rank of the top-64 (blendnt1.0); only the planning budget differs (per-decision rule-A budget in the meta cell, from results/meta_schedule.json)
- DESIGN CONFOUND found at analysis: 16 pushes per episode for every cell = ~19 s of wall time for the fast cells vs ~25 s for 0.5 s, so comparisons after ~19 s and 16-push completion rates are not equal-time. Fixed by RUN-0015b: `time_mpc.py --tag meta_cl22 --cells "blendnt1.0@0.06,blendnt1.0@meta" --steps 22` (same starts/seed/map), compared against meta_cl's 0.5 s cell (`compare_cells.py meta_cl22 "blendnt1.0@0.5" meta_cl`)
- smoke: `--tag smoke3` (2 steps)
- commit a5ccb680 dirty
