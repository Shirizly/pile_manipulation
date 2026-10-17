# RUN-0003 — closed-loop pilot: lyap vs time-aware objectives

- command: `python scripts/run_probe.py --tag exp0070_pilot_v0 --exp EXP-0070 -- python experiments/EXP-0070-time-optimal-mpc/code/time_mpc.py --tag pilot_v0 --cells lyap,time,time_lin,rate --etc-map experiments/EXP-0070-time-optimal-mpc/results/etc_map_v0.json`
- defaults: model nfd_3ch_randlen (broad), CEM pop 1024 elite 0.25, budget 0.5 s, n_cand 1024, bank 16384, 16 pushes, pile-aware bank, legalize + legal fallback, planner seed 0; goals quadrant_0, quadrant_3, two_squares, letter_O/T/S/X; DS-0006 starts 40-47; cells interleaved per chunk of 32 envs
- smoke run before it: `--tag smoke` (2 steps, OLD binned map; superseded, not analysed)
- output `results/pilot_v0.json` (checkpointed every step), analysis `code/analyse_pilot.py pilot_v0` -> `results/analysis_pilot_v0.json`
- commit a5ccb680 dirty

## Session notes
- pilot_v0 was STOPPED (kill of PID 220854) after 2 of 7 chunks: 64 complete episodes (16 quadrant pairs x 4 cells); the 32 incomplete rows of chunk 2 stay in the file with complete=false and are ignored by every analysis.
- pilot_v0b: `time_mpc.py --tag pilot_v0b --cells lyap,rate,time_rerank --starts 40 41 42 43 44 45` (run_probe tag exp0070_pilot_v0b). Machine lost power ~16:35 during chunk 1 (chunk 0 = 32 episodes on disk). Resumed 19:00 with the identical command (run_probe tag exp0070_pilot_v0b_resume, PID 19016); the runner skips completed episodes and re-ran chunk 1 from scratch. 126 complete episodes.
- analysis: `analyse_pilot.py pilot_v0b`, `equal_time.py pilot_v0b`, `equal_time.py pilot_v0`.
