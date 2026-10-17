# RUN-0001 — Genesis push timing vs the analytic action-time model

- command: `python scripts/run_probe.py --tag exp0070_sim_timing --exp EXP-0070 -- python experiments/EXP-0070-time-optimal-mpc/code/sim_timing.py` (meta `runs/exp0070_sim_timing.json`; 3rd invocation is the reported one: the 1st crashed on a cpu/cuda tensor mix, the 2nd put 3 transit targets outside the tray)
- commit a5ccb680, dirty (other agents' uncommitted edits in simple_mpc/learned_mpc.py, adapters.py, etc.; this record's code was untracked new files)
- 6 envs, DS-0006 corpus `n20_scatter_s160a128_L20-70mm_randlenphys` slate 40 start, TRAINING_PHYSICS, config_oracle.yaml; pushes of 20/40/70 mm; free-air transit 0-78 mm at clearance height through `plate_velocity_translation`
- output `results/sim_timing.json`; runtime ~55 s GPU
