# RUN-0011 — user's loop, iteration 2

- automated by `code/queue_loop.sh` right after RUN-0010: `analyse_pilot.py iter1`, `equal_time.py iter1`, then
  `loop_step.py iter1 results/etc_map_m0.json etc_map_m1` = the pre-declared rule (EXPERIMENT.md) -> `results/loop_step_iter1.json`:
  best cell blend1.0, equal-time +0.050 / +0.040 at 12 / 16 s (cond a true), m0's mean Bellman residual on its own transitions
  +0.57 s (cond b false, threshold 0.5) -> lam2 = 0.5; map m1 = gap|td refit on retro + pilot_v0b/v1/b01 + iter1 blend1.0
  episodes (weight 3) -> `results/etc_map_m1.json`
- `time_mpc.py --tag iter2 --cells lyap,blend0.5 --starts 40..45 --map2 results/etc_map_m1.json --map-key "gap|td"` (tag exp0070_iter2)
- commit a5ccb680 dirty
