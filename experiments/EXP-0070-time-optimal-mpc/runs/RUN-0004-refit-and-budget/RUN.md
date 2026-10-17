# RUN-0004 — policy-iteration refit (v1) and the 0.1 s budget pair (b01)

- `code/analyse_pilot.py pilot_v0b` -> `results/pilot_v0b_episodes.json`; `code/refit_merge.py pilot_v0b time_rerank results/etc_map_v0.json etc_map_v1` -> `results/etc_map_v1.json` (+ `etc_map_v1_raw.json`, `etc_map_v1.txt`); quadrant_0/3, letter_T/S/X refit from 6 time_rerank episodes each, two_squares and letter_O kept v0.
- `time_mpc.py --tag pilot_v1 --cells time_rerank --starts 40..45 --etc-map results/etc_map_v1.json` (launched by the aborted `code/queue2.sh`; ran to completion); analysis `equal_time.py pilot_v1 pilot_v0b` (lyap rows from v0b).
- `time_mpc.py --tag pilot_b01 --cells lyap,time_rerank --budget 0.1 --starts 40..45 --etc-map results/etc_map_v0.json` (`code/queue3.sh`).
- commit a5ccb680 dirty; GPU.
