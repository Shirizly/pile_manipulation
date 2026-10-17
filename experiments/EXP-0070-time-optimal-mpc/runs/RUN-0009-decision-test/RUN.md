# RUN-0009 — v2 maps (etc_v2.py) and the one-step decision test with Genesis truth

- `etc_v2.py calib` (run_probe tag exp0070_etcv2_calib) -> `results/etc_v2_calib.json`: model vs true change of both readouts (lyap V, mass gap) for every executed push of pilot_v0/v0b/v1/b01, 5 quantile bins per goal
- `etc_v2.py fit etc_map_m0` -> `results/etc_map_m0.json`: f_g(r) = c r^p for r in {lyap, gap} x {mc, td}, data = retro (1296 recorded episodes) + pilot_v0b + pilot_v1 + pilot_b01 (pilot_v0's collapsed time cells excluded), GroupKFold(4) by start; offsets attached
- `decision_test.py` (tag exp0070_decision_test) -> `results/decision_test.json`, `results/decision_test_rows.json`: 270 states, lyap CEM pool (top 64), selection rules J / blend, every distinct pick executed in Genesis (32 envs per batch, standard execute_action)
- commit a5ccb680 dirty
