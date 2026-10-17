---
id: MODEL-0025
status: active
---
# MODEL-0025 -- ens128: zoom128 + vanilla128 (seed 0 each), mass-balanced mean

Composite; no checkpoint. Members MODEL-0021 (zoom128 s0) and MODEL-0018 (vanilla128 s0). Prediction = per-step mass-balanced mean of
the members' pasted-frame predictions (`wide_planner.WideEns(..., balance=True)`; EXP-0075 key `ens128` in `code/bs_lib.py`, `ens2bal` in `closed_loop.py`).
Offline (EXP-0074 `results/eval_ens_zoom_vanilla_s0_balance.json`): the ensemble of these two with the balance fix is the best wide-domain predictor tested there.
Cost ~ sum of both members.

## Test history
See `tests.md`.
