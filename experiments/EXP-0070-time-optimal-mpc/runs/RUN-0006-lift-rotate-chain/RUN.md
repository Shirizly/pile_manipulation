# RUN-0006 — Genesis tests A (lift), B (rotation), C (chained vs free)

- `chain_probe.py --mode lift` (run_probe tag exp0070_chain_lift; 3rd invocation reported, the first two crashed: too few states loaded / device mix) -> `results/chain_probe_lift.json`
- `chain_probe.py --mode rotate` (tag exp0070_chain_rotate) -> `results/chain_probe_rotate.json`
- `chain_probe.py --mode chain --n-states 1` (smoke, after two device-mix crashes; its state is kept, same code) then `--n-states 24` via `code/queue3.sh` (resumes, 23 more) -> `results/chain_probe_chain.json`, per-state tensors `results/chain_probe_states/*.pt`; `code/chain_analyse.py` -> `results/chain_analyse.json`
- states: pilot_v0b lyap episodes (positions + quaternions), k in 1..6; 32 envs; TRAINING_PHYSICS; rotation simulated at 3 rad/s (servo-driven x/y along the arc, yaw pinned per step); real-rotation time assumed pi rad/s, 20 rad/s^2 (sensitivity pi/2..2 pi in chain_analyse)
- commit a5ccb680 dirty
