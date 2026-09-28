# RUN-0001-sensitivity

One process, one `GenesisOracleEnv` (n_envs = 32, built from
`simple_mpc/config/config_oracle.yaml` with `oracle_config_with_physics(P_train)`,
`record_transitions=False`). Physics switched at runtime before every
condition: `apply_physics(env, phys)` (-> `SandboxManipulation.set_material_properties`)
and `env._real_settle_steps = phys["settle_steps"]`. Every rollout is
`env.rollout_candidates(A (32,1,4), snapshot, use_rollout_fidelity=False, record=False)`.

## Inputs
- States: DS-0006 corpus `Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys`,
  `step(0)`, first row of slates 100..111 (20 cubes, (20, 7) pos + wxyz quat).
- Actions: 32 per state, `sample_actions` (seed `default_rng(1000 + slate)`):
  a random cube, a random direction u; start = cube centre - 8 mm * u, end =
  start + 20 mm * u (blade perpendicular, yaw from travel direction); both
  endpoints inside the tray minus 4 mm; rejected if any cube centre lies under
  the blade's touchdown footprint.
- Perturbations: `torch.Generator(cpu).manual_seed(2000 + slate)`, drawn in
  condition order.

## Conditions (order of execution per state)
ref (P_train), repeat (P_train), P_binned (0.3/0.3/1000, settle 3000),
P_oracle (0.25/0.3/750, settle 100), ref_back (P_train again -- checks the
switch is reversible), state_{0.25,0.5,1.0}mm (xy N(0,s) + yaw N(0,1 deg) per
cube, P_train), act_{0.5,1.0}mm (start/end each coordinate N(0,s)), act_rot1deg
(direction rotated +-1 deg about the start).
P_train = 0.7 / 0.5 box / 450, settle 3000 (= `TRAINING_PHYSICS`).

## Command
```
python scripts/run_probe.py --tag exp0048_sensitivity --out-dir experiments/EXP-0048-physics-sensitivity/runs \
  --threads 4 --timeout 1800 --exp EXP-0048 --artifact-dir experiments/EXP-0048-physics-sensitivity/artifacts \
  -- python -u experiments/EXP-0048-physics-sensitivity/code/sensitivity.py
python experiments/EXP-0048-physics-sensitivity/code/sensitivity.py --analyze   # recompute results/sensitivity.json
```
Commit 3bae8cd7, tree dirty (other agents' work; this run depends only on
`code/sensitivity.py` plus committed library code).

## Launch accident (provenance note)
The first launch (01:37) crashed (CUDA-default generator). The second
(01:40:26, run_probe pid 54294 / python pid 54407) is the run that produced the
data. A third launch (01:43:44) was started by mistake and killed within 2 min
before it wrote any state; its run_probe overwrote `runs/exp0048_sensitivity.json`
and `artifacts/COMMAND.txt` (same argv; its run_id is the one recorded there)
and truncated the log, so the log holds NULs before the surviving process's lines.

## Outputs
- `artifacts/raw.pt` -- per state: start state, actions, and for each condition
  the start state, actions and final positions (checkpointed after every state).
- `results/sensitivity.json` -- the analysis (recomputed after every state).
- Wall clock 3.5-6.5 min per state (11 conditions x 32 pushes), RTX 4070 Laptop, GPU shared with another Genesis job. Killed by the 1800 s timeout during slate 105: 5 states (100-104) complete.

## Switch verification (separate process)
`python scripts/run_probe.py --tag exp0048_verify_switch --out-dir experiments/EXP-0048-physics-sensitivity/runs --threads 4 --timeout 560 --exp EXP-0048 -- python -u experiments/EXP-0048-physics-sensitivity/code/verify_switch.py`
-> `results/verify_switch.json` (8 pushes, slate 100; exit 0 in 313 s).
