# RUN-0001-repeat-pilot

Two invocations of the same script, one per fidelity arm (see EXPERIMENT.md
`provenance`/`design` for the full rationale).

## Configuration

- Base config: `simple_mpc/config/config_oracle.yaml` (`load_oracle_config()`),
  overridden in-script: `num_objects=20`, `settle_steps=60`,
  `reset_warmup_steps=10`, `mpc.rollout_settle_steps=20`.
- `n_envs = N_ACTIONS * N_REPEATS = 3 * 5 = 15` for both invocations.
- `N_STATES = 3` (reduced-fidelity invocation), `N_STATES = 1` (full-fidelity,
  auto-set by the script's `--full-fidelity` flag since a full-fidelity
  rollout costs ~3x more real settle/clearance steps per cell).
- Grid: `(64, 64)`, goal `"corner"`, value fn `control_utility_test.lyapunov`.
- Candidate actions (world metres, `[sx, sy, ex, ey]`, `wkspc_w=0.064`):
  `[-0.6w,0,0.6w,0]` (+x sweep), `[0,-0.6w,0,0.6w]` (+y sweep),
  `[-0.5w,-0.5w,0.3w,0.3w]` (diagonal sweep).

## Commit / dirty state

Commit `a175b981`. Tree was dirty (other agents' concurrent, unrelated work --
NFD training RUN-0010, baseline file edits per the session's git status
snapshot); this run's own code
(`experiments/EXP-0024-genesis-repeat-noise-floor/code/pilot_repeat_noise.py`,
promoted unchanged from `experiments/temp/exp0024-repeat-noise/
pilot_repeat_noise.py`) is the only file its numbers depend on.

## Exact commands

```
cd /home/alon/Code/pile_manipulation
conda activate pme
OMP_NUM_THREADS=4 PYTHONPATH=. python -u \
  experiments/temp/exp0024-repeat-noise/pilot_repeat_noise.py \
  > experiments/temp/exp0024-repeat-noise/stdout.log \
  2> experiments/temp/exp0024-repeat-noise/stderr.log
# reduced-fidelity arm; exit 0

OMP_NUM_THREADS=4 PYTHONPATH=. python -u \
  experiments/temp/exp0024-repeat-noise/pilot_repeat_noise.py --full-fidelity \
  > experiments/temp/exp0024-repeat-noise/stdout_full.log \
  2> experiments/temp/exp0024-repeat-noise/stderr_full.log
# full-fidelity arm; exit 0
```

Logged in `experiments/COMMANDS.jsonl` under `run_id`
`exp0024-pilot-repeat-noise` / `exp0024-pilot-repeat-noise-fullfidelity`.

## Timing

- Reduced-fidelity invocation: 131.0s total (one scene build ~61s incl. kernel
  compile, then 3 states x ~10-13s/rollout).
- Full-fidelity invocation: 72.8s total (one scene build ~61s, one state
  ~10.8s rollout).
- GPU: single RTX 4070 Laptop, 8GB, shared with the protected NFD training
  job (RUN-0010) throughout -- checked `nvidia-smi` before each invocation
  (247MiB used / 8188MiB total, 21-36% util from the training job) and never
  approached contention; no interaction with that job.

## Outputs

- `artifacts/RUN-0001-repeat-pilot/results.json` (reduced-fidelity, 3 states)
- `artifacts/RUN-0001-repeat-pilot/results_fullfidelity.json` (full-fidelity,
  1 state)

Both hold, per state: `v0`, `dv_by_action` (3x5 raw values), `action_means`,
`within_var_per_action`, `within_var_mean`, `between_var`,
`ratio_within_over_between`, `max_repeat_com_spread_mm`, timing.

## Status

Completed, exit 0, both invocations. No failures, no partial cells.
