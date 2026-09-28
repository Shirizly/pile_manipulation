# EXP-0048 -- sensitivity of 20 mm push outcomes to the dataset physics sets (overnight stage C)

Written retrospectively 2026-09-25 from EXPERIMENT.md, code/sensitivity.py and verify_switch.py,
runs/RUN-0001-sensitivity/RUN.md, the command ledgers and the TODO.md overnight plan (stage C),
not before the run. The pre-run plan is quoted where TODO.md records it.

## Why
Existing corpora were collected under different material physics (P_train 0.7/0.5/450, P_binned
0.3/0.3/1000, P_oracle 0.25/0.3/750). Can their 20 mm rows be pooled? Pre-run plan (TODO.md stage C,
"GPU agent, 45 min"): "physics-sensitivity pilot -- same states + actions under each dataset
physics set vs imperceptible state/action perturbations under one physics. Decides: mix /
pretrain-only / drop the other-physics data."

## Design
- States: DS-0006 step 0, first row of slates 100-111 (12 planned), n20 single-layer scatter.
- Actions: 32 per state, perpendicular 20 mm pushes aimed at a random cube (start = cube centre -
  8 mm along the direction), seed default_rng(1000 + slate).
- Conditions (same states and actions, one GenesisOracleEnv, 32 envs, full fidelity, physics
  switched at runtime by apply_physics): ref (P_train), repeat, P_binned, P_oracle, ref_back;
  state jitter 0.25/0.5/1.0 mm (+1 deg yaw); action shift 0.5/1.0 mm; action rotation 1 deg
  (perturbation seed 2000 + slate).
- Baselines / noise floor: repeat and ref_back (the simulator's repeat floor). No do-nothing arm
  (outcomes are compared to the P_train reference, not to the start).
- Metrics: moved-cube final-position diff (mm); |dv_cond - dv_ref| / between-action sd of dv_ref
  (`repeat_noise_ratio`-style, lyapunov on occ_for_scoring, 30 goals); Spearman of dv across the 32
  actions per (state, goal); top-1 match and regret; input-image L1/mass.
- Switch verification (verify_switch.py): 8 pushes from slate 100 under P_train x2, P_binned and an
  extreme low-friction set; material properties read back after each switch.
- Budget: declared 50 min wall clock; spent ~50 min.

## Predictions
From the frontmatter prediction block:
- supports: physics-set cube-diff median <= 1 mm and <= 1.5 x the 0.5 mm perturbation's; dv diff
  median <= 0.1 sd; Spearman median >= 0.95.
- refutes: cube-diff median > 2 x the 1.0 mm action perturbation's, or dv diff median > 0.3 sd, or
  Spearman median < 0.9.

## Deviations
- 5 of 12 planned states (slates 100-104) completed: run_probe's 1800 s timeout killed the job
  during slate 105 (resumable from artifacts/raw.pt).
- Budget declared 50 min against the plan's 45 min.
- Launch accidents: first launch crashed (CUDA-default generator); a third launch started by
  mistake overwrote runs/exp0048_sensitivity.json and artifacts/COMMAND.txt and NUL-truncated the
  log; the data come from the second launch (01:40).
- GPU shared with another Genesis job (3.5-6.5 min per state).
- An orchestrator note (EXPERIMENT.md) revises the reading of the "input change" column: L1 image
  change does not measure perceptibility.
