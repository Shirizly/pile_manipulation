---
id: EXP-0048
title: >
  The material physics set (P_train 0.7/0.5/450 vs P_binned 0.3/0.3/1000 vs
  P_oracle 0.25/0.3/750) changes 20 mm push outcomes about as much as a 0.5 mm
  action jitter -- ~2x the simulator's repeat floor, ~0.04 between-action sd in
  dv, action ranking Spearman 0.99 -- so the datasets can be pooled
tier: T1
mode: exploratory
date: 2026-09-25
hypothesis: null

claim: >
  On n20 single-layer scatter states (DS-0006, 5 states (of 12 planned) x 32 perpendicular
  20 mm pushes aimed at a cube, full-fidelity rollout_candidates), switching the
  material physics from P_train to P_binned or P_oracle changes the final
  positions of pushed cubes by a median <= 1 mm, the lyapunov dv (30 letter /
  quadrant goals) by a median <= 0.1 between-action sd, and leaves the per-
  (state, goal) Spearman ranking of the 32 actions >= 0.95 -- i.e. within the
  range spanned by the simulator's repeat noise and model-imperceptible
  (<= 0.5 mm) state/action perturbations.

prediction:
  supports: "physics-set cube-diff median <= 1 mm and <= the 0.5 mm action/state perturbation's x1.5; dv diff median <= 0.1 sd; Spearman median >= 0.95"
  refutes: "physics-set cube-diff median > 2x the 1.0 mm action perturbation, or dv diff median > 0.3 sd, or Spearman median < 0.9"
  discriminating: true

provenance:
  commit: 3bae8cd7
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0048-physics-sensitivity/code/sensitivity.py
  data: ["DS-0006 (Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys, step 0, slates 100-111)"]
  code_path: "GenesisOracleEnv.rollout_candidates(use_rollout_fidelity=False, record=False), physics switched at runtime by apply_physics + env._real_settle_steps"
  seed: "actions default_rng(1000+slate); perturbations torch cpu Generator(2000+slate)"
  split: "n/a (no fitting)"
  runtime: "~3.6-6 min per state (11 conditions x 32 pushes), RTX 4070 Laptop, GPU shared with another Genesis job"
  runs: [RUN-0001-sensitivity]

budget:
  declared: "50 min wall clock"
  spent: "~50 min"
  outcome: stopped-early

design:
  varied:
    condition: [repeat, P_binned, P_oracle, ref_back, state_0.25mm, state_0.5mm, state_1.0mm, act_0.5mm, act_1.0mm, act_rot1deg]
  held_fixed: {states: "DS-0006 slates 100-111", actions: "32 per state, 20 mm, perpendicular blade", n_envs: 32, fidelity: full, reference: "P_train first call"}
  baselines: [repeat, ref_back]
  metric: "repeat_noise_ratio-style: |dv_cond - dv_ref| / between-action sd(dv_ref); plus cube-position diff, occ_for_scoring L1/mass, Spearman of dv across actions"

noise_floor: "repeat (same physics, second call): moved-cube diff median ~0.36 mm, dv diff median 0.02 sd, Spearman 0.994 -- the known history-dependent plate-yaw residual (INVARIANTS genesis-snapshot-restore-repeat-determinism)"

depends_on: [goal-mask-axis-convention-row-y-col-x, score-occupancy-subpixel-stable]                    # the broken tags genesis-snapshot-restore-repeat-determinism and
                                  # occupancy-dv-subpixel-stable are MEASURED here (repeat arm), not assumed;
                                  # this record bears on benchmark-physics-matches-training
establishes: []

result: "P_binned / P_oracle vs P_train: pushed-cube final diff median 0.78 / 0.76 mm (repeat 0.39, act 0.5 mm jitter 0.76, act 1.0 mm 1.42, state 0.5 mm 0.64), dv diff median 0.047 / 0.049 between-action sd (repeat 0.022, act 0.5 mm 0.044), Spearman 0.991 (repeat 0.995); 5 states x 32 pushes x 30 goals"
verdict: supported
downgrades: [imprecision, indirectness]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If the physics set mattered for the pushes our datasets contain, P_binned /
P_oracle outcomes would sit well outside the cloud spanned by the repeat
floor and by sub-pixel state / action jitter (which a 64x64, 2 mm/pixel
model cannot see or cannot resolve), and the ranking of actions per goal
would reorder. Both physics arms were run on identical states and actions in
the same process, so any difference beyond `repeat`/`ref_back` is the physics.

## What was actually run

Tree dirty at commit 3bae8cd7 (other agents' concurrent edits to skills, docs,
Baselines, env/, simple_mpc/adapters.py, transforms/functional.py); the numbers
depend on this record's uncommitted `code/` plus `simple_mpc/genesis_oracle.py`,
`simple_mpc/learned_mpc.py`, `Genesis/sandbox_manipulation_clean.py` (clean) and
`simple_mpc/adapters.py::occ_for_scoring` / `transforms/functional.py` (dirty).
See `runs/RUN-0001-sensitivity/RUN.md`. One GenesisOracleEnv; physics changed at
runtime. The switch was verified separately (`code/verify_switch.py`,
`results/verify_switch.json`): after `apply_physics` the cube geom friction,
floor geom friction and cube mass read back as set (0.7/0.5/5.6e-5 kg ->
0.3/0.3/1.25e-4 kg), and an extreme friction set (0.02/0.02) moves cubes by up
to 79 mm vs P_train (mean 2.9 mm over all cubes), versus 3.5 mm max / 0.09 mm
mean for a plain P_train repeat -- so the runtime switch takes effect. A 9x
density change alone (4000) changed outcomes no more than a repeat (0.08 mm
mean): at plate speed 0.125 m/s the push is quasi-static and mass barely matters.
Deviation: the run was cut by the time budget; 5 of 12 states (slates 100-104) are in
the numbers (run_probe's 1800 s timeout killed the job during slate 105) (every metric is recomputed from `artifacts/raw.pt` by
`sensitivity.py --analyze`; re-running the script resumes at slate 105 from the checkpoint).

## Numbers

5 states (slates 100-104) x 32 pushes; reference: 9.1 % of cubes move > 1 mm, max 19 mm.

| condition | input change (L1/mass) | moved-cube final diff mm med / p90 | displacement diff mm med / p90 | per-push max diff mm med | frac cubes >2 mm | outcome image L1/mass med / p90 | abs dv diff / between-action sd med / p90 | Spearman med / p10 | top-1 same | top-1 regret / sd mean | pushes with mean diff > repeat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| repeat | 0.000 | 0.39 / 1.27 | 0.39 / 1.27 | 0.53 | 0.004 | 0.014 / 0.042 | 0.022 / 0.109 | 0.995 / 0.988 | 0.93 | 0.003 | 0.00 |
| P_binned | 0.000 | 0.78 / 1.89 | 0.78 / 1.89 | 0.96 | 0.008 | 0.028 / 0.060 | 0.047 / 0.186 | 0.991 / 0.980 | 0.95 | 0.005 | 0.79 |
| P_oracle | 0.000 | 0.76 / 1.89 | 0.76 / 1.89 | 1.03 | 0.008 | 0.027 / 0.062 | 0.049 / 0.187 | 0.991 / 0.980 | 0.93 | 0.009 | 0.79 |
| ref_back | 0.000 | 0.42 / 1.36 | 0.42 / 1.36 | 0.56 | 0.004 | 0.015 / 0.044 | 0.023 / 0.113 | 0.995 / 0.988 | 0.90 | 0.005 | 0.52 |
| state_0.25mm | 0.138 | 0.51 / 1.59 | 0.54 / 1.69 | 0.87 | 0.007 | 0.145 / 0.185 | 0.033 / 0.143 | 0.992 / 0.984 | 0.89 | 0.008 | 1.00 |
| state_0.5mm | 0.224 | 0.64 / 1.67 | 0.61 / 1.62 | 1.31 | 0.006 | 0.226 / 0.264 | 0.041 / 0.157 | 0.991 / 0.981 | 0.90 | 0.006 | 1.00 |
| state_1.0mm | 0.491 | 1.06 / 2.63 | 1.04 / 2.59 | 3.30 | 0.134 | 0.487 / 0.577 | 0.072 / 0.234 | 0.986 / 0.973 | 0.82 | 0.030 | 1.00 |
| act_0.5mm | 0.000 | 0.76 / 1.95 | 0.76 / 1.95 | 0.95 | 0.008 | 0.026 / 0.063 | 0.044 / 0.153 | 0.991 / 0.979 | 0.80 | 0.023 | 0.77 |
| act_1.0mm | 0.000 | 1.42 / 2.96 | 1.42 / 2.96 | 1.76 | 0.029 | 0.048 / 0.098 | 0.091 / 0.280 | 0.980 / 0.963 | 0.70 | 0.044 | 0.89 |
| act_rot1deg | 0.000 | 0.47 / 1.77 | 0.47 / 1.77 | 0.63 | 0.006 | 0.017 / 0.054 | 0.027 / 0.141 | 0.992 / 0.982 | 0.92 | 0.004 | 0.67 |


Cube-diff columns are over cubes that moved > 1 mm in the reference (about 9 %
of cubes; the rest are untouched in every condition). "input change" is the
L1 change of the soft-scored start image (`occ_for_scoring`) relative to its
mass -- the perceptibility of a state perturbation (0 for physics / action arms).
dv = lyapunov(after) - lyapunov(before) on `occ_for_scoring` images, 30 goals,
normalised per (state, goal) by the sd of dv_ref over the 32 actions.

## What would change the verdict

Longer pushes (40-70 mm, where sliding after the blade and pile-to-pile
contact matter more), piled / multi-layer states (friction governs heap collapse), and multi-step chains, where 1 mm differences compound.
Each is one more run of this script with a different state / length set (~40 min).

## Threats

- imprecision: 5 states x 32 actions; per-state medians are shown in
  results/sensitivity.json and are consistent across states, but the top-1 match
  rates (0.7-0.95) are noisy.
- indirectness: only 20 mm perpendicular pushes aimed at a single cube on sparse
  single-layer scatter; P_binned datasets also contain 40-70 mm pushes and
  multi-step chains, where friction (post-release sliding, heap collapse) may
  matter more. The claim is scoped to 20 mm single pushes.
- state_1.0mm jitter can create cube-cube interpenetration that pops on settle
  (per-push max diff 3.3 mm, 10 % of cubes > 2 mm); that arm overstates the
  effect of a 1 mm imperceptible error.
- the settle cap differs (P_oracle 100 steps vs 3000); its effect is folded into
  P_oracle, which is indistinguishable from P_binned here.

## Unrelated findings

- `scripts/run_probe.py` launched twice with the same `--tag` truncates the
  first run's log and overwrites its `<tag>.json` / `COMMAND.txt` while the first
  process keeps writing (log gets a NUL-filled prefix). Seen here; a tag lock or
  per-run suffix would prevent it.
- A command backgrounded with `&` inside a Bash-tool call that is itself moved
  to the background was killed when the wrapper exited (first launch lost).

## Note added 2026-09-25 (orchestrator): what the "input image change" means
The input-change numbers are L1 image differences, and they do not measure
perceptibility. Recomputed on 40 DS-0006 states with the MODEL-INPUT raster (hard
occupancy `occ_from_particles`, 64 px = 2 mm/px, a 5 mm cube covers ~4.9 px):
- jitter 0.25 mm per axis flips ~17 boundary pixels per image (~0.8 per cube; L1/mass 0.17);
- 0.5 mm flips 29; 1 mm flips 59.
The hard rasteriser decides each pixel by a threshold on the cube footprint, so a
1/8-pixel shift flips edge pixels on most cubes (the aliasing that motivated soft truth
scoring, METRICS.md). The soft scorer changes similarly in L1 (0.12 at 0.25 mm) because a
sigma = 2 mm Gaussian shifted by 1/8 of its width changes every pixel slightly.
So sub-pixel shifts are "visible" as aliasing noise in the input. It is NOT true that
only < 0.25 mm is imperceptible. The perturbation sizes here should be read in mm
against the 2 mm pixel: 0.25-1 mm = 1/8-1/2 px.
