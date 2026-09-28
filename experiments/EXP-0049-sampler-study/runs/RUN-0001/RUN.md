# RUN-0001: 96 simulated candidates per (state, sampler), four samplers, scatter + clump states

- Pilot: `scripts/run_probe.py --tag exp0049_pilot -- code/sampler_study.py --pilot` (scatter0, all 4 samplers;
  those files are reused, not re-simulated). Log: `runs/exp0049_pilot.log` at the repo root.
- Main: two processes in parallel (32 envs each, GPU otherwise idle at launch),
  `scripts/run_probe.py --tag exp0049_sim_shard{0,1} --threads 4 --timeout 3000 -- code/sampler_study.py --shard {0,1}/2`.
  Shard 0 = scatter states, shard 1 = clump states. Started 2026-09-25 01:46 CEST, **stopped by PID at
  ~02:24 (time budget)**, so only the first states of each type are complete (see EXPERIMENT.md Numbers
  for n). Logs: `runs/exp0049_sim_shard{0,1}.log`. Commit 3bae8cd7, dirty tree.
- Clump states: `code/sampler_study.py::make_clump_states(sim, 10, seed=0)`, one settle
  (`update_material_state`) under TRAINING_PHYSICS; 32 proposed, 0 rejected (z / tray);
  saved `artifacts/clump_states.pt` (`states (10,20,7)` [x,y,z,qw,qx,qy,qz], stats, physics, provenance).
- Per unit: `artifacts/sim/<state>_<sampler>.pt` (raw sampler starts/stops/angles, raw length and
  perpendicularity error, the executed 4-D actions, final particle positions, timings), written atomically.
- Analysis (CPU): `python experiments/EXP-0049-sampler-study/code/analyse.py` ->
  `results/sampler_study.json` (aggregate per type x sampler + per-state rows).
