# EXP-0049 -- candidate samplers for 20 mm perpendicular pushes on n20 single-layer states (overnight stage D)

Written retrospectively 2026-09-25 from EXPERIMENT.md, code/sampler_study.py and code/analyse.py
(docstrings/constants), runs/RUN-0001/RUN.md, the command ledgers and the TODO.md overnight plan
(stage D), not before the run. The pre-run plan is quoted where TODO.md records it.

## Why
New narrow-domain data (stage F) and planners need a candidate generator for exactly 20 mm
perpendicular pushes; is the pile-aware sampler too limiting? Pre-run plan (TODO.md stage D, "GPU
agent, 60 min"): "sampler study (pile-aware strict / relaxed, placement-aware perpendicular 20 mm,
uniform) on scatter + single-layer-clump states: null-push fraction, particles moved, best-of-N
true value per goal."

## Design
- States: 10 DS-0006 step-0 scatter states (slates 120-129) + 10 constructed single-layer clump
  states (`make_clump_states(sim, 10, seed=0)`, lattice patches, 1-3 clusters, pitch 5.5 mm,
  settled under TRAINING_PHYSICS; artifacts/clump_states.pt). n = 20 cubes.
- Samplers (SandboxManipulation.generate_action_samples, push_length 20 mm): S1 pile_aware
  min_swath 3; S2 pile_aware min_swath 1 + pile_clearance 2.5 mm; S3 placement_aware +
  perpendicular; S4 blind perpendicular (uniform baseline). Post hoc: S1/S2 `_fulllen` (drop pushes
  the sampler shortened) and S3+S1 mixtures (resampled).
- Every candidate converted to a 4-D [sx,sy,ex,ey] action, start and direction kept, length forced
  to exactly 20 mm (yaw perpendicular by construction); 96 per (state, sampler); all simulated with
  rollout_candidates (full fidelity, TRAINING_PHYSICS, 32 envs, 2 shard processes).
- Metrics: lyapunov improvement v0 - v1 on occ_for_scoring (METRICS.md ground-truth scoring),
  mean over 12 goals (letters O T S L C X H Z I, quadrants 0/1/3); best-of-N (N = 8-96, 400
  resamples); null-push fraction; cubes moved; displacement; outcome L1.
- Baselines: S4 blind uniform; improvement 0 = do nothing. Paired across states (sem).
- Budget: declared 70 min wall clock including code (plan: 60 min); spent ~60 min.

## Predictions
Exploratory: no pre-registered prediction.

## Deviations
- Stopped by the time budget (both shards killed at ~02:25): 8/10 scatter and 5/10 clump states
  have all four samplers; incomplete states are dropped from the analysis.
- Planned "best-of-N true value per goal" was reported as the mean over 12 goals; no per-goal
  breakdown (listed under "What would change the verdict").
- Pile-aware shortened wall-limited pushes (125/768 scatter, 145/480 clump S1 pushes; many to
  ~0 mm); these were force-extended to 20 mm toward the wall; the `_fulllen` rows bound the effect.
- Per-unit seed uses python hash((state, sampler)): not reproducible across processes.
- The pilot failed twice before succeeding (ledger); its scatter0 files were reused.
- No simulator repeat was run (each candidate simulated once).
