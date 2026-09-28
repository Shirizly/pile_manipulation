---
id: DS-0013
title: Narrow-domain MULTI-STEP CANDIDATE-SEQUENCE pools ("DS-B") -- n20 single layer, exact 20 mm perpendicular, training physics
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/seqpools_dsB (_{k}_data.pt, k = 0..31)
producer: Genesis/chain_collection.py --mode seqpools --seed 401
---
- **COMPLETE**, produced by the coder (not this record's own work) while the experimenter's R0/R2
  rungs were running: 32 chunks, each one POOL of 64 candidate 3-push SEQUENCES from a single
  shared start state (16 chunks scatter-start, 16 clump-start, alternating) -- 32 pools x 64
  sequences x 3 pushes/sequence = 6,144 rows total (verified by direct load, 2026-09-28: 32
  files x 192 rows/file). Columns: `states, states_, p_starts, p_stops, angles, chain_env,
  chain_step, pool_idx, valid, single_layer, start_kind` -- `pool_idx` is per-CHUNK (always 0
  within a chunk; the chunk index itself IS the pool id across the corpus), `chain_env` (0-63)
  is the candidate SEQUENCE id within its pool, `chain_step` (0-2) is the push index within that
  sequence. Same physics/sampler as DS-0008/9/10/11/12 (TRAINING_PHYSICS 0.7/0.5/450 settle
  3000, `pile_aware` sampler, `min_swath_particles=3`, `push_length=0.02`,
  `Genesis.clump_states:clump_starts`), seed 401 (disjoint from 1/101/102/201/301).
- Command (from `manifest.json.argv`): `Genesis/chain_collection.py --mode seqpools --out
  Genesis/data/narrow_l20_n20/seqpools_dsB --n-chunks 32 --pool 64 --steps 3 --n-envs 32
  --starts mixed --clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
  "min_swath_particles": 3, "push_length": 0.02}' --seed 401`. `manifest.json`: `complete: true`,
  every chunk `valid`/`single_layer_after` == 192 (no rows dropped by either check), wall time
  per chunk ranged ~88-560s (mean ~175s), total ~5,600s (~93 min) collection wall-clock.
- Purpose (docs/experimental_design/retrieval_based_modeling.md section 2.1, "DS-B multi-step
  test pools"): the genuine multi-step RANKING test set EXP-0059's R0 rung explicitly found
  missing (LOG.md's 2026-09-28 mid-task entry: no existing corpus supported multiple candidate
  action SEQUENCES from one shared start scored at horizon > 1 under narrow-domain physics).
  This closes that gap -- each pool's 64 sequences let a multi-step eval rank candidates by
  predicted TERMINAL outcome (after 3 pushes) against the pool's own true terminal states, and
  score per-step (after push 1/2/3) rollout accuracy against each sequence's own true chain.
- Consumers: EXP-0059 task 2 (multi-step headline: terminal `slateN`/`slateN_tough`, per-step
  `slateN`, rollout accuracy per step) -- see `experiments/EXP-0059-retrieval-transition-model/
  results/multistep_eval.json` once run.

**KNOWN DEFECT (2026-09-28, ISS-010):** same `_pile_aware_stops` clamp defect as DS-0008.
Measured: 51.7% of rows illegal at touchdown (0mm SAT overlap; 73.7% at 1mm margin); worse for
clump (0.611) than scatter (0.422). Per-row flags:
`Genesis/data/narrow_l20_n20/seqpools_dsB/_{k}_data_legality.pt`. Full writeup:
`experiments/OPEN_ISSUES.md` ISS-010.
