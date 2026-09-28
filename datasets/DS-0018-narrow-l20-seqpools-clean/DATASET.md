---
id: DS-0018
title: Narrow-domain MULTI-STEP CANDIDATE-SEQUENCE pools, clean by construction -- n20 single layer, exact 20 mm perpendicular, training physics, replaces DS-0013
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/seqpools_v2 (_{k}_data.pt, k = 0..31)
producer: Genesis/chain_collection.py --mode seqpools --seed 2005
---

## What this replaces and why

Supersedes **DS-0013** (seqpools_dsB, 51.7% illegal at touchdown -- ISS-010). Same shape and
size as DS-0013 (32 pools x 64 candidate 3-push sequences from a shared start state = 6,144
rows), fresh seed, collected with the ISS-010-fix sampler (see DS-0015's DATASET.md for the fix
+ `start_gap_range` description -- identical recipe, same physics, same push length; only the
seed and output path differ). DS-0013's own DATASET.md is marked superseded for clean
comparisons; its payload is UNCHANGED.

Command: `python -u Genesis/chain_collection.py --mode seqpools --out
Genesis/data/narrow_l20_n20/seqpools_v2 --n-chunks 32 --pool 64 --steps 3 --n-envs 32 --starts
mixed --clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
"min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}' --seed 2005`.
Physics: TRAINING_PHYSICS (0.7/0.5/450, settle 3000). Resolved gap window: 5-15 mm. No smoke
rerun before launch -- nothing in the sampler changed since the already-verified `train_v2`/
`test_*_v2`/`val_pools_v2` runs.

## Verification (full-scale audit) + clean copy/archive

6,144 rows (32 pools x 64 sequences x 3 pushes):

| check | result |
|---|---|
| illegal touchdowns (exact SAT, 0mm) | **0/6144 (0.0%)** |
| `gap_out_of_window` (last-resort accept, flagged) | **0/6144 (0.0%)** |
| `valid==False` | 6/6144 (0.098%) |
| null (max per-cube xy disp < 1mm) | 3/6144 (0.049%) -- all 3 are a SUBSET of the 6 `valid==False` rows |

**Clean copy + archive (`experiments/EXP-0059-*/code/split_clean_archive.py`, CPU-only,
original payload untouched)**: a rollout set, so a (pool/chunk, chain_env) SEQUENCE with any bad
step is removed WHOLE -- `chain_env` is the sequence id (0-63) WITHIN one chunk's pool (each
`_{k}_data.pt` = one pool), matching `chain_collection.py`'s own seqpools schema (it reuses
chains' `chain_env`/`chain_step` field names plus a `pool_idx` column, per its own docstring).
2,048 sequences total (32 pools x 64) -> **6 whole sequences removed (0.29%), 18 rows -> 6,126
clean rows / 2,042 clean sequences (99.7%)** -- far higher survival than DS-0013's own would-be
split (not computed -- DS-0013 was superseded rather than split, since ISS-010 makes any
per-sequence survival estimate there moot at its ~52%-per-row defect rate). Clean:
`Genesis/data/narrow_l20_n20/seqpools_v2_clean/`. Archive:
`Genesis/data/narrow_l20_n20/seqpools_v2/archive_removed/_{k}_data_removed.pt` (same schema +
`reason` in `{invalid_redraw_exhausted, sequence_contains_bad_step}` + `source_file`/
`source_row`).

Files: `_{k}_data.pt` (chains-mode field names: `states, states_, p_starts, p_stops, angles,
chain_env, chain_step, pool_idx, valid, single_layer, start_kind, gap_out_of_window`) +
`_{k}_config.yaml` (carries `start_gap_resolved: [0.005, 0.015]` in its `chain_collection`
block, `mode: "seqpools"`). `manifest.json`: `complete: true`, 32/32 chunks.
