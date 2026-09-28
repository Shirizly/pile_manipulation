---
id: DS-0016
title: Narrow-domain CLEAN TEST, clean by construction -- same-state pools + chains, n20 single layer, exact 20 mm perpendicular, training physics, replaces DS-0009
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/test_pools_v2 (pools_0.pt), Genesis/data/narrow_l20_n20/test_chains_v2
producer: Genesis/chain_collection.py (--mode pools seed 2002; --mode chains seed 2001)
---

## What this replaces and why

Supersedes **DS-0009** (test_chains 46.2% illegal, test_pools 56.1% illegal at touchdown --
ISS-010). Same shape and sizes as DS-0009 (test_pools: 32 start states x 64 pushes; test_chains:
4 chunks x 32 envs x 8 pushes = 1,024 transitions), fresh seeds, collected with the ISS-010-fix
sampler (see DS-0015's DATASET.md for the fix + `start_gap_range` description -- identical
recipe, same physics, same push length, only the seeds and output paths differ). DS-0009's own
DATASET.md is marked superseded for clean comparisons; its payload is UNCHANGED.

## Verification (full-scale audit)

| check | test_chains (1024 rows) | test_pools (2048 rows) |
|---|---|---|
| illegal touchdowns (exact SAT, 0mm) | **0/1024 (0.0%)** | **0/2048 (0.0%)** |
| `gap_out_of_window` (last-resort accept, flagged) | **0/1024 (0.0%)** | **0/2048 (0.0%)** |
| `valid==False` | 38/1024 (3.71%) | 0/2048 (0.0%; pools mode only ever saves already-`valid` draws by construction) |
| null (max per-cube xy disp < 1mm) | 29/1024 (2.83%) -- all 29 are a SUBSET of the 38 `valid==False` rows | 0/2048 (0.0%) |
| `PileSweepData(exclude_flagged=True)` (test_chains; `_data`+config-yaml schema, chains only -- test_pools' `pools_*.pt` files are read via the unified-benchmark-shape readers, not this class) | drops 38, keeps 986/1024 -- agrees exactly | n/a |

Physics: TRAINING_PHYSICS (0.7/0.5/450, settle 3000). Sampler: `{"pile_aware": true,
"min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}` (resolved
window 5-15 mm). Starts: `--starts mixed`. Never used for training.

Commands:
- chains: `python -u Genesis/chain_collection.py --mode chains --out
  Genesis/data/narrow_l20_n20/test_chains_v2 --n-chunks 4 --steps 8 --n-envs 32 --starts mixed
  --clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
  "min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}' --seed 2001`
- pools: `python -u Genesis/chain_collection.py --mode pools --out
  Genesis/data/narrow_l20_n20/test_pools_v2 --n-chunks 1 --pool 64 --n-envs 32 --starts mixed
  --clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
  "min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}' --seed 2002`

Files: test_pools -- unified benchmark shape (`states, states_, p_starts, p_stops, angles,
pool_idx, valid, gap_out_of_window, start_kind`). test_chains -- training-shaped keys +
`chain_env, chain_step, valid, single_layer, start_kind, gap_out_of_window` +
`_{k}_config.yaml` (carries `start_gap_resolved: [0.005, 0.015]`). Both `manifest.json`:
`complete: true`.

## Clean copy + archive (2026-09-28)

`experiments/EXP-0059-*/code/split_clean_archive.py` (CPU-only, originals untouched).
- **test_pools**: 0/2048 `valid==False` -- already 100% clean, no split needed.
- **test_chains** (rollout set -- a sequence with any bad step is removed WHOLE): 1,024 rows /
  128 eight-step sequences -> **16/128 whole sequences removed (12.5%), 128 rows -> 896 clean
  rows (112 sequences)**. Contrast with DS-0009's own pre-fix `test_chains` split, where 127/128
  sequences were lost (99.2%) at the old ~55% per-row bad rate -- this fixed set's ~3.7% per-row
  `valid==False` rate compounds far less across an 8-step chain. Clean:
  `Genesis/data/narrow_l20_n20/test_chains_v2_clean/`. Archive:
  `Genesis/data/narrow_l20_n20/test_chains_v2/archive_removed/_{k}_data_removed.pt` (same schema
  + `reason` in `{invalid_redraw_exhausted, sequence_contains_bad_step}` + `source_file`/
  `source_row`).
