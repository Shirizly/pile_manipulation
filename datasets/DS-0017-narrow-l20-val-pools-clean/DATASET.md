---
id: DS-0017
title: Narrow-domain VALIDATION pools, clean by construction -- n20 single layer, exact 20 mm perpendicular, training physics, replaces DS-0011
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/val_pools_v2 (pools_0.pt)
producer: Genesis/chain_collection.py --mode pools --seed 2004
---

## What this replaces and why

Supersedes **DS-0011** (val_pools, 47.9% illegal at touchdown -- ISS-010). Same shape as DS-0011
(32 start states x 64 pushes, unified benchmark shape), fresh seed, ISS-010-fix sampler (see
DS-0015's DATASET.md for the fix + `start_gap_range` description). DS-0011's own DATASET.md is
marked superseded for clean comparisons; its payload is UNCHANGED.

Command: `python -u Genesis/chain_collection.py --mode pools --out
Genesis/data/narrow_l20_n20/val_pools_v2 --n-chunks 1 --pool 64 --n-envs 32 --starts mixed
--clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
"min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}' --seed 2004`.
Physics: TRAINING_PHYSICS (0.7/0.5/450, settle 3000). Resolved gap window: 5-15 mm.

**COMPLETE (2026-09-28).** Full-scale audit: **0/2048 illegal, 0/2048 null, 0/2048
`gap_out_of_window`, 0/2048 `valid==False`** -- this set is already 100% clean, no
`archive_removed/` or `_clean` split needed (unlike DS-0015/DS-0016, which each had a handful
of `valid==False` rows to archive). Same recipe/seed family as DS-0015/DS-0016 (TEST chains
seed 2001, TEST pools seed 2002, TRAIN seed 2003, this set seed 2004 -- all fresh and mutually
disjoint).
