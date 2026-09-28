---
id: DS-0011
title: Narrow-domain VALIDATION pools ("DS-A") -- n20 single layer, exact 20 mm perpendicular, training physics
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/val_pools (pools_0.pt)
producer: Genesis/chain_collection.py --mode pools --seed 201 (new seed, disjoint from DS-0008/9/10's 1/101/102)
---
- 32 start states (16 scatter + 16 clump) x 64 pushes, all from the same state -- identical shape
  to DS-0009's `test_pools`, same sampler/physics, DIFFERENT seed. Unified benchmark shape:
  states, states_, p_starts, p_stops, angles, pool_idx, valid.
- Purpose (docs/experimental_design/retrieval_based_modeling.md 2.1 "DS-A val pools"): `slateN`
  for tuning retrieval-model variants WITHOUT touching DS-0009 (the frozen eval set).
- Launched 2026-09-28 by EXP-0059's R0 subagent per the designer's plan; `--n-chunks 1` (single
  atomic chunk covering all 32 states, ~10 min estimated, matches DS-0009 test_pools' own
  timing). Command: `Genesis/chain_collection.py --mode pools --out
  Genesis/data/narrow_l20_n20/val_pools --n-chunks 1 --pool 64 --starts mixed --clump-fn
  Genesis.clump_states:clump_starts --sampler '{"pile_aware": true, "min_swath_particles": 3,
  "push_length": 0.02}' --seed 201`. Physics: TRAINING_PHYSICS (0.7/0.5/450, settle 3000),
  same as DS-0008/9/10.
- Status at record time: collection running in the background (PID logged in
  `experiments/EXP-0059-retrieval-transition-model/LOG.md`); this DATASET.md was written before
  the job's own completion so an ID exists as soon as collection started, per experiment-log's
  "IDs outlive their payload" -- check `Genesis/data/narrow_l20_n20/val_pools/manifest.json`
  (`complete: true/false`) for current status before using it.

**KNOWN DEFECT (2026-09-28, ISS-010):** same `_pile_aware_stops` clamp defect as DS-0008/9.
Measured: 47.9% of rows illegal at touchdown (0mm SAT overlap; 73.6% at 1mm margin). Per-row
flags: `Genesis/data/narrow_l20_n20/val_pools/pools_0_legality.pt`. Full writeup:
`experiments/OPEN_ISSUES.md` ISS-010.

**SUPERSEDED for clean comparisons (2026-09-28):** see **DS-0017**
(`datasets/DS-0017-narrow-l20-val-pools-clean/`) -- fresh val_pools, same recipe/size, new seed,
ISS-010-fix sampler, complete, 0/2048 illegal / null / gap_out_of_window (100% clean, no split
needed). This set's payload is UNCHANGED (still valid for before/after-the-fix comparisons); use
DS-0017 for any new validation work.

**Clean copy + archive (2026-09-28, `experiments/EXP-0059-*/code/split_clean_archive.py`,
CPU-only, originals untouched):** pool set -- bad candidates dropped individually. 2,048 rows ->
**1,136 removed (55.5%: illegal 981 + null 186 - both 30, rounding) -> 912 clean rows**. Clean:
`Genesis/data/narrow_l20_n20/val_pools_clean/pools_0.pt`. Archive:
`Genesis/data/narrow_l20_n20/val_pools/archive_removed/pools_0_removed.pt` (same schema +
`reason` + `source_file`/`source_row`).
