---
id: DS-0009
title: Narrow-domain CLEAN TEST -- same-state pools + chains, n20 single layer, exact 20 mm perpendicular, training physics
status: superseded  # 2026-09-28, ISS-010 illegal touchdowns; replaced by DS-0016
date: 2026-09-25
path: Genesis/data/narrow_l20_n20/test_pools (pools_0.pt), Genesis/data/narrow_l20_n20/test_chains
producer: Genesis/chain_collection.py (--mode pools seed 102; --mode chains seed 101)
---
- test_pools: 32 start states (16 scatter + 16 clump) x 64 pushes, all from the same state.
  Unified benchmark shape: states, states_, p_starts, p_stops, angles, pool_idx, valid.
- test_chains: 4 chunks x 32 envs x 8 pushes (1,024 transitions).
- Sampler, physics and validity checks as DS-0008; different seeds; never used for training.

**KNOWN DEFECT (2026-09-28, ISS-010):** same `_pile_aware_stops` clamp defect as DS-0008.
Measured: test_chains 46.2% illegal at touchdown (0mm SAT overlap; 69.2% at 1mm margin),
test_pools WORSE at 56.1% (76.7% at 1mm) -- pools start fresh from a just-spawned, denser state
with no prior pushes to de-clump it, so illegal rows CLUSTER there relative to chains. Per-row
flags: `Genesis/data/narrow_l20_n20/test_chains/_{k}_data_legality.pt`,
`test_pools/pools_0_legality.pt`. Full writeup: `experiments/OPEN_ISSUES.md` ISS-010.

**SUPERSEDED for clean comparisons (2026-09-28):** see **DS-0016**
(`datasets/DS-0016-narrow-l20-test-clean/`) -- fresh test_chains+test_pools, same recipe/sizes,
new seeds, ISS-010-fix sampler: 0/1024 and 0/2048 illegal respectively, 0 gap_out_of_window.
This set's payload is UNCHANGED (still valid for before/after-the-fix comparisons); use DS-0016
for any new test/comparison work.

**Clean copy + archive (2026-09-28, `experiments/EXP-0059-*/code/split_clean_archive.py`,
CPU-only, originals untouched):**
- **test_pools** (a pool set -- drop bad CANDIDATES individually): 2,048 rows -> **1,279 removed
  (62.5%: illegal 1,149 + null 163 - both 33) -> 769 clean rows**. Clean:
  `Genesis/data/narrow_l20_n20/test_pools_clean/pools_0.pt`. Archive:
  `Genesis/data/narrow_l20_n20/test_pools/archive_removed/pools_0_removed.pt`.
- **test_chains** (a rollout set -- a (chunk, chain_env) SEQUENCE with any bad step is removed
  WHOLE): 1,024 rows / 128 eight-step sequences -> **127/128 whole sequences removed (99.2%),
  1,016 rows -> only 8 clean rows (1 sequence) survive**. At this dataset's ~55% per-row bad
  rate, an 8-step chain surviving intact is rare by construction (`P(all 8 steps good) ~=
  0.45^8 ~= 0.17%`, close to the measured 1/128) -- **this set's clean copy is not usable as a
  rollout test set**; use DS-0016's `test_chains_v2_clean` (87.5% of sequences survive) instead.
  Clean: `Genesis/data/narrow_l20_n20/test_chains_clean/`. Archive:
  `Genesis/data/narrow_l20_n20/test_chains/archive_removed/_{k}_data_removed.pt` (removed rows
  whose OWN step was bad get `reason` in `{illegal, null, illegal_and_null}`; rows removed only
  because a SIBLING step in their sequence was bad get `reason=sequence_contains_bad_step`).
