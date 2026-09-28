---
id: DS-0008
title: Narrow-domain TRAIN chains -- n20 single layer, exact 20 mm perpendicular pushes, training physics
status: superseded  # 2026-09-28, ISS-010 illegal touchdowns; replaced by DS-0015
date: 2026-09-25
path: Genesis/data/narrow_l20_n20/train
producer: Genesis/chain_collection.py --mode chains (run: datasets/DS-0008-narrow-l20-n20-train/runs/ds0008_train.*)
---
- 24 chunks x 32 envs x 8 chained pushes (6,144 transitions), seed 1.
- Starts per chunk: 16 scatter (state_library drop spawns) + 16 single-layer clumps
  (Genesis/clump_states.py).
- Actions: pile-aware (min_swath 3) with push_length 0.02 (EXP-0049's best sampler).
  Wall-shortened draws are REDRAWN, never extended; `valid` flags exact 20 mm +-0.1 mm and
  perpendicular within 0.1 deg.
- Physics TRAINING_PHYSICS (0.7 / 0.5 / 450, settle 3000).
- Files `_{k}_data.pt` with the training keys plus chain_env, chain_step, valid,
  single_layer, start_kind, and `_{k}_config.yaml` (overnight_randlen template +
  chain_collection block). manifest.json has per-chunk counts.
- Readable by PileSweepData (configs/dataset/genesis_narrow_l20_train*.yaml).

**KNOWN DEFECT (2026-09-28, ISS-010):** the `pile_aware` sampler's `_pile_aware_stops` clamp
(`Genesis/sandbox_manipulation_clean.py`) places the tool ON a cube at touchdown far more often
than intended -- measured 44.3% of valid rows (0.689 at a 1mm margin) have the blade footprint
overlapping a cube's rotated-square footprint at `p_start` (SAT test, see
`experiments/EXP-0059-retrieval-transition-model/code/audit_tool_placement.py`). Worse for clump
starts (0.530) than scatter (0.356); falls slightly across chain steps (0.490 -> 0.408). Per-row
flags: `Genesis/data/narrow_l20_n20/train/_{k}_data_legality.pt` (`illegal_0mm`,
`illegal_1mm_margin`; originals untouched). Full writeup: `experiments/OPEN_ISSUES.md` ISS-010.

**SUPERSEDED for clean comparisons (2026-09-28):** the sampler is now fixed (ISS-010 closed --
`Genesis/action_sampling.py::pile_aware_action_batch`, exact-SAT redraw + `start_gap_range`).
Rather than patch this set, the user chose fresh collection at the same recipe/size instead of
DS-0008+DS-0010 combined: see **DS-0015** (`datasets/DS-0015-narrow-l20-train-clean/`),
0/12032 illegal, 0/12032 gap_out_of_window. This set's payload is UNCHANGED and remains valid
for anything that predates the fix, or for before/after comparisons of the fix itself -- use
DS-0015 for any new training/comparison work.

**Clean copy + archive (2026-09-28, `experiments/EXP-0059-*/code/split_clean_archive.py`,
CPU-only, originals untouched):** independent-transition set, so bad ROWS (illegal OR null,
union) were dropped individually, chain metadata kept for provenance. 6,144 rows -> **3,343
removed (54.4%: illegal 2,722 + null 725 - both 104) -> 2,801 clean rows**. Clean copy:
`Genesis/data/narrow_l20_n20/train_clean/` (same per-chunk file layout + config yaml).
Archive: `Genesis/data/narrow_l20_n20/train/archive_removed/_{k}_data_removed.pt` (same schema
+ `reason` in `{illegal, null, illegal_and_null}` + `source_file`/`source_row`).
