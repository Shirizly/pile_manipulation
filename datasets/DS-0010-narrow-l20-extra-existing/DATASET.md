---
id: DS-0010
title: Existing matched-physics rows for the narrow domain (18-22 mm, n20 single layer) -- TRAINING ONLY
status: superseded  # 2026-09-28, ISS-010 illegal touchdowns; replaced by DS-0015
date: 2026-09-25
path: Genesis/data/narrow_l20_n20/extra_18_22
producer: datasets/DS-0010-narrow-l20-extra-existing/extract.py
---
- 5,777 rows from 145,920 scanned (285 files).
- Sources: the overnight_randlen TRAIN split (never its test split) and Sean n20.
- Filter:
  - push length 18-22 mm;
  - perpendicular within 0.1 deg;
  - every cube below z = 15 mm before AND after the push.
- Physics as training (0.7 / 0.5 / 450; settle 500 for scatter spawns, 3000 otherwise --
  EXP-0047). Per-row provenance: source_file, source_rows.
- User rule 2026-09-25: 18-22 mm rows may train, never test.

**Legality audit (2026-09-28, ISS-010):** unlike DS-0008/9/11/12/13 (`pile_aware`), DS-0010 is
essentially clean by the EXACT SAT touchdown-overlap test: only 1/5,777 rows (0.02%) have the
blade footprint overlapping a cube at `p_start`. It DOES sit close to the boundary, though:
33.4% of rows are within a 1mm margin of touching. Consistent with an older, likely
`placement_aware` (not `pile_aware`) sampler whose default clearance is exactly 0 (see
`Genesis/placement_sampling.py::free_placements`'s `clearance=0.0` default) -- not independently
re-verified per source file (`overnight_randlen_train/{mixed,piled}_n20`, `Sean/n20`). Per-row
flags: `Genesis/data/narrow_l20_n20/extra_18_22/_{k}_data_legality.pt`. Full writeup:
`experiments/OPEN_ISSUES.md` ISS-010.
