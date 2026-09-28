---
id: EXP-0047
title: >
  Narrow-domain data inventory (n20, single-layer, 20 mm, perpendicular, chains):
  existing corpora hold ~6-8k matched-physics n20 single-layer 18-22 mm
  perpendicular transitions (overnight_randlen + Sean) but under 300 at exactly
  20 mm and almost no chained narrow pairs; every exact-20 mm corpus is a pile
tier: T0
mode: exploratory
date: 2026-09-25
claim: >
  No existing corpus provides clean exact-20 mm perpendicular pushes on n20
  single-layer states (scatter or clump) with chains; the only in-domain rows
  are the 18-22 mm slice of the randomly-lengthed corpora, so the narrow-domain
  train and test sets must be collected.
provenance:
  commit: 3bae8cd7
  dirty: true
  script: code/inventory.py (+ code/breakdown.py for per-spawn groups)
  data: ["Genesis/data/overnight_randlen", "Genesis/data/Sean", "Genesis/data/slates", "Genesis/data/slates_multistep", "Genesis/data/slates_binned (DS-0001/5/6)", "Genesis/data/cube_spectrum", "Genesis/data/granularity", "Genesis/data/corl", "Genesis/data/dinowm_test", "DS-0007"]
  code_path: "torch.load raw *_data.pt / step*.pt -> per-row L, perp deviation, max z, clump fraction; chain links by state continuity (1 mm)"
  seed: "none"
  split: "not applicable -- whole-corpus census"
  data_commit: "not applicable (data gitignored; counted as on disk 2026-09-25)"
result: >
  n20 single-layer 18-22 mm perpendicular (<2 deg) rows: overnight_randlen 3738
  (191 exact-20), Sean 2424 (103), DS-0006 815 (36), DS-0001 855 (40), DS-0005
  815 (36); slates/slates_multistep/cube_spectrum exact-20 corpora are piles
  (0-4 single-layer rows). Single-layer clumps (clump frac >= 0.5) exist mainly in
  Sean piled/inbetween (9945 rows; 215 in 18-22 mm) and overnight mixed (1537; 77).
  Chained pairs with both pushes in-domain: 147 (overnight) + 325 (Sean) at
  18-22 mm, 1 + 1 at exact 20. See results/inventory.json.
verdict: supported
downgrades: [indirectness]   # single-layer (z<15 mm) and clump (>=50% cubes with a 6 mm neighbour) are thresholds chosen here
grade: moderate
---

Dirty tree: many unrelated working-tree edits by parallel agents (see git status);
this record only reads data and adds its own code/ and results/.
