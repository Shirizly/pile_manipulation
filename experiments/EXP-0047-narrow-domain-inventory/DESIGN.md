# EXP-0047 -- inventory of existing data for the narrow domain (overnight stage B)

Written retrospectively 2026-09-25 from EXPERIMENT.md, code/inventory.py and code/breakdown.py
(docstrings/constants), results/inventory.{json,log} and the TODO.md overnight plan (stage B), not
before the run. The pre-run plan is quoted where TODO.md records it.

## Why
The overnight plan narrows new work to n = 20, single-layer (scatter and clumps), perpendicular,
exactly 20 mm pushes at training physics. Before collecting, find what existing corpora already
provide. Pre-run plan (TODO.md stage B, "CPU agent, 30 min"): "data inventory -- per corpus:
physics, n, layers (single-layer incl. clumps), push length, perpendicularity (+-2 deg), chains;
counts for the narrow domain."

## Design
- Corpora (whole-corpus census as on disk 2026-09-25): Genesis/data/overnight_randlen, Sean,
  slates, slates_multistep, slates_binned (DS-0001, DS-0005, DS-0006), cube_spectrum, granularity,
  corl, dinowm_test, DS-0007 (noted as a derived subset of Sean, not additional data).
- Per row, measured from tensors (never config claims): push length L; exact 20 mm = |L - 20| <
  0.1 mm; band 18-22 mm; perpendicular deviation (< 2 deg and < 0.1 deg); single layer = max cube
  centre z < 15 mm; clump fraction = cubes with an xy neighbour within 6 mm; clump state = single
  layer and clump fraction >= 0.5; chain link = next row's start equals this row's end (< 1 mm).
- Physics table per corpus from recorded config keys (friction, density, settle steps, ...).
- Per-spawn-group breakdown (breakdown.py) for overnight_randlen and Sean n20.
- Output: counts, not a metric; no baselines apply. Checkpointed per corpus (resumable).
- Budget: plan 30 min CPU; spent not recorded.

## Predictions
Exploratory: no pre-registered prediction.

## Deviations
- The single-layer (z < 15 mm) and clump (>= 50% with a 6 mm neighbour) thresholds were chosen in
  this experiment; the plan did not fix them.
- No run log under runs/ and no command-ledger entry; the invocation is not recorded beyond
  results/inventory.log.
