# DS-0007 — same-state pools extracted from Sean's corpus (test set, many states)

**Status:** active
**Payload:** `datasets/DS-0007-sean-same-state-pools/data/<spawn>_n<N>.pt` + `manifest.json` (gitignored, 134 MB)
**Builder:** `build.py` (seconds; `runs/ds0007_build.*`)

## What this is
Groups of rows in `Genesis/data/Sean/**/_k_data.pt` that start from the IDENTICAL
state (positions equal to 0.1 mm; verified max difference 0), kept when a group has
>= 6 pushes. ~7,000 pools, almost all of 8 distinct pushes. No current model was
trained on Sean, so the whole set is held out for now (split it before training on
Sean -- TODO G4a).

| shard | pools | rows | notes |
|---|---|---|---|
| scattered_n20 | 801 | 6910 | pools of 8-16 |
| scattered_n50 | 800 | 6529 | |
| scattered_n100 | 797 | 6360 | >50 objects: existing data only, never re-simulate |
| inbetween_n20 / n50 / n100 | 799 / 788 / 656 | 6391 / 6128 / 5015 | "inbetween" spawn, pilefrac 0.5, 4 layers |
| piled_n20 / n50 | 800 / 761 | 6418 / 5758 | piles deprioritised (artificial here) |

## Physics
Particle friction 0.7, box friction 0.5, density 450, safety_margin 0.005 -- the same
as overnight_randlen (the models' training corpus); tray 128 mm; 5 mm cubes.

## Caveats
- Push lengths span ~0-70 mm (not binned); 9% of pushes move no particle by > 1 mm.
- Pools are small (K ~ 8): per-pool slateN is noisy; power comes from the number of
  states (EXP-0026, EXP-0029). Use goal-averaged, state-resampled statistics.
- Rows are chained transitions in the source files; pools from the same file are
  different (later) states of one episode -- group by `file_idx` when resampling if
  within-episode correlation matters.

## Format (the unified benchmark row format, see the data-collection skill)
Flat rows: `states (R,n,7)`, `states_ (R,n,7)`, `p_starts (R,3)`, `p_stops (R,3)`,
`angles (R,)`, `pool_idx (R,)` (= state id), `file_idx (R,)`, plus `files` (source list).

## Update 2026-10-06 (user decision): the Sean corpus is now the CANONICAL wider-domain TRAINING dataset
This DATASET.md describes the same-state-pool TEST extraction from Sean's files; as of 2026-10-06 the same source corpus
(`Genesis/data/Sean/**/_*_data.pt`) is also the canonical training/validation corpus for wide-domain work (EXP-0074). Splits are
FILE-level (per shard: 10 % test / 5 % val / rest train; `experiments/EXP-0074-wide-domain-zoom-nfd/code/sean_data.py::split_files`),
so DS-0007 pools built from all files overlap the training files -- for any model trained on Sean, evaluate on the test-split files only
(EXP-0074 `code/eval_wide.py` rebuilds pools from the test-split files). Rows within a file are chained (exact state match,
chains of up to 4 pushes; null pushes <1 mm are dropped before linking).
