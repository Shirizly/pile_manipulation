# DS-0002 — real, deduplicated, genuinely-distinct particle states

**Status:** active
**Payload:** `datasets/DS-0002-real-distinct-states/data/states.pt` (gitignored; regenerate with the command below)
**Builder:** `datasets/DS-0002-real-distinct-states/build.py`

## What this is

A pooled, deduplicated corpus of **304,655 distinct real particle states**
(304,655 out of 631,612 raw candidate rows — 326,957 collapsed as
duplicates/near-duplicates, 51.8%), for the two project components that only
need STATES, never transitions: the self-supervised encoder and the
geometric value function.

This exists because every maintained corpus in `Genesis/data/` is either
**slate-shaped** — one settled pile swept by ~1000 near-identical candidate
pushes — or was never checked for cross-file duplication (e.g. the same
settled-state library reused verbatim as the "initial" state across many
independent-transition files). Fitting either component with a row-random
train/validation split on such data silently selects a far-too-small
regularisation strength and overfits — this is the
`readout-cv-folds-slate-aware` invariant in `experiments/INVARIANTS.md`, hit
four separate times before this dataset existed. DS-0002 exists so the next
consumer groups by construction (the `group` field) instead of rediscovering
the bug a fifth time.

## Inclusion rule

- **All distinct initial states**, from every real corpus swept.
- **Post-sweep states ONLY from corpora whose transitions are independent**
  (`overnight_randlen_{train,test}`, `Sean/*`) — those are genuinely diverse,
  not near-duplicates of one shared start.
- **Post-sweep states from slate pools are near-duplicates of one pile** (one
  settled state, ~1000 slightly different candidate pushes) — NOT included
  wholesale. At most 5 post-sweep states are sampled per slate
  (`SLATE_POST_SAMPLE = 5` in `build.py`), uniformly at random across that
  slate's candidates/steps.

## Corpora swept

| corpus | files | rows contributed (pre-dedup) | role |
|---|---|---|---|
| `Genesis/data/overnight_randlen_train/{mixed,piled,scattered}_n{20,50}` | 192 | all `states` + all `states_` | independent transitions |
| `Genesis/data/overnight_randlen_test/{...}` | 21 | all `states` + all `states_` | independent transitions |
| `Genesis/data/Sean/{inbetween,piled,scattered}-*/cube/n{20,50,100}/...` | 1230 (`*_data.pt` only; `.zip` and `*_failed.pt` excluded) | all `states` + all `states_` | independent transitions |
| `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm` | 20 slates | 1 initial + ≤5 sampled post-sweep per slate | slate-shaped, capped |
| `Genesis/data/slates_multistep/{n20_L10mm,n20_L20mm,n20_L40mm}` (base dirs; `*_eval`/`*_train` are filtered aliases, not swept separately) | 150 slates total across 3 tags | 1 initial + ≤5 sampled post-sweep per slate (pooled across steps 0-2 and all envs) | slate-shaped, capped |

## Skipped, with reasons

| corpus | why skipped |
|---|---|
| `Genesis/data/overnight_randlen` (un-split) | superseded by `overnight_randlen_{train,test}` per `docs/CODEMAP.md`; sweeping both would double-count the same rows |
| `Genesis/data/chickpeas`, `chickpeas_no_shuffle_position` | spheres on glass/wood, `.pkl` payload — not the `(n_objects,7)` cube-pose schema this dataset targets |
| `Genesis/data/corl` | mixed sphere/cube legacy corpus; provenance/config not verified within this task's budget |
| `Genesis/data/corl_limited` | cubes, but at swept sizes (5.0/6.75/8.5 mm) — mixing `cube_size` would break the fixed `CUBE_SIZE`/`REST_Z` schema DS-0002 and DS-0003 share; out of scope for this pass |
| `Genesis/data/cube_spectrum`, `Genesis/data/granularity` | deliberate particle-size/count sweeps (granularity spectrum), not representative single-condition states |
| `Genesis/data/dinowm_test`, `dinowm_test_dino_wm` | smoke-test corpora for a different (DINO-WM) pipeline |
| `Genesis/data/foresight` | smoke-test/probe corpus (`pile_smoke`, `probe_dense`), not a maintained training corpus |
| `Genesis/data/mpc_runs` | MPC rollout trajectories (evaluation artefacts), not a states corpus |
| `Genesis/data/slates` (`n20_heap_5mm`) | superseded single-push-length slate corpus per `docs/CODEMAP.md` (4 of 6 length bins starved); `slates_binned`/`slates_multistep` are the maintained slate corpora |

## Deduplication criterion

**Rows with the same `n_objects` whose particle (x,y) positions match to
within 1 mm, up to particle ordering, are the same state.** Concretely:
round each particle's (x,y) to `DEDUP_TOL_M = 0.001` m, sort the per-row
quantized (x,y) keys (particle storage order is not physically meaningful),
hash the resulting canonical byte string; rows with the same hash within a
given `n_objects` group collapse to their first occurrence. Quaternion and z
are **not** part of the key — dedup targets pile *layout*; z is
near-constant for a single layer and yaw is not expected to match exactly
even for the "same" pile across independent runs.

**Per-`n_objects` collapse:**

| n_objects | raw rows | distinct | collapsed | collapse rate |
|---|---|---|---|---|
| 20 | 308,220 | 141,297 | 166,923 | 54.2% |
| 50 | 218,752 | 109,132 | 109,620 | 50.1% |
| 100 | 104,640 | 54,226 | 50,414 | 48.2% |
| **total** | **631,612** | **304,655** | **326,957** | **51.8%** |

The collapse rate is high (~half) — most plausibly explained by
`overnight_randlen`/Sean drawing initial states from a shared, reused
settled-state library across many independent-transition files (same start
state reappearing verbatim as the "before" state of several different pushes)
rather than by resampling a fresh pile every time. This is a genuine finding
about corpus construction, not a dedup bug — the tolerance (1 mm, at least an
order of magnitude tighter than the 1 mm/2.5 mm particle sizes present) makes
false positive collapses very unlikely.

## Schema

`data/states.pt` is a dict:
- `xyz`: `(N, MAX_N, 3)` float32, metres, zero-padded past each row's `n_objects`
- `quat`: `(N, MAX_N, 4)` float32, wxyz, zero-padded past `n_objects`
- `n_objects`: `(N,)` int64 — 20, 50, or 100
- `corpus`, `source_file`, `spawn_mode`, `role` (`initial`/`post_sweep`): `(N,)` lists of str
- `group`: `(N,)` list of str — group id for near-duplicate/shared-origin rows.
  Independent-transition rows are grouped per source **file**
  (`randlen_{train,test}:<spawn_dir>:<filename>`, `sean:<mode>:<filename>`) —
  coarser than per-env, deliberately conservative (never leaks a chained
  sequence across a split). Slate rows are grouped per **slate**
  (`slates_binned:slate<i>`, `slates_multistep:<tag>:slate<i>`).
- `dedup_tol_m`, `n_raw_rows`, `n_distinct_rows`, `dedup_report_per_n_objects`, `skipped_corpora`: provenance metadata

## Regeneration

```
OMP_NUM_THREADS=4 /home/alon/anaconda3/envs/pme/bin/python -u datasets/DS-0002-real-distinct-states/build.py
```

Runtime: ~7 s (pure `torch.load` + numpy dedup, no simulation, no GPU).
