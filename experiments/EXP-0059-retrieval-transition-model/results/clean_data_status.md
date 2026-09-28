# EXP-0059 clean-data status (ISS-010 fix + fresh collection)

Updated: 2026-09-28, mid-task. Overwritten after each step per the coordinator's briefing.

## Scope (current, per coordinator narrowing 2026-09-28)

- Fresh collections ONLY: (a) TEST (DS-0009-shaped: chains+pools), (b) TRAIN (DS-0008-shaped
  replacement, ~12k rows, replaces DS-0008+DS-0010), (d) VAL pools (DS-0011-shaped). Multi-step
  (c) is explicitly OUT of scope for now.
- Old-set flag/split/archive (tasks 2-3): DS-0008, DS-0009 (chains+pools), DS-0011 only.
- Order: (a)+(b) concurrently if GPU allows, then (d). BLOCKED on coordinator confirmation
  before any production launch (see "Blocking" below).

## Task 1 -- sampler fix (code, done; production collection BLOCKED pending confirmation)

- **Bug fixed (ISS-010):** `Genesis/sandbox_manipulation_clean.py::_pile_aware_stops` clamped a
  collision-free `pile_contact_starts` touchdown into the sampling box with NO cube check.
  Replaced the `_apply_pile_aware_starts`+`_pile_aware_stops` pair with
  `_pile_aware_action_legal`, which calls the new Genesis-free
  `Genesis/action_sampling.py::pile_aware_action_batch`: same box-clamp + push-length logic,
  but checks the FINAL touchdown footprint with an exact SAT test
  (`overlaps_rect_pairs_torch`, a torch port of `Baselines.common.cube_overlap.overlaps_rect_pairs`)
  and redraws (fresh heading, never a shortened/lengthened push) any illegal slot.
- **Start-gap sampling (coordinator spec, 2026-09-28):** added `start_gap_range=(lo_margin,
  hi_margin)` sampler config key. The gap (blade FRONT FACE to the first-contact cube's NEAR
  FACE, along the push axis) is sampled uniformly in `[lo_margin, L-hi_margin]` per action
  (`L`=scalar push_length), applied at the SAME point the old fixed distance was
  (`pile_contact_starts`'s own `clearance` argument) via a face-to-face -> centre-based
  conversion (`+ blade_half_width + cube_half`), not a post-hoc shift. `start_gap_range=None`
  reproduces the old fixed-clearance behaviour exactly.
- **Second bug found + fixed during review (coordinator caught via smoke-test numbers):** the
  box clamp moves the touchdown in WORLD (x,y), decoupling the along-push-axis gap from the
  clamped draw's geometry whether or not the clamped point is still legal -- so a legal-but-
  clamped draw was silently keeping an arbitrary (occasionally far-outside-window) gap. Fixed by
  treating "was clamped" as an ADDITIONAL redraw trigger, with a priority score (legal+unclamped
  > legal+clamped > illegal) so the clamp-precision redraw can never regress the hard legality
  guarantee (this itself regressed illegal to 2-5/32 in one smoke run when clamp and illegal
  shared one undifferentiated redraw budget -- caught, fixed, verified 0/256 after).
- Genesis-free unit tests: `tests/test_action_sampling.py` (42 tests, all passing), including
  direct regression tests for the clamp-redraw fix and the face-to-face gap conversion.
- **Real smoke test (seed 9102, scatter+clump mixed, 256 rows, TRAINING_PHYSICS, push_length
  0.02, start_gap_range=[0.005,0.005]):**
  - illegal touchdowns (exact SAT, 0mm): **0/256 (0.0%)**.
  - along-axis gap to first-contact cube, mm, percentiles [5,25,50,75,95]:
    **[5.02, 8.13, 11.47, 13.85, 32.88]**; 81.8% of rows within [5,15]mm +/-1mm slack. The
    upper tail (p95=32.9mm) is from slots where no unclamped-AND-legal candidate was found
    within `max_redraws=20` -- legal (0 illegal overall) but gap not guaranteed exact; 9/256
    rows had no candidate cube ahead in the swath at all (excluded from the gap stats).
  - first-contact-cube displacement, mm, percentiles: **[0.0, 6.0, 8.71, 12.12, 16.03]** -- p5=0
    persists for a small tail, consistent with the same clamp-exhausted slots plus friction
    distributing force away from the nominal first-contact cube.
  - **null rate (max per-cube disp <1mm): 40/256 = 15.6%** -- over the 5% gate; flagging per
    the coordinator's instruction, not yet acted on (no top-up has been run; fresh collection
    was STOPPED before any production launch).
- **STATUS: awaiting coordinator confirmation before launching production (a)/(b)/(d).** A
  first production launch was started and immediately stopped (coordinator STOP) once the gap
  bugs above were found; nothing was written to `Genesis/data/narrow_l20_n20/{test_chains_v2,
  test_pools_v2,train_v2}` (jobs killed during Genesis scene compile, before any chunk wrote).

## Task 2 -- null-transition flagging (old sets: DS-0008, DS-0009 chains+pools, DS-0011) -- DONE

Script: `experiments/EXP-0059-retrieval-transition-model/code/flag_null_transitions.py`
(Genesis-free; threshold 1mm max per-cube xy displacement, same as the retrieval bank's `moved`
flag; max yaw change also recorded per row, not part of the null criterion). Per-row
`_k_data_nullflag.pt` / `pools_k_nullflag.pt` written next to each source file (originals
untouched). Full JSON: `experiments/EXP-0059-retrieval-transition-model/results/null_transition_audit.json`.

| dataset | rows (valid) | illegal (ISS-010, existing audit) | null | illegal AND null |
|---|---|---|---|---|
| DS-0008 train | 6144 | 44.3% | 11.8% (725) | 1.7% (104) |
| DS-0009 test_chains | 1024 | 46.2% | 9.6% (98) | 1.1% (11) |
| DS-0009 test_pools | 2048 | 56.1% | 8.0% (163) | 1.6% (33) |
| DS-0011 val_pools | 2048 | 47.9% | 9.1% (186) | 1.5% (30) |

## Task 3 -- split old sets into clean + archive_removed -- NOT STARTED

Next up once task 1 is confirmed and not competing for turn budget with the smoke-test
diagnosis above.

## Task 4 -- fresh collection -- LAUNCHED 2026-09-28 13:46, running detached

Fixed two more bugs per coordinator review (face-vs-centre gap conversion; box clamp decoupling
the gap regardless of legality -- see `action_sampling.py::pile_aware_action_batch`'s
docstring), raised `max_redraws` 20->200 (retarget: redraw a fresh heading/contact cube rather
than accept an out-of-window gap; last resort accepts + flags `gap_out_of_window` per row, now
saved in every chunk's `.pt` file), and reran the smoke test (seed 9103, 256 rows,
mixed scatter+clump):

- **illegal: 0/256 (0.0%)**. **gap in [5,15]mm (+/-0.5mm slack): 251/251 = 100%** of rows with a
  contact cube (5/256 had none in swath at all -- structural, not a gap-window failure).
  `gap_out_of_window` flag: 0/256. Added redraw cost: negligible (55s vs 53-58s at
  max_redraws=20 -- action sampling is cheap next to the physics step, as expected).
- **null: 5/256 = 1.95%**. Breakdown: (i) no cube ahead in swath = 5/5 of the nulls, (ii) gap
  out of window/clamp-exhausted = 0, (iii) gap in window but contact cube still moved <1mm = 0.
  No physics/footprint anomaly to chase -- every null here is a push that structurally had
  nothing in its path, not a push that touched a cube and failed to move it.
- **LAUNCH CRITERIA (illegal=0, >=95% gap in-window, null<=5%): ALL MET.** Launched (a) TEST
  and (b) TRAIN immediately, detached, per the coordinator's standing go-ahead.

**Running now** (`setsid nohup`, survive this session; sampler
`{"pile_aware": true, "min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}`,
TRAINING_PHYSICS, `--starts mixed`, `Genesis.clump_states:clump_starts`; logged in
`experiments/COMMANDS.jsonl`):

| job | PID | out dir | target rows | seed | log | ETA |
|---|---|---|---|---|---|---|
| (a) TEST chains | 1352383 (wrapper) / 1352387 (python) | `Genesis/data/narrow_l20_n20/test_chains_v2` | 4x32x8=1024 | 2001 | `experiments/EXP-0059-*/runs/test_chains_v2.log` | ~4 min |
| (a) TEST pools | 1352384 (wrapper) / 1352389 (python) | `Genesis/data/narrow_l20_n20/test_pools_v2` | 32x64=2048 | 2002 | `experiments/EXP-0059-*/runs/test_pools_v2.log` | ~10 min |
| (b) TRAIN | 1352385 (wrapper) / 1352388 (python) | `Genesis/data/narrow_l20_n20/train_v2` | 47x32x8=12032 | 2003 | `experiments/EXP-0059-*/runs/train_v2.log` | ~40-45 min |

GPU confirmed healthy running all 3 concurrently: 579MB used / 7.2GB free right after launch.
Check `manifest.json`'s `complete` field in each out dir for status; each chunk is
atomic/resumable (`chain_collection.py` skips finished chunk indices on restart).

**UPDATE 2026-09-28 14:4x -- (a)+(b) confirmed COMPLETE** (`manifest.json` `complete: true` for
all three: test_chains_v2 4/4 chunks, test_pools_v2 1/1, train_v2 47/47). Full-scale audit
(exact SAT illegal test + null + `gap_out_of_window`, same methodology as
`audit_tool_placement.py`/`flag_null_transitions.py`, run directly against the `*_v2` files):

| set | rows | illegal | gap_out_of_window | valid==False | null (subset check) |
|---|---|---|---|---|---|
| TEST chains (seed 2001) | 1024 | **0 (0.0%)** | **0 (0.0%)** | 38 (3.71%) | 29 (2.83%) -- all 29 ⊆ the 38 invalid |
| TEST pools (seed 2002) | 2048 | **0 (0.0%)** | **0 (0.0%)** | 0 | 0 |
| TRAIN (seed 2003) | 12032 | **0 (0.0%)** | **0 (0.0%)** | 373 (3.10%) | 282 (2.34%) -- all 282 ⊆ the 373 invalid |

**Cross-checked against the experimenter's `Genesis/training/dataset.py::
PileSweepData(exclude_flagged=True)` loader** (used concurrently for NFD training): TEST chains
-- loader drops 38 (0 gap_out_of_window + 38 invalid/illegal + 0 additional null), kept
986/1024, agrees exactly. TRAIN -- loader drops 373 (0 + 373 + 0), kept 11659/12032, agrees
exactly. (TEST pools' `pools_0.pt` isn't in `PileSweepData`'s `*_data.pt`+config-yaml schema --
expected, pools are read via the unified-benchmark-shape readers, not this class; its own
audit above already shows 0/0/0/0.)

**(d) VAL pools LAUNCHED 2026-09-28 14:45, running** (PID 1416423 python / 1416420 wrapper,
seed 2004, `Genesis/data/narrow_l20_n20/val_pools_v2`, target 32x64=2048 rows, log
`experiments/EXP-0059-*/runs/val_pools_v2.log`, ETA ~10 min). Same recipe/seed family as
(a)/(b); expected to audit the same way once complete. GPU kept to this one job only (an
experimenter agent is training NFD concurrently) -- 205MB used / 7.6GB free right after launch.

## Task 5 -- registration / doc updates -- DONE for the 3 completed sets (+ DS-0017 placeholder)

- **DS-0015** (`datasets/DS-0015-narrow-l20-train-clean/DATASET.md`) -- fresh TRAIN, replaces
  DS-0008+DS-0010, full-scale audit above.
- **DS-0016** (`datasets/DS-0016-narrow-l20-test-clean/DATASET.md`) -- fresh TEST chains+pools,
  replaces DS-0009, full-scale audit above.
- **DS-0017** (`datasets/DS-0017-narrow-l20-val-pools-clean/DATASET.md`) -- fresh VAL pools,
  replaces DS-0011, written before completion per "IDs outlive their payload"; update once
  `val_pools_v2/manifest.json` shows `complete: true`.
- DS-0008, DS-0009, DS-0011's own DATASET.md files now each note they are SUPERSEDED for clean
  comparisons, pointing at DS-0015/16/17, payloads UNCHANGED.
- `experiments/OPEN_ISSUES.md` ISS-010: **status changed to `closed (sampler fix)`**, full
  writeup of the fix + the two bugs found+fixed during review + what's still open (EXP-0053
  audit, DS-0010 per-file sampler verification, DS-0012/13 not recollected).
- `docs/CODEMAP.md` datasets table: new row for the `*_v2` sets, old `narrow_l20_n20` row
  annotated with the fix + pointer to the replacements.
- `.claude/skills/data-collection/SKILL.md`: `pile_aware`/`start_gap_range` entries updated,
  new trap entry (the clamp defect + fix + the two follow-up bugs' lesson: redraw-priority
  ordering when adding a second condition to an existing redraw loop).

## Task 3 -- clean copy + archive_removed/ -- DONE (2026-09-28, corrected: this IS required)

`experiments/EXP-0059-retrieval-transition-model/code/split_clean_archive.py` (CPU-only,
originals untouched; full counts also in `results/split_clean_archive.json`):

| set | kind | rows | removed | clean | clean dir | archive dir |
|---|---|---|---|---|---|---|
| DS-0008 train | independent (drop rows) | 6144 | 3343 (54.4%) | 2801 | `train_clean/` | `train/archive_removed/` |
| DS-0009 test_pools | pool (drop candidates) | 2048 | 1279 (62.5%) | 769 | `test_pools_clean/` | `test_pools/archive_removed/` |
| DS-0009 test_chains | sequence (drop whole) | 1024/128seq | 1016 rows / **127 seq (99.2%)** | 8 rows / **1 seq** | `test_chains_clean/` | `test_chains/archive_removed/` |
| DS-0011 val_pools | pool (drop candidates) | 2048 | 1136 (55.5%) | 912 | `val_pools_clean/` | `val_pools/archive_removed/` |
| DS-0015 train_v2 | independent (drop rows) | 12032 | 373 (3.10%) | 11659 | `train_v2_clean/` | `train_v2/archive_removed/` |
| DS-0016 test_chains_v2 | sequence (drop whole) | 1024/128seq | 128 rows / **16 seq (12.5%)** | 896 rows / **112 seq** | `test_chains_v2_clean/` | `test_chains_v2/archive_removed/` |
| DS-0016 test_pools_v2 | pool | 2048 | 0 | 2048 | (already clean, no split) | -- |
| DS-0017 val_pools_v2 | pool | 2048 | 0 | 2048 | (already clean, no split) | -- |

Archive schema = same keys as the source file + `reason` (old sets:
`illegal`/`null`/`illegal_and_null`; fresh sets: `invalid_redraw_exhausted`, or
`sequence_contains_bad_step` for a row whose OWN step was fine but a sibling step in its
sequence was bad) + `source_file` + `source_row`.

**Headline finding: DS-0009's OLD `test_chains` is not usable as a clean rollout set** -- at its
~55% per-row bad rate, only 1 of 128 eight-step sequences survives whole-sequence removal
(binomial expectation ~0.17%, matches). **DS-0016's fresh `test_chains_v2` (same shape, fixed
sampler) keeps 112/128 sequences (87.5%)** -- the clearest single number demonstrating the
sampler fix's practical value for anything that needs an intact multi-step sequence, not just a
lower error rate.

Per-file DATASET.md updates (counts + paths) landed in DS-0008, DS-0009, DS-0011, DS-0015,
DS-0016, DS-0017.

## DS-0017 finalized (2026-09-28 14:53)

`val_pools_v2` completed (2048/2048 valid, manifest `complete: true`). Full-scale audit:
**0/2048 illegal, 0/2048 null, 0/2048 gap_out_of_window, 0/2048 `valid==False`** -- already
100% clean, no archive/split needed. DS-0017's DATASET.md updated accordingly.

**ALL FOUR TASK-4 COLLECTIONS NOW COMPLETE AND CLEAN**: DS-0015 (train), DS-0016 (test
chains+pools), DS-0017 (val pools).

## Task 5 -- registration / doc updates -- NOT STARTED
