---
id: DS-0015
title: Narrow-domain TRAIN, clean by construction -- n20 single layer, exact 20 mm perpendicular pushes, training physics, replaces DS-0008+DS-0010
status: active
date: 2026-09-28
path: Genesis/data/narrow_l20_n20/train_v2 (_{k}_data.pt, k = 0..46)
producer: Genesis/chain_collection.py --mode chains --seed 2003 (fresh, disjoint from every prior seed)
---

## What this replaces and why

Supersedes **DS-0008** (train chains, 6,144 rows) + **DS-0010** (extra_18_22, 5,777 rows) as the
single training corpus for this domain (~11,921 rows combined). ISS-010 found that DS-0008's
`pile_aware` sampler placed the tool ON a cube at touchdown in 44.3% of rows (the
`_pile_aware_stops` box-clamp had no cube check); rather than patch DS-0008/0010 with a
replace-and-archive pass, the user (2026-09-28) chose to **recollect fresh, at the fixed
sampler's now-standard recipe**, sized to match the old combined total (11,921) rather than top
up the old rows. DS-0008/0010's own DATASET.md files are marked superseded for clean
comparisons; their payloads are UNCHANGED (still usable for anything that predates this fix, or
for before/after comparisons of the fix itself).

## Sampler fix (the reason this set is legal by construction)

`Genesis/sandbox_manipulation_clean.py::_pile_aware_action_legal` (calling
`Genesis/action_sampling.py::pile_aware_action_batch`) replaces the old
`_apply_pile_aware_starts`+`_pile_aware_stops` pair: the box clamp that used to place a
collision-free touchdown ON a cube with no check is now followed by an exact SAT overlap test
against every cube, redrawing (a fresh heading, never a shortened/lengthened push) any illegal
slot, up to `max_redraws=200`.

**New sampler config key, `start_gap_range` (2026-09-28 coordinator spec):** the gap from the
blade's front face to the near face of the first cube its swath will contact, measured along the
push axis, is SAMPLED per action, uniformly, in `[lo_margin, L - hi_margin]` (here
`start_gap_range=[0.005, 0.005]`, `L=0.02` -> a 5-15 mm gap, so the contacted cube travels
5-15 mm) instead of the old fixed one-particle-width gap. A slot with no legal AND in-window
candidate after `max_redraws` attempts is retargeted (fresh heading/contact cube) rather than
accepted as drawn; only as a last resort is it accepted, flagged `gap_out_of_window=True` in the
saved data (all-False in this set at full scale -- see below).

## Verification (full-scale audit, not just the pre-launch smoke test)

12,032 rows (47 chunks x 32 envs x 8 steps; a few more than DS-0008+DS-0010's 11,921 by
construction of the chunk grid -- use all of it, or subsample, per consumer preference):

| check | result |
|---|---|
| illegal touchdowns (exact SAT, 0mm, same test as `audit_tool_placement.py`) | **0/12032 (0.0%)** |
| `gap_out_of_window` (last-resort accept, flagged) | **0/12032 (0.0%)** |
| `valid==False` (push length/perpendicularity check failed, redraw-exhausted; pre-existing schema field, unrelated to the gap fix) | 373/12032 (3.10%) |
| null (max per-cube xy displacement < 1mm, same threshold as the retrieval bank's `moved` flag) | 282/12032 (2.34%) -- **all 282 are a SUBSET of the 373 `valid==False` rows** (verified directly), not an independent failure mode |
| `Genesis/training/dataset.py::PileSweepData(exclude_flagged=True)` (the experimenter's training loader) | independently confirms: drops 373 invalid/illegal + 0 gap_out_of_window + 0 additional null -- kept 11,659/12,032, agrees exactly with the counts above |

Physics: TRAINING_PHYSICS (particle friction 0.7, box friction 0.5, density 450, settle 3000).
Starts: `--starts mixed` (half `state_library` scatter, half `Genesis.clump_states:clump_starts`
single-layer clumps), same recipe as DS-0008. Push length exactly 0.02 m, perpendicular.

Command: `python -u Genesis/chain_collection.py --mode chains --out
Genesis/data/narrow_l20_n20/train_v2 --n-chunks 47 --steps 8 --n-envs 32 --starts mixed
--clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
"min_swath_particles": 3, "push_length": 0.02, "start_gap_range": [0.005, 0.005]}' --seed 2003`.
Files: `_{k}_data.pt` (training keys + `chain_env`, `chain_step`, `valid`, `single_layer`,
`start_kind`, `gap_out_of_window`) + `_{k}_config.yaml` (carries `start_gap_resolved: [0.005,
0.015]`, the literal resolved window, in its `chain_collection` block). `manifest.json`:
`complete: true`, per-chunk `gap_out_of_window` counts.

For training, use `exclude_flagged=True` (drops the 373 `valid==False`/gap-precision rows,
which subsume every null) rather than re-deriving a filter.

## Clean copy + archive (2026-09-28)

`experiments/EXP-0059-*/code/split_clean_archive.py` (CPU-only, this set's own `Genesis/data/
narrow_l20_n20/train_v2/` payload untouched) additionally materialises the `exclude_flagged`
filter as physical files, so a consumer that just globs a directory (rather than using
`PileSweepData(exclude_flagged=True)`) also gets the clean rows only: 12,032 rows -> **373
removed (3.10%, all `valid==False`) -> 11,659 clean rows**, matching the loader's own count
exactly. Clean: `Genesis/data/narrow_l20_n20/train_v2_clean/` (same per-chunk layout + config
yaml). Archive: `Genesis/data/narrow_l20_n20/train_v2/archive_removed/_{k}_data_removed.pt`
(same schema + `reason=invalid_redraw_exhausted` + `source_file`/`source_row`).
