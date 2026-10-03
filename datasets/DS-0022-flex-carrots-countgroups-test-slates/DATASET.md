---
id: DS-0022
title: FleX carrot piles, object-count-targeted TEST slates -- 200 states x 100 obj-biased pushes from the same settled state, 4 count groups (50 states each), native FleX units
status: active
date: 2026-10-03
path: datasets/DS-0022-flex-carrots-countgroups-test-slates/ ; data/ (raw, gitignored, 1.7 GB) holds manifest.jsonl, run_config.json, <state_idx>/ dirs
producer: ported from the external FleX project (~/Code/dyn-res-pile-manip @ 5c9eca9, dirty) -- collect_true_action_results_parallel.py (copy in EXP-0064 code/), config EXP-0064 code/configs/slates_objbiased_carrots_grouped.yaml (= data/run_config.json)
---

## What it is

Test corpus of EXP-0064. Same-state slate format as DS-0019 (schema
`true_action_slates_FORMAT.md`, shipped alongside): per state, 100
independently sampled `obj_biased` pushes, each replayed from the same
settled initial state, stratified over 6 push-length bins (edges 0.96 / 2.40 /
3.85 / 5.29 / 6.73 / 8.17 / 9.62, as DS-0019). States 0-49 / 50-99 / 100-149 /
150-199 target 10-30 / 50-70 / 100-150 / 400-500 carrots (`count_target`,
single centred blob). 2 parallel envs per batch (`n_envs 2`).

## Units and frame

As DS-0021: native FleX units, **stored actions are (x, -z)** (re-measured
2026-10-03: 99.9 % of in-path particles move under z = -a1 vs 81 % literal;
cos(displacement, push) 0.984 vs -0.057).

## Audit (2026-10-03)

| check | result |
|---|---|
| actions | 20,000 planned (4,998 + 3x5,000), **17,761 valid**; valid share 96.2 / 92.8 / 87.7 / 78.6 % by group (solver explosions rise with pile size) |
| **escaped particles in rows marked valid** | 0.56 / 1.90 / 2.1 / see EXP-0064 results per state (all groups, rising with N) -- filter with `escape_abs` 10 |
| pool size per state | 62-100 clean (escape-filtered) candidates; group means fall with N -> slateN not comparable across groups at full pool; use the K = 62 equalised read (EXP-0064 RUN-0006) |
| pile footprint | grows with count (single blob); fits the DS-0019 grid +-7.2 |

Pool size differs from every Genesis corpus and from DS-0019 (86-191):
slateN is not comparable across corpora.

## Regeneration

Not regenerable from this repo (PyFleX). Source command RECONSTRUCTED:
`collect_true_action_results_parallel.py --config config/data_gen/slates_objbiased_carrots_grouped.yaml`.
Log: EXP-0064 `artifacts/RUN-0002-collect-test/`. Identical copy remains in the source repo
(`data/true_action_slates_objbiased_carrots_grouped`).
