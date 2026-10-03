---
id: DS-0021
title: FleX carrot piles, object-count-targeted TRAIN trajectories -- 2000 states x 10 sequential obj-biased pushes, 4 count groups (10-30 / 50-70 / 100-150 / 400-500 carrots), native FleX units
status: active
date: 2026-10-03
path: datasets/DS-0021-flex-carrots-countgroups-train/ ; data/ (raw, gitignored, 2.3 GB) holds manifest.jsonl + <state_idx>/ dirs
producer: ported from the external FleX project (~/Code/dyn-res-pile-manip @ 5c9eca9, dirty -- collector files uncommitted there) -- collect_transitions.py (copy in EXP-0064 code/), config EXP-0064 code/configs/transitions_carrots_grouped.yaml
---

## What it is

Training corpus of EXP-0064 (object-count GNN study). Sequential ("continuing
trajectory") format -- schema in `true_action_transitions_FORMAT.md` (shipped
alongside, unchanged). 2000 states, each followed by 10 pushes with the
`obj_biased` sampler (start near a particle + jitter, end uniform, margin-aware;
NOT the perpendicular 20 mm Genesis convention). States 0-499 / 500-999 /
1000-1499 / 1500-1999 target 10-30 / 50-70 / 100-150 / 400-500 carrots
(`count_target` init_pos, `target_num_carrots` and `count_group` per
`state_init` record, drawn seeded by (seed, state_idx)). Every pile is a
single centred blob (rand_blob footprint sized to the target count); there is
no scatter/spread variant -- object count is confounded with pile footprint.

## Units and frame

Native FleX units, same as DS-0019/DS-0020 (global_scale 24, wkspc_w 5,
action_margin 1.6, particle_r 0.125). **Stored actions are (x, -z) of the
particle files** (invariant `flex-action-frame-neg-z`), re-measured here
2026-10-03 (EXP-0064 `code/frame_check.py`): mean cos(moved-particle
displacement, push direction) 0.992 under z = -a1 vs 0.004 literal. EXP-0064's
as-run training read them literally (see EXP-0064 issues.md, I-1).

## Audit (2026-10-03, this repo)

| check | result |
|---|---|
| records | 2000 state_init, 20000 transitions, all `valid: true` |
| carrots per group (target) | 10-30, 50-70, 100-150, 400-500 exactly |
| particles per state, median [min, max] | 433 [67, 1465] / 1254 [332, 3427] / 2575 [705, 7081] / 9193 [2829, 23068] -- per-carrot scale is randomised, so particles per object vary ~5x within a group |
| push length (FleX units) | min 0.01, quartiles 2.39 / 3.71 / 5.09, max 9.48 |
| **escaped particles (any \|x\| or \|z\| > 10) in rows marked valid** (1500-row sample) | **13/375 (3.5 %), 18/361 (5.0 %), 32/373 (8.6 %), 57/391 (14.6 %)** by group -- rises with object count; the collector's explosion check did not catch these. Train on this corpus only with an escape filter (DS-0019's `escape_abs` 10) |
| colour images | `*_color.png` per frame (720x720); no depth images |

## Regeneration

Not regenerable from this repo (needs PyFleX + the source repo's
`env/flex_env_multi.py` `count_target` branch, copied for reference to
EXP-0064 `code/ported_reference/`). Source command (from the source report,
RECONSTRUCTED): 2 shards of `collect_transitions.py --config
config/data_gen/transitions_carrots_grouped.yaml` under `run_transitions_shard.sh`,
then `collect_transitions.py --merge`. Logs: EXP-0064 `artifacts/RUN-0001-collect-train/`.
The source repo still holds an identical copy at `data/true_action_transitions_carrots_grouped`.
