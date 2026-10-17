---
id: DS-0023
title: Same-state candidate pools from closed-loop VISITED states -- 48 states after 4 / 8 / 12 legal CEM pushes of the EXP-0039 RUN-0002 zoo on 8 letter goals, 64 legal pile-aware 20 mm candidates + the planner's own legalised top-16 per state, training physics, soft truth from stored particle states
status: pending                  # set to active once collection completes (EXP-0068 RUN-0002)
date: 2026-10-05
path: datasets/DS-0023-visited-state-pools/data (starts.pt, pools_state<i>.pt per state, pools_0.pt merged, manifest.json)
producer: experiments/EXP-0068-visited-state-metrics/code/collect_pools.py (states: code/select_states.py from EXP-0068 RUN-0001)
---

## What it is
The first benchmark pool set whose START states are states a learned-model planner actually REACHED, not spawns.
EXP-0068 RUN-0001 re-ran EXP-0039 RUN-0002's CEM arm (8 OCC models x 8 letters O T S L X Z C H x DS-0006 scatter starts
40 / 48, 13 x 20 mm pushes, CEM pop 1024 / elite 0.25 / 0.5 s, legalize + legal fallback, planner seed 0,
TRAINING_PHYSICS) with full-state recording (`--record-quats`, new) and the planner's 16 best-predicted candidates per
decision (`topk 16`, new). `select_states.py` takes 48 of those states by a fixed rule (DESIGN.md): 24 cells = letter x
k in {4, 8, 12}, 2 states per cell from 2 different models at the 2 different starts; every model 6 states, every start
24, no episode twice, 6 different models per letter. Each state carries `meta` (model, goal, start, k, layer count,
legality, the executed push's pred/true dv) and the planner's `topk_actions` / `topk_dv` at that step.

## Pools (unified benchmark shape + extra columns)
Per state: broadcast to 32 envs (`set_particle_state` + `update_material_state`; `settle_drift_m` records how far the
re-settle moved any cube), 64 candidates from DS-0016's sampler (`pile_aware, min_swath_particles 3, push_length 0.02,
start_gap_range [0.005, 0.005]`; exact 20 mm +-0.1 mm and perpendicular enforced by redraw, `Genesis/chain_collection
.check_actions`), plus the 16 planner candidates passed through `Genesis.action_sampling.legalize_pushes` with the closed
loop's own parameters (cube half 2.5 mm, blade half-length 20 mm, half-width 1 mm, box = tray minus 4 mm) -- `source` 1,
`legal_ok` / `legal_shift_m` recorded, un-legalisable ones kept and flagged. Every candidate is simulated from the
identical settled state in sub-batches of 32 (`execute_action` + `update_material_state`). Columns: `states (R, 20, 7)`
(the settled start, repeated), `states_ (R, 20, 7)`, `p_starts / p_stops (R, 3)`, `angles (R,)`, `pool_idx (R,)` (state
id 0..47), `valid`, `gap_out_of_window`, `source` (0 sampler / 1 planner), `legal_ok`, `legal_shift_m`,
`planner_pred_dv` (NaN for sampler rows), `meta` (per pool), `start_kind` ("visited"). Physics: TRAINING_PHYSICS
(0.7 / 0.5 / 450, settle 3000). Truth is scored from `states_` with the soft rasteriser (METRICS.md), never stored.

## Known limits
- The states are produced by 8 narrow-corpus pilot models and 2 broad models (user decision 2026-10-05: narrow data is a
  pilot instrument); the corpus is model-agnostic only in the sense that no model contributes more than 6 of 48 states.
- 2 DS-0006 starts only (40, 48); letter diversity comes from the goals, not the spawns.
- Planner candidates are legalised by translation (as the closed loop does), so they are not exactly what the planner
  scored; `legal_shift_m` says how far each moved. Sampler candidates are legal by construction.
- Verification counts (illegal touchdowns, null rows, `valid==False`, layer-1 cubes, settle drift) are in `manifest.json`
  and EXP-0068 `results/analysis.md`; fill in below once collected.

## Verification
pending

## Commands
See `experiments/EXP-0068-visited-state-metrics/code/queue.sh` (RUN-0001 closed loop, select, RUN-0002 pools) and
`experiments/COMMANDS.jsonl` (`EXP-0068/RUN-000{1,2}/...`).
