# RUN-0002 -- DS-0023 collection: same-state candidate pools from the 48 visited states

- **Select** (CPU): `python -u code/select_states.py --episodes results/cl_full.json --out datasets/DS-0023-visited-state-pools/data/starts.pt`
  (rule in `../../DESIGN.md`; prints the per-model / per-start / per-letter / per-k counts and the layer / legality audit).
- **Collect** (GPU): `python -u code/collect_pools.py --starts datasets/DS-0023-visited-state-pools/data/starts.pt --out
  datasets/DS-0023-visited-state-pools/data --pool 64 --n-envs 32` (sampler = DS-0016's; seed 68; one atomic
  `pools_state<i>.pt` per state, resumable; merged `pools_0.pt`; `manifest.json` with per-state counts and timing).
- **Code / ledger:** as RUN-0001 (`EXP-0068/RUN-0002/select`, `EXP-0068/RUN-0002/pools`).
- **Status:** NOT RUN (paused 2026-10-05, see below)

## Paused 2026-10-05 (user), 14:45 CEST
Not run. RUN-0001 (`cl_full`) was launched once at 14:36 (CUT design: `--starts 40`, 13 pushes, 64 episodes) and killed by
the coordinator 24 s later, before any checkpoint; `results/` and `datasets/DS-0023-*/data/` are empty. It was a FIRST
run, not a re-run: no EXP-0068 closed-loop states ever existed (the session that pre-registered this record waited on
EXP-0067's GPU lock for 5.5 h and was cut off before the GPU freed). Resume: `code/queue.sh` (see EXPERIMENT.md "Paused").
