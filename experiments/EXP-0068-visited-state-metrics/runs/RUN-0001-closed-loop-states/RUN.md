# RUN-0001 -- closed-loop re-run with full-state recording (states for DS-0023)

- **Purpose:** supply re-simulable VISITED states (xyz + quaternions) and the planner's top-16 candidates per decision;
  the closed-loop ORDER is not re-measured here (2 starts). Pre-registered in `../../DESIGN.md`.
- **Command** (`code/queue.sh`, step `cl_full`; pilot `pilot` = 1 model x 1 letter x 2 starts x 2 pushes, timing + "quats /
  topk_actions recorded" check only, no outcome read):
  `python -u experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py --tag cl_full --results-dir results
  --models <8 zoo models> --planners cem --goals letter_O letter_T letter_S letter_L letter_X letter_Z letter_C letter_H
  --starts 40 48 --steps 13 --budget 0.5 --n-cand 1024 --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25,
  "push_len": 0.02, "topk": 16}}' --legalize --legalize-fallback --planner-seed 0 --record-states --record-quats`
  (interpreter `/home/alon/anaconda3/envs/pme/bin/python`, `OMP_NUM_THREADS=4`, GPU exclusive under `experiments/temp/gpu.lock`).
- **Code:** commit a5ccb680, dirty (this record's `learned_mpc.py` topk / quats recording, the driver's `--record-quats`,
  EXP-0066's uncommitted `legalize_fallback` / `planner_seed`); ledger `EXP-0068/RUN-0001/*` in `experiments/COMMANDS.jsonl`.
- **Data:** DS-0006 starts 40 / 48 (`Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys`), TRAINING_PHYSICS.
- **Artifacts:** `results/cl_full.json` (driver checkpoint, rewritten after every push; `states`, `quats`, `topk_actions`,
  `topk_dv`, `legal_ok`, `legal_fallback` per episode), `logs/cl_full.log` / `.pid`, `results/cl_full.done`.
- **Status:** NOT RUN (paused 2026-10-05, see below)

## Paused 2026-10-05 (user), 14:45 CEST
Not run. RUN-0001 (`cl_full`) was launched once at 14:36 (CUT design: `--starts 40`, 13 pushes, 64 episodes) and killed by
the coordinator 24 s later, before any checkpoint; `results/` and `datasets/DS-0023-*/data/` are empty. It was a FIRST
run, not a re-run: no EXP-0068 closed-loop states ever existed (the session that pre-registered this record waited on
EXP-0067's GPU lock for 5.5 h and was cut off before the GPU freed). Resume: `code/queue.sh` (see EXPERIMENT.md "Paused").
