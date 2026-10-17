# RUN-0003 -- scoring every model on DS-0023 (visited) and DS-0016 (spawn), and the analysis

- **Score** (GPU): `python -u code/score_pools.py --corpus ds0023 --models <8 zoo + linear boot1 boot2 + nfd_3ch_randlen_seed1>`
  and `--corpus ds0016` -> `results/scores_ds0023.json`, `results/scores_ds0016.json` (per-row swept-region error sums for
  accuracy_1 / blur2, sliced-W1, per pool x letter vt / vp and |mass error|); predicted occupancies float16 in
  `artifacts/RUN-0003/pred_<corpus>_<model>.npz`.
- **Analyse** (CPU): `CUDA_VISIBLE_DEVICES= python -u code/analyse.py` -> `results/analysis.{json,md}` (pre-registered
  quantities, DESIGN.md).
- **Code / ledger:** as RUN-0001 (`EXP-0068/RUN-0003/score_ds0023`, `score_ds0016`, `analyse`).
- **Status:** NOT RUN (paused 2026-10-05, see below)

## Paused 2026-10-05 (user), 14:45 CEST
Not run. RUN-0001 (`cl_full`) was launched once at 14:36 (CUT design: `--starts 40`, 13 pushes, 64 episodes) and killed by
the coordinator 24 s later, before any checkpoint; `results/` and `datasets/DS-0023-*/data/` are empty. It was a FIRST
run, not a re-run: no EXP-0068 closed-loop states ever existed (the session that pre-registered this record waited on
EXP-0067's GPU lock for 5.5 h and was cut off before the GPU freed). Resume: `code/queue.sh` (see EXPERIMENT.md "Paused").
