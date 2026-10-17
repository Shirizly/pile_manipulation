# RUN-0001 — EXP-0066 closed-loop power study (NFD narrow v2 x 3 seeds vs linear narrow v2 res64 x 3 instances)

- **Started:** 2026-10-04 20:39 CEST (queue `code/queue.sh`, pid `logs/queue.pid`); cost pilot 20:34 (`results/pilot/`).
- **Code:** commit a5ccb680, **dirty** -- uncommitted: `simple_mpc/learned_mpc.py` (legal_fallback, legalize_fallback=,
  planner_seed=), `experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py` (`--legalize-fallback`,
  `--planner-seed`, `legalize(acts, envs=)`), `Baselines/LinearForesight/fit_switched.py` (`--bootstrap-seed`),
  `simple_mpc/adapters.py` (`linear_narrow_l20_v2_res64_boot{1,2}`), `tests/test_learned_mpc.py`, `docs/CODEMAP.md`,
  `docs/ARCHITECTURE.md`.
- **Linear refits (GPU, 41 s each, 20:31-20:32):** `PYTHONPATH=. python -u Baselines/LinearForesight/fit_switched.py
  --n-bins 1 --res 64 --train-cfg configs/dataset/genesis_narrow_l20_train_v2.yaml --test-cfg
  configs/dataset/genesis_narrow_l20_test_chains_v2.yaml --bootstrap-seed {1,2} --out
  Baselines/LinearForesight/runs/operator_narrow_l20_v2_res64_boot{1,2}.pt` (logs `logs/fit_linear_boot{1,2}.log`;
  held-out swept-region accuracy 0.2798 / 0.2786 vs the full-data operator's 0.332).
- **Cells:** one driver call per cell, in DESIGN.md's order; exact argv per cell in `logs/<tag>.log` head / `queue.sh`:
  `python -u experiments/EXP-0043-batched-closed-loop/code/batched_closed_loop.py --tag <tag> --results-dir results
  --models <model> --planners cem --goals letter_O letter_T letter_S letter_X --starts 40..71 --steps 16 --budget 0.5
  --n-cand 1024 --cells '{"l20": {"cem_pop": 1024, "cem_elite_frac": 0.25, "push_len": 0.02}}' --legalize
  --legalize-fallback --planner-seed <ps> --record-states`. Interpreter `/home/alon/anaconda3/envs/pme/bin/python`,
  `OMP_NUM_THREADS=4`, GPU exclusive (RTX 4070 8 GB; `nvidia-smi` idle before the start).
- **Data:** DS-0006 starts 40-71 (`Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys`), EXP-0046
  `vstar.json` per-letter optimum.
- **Artifacts:** `results/<tag>.json` (driver checkpoint, rewritten atomically after every push; `states` recorded),
  `results/<tag>.done`, `logs/<tag>.log` / `.pid`, ledger lines `EXP-0066/RUN-0001/<tag>` in `experiments/COMMANDS.jsonl`.
- **Analysis:** `python code/analyse.py` -> `results/analysis.{json,md}`.
- **Status:** complete, 12 / 12 cells, every episode complete; cells 24-28 min each; 2026-10-04 20:39 -> 2026-10-05 01:56 CEST (5.3 GPU-h). No cell skipped by the budget guard. Analysis 02:00; `results/analysis.{json,md}`.
