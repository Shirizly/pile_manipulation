# RUN-0007 — NFD ranking of chained vs free pushes (1 and 2 steps)

- `code/chain_model_eval.py nfd_3ch_randlen` (via `code/queue3.sh`) -> `results/chain_model_eval_nfd_3ch_randlen.json`, `results/chain_model_eval.txt`
- inputs: RUN-0006's `results/chain_probe_states/*.pt` (24 states x 32 free + 32 chained Genesis outcomes)
- note: `capture` is ill-defined for the chained set (best true dV often >= 0 or near 0) -- report Spearman and MAE only
- commit a5ccb680 dirty; GPU < 1 min
