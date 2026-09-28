# RUN-0001 — eval_report --truth-scoring soft, default and many goal sets

- Commit 3bae8cd7, dirty (soft-scoring code of this session). Via run_probe: exp0028_soft_scoring.json; code/run_both.sh (CPU, CUDA hidden). ~75 min.
- Outputs: artifacts/RUN-0001-soft-{default,many}/report.json (checkpointed after every model). Comparison: code/compare.py -> results/compare.json (run inline, seconds).
