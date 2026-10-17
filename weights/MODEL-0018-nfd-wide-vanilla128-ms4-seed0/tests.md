# MODEL-0018 test history
- wide Sean test (8 held-out shards, pasted 64x64 frame, no balance fix): acc1 0.572 / slateN 0.924 / roll1-4 0.563 0.503 0.467 0.453 -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_w128_f8_ms4.json`
- val-selected post-hoc mass balance (none vs balance, test) -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/posthoc_fix_val.md`, `posthoc_fix_test_*.json`
- ensembles / hard gating / per-division table / narrow-test re-score -> EXP-0074 -> `results/eval_ens_*.json`, `gating_bal.json`, `divisions.md`, `eval_narrowtest_widetrained_*`
- member of MODEL-0025 (ens128)
- closed-loop planning model (ens/greedy/budget study; see EXP-0075 README/LOG for which arms used it) -> EXP-0075 (no RUN ids; results/ + LOG.md) (key `vanilla128` in code/bs_lib.py / closed_loop.py)
