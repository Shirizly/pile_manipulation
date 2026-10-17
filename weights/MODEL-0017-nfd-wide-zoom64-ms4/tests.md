# MODEL-0017 test history
- wide Sean test (8 held-out shards, pasted 64x64 frame, no balance fix): acc1 0.534 / slateN 0.896 / roll1-4 0.524 0.475 0.441 0.428 -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_z64_f8_ms4.json`
- val-selected post-hoc mass balance (none vs balance, test) -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/posthoc_fix_val.md`, `posthoc_fix_test_*.json`
- closed-loop planning model (ens/greedy/budget study; see EXP-0075 README/LOG for which arms used it) -> EXP-0075 (no RUN ids; results/ + LOG.md) (key `zoom64` in code/bs_lib.py / closed_loop.py)
