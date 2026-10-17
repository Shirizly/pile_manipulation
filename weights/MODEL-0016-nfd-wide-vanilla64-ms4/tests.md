# MODEL-0016 test history
- wide Sean test (8 held-out shards, pasted 64x64 frame, no balance fix): acc1 0.534 / slateN 0.907 / roll1-4 0.525 0.457 0.418 0.401 -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_w64_s0_ms4.json`
- val-selected post-hoc mass balance (none vs balance, test) -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/posthoc_fix_val.md`, `posthoc_fix_test_*.json`
- closed-loop planning model (ens/greedy/budget study; see EXP-0075 README/LOG for which arms used it) -> EXP-0075 (no RUN ids; results/ + LOG.md) (key `vanilla64` in code/bs_lib.py / closed_loop.py)
