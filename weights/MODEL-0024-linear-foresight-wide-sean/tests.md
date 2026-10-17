# MODEL-0024 test history
- wide Sean test (8 shards, pasted 64x64 frame): acc1 0.539 / slateN 0.896 / roll1-4 0.538 0.461 0.423 0.410 -> EXP-0074 -> `experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_lf_wide.json`
- hard-gated / ensemble with NFD pair (never selected) -> EXP-0074 -> `results/gating_bal.json`, `eval_ens3_lf_gated.json`
- closed-loop planning model (ens/greedy/budget study; see EXP-0075 README/LOG for which arms used it) -> EXP-0075 (no RUN ids; results/ + LOG.md) (LF planner in code/bs_lib.py)
