# RUN-0014 -- input-resampling control

Command: `COMMAND.txt`. Stdout: `stdout.log`. Raw JSON:
`artifacts/RUN-0014-input-resampling-control/input_resampling_control.json`.
Follow-up fidelity check: `fidelity_64_96.log`,
`artifacts/RUN-0014-input-resampling-control/fidelity_64_vs_96.json`.

Extended `experiments/EXP-0022-warped-nfd-push-frame/code/wall_proximity_stratified_eval.py`
with a `--degraded-baseline` flag (`run_predictor_degraded_input`,
`bootstrap_delta3`, `strat_table3`) rather than forking a new script. CPU
only (asserted in-script); GPU job RUN-0010 was left alone. No training.

Ran on `randlen_test` (10751 transitions), 5 quantile length bins (same
edges as RUN-0012/0013 by construction -- frame-independent, same corpus,
same `--n-bins 5`). Clean-input baseline reproduced the published 0.4564
world-frame accuracy exactly (mandatory reproduction check, passed).

Verdict and full tables: `results/input_resampling_control.md`. Short
version: input resampling reproduces the DIRECTION of the length-dependent
shape (monotonic shrink with push length) but not its magnitude (overshoots
the warped model's actual deficit 1.6x-5.7x, growing with length) or its
steepness (its own delta only falls to 44% of its bin-0 value by the longest
bin, vs the warped model's 12%) -- a partial, not a full, confirmation.
Model-free canon_res=64-vs-96 identity round-trip fidelity: 96 removes ~28%
of the round-trip RMS degradation relative to 64 (both full-image and
swept-region measures agree).
