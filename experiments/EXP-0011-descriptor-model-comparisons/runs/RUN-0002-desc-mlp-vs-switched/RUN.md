# RUN-0002-desc-mlp-vs-switched

Single small MLP over the 94-dim D-all-local push-frame descriptor basis
(identical basis weights/MODEL-0002-descriptor-only-D-all-local was fit on)
vs. that fitted switched-linear operator, on one-step descriptor_accuracy
and (separately, underpowered) slateN.

- Code: `experiments/EXP-0011-descriptor-model-comparisons/code/
  desc-mlp__{desc_features,metric,train_mlp,eval_control}.py` (originally
  `experiments/temp/desc-mlp/*.py`)
- Commit: 6ea03278 (dirty tree at run time)
- Command (reconstructed): `python -u experiments/temp/desc-mlp/train_mlp.py`
  then `python -u experiments/temp/desc-mlp/eval_control.py`
- Data: `Genesis/data/overnight_randlen_train` (all 5 groups, 192 files,
  98304 rows) / `overnight_randlen_test` (21 files, 10752 rows) for
  training/descriptor_accuracy; `Genesis/data/slates_binned/
  n20_scatter_s20a1000_L20-70mm` for the control-metric eval.
  `slates_multistep` was NOT loaded in this run (see EXPERIMENT.md
  Threats/incomplete-design) -- confirmed by inspection of
  `eval_control.py::main()`, which has no `slates_multistep` load call.
- Fitted object: `artifacts/RUN-0002-desc-mlp-vs-switched/mlp_checkpoint.pt`
  (state dict + config + train-set normalisation stats), plus cached
  descriptor tensors under `artifacts/RUN-0002-desc-mlp-vs-switched/cache/`.
- Output: `results/results_train.json` (tuning sweep + descriptor_accuracy),
  `results/results_control.json` (slateN, 9 cells x 4 models).
- Status: completed within its declared 75min/~150k token budget, but with
  the slates_multistep control cell out of scope for that budget (named as
  a gap, not hidden).
