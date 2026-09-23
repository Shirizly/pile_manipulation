# RUN-0001-desc-frame-compare

Raw-frame vs action-frame (push-frame-warped) analytic descriptors under a
shared switched-linear (6-bin) fit. Same descriptor formula
(`dmdc_baseline.occupancy_descriptors`, D=87) applied to two different
frames.

- Code: `experiments/EXP-0011-descriptor-model-comparisons/code/desc-frame-compare__run_experiment.py`
  (originally `experiments/temp/desc-frame-compare/run_experiment.py`)
- Commit: 6ea03278 (dirty tree at run time; uncommitted changes were this
  script itself and its outputs, not project code)
- Command (reconstructed from `experiments/temp/desc-frame-compare/run.log`,
  not recorded verbatim at run time):
  `python -u experiments/temp/desc-frame-compare/run_experiment.py`
- Data: `Genesis/data/overnight_randlen_{train,test}` (n20 groups only: 135
  train files/69120 rows, 15 test files/7680 rows), `Genesis/data/
  slates_multistep/n20_{L20mm,L40mm}`, `Genesis/data/slates_binned/
  n20_scatter_s20a1000_L20-70mm`.
- Fitted objects: `artifacts/RUN-0001-desc-frame-compare/operators/
  frame{A,B}_lam{1e-4,1e-3,1e-2}.pt` (6 bundles, each `A` (6x87x87),
  mu/sigma, bin_edges, config).
- Output: `results/results_desc_frame_compare.json` (every one of the 27
  control-metric cells, plus the 3x2 ridge-sweep accuracy table).
- Status: completed, over its declared 60min/~140k token budget (~180min/
  180k+ tokens spent) -- flagged mid-run by the coordinator, reported as-is
  per instruction (not a partial result).
- Log: `experiments/temp/desc-frame-compare/run.log`.
