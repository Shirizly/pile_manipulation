# RUN-0004 — file-disjoint grouped lambda selection + full lambda/encoder diagnostic table
- experiment: EXP-0020 · commit 6ea03278 (dirty)
- commands:
  `OMP_NUM_THREADS=4 python -u code/fit_switched_grouped_lam.py --latents artifacts/RUN-0002 --out artifacts/RUN-0004`
  then `OMP_NUM_THREADS=4 python -u ../EXP-0019-lejepa-encoder-pushlen-switched/code/diag_lam_and_encoder.py artifacts/RUN-0002 ../EXP-0016-lejepa-random-encoder-floor/artifacts/RUN-0001 results/diag_lam_encoder.json` (the latter UNCHANGED)
- `fit_switched_grouped_lam.py` is the only new code in this experiment: it imports ridge_fit/feats/r2/fit_bins/predict from `fit_switched_hard.py` and replaces ONLY the inner validation split with file-disjoint 5-fold folds (fold of row i = (i // 512) % 5; 512 = 98304/192 asserted against the file count).
- closed-form ridge, float64, CUDA; deterministic; status: completed.
- outputs: `artifacts/RUN-0004/{operators_grouped.pt,switched_metrics_grouped.json}`, `results/diag_lam_encoder.json`.
