# RUN-0003 — hard push-length-gated switched-linear latent dynamics + diagnostics
- experiment: EXP-0019 · commit 6ea03278 (dirty)
- command: `python -u code/fit_switched_hard.py --latents artifacts/RUN-0002 --out artifacts/RUN-0003`
  then `python -u code/diag_lam_and_encoder.py artifacts/RUN-0002 ../EXP-0016-lejepa-random-encoder-floor/artifacts/RUN-0001 results/diag_lam_encoder.json`
- closed-form ridge, float64, CUDA; deterministic; status: completed
- outputs: `artifacts/RUN-0003/{operators.pt,switched_metrics.json}`, `results/diag_lam_encoder.json`
