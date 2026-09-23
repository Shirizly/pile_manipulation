# RUN-0003 — hard push-length-gated switched-linear latent dynamics (EXP-0019 method verbatim)
- experiment: EXP-0020 · commit 6ea03278 (dirty)
- command: `OMP_NUM_THREADS=4 python -u ../EXP-0019-lejepa-encoder-pushlen-switched/code/fit_switched_hard.py --latents artifacts/RUN-0002 --out artifacts/RUN-0003` (script UNCHANGED, row-random inner lambda split included, for apples-to-apples comparability with EXP-0019 RUN-0003)
- closed-form ridge, float64, CUDA; deterministic; status: completed.
- outputs: `artifacts/RUN-0003/{operators.pt,switched_metrics.json}`.
