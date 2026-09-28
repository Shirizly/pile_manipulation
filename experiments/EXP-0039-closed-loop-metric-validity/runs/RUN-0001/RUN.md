# RUN-0001 — 192 closed-loop episodes

- Command: gpu_queue2.sh -> `scripts/run_probe.py --tag exp0039_closed_loop --exp EXP-0039 --out-dir experiments/EXP-0039-closed-loop-metric-validity/runs/RUN-0001 --threads 4 --timeout 54000 -- python experiments/EXP-0039-closed-loop-metric-validity/code/closed_loop.py` (defaults = addendum 3: 4 models x rank/gd/cem x 4 goals x 4 starts, 1.0 s, 8 steps, 64 candidates).
- 2026-09-24 12:49-15:37 (10065 s), exit 0; commit 3bae8cd7, dirty.
- Output: results/episodes.json (rewritten atomically after every step; 192/192 complete).
- Analysis: `CUDA_VISIBLE_DEVICES="" python experiments/EXP-0039-closed-loop-metric-validity/code/analyse.py` -> results/analysis.json; linear_switched_soft DS-0006 predictions cached in artifacts/RUN-0001/.
