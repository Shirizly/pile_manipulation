# RUN-0006 — extended eval-only re-score on DS-0022 (this repo)

- what: `code/eval_extended.py` (per-state npz, atomic, manifest) then `code/summarize_extended.py`. Models: MODEL-0011 with action z sign +1 (as run), MODEL-0012 with -1 (corrected), `persistence`, `field` (s0 + corrected action field). Seeded FPS x3 reps; 17 point goals (goal 0 = the as-run goal); all-particle truth via nearest-node carry; mask goals random_quadrant/ring_O/T x lyapunov/mass_in_region/signed_mass on a 64x64 +-7.2 binary disk raster (row x, col -z); escaped rows flagged; K-equalised subsets (K = smallest clean pool).
- commit d72bb304, dirty; GPU (shared with RUN-0005 training).
- commands (exact argv in `artifacts/RUN-0006-eval-extended/*.json` / COMMAND.txt):
  - `python scripts/run_probe.py --tag exp0064_run0006_asrun ... -- python code/eval_extended.py --out artifacts/RUN-0006-eval-extended/asrun --fps-reps 3 --n-extra-goals 16`
  - `python scripts/run_probe.py --tag exp0064_run0006_fixed ... -- python code/eval_extended.py --out artifacts/RUN-0006-eval-extended/fixed --models gnn_fixed:-1:weights/MODEL-0012-*/checkpoint.pth --fps-reps 3 --n-extra-goals 16`
  - `python code/summarize_extended.py --dirs artifacts/RUN-0006-eval-extended/asrun artifacts/RUN-0006-eval-extended/fixed --out results/`
- output: `artifacts/RUN-0006-eval-extended/{asrun,fixed}/state_*.npz`; `results/extended_summary.{md,json}`.
