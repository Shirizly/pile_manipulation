# RUN-0005-multistep-cv-nfd-lam0.7

Directly executes RUN-0004's own "What would change the verdict" fix: RUN-0004
found the fine-tuning-vs-untrained control benefit positive-signed but
unresolved at n=15-30 held-out slates (best cell `mass_in_region` lam=0.7,
+0.048±0.034, ~1.4×sem) and diagnosed the bottleneck as POWER, not effect
size — the 35/15 split only ever holds out 15 of the corpus's 50 slates.
This run replaces the single 35/15 split with **5-fold cross-validation over
all 50 slates** (seed 0, folds disjoint, asserted to exactly cover 0..49):
train on 40, test on the other 10, rotate so every slate is a test slate
exactly once. Pooling all 5 folds gives every one of the 50 slates × 2
datasets = **100 paired observations**, each scored by a model that never
saw that slate in training — instead of 30.

Lambda is FIXED at 0.7, not swept — RUN-0003 found the lambda sweep
unresolvable ({0.3,0.5,0.7,0.9} all within ~0.01 of each other); re-sweeping
5× as expensive under CV would not have changed that conclusion, so it was
not repeated. Same fine-tuning recipe as RUN-0003/RUN-0004 in every other
respect: full-image (unmasked) MSE, `L=λL1+λ²L2+λ³L3`, Adam, batch=256,
100 epochs, always initialised from `Baselines/NFD/runs/nfd_3ch/unet_best.pth`
(never from scratch), per fold.

**Result: the effect now clears 2σ for both discriminating goals.** Pooled
(n=100) paired per-slate capture difference (fine-tuned − untrained,
closed-loop): `mass_in_region` +0.0459±0.0180 (**+2.54σ**),
`signed_mass_in_region` +0.0341±0.0144 (**+2.37σ**). `lyapunov` (reference
row, carried forward per task instruction, not a discriminating goal since
it was already near ceiling untrained): +0.0162±0.0081 (+2.00σ, borderline,
consistent with the ceiling effect already diagnosed in RUN-0004). The
control-benefit half of C-012 moves from `narrowed` to `supported`.

Per-fold spread is real but does not change the sign anywhere: `mass_in_region`
per-dataset paired diffs are n20_L20mm +0.064±0.029 (n=50) and n20_L40mm
+0.028±0.022 (n=50) — both positive, n20_L20mm carries more of the pooled
effect. `signed_mass_in_region` is similarly split: n20_L20mm +0.065±0.024,
n20_L40mm +0.003±0.015 (essentially flat alone). This per-dataset
inconsistency is flagged (`inconsistency` downgrade), not smoothed over —
the pooled headline is real but not uniform across corpora.

Pooled (CV, every row scored by the fold that held its slate out) per-step
closed-loop accuracy: step1 0.443, step2 0.324, step3 0.243. This is HIGHER
than RUN-0003's untrained→trained step-3 range (+0.092→+0.209 to +0.216) —
expected and not a like-for-like improvement: CV folds train on 40 slates
each (80 rows more per dataset than RUN-0003's 35-slate/single-split train
set), so a same-recipe, larger-training-set model scoring somewhat higher
step-3 accuracy is consistent with more training data, not a different
finding about the objective itself.

- **Command**: see `COMMAND.txt`.
- **Commit**: `0ddab20f` (dirty tree — same pre-existing, unrelated
  docs/skills reorganisation as RUN-0001..RUN-0004).
- **Device**: single GPU (RTX 4070 Laptop 8GB), 1665s (~28min) wall clock
  total — 5 folds × 100 epochs × 4000 iters/fold (batch=256, 10240 train
  rows/fold across both datasets), ~332s/fold.
- **Data / split**: raw-file loader (`rollout.py::load_dataset`), same as
  every prior run in this record. NEW: 5-fold split (not RUN-0002/0003/0004's
  35/15), seed 0, `np.array_split` of a `permutation(50)`, folds asserted
  disjoint AND to exactly cover `range(50)`.
- **Degeneracy check**: frac(dv_true==0) per (dataset, value-fn): all
  ≤0.065 (`mass_in_region` n20_L20mm 0.065, n20_L40mm 0.008;
  `signed_mass_in_region` 0.026/0.003; `lyapunov` 0.006/0.0) — no cell
  excluded as degenerate.
- **Wins/losses/ties**: METRICS.md's cross-model-agreement definition
  (fine-tuned vs. untrained NFD, pooled n=100): `mass_in_region`
  40W/25L/35T, `signed_mass_in_region` 33W/26L/41T, `lyapunov` (reference)
  25W/13L/62T (mostly ties — expected near ceiling).
- **K=32 reference** (METRICS.md, k=32/reps=50): reported per-dataset in
  `artifacts/.../results_cv.json` (`k32_trained_mean`/`k32_untrained_mean`
  per value-fn/dataset); consistent in sign with the full-pool numbers
  above.
- **Status**: completed, no errors, no non-finite losses in any fold.
- **Outputs**: `artifacts/RUN-0005-multistep-cv-nfd-lam0.7/{results_cv.json,
  train_cv.log}`, `code/multistep-cv__train_eval_cv.py`,
  `results/RUN-0005-multistep-cv-nfd-lam0.7-RESULTS.md`. Per-fold checkpoints
  (`nfd_fold{0..4}_lam0.7.pth`) remain under
  `experiments/temp/multistep-cv/` (not promoted to `weights/` — these are
  5 fold-specific models, not a single reusable instance; RUN-0003's
  `nfd_lam0.9.pth`, already promoted to `weights/MODEL-0003-...`, remains
  the reusable single-split artifact).
