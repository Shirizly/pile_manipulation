# 5-fold cross-validation of NFD multistep fine-tuning (lambda=0.7 fixed)

Motivation: RUN-0004 found the fine-tuning-vs-untrained control benefit
positive-signed in every value-fn/lambda cell but never clearing 2×sem at
n=15-30 held-out slates (best: `mass_in_region` lam=0.7, +0.048±0.034,
~1.4 sem) — diagnosed as a power problem (only 30 pooled slate observations
available from a single 35/15 split), not an effect-size problem. This run
fixes that by 5-fold cross-validating over all 50 slates × 2 datasets
(seed 0, folds disjoint, exactly cover 0..49): train on 40, test on the
other 10, rotate so every slate is a test slate exactly once. Lambda fixed
at 0.7 (not swept — RUN-0003's sweep was unresolvable, all lambdas within
~0.01 of each other).

Code: `train_eval_cv.py`. Results: `results_cv.json`, `train_cv.log`.

## Fold verification

Folds (seed=0): `[[2,3,4,18,20,23,24,26,34,36], [1,6,10,11,21,22,27,30,45,47],
[0,8,16,19,28,35,37,38,43,46], [9,12,13,17,25,32,40,42,44,48],
[5,7,14,15,29,31,33,39,41,49]]` — 5 folds of 10, asserted disjoint and to
exactly cover `range(50)`. Both datasets (`n20_L20mm`, `n20_L40mm`) use the
same slate-index-based fold assignment (both have `slate_idx` 0..49,
verified in-script).

## Headline: pooled paired per-slate capture difference (fine-tuned − untrained, closed-loop)

| value fn | pooled mean±sem (n=100) | sigma | wins/losses/ties | degeneracy (L20mm/L40mm) |
|---|---|---|---|---|
| `mass_in_region` | **+0.0459 ± 0.0180** | **+2.54σ** | 40/25/35 | 0.065 / 0.008 |
| `signed_mass_in_region` | **+0.0341 ± 0.0144** | **+2.37σ** | 33/26/41 | 0.026 / 0.003 |
| `lyapunov` (reference only) | +0.0162 ± 0.0081 | +2.00σ | 25/13/62 | 0.006 / 0.0 |

Both discriminating goals (`mass_in_region`, `signed_mass_in_region`) now
clear 2σ. `lyapunov` sits exactly at the 2σ boundary — consistent with its
already-diagnosed near-ceiling untrained baseline (RUN-0001/RUN-0004: 0.967
untrained, 15/0/0 wins vs init at every lambda) leaving it little headroom
to move even with 100 pooled slates; it is not one of the discriminating
goals per the task design and is carried as a reference row only.

## Per-dataset breakdown (paired diff, mean±sem, n=50 each)

| value fn | n20_L20mm | n20_L40mm |
|---|---|---|
| `mass_in_region` | +0.0642 ± 0.0286 | +0.0276 ± 0.0219 |
| `signed_mass_in_region` | +0.0653 ± 0.0239 | +0.0029 ± 0.0151 |
| `lyapunov` | +0.0084 ± 0.0055 | +0.0240 ± 0.0152 |

Both datasets are positive-signed for `mass_in_region`; for
`signed_mass_in_region`, `n20_L40mm` alone is not distinguishable from zero
(~0.2σ) while `n20_L20mm` alone is (~2.7σ) — the pooled result is real but
not uniform across corpora (flagged as `inconsistency`, not smoothed over).

## Pooled per-step closed-loop accuracy (every row scored by the fold that held its slate out)

| step | pooled accuracy |
|---|---|
| 1 | 0.443 |
| 2 | 0.324 |
| 3 | 0.243 |

RUN-0003 reference (single 35/15 split, untrained → trained): step3
+0.092 → +0.209 to +0.216. RUN-0005's step-3 number (0.243) is somewhat
higher than RUN-0003's trained range, but this is NOT a like-for-like
comparison of the objective: CV folds train on 40 slates (10240 rows across
both datasets) vs. RUN-0003's 35-slate single split (~8960 rows) — a
same-recipe model trained on ~14% more data scoring modestly higher is
consistent with the larger training set, not a different finding about the
multi-step objective.

## Per-fold training summary

All 5 folds trained without a non-finite loss; 4000 iters/fold (100 epochs
× 40 iters/epoch), ~332s/fold, final training loss 0.0097-0.0106 (tight
spread across folds — no fold diverged or under-trained relative to the
others).

## Bottom line

The power fix worked: pooling all 50 slates via 5-fold CV (n=100 vs.
RUN-0004's n=30) moved both discriminating goals from ~1.4σ to >2.3σ,
**resolving the control-benefit question positive**, not as a null. The
effect is modest in absolute size (paired capture gain ~0.03-0.05) and not
perfectly uniform across the two datasets, but it is no longer a
power-limited null — C-012 moves from `narrowed` to `supported` in
`REGISTER.md`.
