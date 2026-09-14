# Re-scoring terminal (horizon-3) regret under mass_in_region / signed_mass_in_region

Motivation: the multi-step terminal regret numbers in EXP-0010 (rollout.py,
train_nfd_multistep.py) were measured mostly under `lyapunov`, where they
are saturated (untrained NFD closed-loop already 0.967, 15/0/0 wins vs
init at every lambda) -- a 2.2x closed-loop step-3 `accuracy` improvement
from fine-tuning therefore produced no measurable terminal-slateN change.
This re-scores the SAME horizon-3 chains under `mass_in_region` and
`signed_mass_in_region` (both already defined in METRICS.md, reused
verbatim from `Baselines/common/goals.py`), restricted to the held-out
15-slate test split (seed 0, 35/15 of 50 slates) used to train the
multistep NFD/linear models -- so, unlike the original rollout.py run
(which pooled all 50 slates), every model here, including the untrained
baselines, is scored on identically the SAME pool.

Code: `rescore_regret.py`. Results: `results_regret_massgoals.json`.

## Split verification

`make_split(seed=0)` (reused from `train_nfd_multistep.py`) gives test
slates `[5, 7, 9, 12, 13, 14, 15, 29, 31, 33, 39, 40, 41, 42, 49]` out of
50 (0..49), asserted disjoint from train and asserted that both datasets'
`slate_idx` values are exactly `{0..49}` (so the index-based split lines
up with the actual slate identity in both corpora). 15 slates x 128 envs
= 1920 test rows per dataset.

## Degeneracy check: frac(dv_true == 0), per (dataset, value-fn)

`dv_true = V(occ3_true) - V(occ0)`, fraction of the 1920 test rows where
this is exactly (< 1e-9) zero:

| dataset | lyapunov | mass_in_region | signed_mass_in_region |
|---|---|---|---|
| n20_L20mm | 0.007 | 0.067 | 0.027 |
| n20_L40mm | 0.000 | 0.006 | 0.004 |

All well under any reasonable degeneracy threshold (e.g. EXP-0008's 0.90)
-- no cell is excluded as degenerate.

## Terminal slateN, closed-loop (headline), mean capture ± sem (n=15 slates), K=32 exact-form not used (MC, reps=50, without replacement)

**n20_L20mm**

| model | lyapunov | mass_in_region | signed_mass_in_region |
|---|---|---|---|
| model0001_switched | 0.937±0.017 (k32 0.954) | 0.899±0.033 (k32 0.930) | 0.868±0.040 (k32 0.935) |
| model0001_global | 0.789±0.048 (k32 0.830) | 0.738±0.060 (k32 0.747) | 0.736±0.053 (k32 0.772) |
| hybrid94 | 0.937±0.017 (k32 0.953) | 0.899±0.033 (k32 0.928) | 0.876±0.041 (k32 0.933) |
| nfd_untrained | **0.967±0.011** (k32 0.942) | **0.821±0.053** (k32 0.879) | 0.867±0.042 (k32 0.872) |
| nfd_lam0.3 | 0.961±0.014 | 0.876±0.040 | 0.941±0.026 |
| nfd_lam0.5 | 0.964±0.013 | 0.891±0.040 | 0.923±0.036 |
| nfd_lam0.7 | 0.973±0.012 | 0.912±0.035 | 0.912±0.036 |
| nfd_lam0.9 | 0.974±0.012 | 0.910±0.036 | 0.941±0.026 |
| persistence | -0.167±0.180 | -0.046±0.099 | -0.085±0.108 |
| random | 0.102±0.174 | 0.176±0.092 | 0.145±0.104 |

**n20_L40mm**

| model | lyapunov | mass_in_region | signed_mass_in_region |
|---|---|---|---|
| model0001_switched | 0.996±0.002 | 0.876±0.023 | 0.919±0.015 |
| model0001_global | 0.999±0.001 | 0.865±0.023 | 0.927±0.015 |
| hybrid94 | 0.994±0.003 | 0.883±0.016 | 0.922±0.014 |
| nfd_untrained | 0.935±0.034 | **0.856±0.026** | 0.918±0.018 |
| nfd_lam0.3 | 0.990±0.004 | 0.845±0.028 | 0.901±0.019 |
| nfd_lam0.5 | 0.988±0.005 | 0.842±0.025 | 0.903±0.018 |
| nfd_lam0.7 | 0.986±0.006 | 0.861±0.020 | 0.917±0.012 |
| nfd_lam0.9 | 0.989±0.005 | 0.859±0.026 | 0.918±0.019 |
| persistence | -0.329±0.191 | -0.088±0.099 | -0.093±0.107 |
| random | 0.173±0.165 | -0.123±0.088 | -0.097±0.091 |

Every trained/untrained model shows wins=15,losses=0,ties=0 at K=N against
the implicit random/mean reference (this is the well-known K=N determinism
noted in METRICS.md — not informative about model-vs-model differences by
itself). For MODEL-0002 (descriptor-only, teacher-forced only, cannot
close the loop across steps): mass_in_region 0.594±0.081 (L20mm) /
0.739±0.050 (L40mm); signed_mass_in_region 0.537±0.089 / 0.696±0.049 —
well below the image models, consistent with its coarse point-mass value
readout.

## Cross-model-agreement wins/losses/ties (METRICS.md definition, NOT the strict-argmax-match one)

vs. `persistence`, closed-loop, n=15 slates per dataset — every image
model (untrained AND fine-tuned) beats persistence 13-15/15 across every
value function and both datasets; `random` is much closer to persistence
(6-11/15). This just confirms all image models chain meaningfully better
than "nothing moved" — it does not resolve fine-tuning-vs-untrained (see
next section, which is the load-bearing comparison for this task).

## Does fine-tuning help control, above noise? (the load-bearing test)

Cross-model-agreement wlt (fine-tuned lambda vs. untrained NFD, closed-loop)
gives small, noisy integer counts at n=15 (e.g. n20_L20mm mass_in_region
lam=0.5: 7W/2L/6T) — not by itself a strong statement. The more sensitive
test is the **paired per-slate slateN-capture difference** (fine-tuned minus
untrained, same slate, same value fn):

**Pooled across both datasets (n=30 slates), closed-loop:**

| value fn | lam=0.3 | lam=0.5 | lam=0.7 | lam=0.9 |
|---|---|---|---|---|
| lyapunov | +0.024±0.020 | +0.025±0.020 | +0.029±0.020 | +0.031±0.019 |
| mass_in_region | +0.022±0.031 | +0.028±0.031 | **+0.048±0.034** | +0.046±0.034 |
| signed_mass_in_region | +0.029±0.024 | +0.021±0.020 | +0.023±0.021 | +0.037±0.024 |

Every point estimate is positive, but none clears 2×sem (best case,
mass_in_region lam=0.7: 0.048/0.034 ≈ 1.4 sem). Per-dataset the picture is
less consistent: n20_L20mm alone shows a real-looking mass_in_region gain
(+0.09±0.055, lam=0.7, ~1.6 sem) while n20_L40mm alone shows ~0 or slightly
negative (-0.01 to +0.006). The pooled positive trend is driven mostly by
n20_L20mm.

## Direct answers

- **Is untrained NFD off-ceiling under the mass goals?** Yes, clearly.
  `mass_in_region` untrained NFD closed-loop is 0.821 (L20mm) / 0.856
  (L40mm), well below `lyapunov`'s 0.967/0.935 — real headroom exists,
  confirming the premise that motivated this run.
- **Does multi-step fine-tuning now show a measurable control benefit, and
  is it above noise?** A small, consistently-positive-signed effect appears
  under both mass goals (and even under lyapunov, which is nominally
  saturated but still shows a small positive paired drift), but at n=15
  slates per dataset (n=30 pooled) it does **not** clear a 2-sem bar in any
  cell. This is a genuine null / power-limited result, not a clean win: the
  headroom recovered by using a more discriminating goal function did NOT,
  at this sample size, resolve into a statistically distinguishable control
  benefit from fine-tuning. More slates (a wider held-out pool) would be
  the natural next step to sharpen this.
- **Does the lambda ordering become resolvable?** No. Differences between
  adjacent lambdas are of the same order as (or smaller than) the
  fine-tuned-vs-untrained gap itself (~0.01-0.03 capture units, sem
  ~0.02-0.06) — not resolvable at this n, same conclusion as the original
  `lyapunov`-only runs.
- **Are `mass_in_region`/`signed_mass_in_region` more discriminating than
  `lyapunov` here?** Yes for revealing that untrained NFD is off-ceiling
  (0.82-0.86 vs 0.94-0.97) — that part of the premise holds. But the extra
  headroom did not translate into a resolvable fine-tuning benefit at this
  sample size; the dataset's fundamental limitation is less "value function
  saturation" and more "only 15-30 held-out slates" (effective-n problem
  METRICS.md already flags generically for slateN at K=N).

## Folding in

`experiments/EXP-0010-multistep-horizon3-rollout/` existed with a complete
`EXPERIMENT.md` and `python scripts/check_register.py` exited 0 (only
pre-existing warnings) at the time this run finished — folded in as
RUN-0004, see that record's `runs/RUN-0004-*/RUN.md` and the updated
`EXPERIMENT.md` "What would change the verdict" section (this run directly
executes the cheapest-check idea proposed there).
