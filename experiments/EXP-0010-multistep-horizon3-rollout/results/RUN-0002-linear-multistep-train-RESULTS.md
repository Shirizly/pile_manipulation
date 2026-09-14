# Multi-step (closed-loop rollout) gradient training of the linear operator

Code: `train_multistep.py` (+ `recompute_wlt.py`, a post-hoc bugfix pass, see
below). Raw output: `results_multistep_train.json`. Fitted operators:
`operator_{switched,global}_lam{0.3,0.5,0.7,0.9}.pt`. Run:
`python -u train_multistep.py`, 765s on GPU (150 Adam iters x 4 lambdas x 2
operator kinds), commit 0ddab20f (dirty tree; provenance block in the JSON).

## Setup

**Objective**: `L = lam*L1 + lam^2*L2 + lam^3*L3`, `Lk` = masked MSE
(`fit_linear_foresight.swept_region_mask` for step k's OWN action, applied
per step, not unioned) between the closed-loop rollout prediction at step k
and the true occupancy, gradients flowing through all 3 steps. Adam,
lr=2e-3, 150 iterations, full-batch (no minibatching).

**Initialisation**: the closed-form ridge-toward-identity solution,
`weights/MODEL-0001-stage2-visual-switched/checkpoint.pt` (6-bin switched
operator set + 1 global/unswitched operator, each 1024x1024, RES=32 push
frame). Both operator families are trained multi-step, independently, at
each lambda — 4 lambdas x 2 kinds = 8 training runs.

**Data / split**: raw `_{batch}_data.pt` files loaded directly (same
`load_dataset` as `multistep-rollout/rollout.py`; NOT the `*_eval` configs,
which apply `min_push_length_m` filtering that drops rows and breaks
row==env alignment across steps — inherited finding). `n20_L20mm` and
`n20_L40mm` only (`n20_L10mm` excluded per task instruction). Split by
`slate_idx`, seed 0, 35 train / 15 test, **same split index set applied to
both datasets** (both have slate_idx 0..49):

```
train (35): [0,1,2,3,4,6,8,10,11,16,17,18,19,20,21,22,23,24,25,26,27,28,30,32,34,35,36,37,38,43,44,45,46,47,48]
test  (15): [5,7,9,12,13,14,15,29,31,33,39,40,41,42,49]
```

Disjointness verified by assertion in `make_split` (`set(train) & set(test)
== {}`) — passes. Combined training set: 8960 rows (4480 n20_L20mm + 4480
n20_L40mm, 35 slates x 128 envs each). Held-out test: 1920+1920 rows (15
slates x 128 envs each), scored per dataset.

## Loss at initialisation vs. after training (train set, masked MSE)

| | L1 | L2 | L3 |
|---|---|---|---|
| **init (closed-form), switched** | 0.03331 | 0.03080 | 0.02941 |
| **init (closed-form), global** | 0.04279 | 0.04018 | 0.03916 |

After 150 iters, every lambda/kind combination drove L2 and L3 down
substantially (e.g. switched lam=0.9: L1 0.0333->0.0285 (+14%), L2
0.0308->0.0199 (-35%), L3 0.0294->0.0136 (-54%); global lam=0.9: L1
0.0428->0.0346, L2 0.0402->0.0260, L3 0.0392->0.0206). **The training
objective itself was optimised successfully and monotonically** at every
lambda — full loss traces in `results_multistep_train.json` /
`init_train_losses`. This makes what follows an honest optimisation
*succeeding on its stated objective* and still failing at the actual goal.

## Spectral radius / largest singular value, per bin, before vs after

| bin | init sr | init sv | lam=0.3 sr | lam=0.5 sr | lam=0.7 sr | lam=0.9 sr |
|---|---|---|---|---|---|---|
| switched bin0 | 1.030 | 1.79 | 1.030 (unchanged) | 1.030 | 1.030 | 1.030 |
| switched bin1 | 1.015 | 1.61 | **3.50** | **4.08** | **4.16** | **4.41** |
| switched bin2 | 1.017 | 2.36 | 1.017 (unchanged) | 1.017 | 1.017 | 1.017 |
| switched bin3 | 1.028 | 2.95 | **3.81** | **4.12** | **3.74** | **3.51** |
| switched bin4 | 1.026 | 3.04 | 1.026 (unchanged) | 1.026 | 1.026 | 1.026 |
| switched bin5 | 1.061 | 4.24 | 1.061 (unchanged) | 1.061 | 1.061 | 1.061 |
| global | 1.014 | 1.65 | **3.85** | **4.69** | **4.23** | **4.18** |

Bins 0/2/4/5 are byte-identical to init at every lambda — because
`n20_L20mm` and `n20_L40mm` are each single-push-length collections
(confirmed in the rollout report: L20mm rows sit almost entirely in bin 1,
L40mm in bin 3), so **those 4 bins receive zero training rows and zero
gradient**; only bins 1, 3, and the global operator ever see data here.
Every operator that *does* receive gradient develops a spectral radius
3.5-4.7x the identity-adjacent init (~1.0-1.06) — **multi-step training
makes the operators MUCH less stable, not more**, the opposite of the
hoped-for effect. Full singular values in the JSON.

## Held-out accuracy + terminal slateN, per lambda

Baselines for reference (established, full 50-slate closed-loop numbers):
model0001_switched +0.168/+0.110/+0.094, model0001_global +0.031/-0.065/-0.104
(diverges), nfd +0.351/+0.181/+0.099. This run's **15-slate test-subset**
closed-form numbers (below, "lam=0 (init)") match those closely, confirming
the split/harness reproduce the established baseline before any multi-step
training is applied.

**n20_L20mm** (step1/step2/step3 accuracy, then terminal lyapunov-capture,
mass_in_region-capture, mean±sem over 15 test slates):

| operator | lambda | acc(1,2,3) | lyapunov | mass_in_region |
|---|---|---|---|---|
| switched | 0 (init) | +0.165/+0.107/+0.090 | 0.937±0.017 | 0.899±0.033 |
| switched | 0.3 | +0.198/+0.055/-0.015 | -0.142±0.141 | -0.060±0.050 |
| switched | 0.5 | +0.200/+0.054/-0.036 | -0.392±0.150 | -0.079±0.059 |
| switched | 0.7 | +0.201/+0.061/-0.034 | -0.398±0.103 | -0.221±0.053 |
| switched | 0.9 | +0.201/+0.071/-0.023 | -0.664±0.153 | -0.280±0.051 |
| global | 0 (init) | +0.027/-0.068/-0.109 | 0.789±0.048 | 0.738±0.060 |
| global | 0.3 | +0.129/+0.001/-0.062 | -0.271±0.144 | +0.298±0.111 |
| global | 0.5 | +0.130/-0.004/-0.088 | -0.282±0.167 | +0.203±0.114 |
| global | 0.7 | +0.136/+0.016/-0.060 | -0.405±0.153 | -0.408±0.020 |
| global | 0.9 | +0.145/+0.044/-0.022 | -0.767±0.136 | -0.209±0.115 |

**n20_L40mm** (same layout):

| operator | lambda | acc(1,2,3) | lyapunov | mass_in_region |
|---|---|---|---|---|
| switched | 0 (init) | +0.410/+0.287/+0.166 | 0.996±0.002 | 0.876±0.023 |
| switched | 0.3 | +0.455/+0.258/+0.061 | -0.203±0.152 | -0.002±0.084 |
| switched | 0.5 | +0.455/+0.273/+0.074 | -0.576±0.115 | -0.000±0.113 |
| switched | 0.7 | +0.455/+0.296/+0.116 | -0.649±0.130 | -0.158±0.092 |
| switched | 0.9 | +0.454/+0.313/+0.150 | -0.531±0.167 | +0.011±0.107 |
| global | 0 (init) | +0.348/+0.235/+0.117 | 0.999±0.001 | 0.865±0.023 |
| global | 0.3 | +0.406/+0.243/+0.064 | -0.367±0.107 | -0.157±0.085 |
| global | 0.5 | +0.405/+0.243/+0.046 | -0.626±0.109 | -0.300±0.025 |
| global | 0.7 | +0.401/+0.257/+0.067 | -0.747±0.100 | -0.249±0.073 |
| global | 0.9 | +0.392/+0.263/+0.086 | -0.709±0.098 | -0.322±0.012 |

## Wins / losses / ties vs. the closed-form init (cross-model-agreement tie
definition, METRICS.md)

Per METRICS.md, ties = the two models (trained vs. init) pick the SAME
candidate row on a slate — not "matches the oracle" (the rollout run's
strict-argmax definition, flagged there as reading 0 everywhere and
explicitly not to be repeated). `wins`/`losses` then compare the two
models' CHOSEN rows' true terminal value directly.

**Note on process**: the first pass (`train_multistep.py`'s own
`wins_losses_ties`) had a sign bug for the COST metric (lyapunov): it
negated the two models' predictions to pick the argmin correctly, but then
compared the raw (non-negated) true values, inverting the win/loss sense
for lyapunov only (`mass_in_region`, a plain higher-is-better VALUE metric,
was unaffected and already correct). Caught by cross-checking against the
`lyap_mean`/`mir_mean` capture numbers above (which are correct — same
formula the rollout script used) — the naive win-rate did not agree with a
mean-capture number that had gone strongly negative. Fixed and recomputed
(no retraining needed) in `recompute_wlt.py`; both tables agree below.

| lambda | operator | dataset | lyapunov w/l/t (n=15) | mass_in_region w/l/t (n=15) |
|---|---|---|---|---|
| 0.3 | switched | L20mm | 0/15/0 | 0/15/0 |
| 0.3 | switched | L40mm | 0/15/0 | 0/15/0 |
| 0.3 | global | L20mm | 1/14/0 | 2/12/1 |
| 0.3 | global | L40mm | 0/15/0 | 0/15/0 |
| 0.5 | switched | L20mm | 0/15/0 | 0/15/0 |
| 0.5 | switched | L40mm | 0/15/0 | 1/14/0 |
| 0.5 | global | L20mm | 0/15/0 | 2/13/0 |
| 0.5 | global | L40mm | 0/15/0 | 0/15/0 |
| 0.7 | switched | L20mm | 0/15/0 | 0/15/0 |
| 0.7 | switched | L40mm | 0/15/0 | 0/15/0 |
| 0.7 | global | L20mm | 0/15/0 | 0/15/0 |
| 0.7 | global | L40mm | 0/15/0 | 0/15/0 |
| 0.9 | switched | L20mm | 0/15/0 | 0/15/0 |
| 0.9 | switched | L40mm | 0/15/0 | 0/15/0 |
| 0.9 | global | L20mm | 0/15/0 | 1/13/1 |
| 0.9 | global | L40mm | 0/15/0 | 0/15/0 |

The multi-step-trained operator **loses to the closed-form init on
essentially every held-out slate**, at every lambda, on both value
functions, for both operator kinds. Effective n stays the full 15 (no
argmax-agreement ties fired here) — this is not a power problem, it is a
clean, resolvable result.

## Answers

**Does multi-step training beat the closed-form one-step operator at
horizon 3?** No — decisively not. Step-3 image accuracy is worse than init
at every lambda for the switched operator on n20_L20mm (init +0.090 ->
trained -0.015 to -0.036) and worse-or-flat on n20_L40mm (init +0.166 ->
trained +0.061 to +0.150, i.e. still below init except lam=0.9 which gets
close but doesn't beat it). Terminal `slateN` is far worse: lyapunov
capture collapses from +0.94/+1.00 (init, L20/L40) to consistently
**negative** (-0.14 to -0.77) at every lambda — worse than random ranking,
not just worse than the oracle. This is well above any noise floor (sem
~0.10-0.17 against effect sizes of 1-1.7 in capture units, and the
win/loss table shows 0/15 or near-0/15 wins, not a marginal split).

**Does it fix the global operator's divergence?** No — it makes the global
operator's terminal ranking much worse while its raw step-3 IMAGE accuracy
looks superficially "fixed" (init -0.109 -> trained -0.022 to -0.088 on
L20mm, less negative). That apparent fix is illusory: the operator's
spectral radius roughly quadruples (1.01 -> 3.85-4.69) and its terminal
lyapunov capture goes from an already-mediocre +0.79 to -0.27 to -0.77 —
i.e. the *pixel-accuracy* symptom improved slightly while the *actual
control-relevant* failure (ranking candidate actions) got catastrophically
worse. Multi-step training as run here does not fix instability; it trades
one failure mode (compounding blur) for a worse one (unconstrained
spectral blowup in the only directions the masked multi-step loss touches).

**Which lambda is best, and is the difference resolvable?** Among
uniformly bad outcomes, the ordering is not flat: higher lambda tends
toward less-bad step-3 image accuracy (e.g. switched/L40mm step3: 0.061 ->
0.074 -> 0.116 -> 0.150 as lambda goes 0.3->0.9) but WORSE terminal
lyapunov capture in most cells (switched/L20mm lyapunov: -0.14 -> -0.39 ->
-0.40 -> -0.66 as lambda increases) — the two metrics disagree on which
lambda is "best", and every lambda loses to the lambda=0 (closed-form)
baseline on the metric that matters for control (terminal ranking). There
is no lambda in {0.3,0.5,0.7,0.9} that is competitive with the
initialisation, so "which lambda is best" has no practically meaningful
answer here beyond "less lambda is closer to the (still-losing) starting
point."

**Does multi-step training cost anything at step 1?** No — step 1 actually
IMPROVES slightly at every lambda for both operator kinds (switched/L20mm:
+0.165 -> +0.198/+0.200/+0.201/+0.201; global/L20mm: +0.027 -> +0.129 to
+0.145). The expected step1-for-step3 trade did not materialise in the
direction hoped; instead training bought a small step-1 gain and lost
catastrophically at step 3 and in terminal ranking.

## Interpretation — why, not just what

Bins 0/2/4/5 get literally zero gradient (both corpora are single-push-
length, so all training rows land in bins 1 and 3 — the same bin-coverage
gap the rollout report already flagged). Where gradient DOES reach an
operator (bins 1, 3, global), 150 unconstrained Adam iterations pull its
spectral radius from ~1.0 to 3.5-4.7 while still reducing the MASKED
training loss the task specified — the objective has no term that
penalises what the operator does to directions/pixels outside the swept
region mask, so it is free to develop large-gain behaviour there, and nothing
in the specified `L1/L2/L3` objective constrains it. The closed-form
operator being trained from IS regularised (ridge-toward-identity); the
multi-step gradient objective, exactly as specified, is not. This reads as
a straightforward, honest negative result about the SPECIFIED objective and
training recipe (unregularised Adam on masked multi-step MSE), not a claim
that multi-step training can never help — a regularised variant (e.g.
ridge-toward-the-closed-form-solution added to the loss, or a trust-region
step-size limit) was out of scope for this task's exact objective and
was not attempted.

## Deliverables

- `train_multistep.py` — training + evaluation harness (this experiment)
- `recompute_wlt.py` — post-hoc bugfix pass for the wins/losses/ties table
  (see note above); does not retrain, only re-scores saved operators
- `results_multistep_train.json`, `wlt_vs_init_fixed.json` — raw output
- `operator_switched_lam{0.3,0.5,0.7,0.9}.pt`, `operator_global_lam{0.3,0.5,0.7,0.9}.pt`
  — 8 fitted operators, each file self-documenting its config/provenance
  in a `note` field
