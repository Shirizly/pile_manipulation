---
# ---- identity -------------------------------------------------------------
id: EXP-0010
title: >
  NFD fine-tuned through a closed-loop 3-step rollout roughly doubles step-3
  accuracy with no step-1 trade-off; the identical objective on a masked
  linear operator collapses terminal slateN instead; the control benefit of
  the NFD result is unmeasured because slateN is already at ceiling untrained
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On slates_multistep (n20_L20mm + n20_L40mm, 128 independently-diverging
  real 3-step trajectories per slate, row index = stable trajectory
  identity), training a predictor through the closed-loop 3-step rollout
  objective L = lam*L1 + lam^2*L2 + lam^3*L3 (Lk = per-step MSE against the
  true occupancy, gradients through all 3 steps, init from the model's own
  single-step fit) improves held-out closed-loop step-3 image `accuracy` by
  more than +0.05 absolute over that same model's untrained/closed-form
  single-step baseline, for AT LEAST ONE of {linear switched/global operator
  with swept-region-masked MSE, NFD (~30k params) with full-image unmasked
  MSE}, lambda in {0.3,0.5,0.7,0.9}; and separately, whether any such
  accuracy gain also raises terminal slateN control capture, or instead
  collapses it, depends on the (predictor, masking) combination and on
  whether slateN was already near ceiling untrained on this corpus.

prediction:
  supports: >
    at least one (predictor, masking) combination shows held-out step-3
    closed-loop accuracy improve by >0.05 absolute over its own untrained/
    closed-form init, WITHOUT terminal slateN (lyapunov or mass_in_region
    capture) falling below the persistence/random floor baselines
  refutes: >
    every combination either fails to clear the +0.05 step-3 accuracy bar,
    or clears it only by also driving terminal slateN below the persistence/
    random floors (i.e. accuracy and control point the same direction, no
    combination is a clean win)
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true
  data_commit: unrecorded
  script: experiments/EXP-0010-multistep-horizon3-rollout/code/multistep-rollout__rollout.py
  data: ["Genesis/data/slates_multistep raw _{batch}_data.pt files, n20_L20mm + n20_L40mm (n20_L10mm excluded throughout, per prior EXP-0006/EXP-0008 contamination finding)"]
  code_path: experiments/EXP-0010-multistep-horizon3-rollout/code/
  seed: 0
  split: >
    RUN-0001: no split, full 50 slates x 128 envs x 3 steps, eval only.
    RUN-0002/RUN-0003: 35 train / 15 test slates, seed 0, the SAME split
    index set applied to both n20_L20mm and n20_L40mm (both have slate_idx
    0..49); disjointness asserted in-script.
  runtime: >
    RUN-0001 ~5s GPU. RUN-0002 765s GPU (150 Adam iters x 4 lambdas x 2
    operator kinds). RUN-0003 ~26min GPU (100 epochs x 4 lambdas,
    batch=256, ~5min/lambda, ~3.1GB observed). RUN-0004 ~10s GPU
    (inference-only re-scoring, no retraining — reuses RUN-0001's/
    RUN-0003's models/checkpoints as-is). RUN-0005 ~28min GPU (5-fold CV,
    100 epochs/fold, lambda FIXED at 0.7, ~332s/fold).
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005]

budget:
  declared: >
    Not pre-declared per-run — all three runs were ad-hoc `experiments/temp/`
    exploration (see experiments/TEMP_LOG.md) before this promotion. This
    promotion task itself was budgeted 60 min wall-clock / ~140k tokens.
  spent: >
    Original runs: ~5s + 765s + ~26min GPU wall-clock (~31min total), across
    3 separate sessions on 2026-09-13. Promotion (this record): well within
    the 60min/140k budget. RUN-0004 (separate 2026-09-14 task, 60min/130k
    budget): ~10s GPU wall-clock (inference-only), well within budget;
    folded into this record at task completion. RUN-0005 (separate
    2026-09-14 task, 2.5h budget): ~28min GPU wall-clock (5-fold CV
    training), well within budget.
  outcome: within

design:
  varied:
    predictor: [linear_switched, linear_global, nfd]
    loss_mask: [swept-region-masked (linear), full-image-unmasked (NFD)]
    lambda: [0.3, 0.5, 0.7, 0.9]
  held_fixed:
    objective_form: "L = lam*L1 + lam^2*L2 + lam^3*L3, gradients through all 3 rollout steps"
    data: [n20_L20mm, n20_L40mm]
    horizon: 3
    init: "each model's own pre-existing single-step fit (closed-form ridge-toward-identity for the linear operators; single-step-trained UNet checkpoint for NFD)"
    goal: "corner (terminal slateN scoring)"
    value_fn (RUN-0004 only): [lyapunov, mass_in_region, signed_mass_in_region]
  held_fixed (RUN-0005 only): >
    lambda FIXED at 0.7 (not re-swept under CV — RUN-0003's sweep was
    unresolvable, all lambdas within ~0.01); split changed from RUN-0002/
    0003/0004's single 35/15 to 5-fold CV over all 50 slates (seed 0, folds
    disjoint, cover 0..49), rest of the recipe (full-image unmasked MSE,
    Adam, batch=256, 100 epochs/fold) identical to RUN-0003.
  baselines: [persistence, random, "own untrained/closed-form single-step init (lambda=0 reference row)"]
  metric: "accuracy"

noise_floor: >
  Run-to-run variation on the SAME quantity (untrained NFD closed-loop
  step1/2/3 accuracy, n20_L20mm) measured independently in RUN-0001
  (+0.351/+0.181/+0.099) vs RUN-0003 (+0.344/+0.174/+0.092): ~0.007
  absolute spread — this is the noise floor for any accuracy comparison
  in this record, and is the same order of magnitude as the NFD-vs-
  switched teacher-forced gap reported in EXP-0009, which therefore
  remains unresolved by that record's single-seed comparison.

depends_on: [push-frame-warp-roundtrip, occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x]
establishes: [slates-multistep-row-is-trajectory-identity, slates-multistep-single-pushlen-bin-starvation]

# ---- outcome --------------------------------------------------------------
result: >
  NFD/unmasked: step-3 closed-loop accuracy +0.092 (untrained) -> +0.209 to
  +0.216 (all 4 lambdas, ~2.2x), step-1 and teacher-forced ALSO improve
  slightly (no trade); terminal slateN untrained was ALREADY at ceiling
  (lyapunov 0.967, 15/0/0 wins vs init at every lambda) so it stays flat —
  supports the claim via the NFD/unmasked combination, but the control axis
  is unmeasured (saturated), not confirmed. Linear/masked: training loss
  fell monotonically but held-out terminal slateN COLLAPSED from +0.937 to
  negative at every lambda (0/15 wins), spectral radius 1.01->3.85-4.69 —
  this combination clears no accuracy bar cleanly (step-3 image accuracy
  also got WORSE at every lambda for the switched operator) and independently
  fails the "no slateN collapse" condition. Lambda is not resolvable in
  either run (differences ~0.01, single seed per lambda).
  RUN-0004 (re-scoring, no retraining): confirms untrained NFD IS off-ceiling
  under mass_in_region (0.821/0.856 vs lyapunov's 0.967/0.935 on the SAME
  held-out 15-slate split used to train), so the "saturated, can't measure"
  caveat above is now testable — but the extra headroom does NOT resolve
  into a statistically distinguishable control benefit: paired per-slate
  capture difference (fine-tuned - untrained, closed-loop, pooled n=30
  across both datasets) is positive-signed in every lambda/value-fn cell
  but never clears 2×sem (best: mass_in_region lam=0.7, +0.048±0.034,
  ~1.4 sem), and per-dataset the signal is inconsistent (n20_L20mm alone
  ~1.6 sem, n20_L40mm alone ~0). Genuine null / power-limited at n=15-30
  slates, not a clean resolution either way — the control axis is now
  MEASURED (not saturated) but still UNRESOLVED (not enough slates).
  RUN-0005 (5-fold CV over all 50 slates, lambda FIXED at 0.7, no re-sweep):
  fixes the power problem directly — pooling all 5 folds gives n=100 paired
  slate observations (vs. RUN-0004's n=30). Both discriminating goals now
  clear 2 sigma: mass_in_region +0.0459±0.0180 (+2.54 sigma),
  signed_mass_in_region +0.0341±0.0144 (+2.37 sigma); lyapunov (reference
  row only, near ceiling) +0.0162±0.0081 (+2.00 sigma, borderline as
  expected from its lack of headroom). Per-dataset the effect is
  positive-signed for mass_in_region in both corpora, but for
  signed_mass_in_region n20_L40mm alone is flat (~0.2 sigma) while
  n20_L20mm alone is strong (~2.7 sigma) — the pooled result is real but
  not uniform across datasets. Pooled per-step closed-loop accuracy (every
  row scored by the fold that held it out): +0.443/+0.324/+0.243 — step3
  somewhat above RUN-0003's trained range (+0.209 to +0.216) because CV
  folds train on 40 slates vs. RUN-0003's 35, not a different finding
  about the objective. The control-benefit half of C-012 is now resolved
  positive, not a null: this record's own "what would change the verdict"
  fix (more held-out slates via CV) worked.
verdict: supported
downgrades: [incomplete-design, indirectness, provenance, untested-dependency, inconsistency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If closed-loop-rollout training could not help ANY predictor/masking
combination, every trained variant would be within noise of (or worse than)
its own untrained init on held-out step-3 accuracy — that is what the linear/
masked run shows. NFD/unmasked instead shows a large (~2.2x), monotonic,
noise-floor-clearing improvement with no step-1 cost, which is the signature
the "supports" branch names. Whether that improvement is also a CONTROL
improvement is a separate, harder question this design can only partially
answer here: `slates_multistep`'s terminal slateN is already near-ceiling
untrained (NFD lyapunov 0.967, 15/0/0 wins over the untrained init at every
lambda), so a ranking metric with no headroom left cannot discriminate
"control got better" from "control was already maxed out" — this is stated
plainly as a limitation, not glossed over.

## What was actually run

Five runs. The first three were promoted together from `experiments/temp/`
because they test one question (does training through the closed-loop
rollout objective help, and for which predictor/masking choice) using a
shared dataset, split convention, and objective form; RUN-0004 and RUN-0005 were added in later sessions specifically to close
the control-axis gap RUN-0001/RUN-0003 left open (see "What would change
the verdict", below — now fully answered). **Nothing was re-run or
recomputed for the original promotion — all numbers in "Numbers" below for
RUN-0001/0002/0003 are taken verbatim from each run's own `RESULTS.md` (now
the source `results/RESULTS.md` files under this record), which remain
authoritative for their own detail. RUN-0004 and RUN-0005 ARE newly
computed (RUN-0004: re-scoring existing checkpoints under new value
functions, no retraining; RUN-0005: new 5-fold-CV training, lambda fixed
at 0.7).**

- **RUN-0001** (`code/multistep-rollout__rollout.py`): closed-loop 3-step
  rollout evaluation of 4 EXISTING models (no training) —
  `model0001_switched`, `model0001_global`, `hybrid94`, `nfd` — plus
  MODEL-0002 (descriptor-only) teacher-forced only. Established: (a) the
  dataset's row-index-is-trajectory-identity property (see `establishes`),
  (b) that both corpora are single-push-length, so a 6-bin switched
  operator sees ~1 bin's worth of variation across the whole rollout (see
  `establishes`), (c) the closed-loop-vs-teacher-forced compounding
  baseline every other run here is read against.
- **RUN-0002** (`code/multistep-train__train_multistep.py` +
  `code/multistep-train__recompute_wlt.py`): gradient-trained the linear
  operator (switched AND global) through the SAME closed-loop objective,
  using **swept-region-masked** MSE per step (the natural choice for a
  linear pixel-space operator, since that is what its closed-form fit
  already targets), lambda swept, init from `MODEL-0001`'s closed-form
  solution. `recompute_wlt.py` is a post-hoc, no-retraining bugfix pass:
  the first win/loss/tie pass had a sign error for the lyapunov (cost)
  metric, caught by cross-checking against the independently-computed
  mean-capture numbers, and was corrected without retraining.
- **RUN-0003** (`code/multistep-nfd__train_nfd_multistep.py`): the SAME
  objective form, same data/split convention, applied to NFD (a ~30k-param
  3-channel UNet) using **full-image unmasked** MSE per step — deliberately
  the opposite masking choice from RUN-0002, motivated by RUN-0002's own
  diagnosis (see below) that the masked objective under-constrains the
  operator outside the swept region.
- **RUN-0004** (`code/multistep-regret__rescore_regret.py`, no retraining):
  re-scores RUN-0001's 4 image models and RUN-0003's 4 fine-tuned NFD
  checkpoints under `mass_in_region`/`signed_mass_in_region` (in addition
  to `lyapunov`), on the SAME held-out 15-slate test split used to train
  RUN-0002/RUN-0003 (unlike RUN-0001, which pooled all 50 slates —
  RUN-0004 restricts every model, including the untrained baselines, to
  the held-out pool for a fair comparison). Confirms untrained NFD is
  off-ceiling under the mass goals (0.821-0.856 vs. lyapunov's 0.935-0.967)
  but finds the resulting fine-tuning-vs-untrained control gain does not
  clear 2×sem at n=15-30 slates in any value-fn/lambda cell — a genuine
  null/power-limited result, not a resolution.
- **RUN-0005** (`code/multistep-cv__train_eval_cv.py`): fixes RUN-0004's
  power problem directly by 5-fold cross-validating over all 50 slates
  (lambda fixed at 0.7, not re-swept). Pooling all 5 folds gives n=100
  paired slate observations instead of RUN-0004's n=30; both discriminating
  goals (`mass_in_region`, `signed_mass_in_region`) now clear 2σ — the
  control-benefit half of C-012 is resolved positive, not a null.

**Scoping decisions carried over from the source runs, stated once here
rather than three times:**
- `n20_L10mm` excluded from all three runs (prior EXP-0006/EXP-0008
  contamination finding for this dataset).
- MODEL-0002 (descriptor-only) closed-loop rollout **out of scope** for all
  three runs — see Threats/`incomplete-design` for why (structural, not a
  time-budget omission).
- GNN/SchenckCNN excluded from RUN-0001 pending EXP-0009 (that record now
  exists; not re-run here).
- RUN-0001's terminal-slateN "ties" column uses a strict argmax-match
  definition (reads 0 everywhere at float precision) rather than
  METRICS.md's cross-model-agreement definition; RUN-0002 and RUN-0003 use
  the correct METRICS.md win/loss/tie definition. **This is a mixed
  convention across runs within the same record** — flagged here and in
  Threats (`provenance`), not silently reconciled, since RUN-0001's ties
  column is not load-bearing for its own headline (mean±sem capture is) but
  would be misleading if compared directly against RUN-0002/RUN-0003's
  win/loss counts.
- RUN-0002/RUN-0003 lambda sweeps are single-seed per lambda; differences
  between adjacent lambdas (~0.01) are explicitly flagged as not resolvable
  at this budget in both source RESULTS.md files.

## Numbers

**Closed-loop step-3 accuracy, untrained/init vs. best available trained
variant, n20_L20mm** (the comparison the claim turns on):

| predictor | masking | untrained/init step3 | trained step3 (lambda range) | delta |
|---|---|---|---|---|
| nfd | full-image unmasked | +0.092 (RUN-0003) / +0.099 (RUN-0001, independent measurement) | +0.209 to +0.216 (all 4 lambdas) | +0.117 to +0.124 |
| linear switched | swept-region masked | +0.090 (RUN-0002 15-slate subset) / +0.094 (RUN-0001, full 50) | -0.036 to +0.116 across (lambda x dataset); on n20_L20mm specifically: -0.015 to -0.023 (WORSE) | negative on L20mm |
| linear global | swept-region masked | +0.027 (RUN-0002) / -0.104 (RUN-0001 CL, already diverging) | -0.022 to -0.062 on L20mm (still negative) | small nominal gain over an already-bad baseline, still net negative |

**Terminal slateN (lyapunov capture, mean, held-out), untrained/init vs.
trained:**

| predictor | untrained/init | trained (lambda range) | wins vs. init (of 15) |
|---|---|---|---|
| nfd | 0.967 | 0.961-0.974 (flat, within noise of ceiling) | not separately tabulated — already saturated |
| linear switched | 0.937 (RUN-0002 15-slate) | -0.14 to -0.66 (COLLAPSED) | 0/15 at every lambda |
| linear global | 0.789 (RUN-0002 15-slate) | -0.27 to -0.77 (COLLAPSED) | 0/15 to 1/15 at every lambda |

Full sweep (both datasets, both value functions, all lambdas, spectral
radii per bin) is in each run's own `results/RESULTS.md` — reproduced in
full there, not cherry-picked here.

**RUN-0004: terminal slateN under `mass_in_region`, closed-loop, held-out
15-slate split (untrained NFD vs. best-lambda fine-tuned NFD, mean±sem):**

| dataset | untrained NFD | best fine-tuned lambda | paired capture diff (pooled n=30, lam=0.7) |
|---|---|---|---|
| n20_L20mm | 0.821±0.053 | 0.912±0.035 (lam=0.7) | +0.048±0.034 (~1.4×sem, pooled both datasets) |
| n20_L40mm | 0.856±0.026 | 0.861±0.020 (lam=0.7) | not significant per-dataset either (n20_L40mm alone ≈0) |

(`signed_mass_in_region` and `lyapunov` pooled paired diffs are similarly
positive-signed but sub-2-sem at every lambda; full table in
`results/RUN-0004-regret-massgoals-rescoring-RESULTS.md`.)

**RUN-0005: 5-fold CV over all 50 slates (lambda fixed at 0.7), pooled paired
capture diff (fine-tuned − untrained, closed-loop, mean±sem, n=100):**

| value fn | pooled diff | sigma | wins/losses/ties |
|---|---|---|---|
| `mass_in_region` | +0.0459±0.0180 | +2.54σ | 40/25/35 |
| `signed_mass_in_region` | +0.0341±0.0144 | +2.37σ | 33/26/41 |
| `lyapunov` (reference only) | +0.0162±0.0081 | +2.00σ | 25/13/62 |

Full per-fold and per-dataset breakdown in
`results/RUN-0005-multistep-cv-nfd-lam0.7-RESULTS.md`.

## What would change the verdict

**Addressed by RUN-0005.** RUN-0004 found a value function with real
headroom (`mass_in_region`) and a consistently positive-signed effect, but
at n=15 slates per dataset (n=30 pooled) it was not distinguishable from
noise, and diagnosed the fix as more held-out slates rather than a
different value function. RUN-0005 applied that fix directly: 5-fold CV
over all 50 slates (instead of a single 35/15 split) triples the pooled
paired-observation count to n=100, and both discriminating goals
(`mass_in_region` +2.54σ, `signed_mass_in_region` +2.37σ) now clear 2σ.
The control-benefit half of C-012 is resolved positive — this record no
longer has an open "what would change the verdict" item on this axis.

The specific gap this record could not close before RUN-0004: **a slates_multistep-like corpus
where the untrained model is NOT already at slateN ceiling**, so a control
metric with headroom could actually detect whether NFD's accuracy gain is
also a ranking gain. Cheapest check: re-run RUN-0003's exact fine-tuned
checkpoints (already saved, `nfd_lam0.9.pth` etc.) against the harder,
wider-goal corpus already built for EXP-0008 (`slaten-broad`'s overnight
holdout, ring/T/quadrant shapes) — no new training needed, ~10-20 min of
eval. If slateN there is not already 15/0/0 for untrained NFD, this would
directly resolve whether the accuracy gain transfers to control. (This
remains a separate, un-run cheap check — RUN-0005 resolved the power
problem on `slates_multistep` itself, not this out-of-corpus question.)

A regularised variant of the masked linear objective (e.g. a ridge-toward-
closed-form penalty, or a trust-region step-size cap) was explicitly out of
scope for RUN-0002 and was not attempted — RUN-0002 refutes the objective
AS SPECIFIED (unconstrained masked MSE), not multi-step training on linear
operators in general; a regularised re-run is the natural next cell and is
NOT run here.

## Threats

- Considered and downgraded away: **imprecision** was carried through
  RUN-0004 (fine-tuning-vs-untrained control benefit under `mass_in_region`/
  `signed_mass_in_region` positive-signed but not clearing 2×sem at n=30
  pooled) but is DROPPED as of RUN-0005 — the fix RUN-0004 itself proposed
  (more held-out slates via cross-validation, not a bigger effect) was
  applied, and the pooled effect (n=100) now clears 2σ for both
  discriminating goals (mass_in_region +2.54σ, signed_mass_in_region
  +2.37σ). The lambda-sweep imprecision (~0.01 between adjacent lambdas,
  single seed per lambda, RUN-0002/RUN-0003) is a separate, narrower
  sub-question — RUN-0005 did not re-sweep lambda (fixed at 0.7,
  deliberately, per its own design) so it neither confirms nor further
  tests that specific imprecision; it is not restated as a record-level
  downgrade because it was never load-bearing for the record's own
  Numbers (which report the full sweep, not a single winning lambda).
- **inconsistency** (NEW as of RUN-0005): the pooled n=100 control-benefit
  effect is real but not uniform across the two datasets — for
  `signed_mass_in_region`, `n20_L20mm` alone is ~2.7σ but `n20_L40mm` alone
  is ~0.2σ (flat); `mass_in_region` is positive-signed in both datasets but
  with a 2-3x difference in magnitude (+0.064 vs +0.028). The pooled
  headline is not an artifact of pooling opposite-signed effects (both
  datasets are non-negative everywhere), but claiming it holds equally
  in both corpora would overstate the evidence.
- **incomplete-design**: MODEL-0002 (descriptor-only) closed-loop rollout
  was descoped structurally, not for budget reasons — its 94-dim descriptor
  is built in the CURRENT step's push-frame (action-relative), and no
  transform exists in this repo to re-express its shape terms (moments2,
  band_mass, dft) under an arbitrary SE(2) reframing between steps with
  differing actions (only the COM, a point, has an existing reframing
  primitive). GNN/SchenckCNN were also excluded from the rollout/training
  runs (only benchmarked for timing in EXP-0009, not for multi-step
  training here). Both corpora used are single-push-length, so 4 of the
  switched operator's 6 bins receive zero training rows in RUN-0002 —
  confirmed byte-identical to init — meaning this record cannot speak to
  whether multi-step training helps a genuinely bin-diverse (`randlen`)
  corpus at all.
- **indirectness**: the metric that clearly improved (NFD `accuracy`) is a
  proxy for the metric that decides (`slateN`); per the project's standing
  rule, `accuracy` comparisons across model TYPES or across this much
  architectural change are suspect on their own, and here the control
  metric could not even be exercised (ceiling). The headline is stated as
  an accuracy result with an unmeasured control implication, not inflated
  into a control claim.
- **provenance**: RUN-0001 used a strict argmax-match tie definition for its
  terminal-slateN "ties" column (reads 0 everywhere, effectively unused);
  RUN-0002/RUN-0003 used METRICS.md's cross-model-agreement win/loss/tie
  definition. This is a genuine mixed-definition situation across runs
  filed under the same record — RUN-0001's ties column is not used for any
  claim in this record's "Numbers" section (its mean±sem capture numbers
  are used instead, which do not depend on the tie definition), so nothing
  above is computed from the mismatched definition, but the mismatch itself
  is real and is why this downgrade is claimed rather than silently
  reconciled.
- **untested-dependency**: `depends_on` cites `occ-rasteriser-consistency`
  and `goal-mask-axis-convention-row-y-col-x`, both `unchecked` in
  `INVARIANTS.md` (believed true, no automated test asserts either). This
  record does not establish those tags itself, so it takes the downgrade
  rather than assuming they hold.
- Considered and dismissed: **selection** — every lambda in the swept set
  is reported for both RUN-0002 and RUN-0003 (full sweep in "Numbers" and
  in each run's own results), not the best cell alone.
- Note: for RUN-0001/0002/0003, **inconsistency** was considered and
  dismissed — the linear/masked collapse is consistent across every lambda
  and both operator kinds (switched, global) and both datasets; the
  NFD/unmasked accuracy improvement is consistent across every lambda and
  both datasets. RUN-0005 is where this domain becomes live for the
  record (see above) — its per-dataset spread on the control-benefit
  question, not the earlier accuracy results, is what earns the downgrade.

## Unrelated findings

- RUN-0001 found `model0001_global` (the unswitched operator) diverges to
  net-negative closed-loop image accuracy by step 3 on n20_L20mm even
  WITHOUT any multi-step training (-0.104, worse than persistence) — a
  pre-existing property of the untrained global operator under closed-loop
  feeding, not something RUN-0002's training caused (RUN-0002 made its
  terminal ranking much worse but its raw step-3 accuracy look
  superficially less bad, -0.109 -> -0.022 to -0.088 — flagged in RUN-0002's
  own RESULTS.md as an illusory fix: the operator's spectral radius roughly
  quadruples in the process).
- RUN-0002's first win/loss/tie pass had a sign bug for the lyapunov (cost)
  metric specifically (mass_in_region, a plain value metric, was
  unaffected) — caught by cross-referencing against the independently
  computed mean-capture numbers rather than trusted on its own; see
  `code/multistep-train__recompute_wlt.py`. Anyone reusing
  `train_multistep.py`'s own `wins_losses_ties` function directly (not
  through the recompute pass) for a COST-sense value function should be
  aware the original function has this bug.
- NFD's tiny parameter count (~30k) meant batch=256 fit comfortably in
  ~3.1GB — no gradient checkpointing or batch-size compromise was needed
  for RUN-0003, unlike some other NFD experiments in this repo that note
  GPU memory pressure.
