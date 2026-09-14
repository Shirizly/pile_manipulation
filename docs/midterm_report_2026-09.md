# Mid-term report — model comparison for pile manipulation

**Date:** 2026-09-14 · **Evidence:** EXP-0004 … EXP-0010 · **Status of all records:** T1,
`mode: exploratory`, grade `very-low` (see *Global limitations*).

Numbers are quoted only where a claim is meaningless without them; each is tied to the
record that owns it, so an invalidation there propagates here.

---

## 1. Under a real time budget, NFD and the switched-linear visual operator are the
## best controllers; GNN and SchenckCNN are not competitive

**How.** Eight model configurations were timed end-to-end on one GPU (warm-up,
`cuda.synchronize()`, median+IQR), counting every per-candidate action-dependent cost
(plate rasterisation, push-frame warp, descriptor computation, goal-value evaluation) but
not the one-time state rasterisation. Each model was then allowed only as many of a slate's
128 candidate actions as it could evaluate within a fixed wall-clock budget, and scored by
`slateN` normalised against the **full** 128-candidate oracle, so a slower model is
penalised for the actions it never saw. Budgets: 2.5 / 5 / 10 ms.

**What.** At 5 ms and above, NFD scores 0.979 and the switched-linear visual operator 0.972,
both ranking the full pool. GNN (0.713) and SchenckCNN (0.863) are squeezed to 8–24
candidates and never recover at any budget tested. The cheapest model is not the best
controller: once the budget is loose enough for everyone to reach 128 candidates, per-candidate
prediction quality decides, and the unswitched operator trails at 0.952.

**Limitations.** Scored on the `corner` goal shape only. Measurement precision is ~5 %
relative, so the NFD-vs-switched gap (0.007) is **not resolved**. Model prediction code paths
are not unified (`provenance` downgrade). At 2.5 ms several models cannot evaluate even one
candidate, which reflects fixed overhead rather than throughput. — **EXP-0009**

---

## 2. Analytic descriptors add nothing on top of a full-resolution visual state

**How.** Switched-linear operators were fit on a 32×32 canonical visual state, on that state
concatenated with 14- and 94-dimensional push-frame descriptor sets, and on descriptors alone;
compared in image space and then as action rankers on real same-state slates.

**What.** In image space the three visual variants are separated by 0.0016 — noise. As action
rankers they are **identical**: across 60 slates the plain visual operator and both hybrids
pick the same action every time, zero disagreements. The descriptors are computed from the same
occupancy grid the visual block already sees in full. Under a time budget the hybrids are
strictly worse, because the descriptor computation costs real milliseconds for no gain.

**Limitations.** Established for linear operators on this state representation; it does not
follow that descriptors are useless to a model with a lossy state (see §3). — **EXP-0005,
EXP-0006, EXP-0009**

---

## 3. Descriptors help in inverse proportion to what the state already carries

**How.** The same descriptor sets were appended to a 64-dimensional learned encoder latent
instead of to raw pixels, and the image-space comparison repeated.

**What.** +1 % on top of full-resolution pixels; **+24 %** on top of the lossy latent. The
learned latent itself loses to raw pixels by roughly 3× (0.060 vs 0.170) — a reconstruction
bottleneck discards spatial detail a 1024-pixel linear operator still exploits, and its
reconstruction error compounds with its prediction error.

**Limitations.** One encoder configuration, no latent-width sweep. — **EXP-0005**

---

## 4. Image-space accuracy and control usefulness are only loosely coupled

**How.** A descriptor-only predictor — which cannot reconstruct an image and therefore scores
exactly 0.000 image-space accuracy — was used to rank actions by mapping its predicted
centre of mass into the world frame and evaluating the goal's distance field there.

**What.** It scores `slateN` **+0.461** on the corpus it was fit on, far above persistence
(−0.093) and random (+0.120), with 35 wins / 5 losses / 20 ties over 60 slates. A model that
one metric calls worthless is usable for control under the other.

**Limitations.** The claim is **narrowed**, not clean: on a held-out corpus the same model
degrades sharply and reverses sign on annular targets (−0.313 on `ring_O`/lyapunov). The
readout is a point-mass approximation that suits a distance-field goal; whether it generalises
to goals requiring spatial extent is untested. — **EXP-0006, EXP-0008**

---

## 5. Switching operators by push length beats a single global operator, conditionally

**How.** Per-bin linear operators (6 equal-width push-length bins) compared against one
operator fit across all lengths, on descriptor, visual and latent state representations, and
across goal shapes and value functions.

**What.** The switched operator wins on every state representation. The mechanism is visible
in the failure case: one operator averaged over all push lengths overreacts to short pushes,
scoring −0.49 on the shortest-push bin where the switched operator scores ≈0. Pooled control
score 0.938 vs 0.911.

**Limitations.** Conditional on excluding the `n20_L10mm` slate dataset, whose pushes all fall
in the weakest bin and which reversed the result when included. Goal-dependent: the switched
operator **loses** on `stripe` under the Lyapunov value function while winning under both mass
value functions. On the multi-step corpora, which are single-push-length, the measured gain is
operator *specialisation*, not bin *switching*. — **EXP-0004, EXP-0006, EXP-0008**

---

## 6. Closed-loop rollout error compounds steeply; terminal ranking is far more robust

**How.** Models were rolled out three steps on genuine 3-step trajectories, feeding each
prediction back as the next input, and compared against teacher-forced prediction (ground
truth re-injected each step) on the same chains.

**What.** Teacher-forced accuracy is flat across steps for every model. Closed-loop, NFD loses
~3.5× of its accuracy by step 3 (+0.351 → +0.099), and the **unswitched linear operator
diverges** — −0.104 at step 3, worse than predicting nothing moved, its spectral radius being
above 1 and applied three times. Ranking whole action sequences by predicted terminal state is
much more forgiving (~11 % worst-case relative loss) because it needs correct ordering, not
pixel fidelity.

**Limitations.** Two single-push-length corpora. Tie statistics in this run used a
non-standard definition and are not comparable to §1/§4. — **EXP-0010**

---

## 7. A descriptor state cannot be rolled out closed-loop at all

**How.** Attempted to chain the 94-dimensional push-frame descriptor state across three steps
with differing actions.

**What.** Structurally impossible with what exists: the descriptors are defined *relative to
the action's push frame*, and no transform exists to reframe shape terms between steps whose
actions differ. Only teacher-forced evaluation is defined for this model.

**Limitations.** A reframing transform could be written; the finding is that action-relative
features do not chain for free, not that they can never chain. — **EXP-0010**

---

## 8. Training a predictor through the rollout improves multi-step accuracy — for a network,
## not for a linear operator

**How.** Both a switched-linear operator and NFD were trained through the differentiable
3-step closed-loop rollout with an exponentially decaying objective,
`L = λL₁ + λ²L₂ + λ³L₃`, initialised from the existing one-step fit, λ swept over
{0.3, 0.5, 0.7, 0.9}, evaluated on held-out slates.

**What.** For NFD with a **full-image** MSE loss: step-3 closed-loop accuracy improves
**2.2×** (+0.092 → +0.216) with no cost to step-1 or teacher-forced accuracy. For the linear
operator with a **swept-region-masked** MSE loss: training loss fell monotonically while
held-out control score collapsed to negative at every λ, and the global operator's spectral
radius roughly quadrupled. The masked objective leaves the operator unconstrained outside the
mask, which is precisely where the value function reads, and discards the ridge-toward-identity
regularisation the closed-form fit provided.

The control benefit is real: under 5-fold cross-validation over all 50 slates (n=100 paired
held-out slates, λ fixed at 0.7), the fine-tuned model beats the untrained one by
**+0.046 ± 0.018 (2.5σ)** on `mass_in_region` and **+0.034 ± 0.014 (2.4σ)** on
`signed_mass_in_region`.

**Limitations.** λ is **not resolvable** — all four values land within ~0.01, single seed each.
The control effect is **not uniform across datasets**: on `signed_mass_in_region` it is strong
on `n20_L20mm` (~2.7σ) and flat on `n20_L40mm` (~0.2σ). Under the Lyapunov value function the
effect is borderline (2.0σ) because that goal is near ceiling. Four of six operator bins
received zero gradient on these single-push-length corpora, so the switched multi-step result
is additionally data-limited. — **EXP-0010**

---

## 9. The design in `hybrid_linear_latent.md` collapses as specified

**How.** Implemented the document's Stage-2 objective directly — a FiLM-conditioned latent
predictor trained against `‖ẑ′ − E(T_a(X))‖²` on the analytic geometric target.

**What.** The encoder collapses: latent spread falls to ~2e-6 and model and baseline losses go
to zero together, yielding a meaningless 39× "improvement". The objective is trivially
minimised by mapping every state to a constant. With a variance regulariser added, the
predictor beats the do-nothing baseline by ~10× (loss ratio 0.095) on the same target.

**Limitations.** The analytic target used was a whole-pile rigid translation, the easiest form
of the transform; this margin does not preview the real-dynamics case. The anti-collapse term
is not part of the document as written. — **EXP-0007**

---

## Global limitations

- **Every record is `mode: exploratory`, grade `very-low`.** No prediction was committed to
  git before any run, several configurations were selected after seeing results, and runs
  executed against an uncommitted tree, so provenance shas do not reconstruct them.
- **`slateN`'s weakness is statistical power, not bias.** At full pool size two decent models
  very often pick the same action; tie rates of 50–90 % collapse effective sample size, and
  several reported comparisons are unresolved for this reason rather than genuinely null.
- **Image-space `accuracy` is a weak instrument.** It ranked the same models three different
  ways under three defensible definitions before input and target were separated, and it
  disagrees with the control metric across model families. It is used here only within a model
  type, and never as a verdict.
- **Three known defects** are recorded as `broken` invariants: an out-of-memory fault in
  `dmdc_baseline.apply_operators` at large descriptor dimension, a ledger-path mismatch in
  `scripts/run_probe.py`, and the missing anti-collapse mechanism in §9's design document.
