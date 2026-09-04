# Ideas log — signal vs detail: does the operator get the physics right and the pixels wrong?

**Opened:** 2026-09-05 · **Register claim:** C-030 · **Status:** LIVE
**Anchors:** EXP-0003 (blur), EXP-0002/EXP-0006 (regime margins),
`reports/linear_foresight_report.md` §2.4 (ranking vs pixel error),
`docs/prediction_difficulty_hypotheses.md` H-A2, H-C1, H-C2.

---

## 1. The idea

In a constrained domain — fixed-length sweeps, contact-aware sampling, piled
objects — the linear operator beats warped persistence by a wide margin
(EXP-0002/0006: +0.23…+0.31 over mean-delta, and 54–64% of persistence's error)
and does creditably in MPC. Yet its predicted images look noisy and it is
nowhere near reproducing the exact post-sweep state.

**The hypothesis: those two facts are not in tension, because per-pixel error is
mostly measuring something MPC does not consume.** A push moves material; what
a controller needs is *where the mass went*. Whether an individual 2.5-px cube
silhouette lands one cell left or right is unpredictable in principle (it
depends on grain-scale contact detail the grid does not represent — C-007,
C-009) and irrelevant in practice.

If that is right, then:

- pixel rms **understates** good models and **cannot rank** them for control;
- the blur effect in EXP-0003 (89.4% → 56.1% as σ goes 0 → 1.5) is not a
  measurement artifact but a crude, isotropic version of the right metric —
  blur suppresses exactly the band nobody can predict;
- there exists a metric under which one-step predictive accuracy *does* rank
  models the way MPC performance ranks them, and picking models by it would be
  better than picking by rms.

That last sentence is the testable payoff, and it is what makes this worth
doing rather than an argument about definitions.

## 2. Why pixel error is the wrong instrument — the double penalty

The failure has a name in the forecast-verification literature. A prediction
that is right in every respect except displaced by one grid cell is penalised
**twice**: once as a miss where the truth is, once as a false alarm where the
prediction is. It scores *worse* than a bland prediction that puts diffuse mass
everywhere and commits to nothing. Meteorology hit this with precipitation
forecasts — spatially intermittent, high-frequency fields — and built a whole
family of spatial verification methods in response. Our occupancy fields have
the same statistics, and we have been using the metric that literature
abandoned.

This also explains an inversion already recorded and never satisfactorily
accounted for: `linear_foresight_report.md` §2.4 found the geometric heuristic
**ranking actions better** than the fitted operator despite **losing to it on
per-pixel error**. A transport model gets the direction of mass flow right while
its pixel detail is worse — and direction is what `dV` depends on. That is this
hypothesis, already observed once.

## 3. Metrics worth borrowing

Ordered by expected value ÷ implementation cost. All are computed on the same
held-out one-step predictions, so adding one is cheap once the harness exists.

| # | Metric | What it measures | Cost | Notes |
|---|---|---|---|---|
| M1 | **Fractions Skill Score (FSS)** | agreement of *fractional coverage* within a neighbourhood of radius r, swept over r | trivial (box filter + one ratio) | The standard double-penalty fix. Sweeping r gives a **skill-vs-scale curve** and a "believable scale" — the radius above which the model has skill. That curve *is* the signal/detail decomposition, measured rather than assumed. Start here. |
| M2 | **Sliced / entropic Wasserstein (EMD)** | how far mass had to move to turn prediction into truth | moderate; `simple_mpc/ot_planner.py` and `compare_model_emd.py` exist | The natural "did the mass go to the right place" metric, and insensitive to texture by construction. Sliced version is cheap enough for a sweep. |
| M3 | **SAL (Structure–Amplitude–Location)** | three separate numbers instead of one | low | Decomposes error into *how much* material, *where* it is, and *what shape* it is. The prediction is that MPC cares about A and L and not S. Being able to say that quantitatively is worth more than any single scalar. |
| M4 | **Displacement–amplitude decomposition via field alignment** | fit the warp that best aligns prediction to truth; report warp size (location error) and post-alignment residual (structure error) | moderate (optical flow, or reuse the SE(2) warp machinery) | The sharpest form of the hypothesis: if post-alignment residual is large but warp size is small, the model has the physics right and the texture wrong — exactly the claim. |
| M5 | **Scale-decomposed error** (Laplacian pyramid or power spectrum) | error per spatial-frequency band | trivial | Directly answers "which bands does the model get right?" and connects to H-B1 (band-decomposed operators). The cheapest way to test the mechanism behind the blur result. |
| M6 | **MS-SSIM** | multi-scale structural agreement | trivial (library) | Cheap, well known, weak physical interpretation. Include as a control: if it behaves like FSS, the effect is generic; if not, the physics-specific metrics are earning their keep. |

Deliberately **not** on the list: perceptual/learned metrics (LPIPS and
relatives). They need a trained network, carry their own dataset biases, and
would put a learned component inside the instrument used to judge learned
models.

## 4. The experiment that decides it

The hypothesis is not "EMD is nicer than rms". It is **"a metric exists under
which one-step accuracy predicts MPC performance, and rms is not it."** So the
decisive experiment is a *meta*-experiment about metrics, not about models:

1. Assemble a **spectrum of predictors** spanning quality: persistence, warped
   persistence (identity), mean-delta, the linear operator, rank-truncated
   operators (r = 2, 4, 8, 16, 64), the geometric heuristic, a UNet if a
   corrected checkpoint exists, and the oracle.
2. For each, compute **every metric in §3** on held-out one-step predictions.
3. For each, measure **actual control performance** — best-of-N realised `dV`
   on simple convex goals, and Lyapunov descent curves.
4. Ask: **which accuracy metric rank-correlates best with control performance
   across the spectrum?** That correlation is the deliverable.

**The design problem, and the fix.** A rank correlation over ~8 real models is
weak evidence. The fix is **synthetic degradations**: take one good prediction
and corrupt it in controlled, physically-meaningful ways —
  - displace it by k pixels (pure location error),
  - scale its delta by α (pure amplitude error),
  - add high-frequency noise at a chosen band (pure texture error),
  - blur it (remove high-frequency signal *and* detail),
  - swap in a neighbouring transition's delta (plausible but wrong physics).
Each degradation has a *known* character, so we can ask which metrics are
sensitive to which, and — critically — **which degradations actually hurt
MPC**. If high-frequency noise costs a lot of rms and nothing in realised `dV`,
while a 2-px displacement costs little rms and a lot of `dV`, the hypothesis is
confirmed directly and with a large, cheap sample. This is a much stronger
instrument than eight real models and it runs on existing data.

**Goals should be simple and convex** — a disc, a square region, a half-plane,
or `control_utility_test.py`'s existing `center` / `point`. Letter shapes add
a representation question on top of the metric question and should wait.

## 5. Predictions, stated in advance

If the hypothesis holds:
- **P1. ~~FSS skill rises sharply with neighbourhood radius and saturates by
  r ≈ 3–5 px.~~ TESTED AND REFUTED — EXP-0007.** It was also **badly worded**:
  "FSS skill" could mean raw FSS, FSS minus a baseline's FSS, or FSS against
  the usable-skill threshold, and those three disagree in shape. A prediction
  must name the exact quantity, not just its direction. Measured: the
  operator's raw FSS is **0.888 at r=1** against a usable-skill threshold of
  0.549, so its believable scale is ≤1 px — it has fine-scale skill, rather
  than acquiring skill only above some radius.
  **P1′ (replacement, untested).** The operator's error *as a fraction of the
  signal in that band* falls by ≥2x from the finest band to the ~8 px band.
  EXP-0007's M5 arm measures 0.686 → 0.221, a 3.1x fall, which supports P1′ —
  but it was run as a secondary arm and wants a fold sweep before being leaned
  on.
- **P2.** Under synthetic degradation, rms is *more* sensitive to
  high-frequency noise than to small displacement; realised `dV` is the
  reverse. The two orderings visibly cross.
- **P3.** Across the model spectrum, FSS and EMD rank-correlate with realised
  `dV` above ~0.8; rms correlates below ~0.5, and may be near zero.
- **P4.** SAL's L and A components correlate with `dV`; S does not.

**Already learned (EXP-0007):** the strong form of this idea — "the fine band
is noise no model can predict" — is **wrong**. The operator predicts the finest
band substantially better than persistence does. The surviving, weaker, and
still-interesting form is that fine detail is *harder* (3x the relative error
of the 8 px band), and the open question is whether that residual matters for
control at all. So P3 below carries the whole programme now; P1 no longer does.

If instead **rms ranks models as well as anything else**, the hypothesis is
refuted, the blur result is a target-difficulty artifact after all, and the
right conclusion is that the operator genuinely is a poor model that happens to
be better than the alternatives. That is a perfectly good outcome and must be
reportable — P3 is the discriminating prediction and it can fail.

## 5b. What the first two experiments established (2026-09-05)

Both P1 and P2 are refuted as worded, and the programme is **better off**: the
mechanism it assumed is wrong, and the mechanism the data shows is more useful.

**Wrong:** "the fine band is unpredictable detail". The operator predicts the
finest band far better than persistence (FSS 0.888 vs 0.342 against a 0.549
usable threshold, EXP-0007).

**Wrong:** "rms over-penalises high-frequency noise". It **under**-penalises it
(EXP-0008).

**Right, and this is the finding — the dissociation is by error TYPE:**

| error type | cost in rms | cost in control utility | verdict on rms |
|---|---|---|---|
| amplitude (halve the delta) | +0.05 | **none** (dV Spearman 0.474 → 0.472) | over-penalises |
| blur σ=2 | +0.07 | **none** (0.474 → 0.473) | over-penalises |
| displacement 1–4 px | +0.06…+0.14 | mild (0.474 → 0.339) | roughly aligned |
| high-frequency noise | +0.02…+0.20 | **catastrophic** (0.474 → 0.012, slate4 goes negative) | badly under-penalises |

**The mechanism: ranking is destroyed by variance, not by bias.** Amplitude,
blur and displacement perturb every candidate the same way, so they cancel when
candidates are compared. Noise is independent per candidate and does not cancel;
it swamps the small between-candidate differences `dV` depends on.

**So the programme's question changes.** It is no longer "which metric replaces
rms" — a global metric swap was tried and failed (FSS loses to rms on pooled
correlation, EXP-0008). It is:

> **A predictor for MPC should be selected for low variance, not low error.
> What is the right variance-sensitive selection criterion, and does it pick a
> different model than rms does?**

That reframing also predicts something checkable and cheap: heavier shrinkage
should improve control utility past the point where it starts hurting rms.
If true, the current ridge strength is tuned on the wrong objective.

**Methodological note worth carrying:** EXP-0008's pooled rank correlation over
14 degraded models found "rms wins", while the same table shows a large
type-specific dissociation. **A pooled correlation cannot detect a
type-specific effect** — it averages over exactly the contrast of interest.
Report per-type comparisons at matched error magnitude.

## 6. Risks and confounds

- **Circularity.** If a metric is *defined* in terms of transported mass and
  the control cost is *also* defined in terms of transported mass, a
  correlation between them is arithmetic, not evidence. Guard: the control
  measurement must be **realised** `dV` from executed pushes, not predicted
  `dV`; and the goal-cost functional must not be one of the metrics under test.
  EMD is the most exposed to this and needs the sharpest guard.
- **Ranking on different states.** §2.4's caveat still stands: candidate slates
  drawn from different states confound state quality with action quality. Same-
  state slates need a small targeted collection (~50 states × 16 actions).
- **Small model spectrum.** Mitigated by the synthetic degradations, but the
  real-model correlation should be reported separately and honestly, not pooled
  with the synthetic one.
- **`grid-convention` is still broken.** Anything using the `PileSweepData`
  raster — including any UNet arm — is transposed (EXP-0001). Until that is
  fixed this programme should use the re-rasterised path only, and must not
  include a UNet in the spectrum.

## 7. Entry points

- `control_utility_test.py` — Lyapunov weights, `dV`, ranking metrics, existing
  simple goals.
- `compare_model_emd.py`, `simple_mpc/ot_planner.py` — EMD/Sinkhorn machinery.
- `occupancy_foresight.py`, `model_zoo.py` — the fitted-model spectrum.
- `scripts/probes/regimes.py` — the constrained domain this is all about.
- Data: `Genesis/data/cube_spectrum/n20` (5120 transitions, piled, fixed 20 mm
  contact-aware pushes) is the constrained domain; `n30` is the replicate.
