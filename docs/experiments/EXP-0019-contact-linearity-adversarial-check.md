---
# ---- identity -------------------------------------------------------------
id: EXP-0019
title: >
  C-009's 84%/58% survives a trivial-baseline check; C-007's "essentially all
  the nonlinearity is one scalar" survives booster-tuning but fails on a
  threshold-sensitive target (max displacement), where the linear share falls
  from 95% to 73% as contact grows -- the opposite of what "one scalar
  explains it" predicts
tier: T1
mode: exploratory
date: 2026-09-05
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  C-007 ("within a contact stratum, linear R2 matches or beats boosted,
  91-110% share, `linear_foresight_report.md` §2.6") and C-009 ("band
  displacement is 84% predictable from OCC features, 58% linearly", §2.3)
  were never re-measured or graded. Three specific failure modes were tested:
  (1) is C-009's 84%/58% dominated by the near-deterministic ~4.6% of
  transitions where the band is empty (y=0 exactly)? (2) is C-007's
  within-stratum match an artifact of an undertuned booster (250 iters,
  lr 0.06, no depth cap, no early stopping)? (3) does C-007 hold only because
  "mean displacement in band" averages over ~15-25 particles, smoothing away
  the threshold/contact-decision nonlinearity the granular-mechanics
  literature predicts -- tested by repeating the same stratified fit on MAX
  per-particle displacement in the band, a target one outlier particle can
  dominate.

prediction: null   # exploratory: the outcome was seen (scratch runs) before this text was finalised; see "What was actually run"

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 7338d66a
  dirty: true                     # scripts/check_register.py was modified by a concurrent agent (EXP-0018) throughout this session; nothing this record touches was uncommitted at run time -- the probe script itself was committed (ebb89333, then 7338d66a) before every number below was produced
  data_commit: unrecorded          # Genesis/data/foresight/{L040,L040b} predate the 2026-09-05 dataset-provenance stamp; no `provenance:` block in their _N_config.yaml
  script: scripts/probes/exp0019_contact_linearity.py
  data: ["Genesis/data/foresight/L040*/cube/n50/size0.005/_*_data.pt"]
  code_path: particle-based (raw states/p_starts/p_stops tensors; no rasteriser, no occupancy grid, no canonical warp -- same as the original §2.3/§2.6 measurement)
  seed: 0
  split: "GroupKFold(n_splits=5), grouped by run/file (24 runs, 7680 transitions total) -- deterministic, no shuffling"
  runtime: "~90s per script invocation, CPU, OMP_NUM_THREADS=4"

budget:
  declared: "70 min, 160k tokens"
  spent: "~65 min, ~145k tokens"
  outcome: within

design:
  varied: {test: [trivial-baseline, booster-tuning, target (mean vs max displacement)],
           contact_stratum: [smallest, middle-low, middle-high, largest],
           booster: [weak (250 iter, lr 0.06, original), strong (1000 iter, lr 0.03, depth 6, early-stopping)]}
  held_fixed: {feature_set: "OCC (grid-visible)", dataset: "L040+L040b scattered 50-cube, n=7680", cv: "GroupKFold(5) by run", ridge: RidgeCV(alphas 1e-3..1e4)}
  baselines: [noise control (Gaussian random features, from variance_decomposition.py), n_in_band-alone (1-feature trivial predictor), contact-binary-alone]
  metric: "R^2 (5-fold mean, held-out), RMSE in mm (held-out), and per-fold sd of R^2 as the noise floor"

noise_floor: >
  Per-fold sd of R^2 across the 5 GroupKFold splits, measured directly (not
  assumed): 0.012-0.033 for the linear model, 0.014-0.043 for boosted, across
  the four contact strata (both targets). Any linear-vs-boosted R^2 gap
  smaller than ~2x this is not resolved.

depends_on: [episode-split, settled-state]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  C-009 survives: restricting to n_in_band>0 (removing the deterministic
  zero-contact 4.6%) barely moves the share (69%->76%, in the direction of a
  LARGER gap, not smaller) -- linear R2 0.576->0.609, boosted 0.836->0.801.
  C-007's mean-displacement finding survives booster tuning (4x the trees,
  depth 6, early stopping changes boosted R2 by <=0.02 in every stratum,
  share stays 91-110%) and holds in absolute RMSE too (linear RMSE < boosted
  RMSE in every one of the 4 strata). But on MAX per-particle displacement in
  the same strata, the pattern C-007 claims is ABSENT: linear share falls
  monotonically 95% -> 87% -> 80% -> 73% as contact grows, with boosted
  beating linear on RMSE too (e.g. stratum 3: RMSE 5.03mm linear vs 3.98mm
  boosted) -- the opposite trend from "share stays ~100% because it's all one
  scalar", and the gap widens rather than shrinks with more contact.
verdict: refuted
downgrades: [untested-dependency, imprecision]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If C-009's 84%/58% were a trivial-baseline artifact, restricting to
contact>0 (removing the exactly-zero cases both models get for free) would
collapse the boosted-vs-linear gap toward zero. It did not -- the gap widened
slightly. If C-007's within-stratum match were an undertuned-booster
artifact, a 4x-larger, depth-capped, early-stopped booster would close the
gap in favour of boosted. It did not move the numbers by more than the noise
floor. If C-007's "one scalar" picture were target-independent, the same
stratification on a differently-shaped target (max instead of mean
displacement -- both scalar, both computed from the identical band-particle
set) should show the same near-100% share. It does not: it declines with
contact, which is exactly the pattern predicted by granular threshold/jamming
mechanics (more particles in the swath -> more opportunities for a
contact-chain/blocking event the mean smooths over but the max does not).

## What was actually run

Three checks against `variance_decomposition.py` / `density_stratified.py`'s
own feature/target construction (imported directly, not reimplemented), on
the exact dataset and glob §2.3/§2.6 used (`Genesis/data/foresight/L040*`,
n=7680, 24 runs). The other datasets named in the task
(`scatter_contact`, `cube_spectrum/n20`, `n30`, `slates/n20_heap_5mm`) were
not touched -- this reruns the ORIGINAL claims on their ORIGINAL dataset
first, since that is the cheapest test that could have invalidated the whole
plan (a different dataset would not even reproduce the baseline numbers to
compare against). That reproduction was exact: OCC linear/boosted R2 =
0.576/0.836, matching §2.3 to 3 decimal places, and the 4-bin contact
stratification matched §2.6 to the same precision (91/103/110/109%) --
run first as a feasibility/reproduction check, not a hypothesis probe, so it
does not compromise the tests that follow.

The numbers seen first in throwaway scratch scripts were identical to the
numbers below once ported into the committed probe (deterministic pipeline:
fixed random_state, non-shuffling GroupKFold) -- reported here as
`mode: exploratory` because the outcome was known before the script was
committed, per the skill's rule, not because the pipeline is stochastic.

One test that was planned but not run: repeating this on the piled/
contact-sampled datasets (`cube_spectrum`, `slates`) that §2.9 used to show
the paper's claim holds under contact-aware sampling. That is a natural
follow-up (see below) but was out of scope for re-testing §2.3/§2.6
specifically, and the budget was spent on three checks against the original
data rather than one check spread across four datasets.

## Numbers

**Test 1 -- C-009 trivial-baseline check** (OCC feature set, mean band
displacement, n=7680, 24 runs):

| condition | n | linear R2 | boosted R2 | share |
|---|---|---|---|---|
| n_in_band alone (1 feature) | 7680 | 0.173 | 0.253 | 68% |
| OCC full set, all transitions (= original §2.3) | 7680 | 0.576 | 0.836 | 69% |
| OCC full set, contact>0 only | 7326 | 0.609 | 0.801 | 76% |

4.6% of transitions (354/7680) have `n_in_band==0` and displacement exactly
0 for every model. A single trivial feature (contact count alone) gets only
17-25% R2 -- nowhere near 84%.

**Test 2/3 -- C-007 booster-tuning and target-sensitivity** (OCC feature
set, 4 contact-amount quartiles, n=7680):

Target = mean displacement in band (the original C-007 target):

| stratum | n | sd(y) | linear R2 (sd) | boosted-weak R2 (sd) | share | RMSE_L | RMSE_G | boosted-strong R2 | share(strong) |
|---|---|---|---|---|---|---|---|---|---|
| smallest | 1829 | 11.60 | 0.708 (0.012) | 0.781 (0.025) | 91% | 6.24 | 5.40 | 0.781 | 91% |
| mid-low | 1909 | 8.80 | 0.822 (0.014) | 0.797 (0.014) | 103% | 3.71 | 3.96 | 0.781 | 105% |
| mid-high | 1789 | 6.76 | 0.810 (0.017) | 0.737 (0.028) | 110% | 2.93 | 3.44 | 0.733 | 110% |
| largest | 2153 | 5.39 | 0.782 (0.027) | 0.718 (0.033) | 109% | 2.50 | 2.84 | 0.709 | 110% |

Target = max per-particle displacement in band (threshold-sensitive
alternative):

| stratum | n | sd(y) | linear R2 (sd) | boosted-weak R2 (sd) | share | RMSE_L | RMSE_G | boosted-strong R2 | share(strong) |
|---|---|---|---|---|---|---|---|---|---|
| smallest | 1829 | 14.56 | 0.756 (0.018) | 0.798 (0.025) | 95% | 7.16 | 6.51 | 0.810 | 93% |
| mid-low | 1909 | 13.06 | 0.648 (0.028) | 0.746 (0.035) | 87% | 7.73 | 6.57 | 0.746 | 87% |
| mid-high | 1789 | 9.85 | 0.566 (0.034) | 0.705 (0.020) | 80% | 6.43 | 5.30 | 0.703 | 81% |
| largest | 2153 | 7.10 | 0.494 (0.030) | 0.681 (0.043) | 73% | 5.03 | 3.98 | 0.680 | 73% |

The 10x-larger, depth-capped, early-stopped booster ("strong") changes
boosted R2 by at most 0.02 vs the original ("weak") config in every cell of
both tables -- the booster used to produce the original 91-110%/95-73% shares
was not undertuned.

## What would change the verdict

- **Re-run on `cube_spectrum`/`slates` piled data with the same stratified
  max-displacement probe.** §2.9 showed the mean-displacement linear share
  reaches ~100-108% under contact-aware sampling; whether max-displacement
  shows the same residual gap there (narrowing with contact, as here, or
  disappearing) would say whether the mean-vs-max discrepancy is a scattered-
  data artifact or a general property of this granular system. Cost: ~10 min
  CPU, script already supports `--glob`.
- **A target that is more directly "threshold-like" than max-displacement**
  -- e.g. fraction of band particles displaced beyond some multiple of a
  cube width, which is closer to what "sharply nonlinear contact decision"
  in §2.3's own residual analysis describes. ~15 min to add.
- **Seed variation on the booster's `random_state`** (currently fixed at 0
  throughout, original included) to check the max-displacement gap is not a
  single-seed fluke. ~5 min.

## Threats

- `untested-dependency` (`settled-state`): this measures particle
  displacement between `states` and `states_`; if `states_` is not a fully
  settled pile (unchecked for the rigid-cube path per INVARIANTS.md), a
  residual-motion component could inflate variance in ways that interact
  with contact amount. Not expected to explain a monotonic 22-point trend,
  but not ruled out either.
- `imprecision`: no seed sweep on the booster or on fold assignment (GroupKFold
  is deterministic here, so this is really "single configuration," not
  "single random draw"). The per-fold sd reported as the noise floor answers
  fold-to-fold variability but not model-seed variability.
- Considered and dismissed: that the max-displacement finding is itself a
  boosted-overfitting artifact in the *opposite* direction (booster
  hallucinating skill on a noisier target). Dismissed because the strong
  booster's R2 barely moves from the weak one's (max 0.02 difference) in
  every stratum, and because the gap shows up in RMSE (a metric with no R2-style
  denominator-shrinkage failure mode) in the same direction and magnitude.
- Considered and dismissed: that the max-displacement gap is really just
  "one outlier particle is unpredictable, and averaging over more of the
  swath at low contact makes max~mean." This is plausible as a MECHANISM
  (and is consistent with, not contradicting, "granular systems have
  threshold events the mean papers over") but does not change the
  measurement: whatever the mechanism, the linear share for a legitimate
  scalar target is not "essentially always ~100%" as C-007 states.
- Dirty tree: `scripts/check_register.py` was modified by a concurrent
  session (EXP-0018) for the whole run; it is not imported by, or relevant
  to, this record's script.

## Unrelated findings

- `density_stratified.py --by contact` hardcodes its target to mean band
  displacement (`y = mean disp`); there is no CLI flag to select max/forward
  displacement, so testing those required a separate script
  (`scripts/probes/exp0019_contact_linearity.py`) rather than a flag addition
  to the existing one. Anyone extending this probe to more targets will hit
  the same wall.
- `variance_decomposition.py`'s `DEFAULT_GLOB` (`L040*`) matches two
  directories, `L040` (8 files) and `L040b` (16 files) -- neither name nor
  any docstring says the reported n=7680 spans both; this is easy to miss
  when trying to reproduce the number from a fresh glob.
