---
# ---- identity -------------------------------------------------------------
id: EXP-0026
title: >
  The K=4 objection fails: from top-1-of-4 to top-1-of-31 the ridge
  operator holds ~96% of the oracle's advantage and the UNet's edge stays
  ~1.6 points -- but selection pressure DOES amplify one error type,
  independent per-candidate noise, and only that one
tier: T1
mode: confirmatory
date: 2026-09-06
hypothesis: C-030

# ---- the claim ------------------------------------------------------------
claim: >
  Every control number in this register is `slate4` -- pick the best of 4
  candidates drawn at random from a ~32-action same-state slate -- while a
  real MPC step ranks hundreds to thousands. If K=4 is what makes both model
  classes look near-ceiling (C-045: "the linear operator already captures 95%
  of the oracle's advantage, and the UNet's edge hardly matters"), then
  raising K on the SAME data, the SAME predictions and the SAME slates should
  pull the linear operator away from the oracle and open up the UNet-linear
  gap.

prediction:                       # committed in 622c495f, BEFORE the run
  supports: >
    linear `slateK` at K=31 (the whole slate) is <= 0.90, i.e. at least 5
    points below its `slate4` of 0.950; AND the paired UNet-linear gap at
    K=31 is at least twice its K=4 value (+0.037 or more), against the paired
    sem over 49-50 slates.
  refutes: >
    linear `slateK` at K=31 stays >= 0.94 and the paired gap does not exceed
    gap(4) + 1 sem. Then K=4 was not the problem, and C-045 stands on much
    stronger evidence.
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 9e78da53
  dirty: false
  dirty_note: >
    Every number here was regenerated after the probes were committed, so the
    sha contains the code that produced them. An earlier identical run under
    622c495f (the pre-registration commit) reported dirty=false only because
    `utils.git_provenance()` does not count untracked files; the probes were
    untracked then. Outputs are bit-identical between the two runs (seeded,
    CPU-deterministic), which is why the earlier ones were simply overwritten.
  data_commit: >
    slates/n20_heap_5mm collected under 006004d0..dbf21ba2 (EXP-0012);
    cube_spectrum/n20 predates provenance stamping (`dataset-provenance`
    unrecorded for pre-2026-09-05 data).
  script: >
    scripts/probes/exp0026_selection_pressure.py (stage 1: cache per-candidate
    dV), scripts/probes/exp0026_kcurve.py (stage 2: sweep K). Both new here.
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt (fit + UNet train, unchanged
          from EXP-0024)",
         "Genesis/data/slates/n20_heap_5mm/*_data.pt (all 50 slates, 1597
          candidates after min_push_length_m=0.0199)"]
  code_path: >
    registry.dataset_registry PileSweepData (type: genesis) throughout --
    EXP-0024's exact path, for the linear fit, the UNet checkpoint
    (runs_exp0024/unetfilm_cube_spectrum_n20, reused unmodified, not
    retrained) and the slate evaluation alike. The degradation arms are
    rebuilt in THIS path from EXP-0025's recipe functions rather than reusing
    EXP-0025's occupancy_foresight-path numbers; see Threats.
  seed: 0
  split: >
    cube_spectrum/n20 train split for the fit (identical to EXP-0024). Slates
    evaluated in full: the headline uses all 50 (via the new
    genesis_cube_spectrum_n20_slates_all.yaml, val_pct=0/test_pct=0,
    split="train"), and EXP-0024's 49-slate subset is reported alongside as a
    reproduction check.
  runtime: "stage 1 ~50 s CPU per cache; stage 2 ~3 min CPU for 18 arms x 6 K x 2000 draws x 50 slates"

budget:
  declared: "half a day of analysis, CPU only, no new data and no training"
  spent: "~1 h wall-clock, ~120k tokens"
  outcome: within

design:
  varied:
    K: [2, 4, 8, 16, 24, 31]
    model: [persistence, mean-delta, "linear (ridge->I)", UNetFilm, oracle]
    degradation: ["displacement k=1/2/4", "amplitude a=0.5/0.75/1.25/1.5",
                  "hf-noise m=0.5/1.0/2.0", "blur s=1.0/2.0", wrong-physics]
  held_fixed:
    predictions: >
      identical to EXP-0024 -- same fit (res=64, crop=1.0, ridge=1.0 toward
      identity), same UNet checkpoint, same slate data, same Lyapunov weights.
      NOTHING about the models changed; only the slate size K and the sampler.
    sampler: "subsets drawn WITHOUT replacement (torch.randperm), 2000 draws per slate per K; K=n_slate is the single exact subset"
    subsets: "one generator per slate, reset per model, so every model is scored on the identical draws"
    goal: "corner (center reported and confirmed degenerate again, not used)"
    noise_seed: 1234
  baselines: [persistence, mean-delta, oracle]
  metric: "slateK, regret_dv, pick_pctile (docs/experiments/METRICS.md), with slate4 as the anchor to the register"

noise_floor: >
  Paired across the 50 shared slates -- the sem of the per-slate difference,
  never the across-slate sd (handoff trap 1; this is the error that flipped
  C-008 and C-045). Measured here for UNet-linear: sem 0.0028 at K=4 rising to
  0.0068 at K=31 (fewer effective draws as K approaches the slate size).
  Bootstrap over slates (10k resamples) gives the CI on each model's own mean.

depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  PREDICTION REFUTED on both clauses. linear slateK: 0.958 (K=4) -> 0.971
  (K=8) -> 0.968 (K=16) -> 0.958 (K=31), never below 0.94 and never 5 points
  down; UNet: 0.974 -> 0.982 -> 0.979 -> 0.976. Paired UNet-linear gap:
  +0.0164 (K=4, sem 0.0028, t 5.9, 40/50) vs +0.0177 (K=31, sem 0.0068,
  t 2.6, 24 wins / 10 losses / 16 ties) -- flat, not doubled, and below
  gap(4)+1sem = +0.0192. So top-1-of-31 is no harder for these models,
  relatively, than top-1-of-4. BUT the unbounded companion moves: regret_dv
  grows 2.7x for both models over the same range (linear 0.0012 -> 0.0032,
  UNet 0.0007 -> 0.0019), so more candidates do leave more value on the
  table -- the captured FRACTION is what is scale-free. The degradation
  sweep explains why, and is the finding worth keeping: K-dependence is
  entirely a property of the ERROR TYPE. Systematic errors are K-invariant
  (blur s=1.0: 0.951 -> 0.960 from K=4 to K=31, i.e. free at every K;
  amplitude a=1.5: 0.951 -> 0.954; displacement k=4: 0.889 -> 0.881), while
  independent per-candidate noise degrades monotonically with K (hf-noise
  m=0.5: 0.916 -> 0.861; m=1.0: 0.878 -> 0.802), and its EXCESS regret over
  the undegraded operator grows 4.7x (m=0.5: +0.0015 at K=4 -> +0.0070 at
  K=31). Two by-products: (1) `rank_metrics` samples slates WITH replacement,
  so every `slate4` in this register is 0.5-1.1 points low (linear 0.9503 vs
  0.9583 without replacement -- and the with-replacement value reproduces
  EXP-0024's 0.950/0.969/0.796 exactly, confirming the pipeline is identical);
  (2) EXP-0024's missing 50th slate was a split artifact, not the
  push-length filter its Unrelated findings blamed.
verdict: refuted
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

**Tier note.** Scoped as a T2 gate — the prediction block was committed in
`622c495f` before any of this ran — but filed as **T1** for the same reason
EXP-0024 was: `scripts/check_register.py` forbids a T2 record from citing a
`depends_on` tag that is not `holds`/`fixed`, and this record inherits
EXP-0024's dependency on `settled-state`, which is `unchecked` for the
rigid-cube path.

## Why this test discriminates

C-045 says the UNet's control edge over a ridge operator is real but small
"against a near-saturated ceiling", and the handoff built a whole programme on
the consequence — that this domain has no headroom, so model-class comparisons
here cannot produce large effects. Every number behind that is `slate4`. If the
ceiling is an artifact of asking for the best of 4 rather than the best of
many, then raising K on data already collected must move it, and no new
scenario, dataset or training run is needed to find out. If raising K does not
move it, the ceiling is a property of the domain and the objection is closed.

The design changes exactly one thing — the number of candidates the model is
asked to choose between — while the predictions, the fit, the checkpoint, the
slates and the metric are byte-identical to EXP-0024's.

## What was actually run

Two stages, no GPU, no refit, no retrain.

**Stage 1** (`exp0026_selection_pressure.py`) recomputes EXP-0024's
per-candidate `dv_pred`/`dv_true` vectors and caches them. EXP-0024 kept only
per-slate summaries, which is why the K sweep was not available from its
outputs. `--degradations` additionally rebuilds EXP-0025's spectrum from the
linear operator's predicted delta, with that record's recipe functions imported
unchanged (`finest_band_noise`, `_gaussian_blur2d`, `build_pyramid`) and its
noise seed (1234).

**Stage 2** (`exp0026_kcurve.py`) sweeps K. Three deliberate differences from
`control_utility_test.rank_metrics`:

1. **Subsets are drawn without replacement.** `rank_metrics` uses
   `torch.randint`, so its "slate of 4" is 4 draws *with* replacement (~3.44
   distinct actions) and its `slate16` (~13 distinct) is not a 16-candidate
   test at all. At K = slate size the without-replacement sampler gives the
   exact top-1-of-everything test, which is the point of the experiment.
2. **All models see the identical subsets**, so the comparison is paired at
   the level of the individual draw.
3. **Two unbounded companions** are reported beside the bounded capture
   fraction — `regret_dv` and `pick_pctile` — because a bounded score's
   difference is forced toward zero and `slateK`'s own denominator grows with
   K. This is the trap that cost EXP-0007 its conclusion, and here it is
   load-bearing: the two families of metric genuinely disagree about whether
   large K is harder.

**All 50 slates.** `configs/dataset/genesis_cube_spectrum_n20_slates_all.yaml`
is new. The existing slate config sets `test_pct: 99` with a comment saying
that puts every file in test; it does not.
`PileSweepData._assign_group_splits` computes `round(50 * 99/100) = 49` test
groups and assigns the remaining group to `train`, and its own overflow clamp
caps the count at `num_groups - 1` regardless — so the registry path **cannot**
put every file in `test` whatever `test_pct` says. `val_pct: 0`/`test_pct: 0`
with `split="train"` is the only setting that loads all 50; the slates are held
out by construction (a physically separate collection run), so the split name
they carry is immaterial.

## Numbers

**`slateK` — fraction of the oracle's advantage captured** (goal=corner, 50
slates, ± is half the 95% bootstrap CI over slates):

| model | K=2 | K=4 | K=8 | K=16 | K=24 | K=31 |
|---|---|---|---|---|---|---|
| persistence | 0.004 | 0.001 | 0.001 | −0.000 | −0.000 | −0.008 |
| mean-delta | 0.627 | 0.806 | 0.828 | 0.829 | 0.824 | 0.814 |
| **linear (ridge→I)** | 0.835 | **0.958** | 0.971 | 0.968 | 0.962 | **0.958** |
| **UNet** | 0.873 | **0.974** | 0.982 | 0.979 | 0.977 | **0.975** |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

K=2 is on a different scale and should not be read as part of the trend: with
two candidates the capture value is exactly +1 or −1, so its mean is
`2·P(correct) − 1`, floored at −1 rather than 0.

**Paired UNet − linear**, per slate:

| K | mean | sd | sem | t | wins / losses / ties |
|---|---|---|---|---|---|
| 4 | +0.0164 | 0.0199 | 0.0028 | 5.85 | 40 / 10 / 0 |
| 8 | +0.0116 | 0.0125 | 0.0018 | 6.54 | 40 / 10 / 0 |
| 16 | +0.0110 | 0.0194 | 0.0027 | 4.02 | 37 / 13 / 0 |
| 24 | +0.0145 | 0.0322 | 0.0046 | 3.19 | 36 / 13 / 1 |
| 31 | +0.0177 | 0.0484 | 0.0068 | 2.58 | 24 / 10 / **16** |

The falling `t` at large K is not a weakening effect: it is 2000 draws
collapsing to one deterministic draw, plus **16 slates where both models pick
the same single best action** and the paired difference is exactly zero. On
non-tied slates the UNet still wins 24 of 34.

**`regret_dv` — value left on the table, Lyapunov units** (lower better):

| model | K=4 | K=8 | K=16 | K=31 | K=31 / K=4 |
|---|---|---|---|---|---|
| linear | 0.0012 | 0.0016 | 0.0023 | 0.0032 | 2.7× |
| UNet | 0.0007 | 0.0010 | 0.0015 | 0.0019 | 2.7× |
| mean-delta | 0.0078 | 0.0109 | 0.0124 | 0.0143 | 1.8× |

**`pick_pctile` — fraction of the offered candidates better than the pick**
(scale-free; lower better): linear 0.038 → 0.035, UNet 0.028 → 0.030 from K=4
to K=31. Both models pick inside the top ~4% of whatever they are shown,
independently of how much they are shown.

**The degradation sweep — where K *does* bite** (`slateK`, goal=corner):

| arm | K=4 | K=8 | K=16 | K=31 | Δ(31−4) | excess regret_dv vs undegraded, K=4 → K=31 |
|---|---|---|---|---|---|---|
| linear (undegraded) | 0.958 | 0.971 | 0.968 | 0.958 | 0.000 | — |
| amplitude a=0.5 | 0.956 | 0.970 | 0.968 | 0.959 | +0.001 | +0.0000 → −0.0001 |
| amplitude a=1.5 | 0.951 | 0.966 | 0.964 | 0.954 | +0.003 | +0.0002 → +0.0003 |
| blur s=1.0 | 0.951 | 0.965 | 0.966 | 0.960 | **+0.009** | +0.0002 → −0.0002 (freer at K=31) |
| blur s=2.0 | 0.941 | 0.955 | 0.952 | 0.942 | +0.001 | +0.0006 → +0.0012 |
| displacement k=1 | 0.952 | 0.967 | 0.963 | 0.948 | −0.004 | +0.0002 → +0.0007 |
| displacement k=4 | 0.889 | 0.907 | 0.894 | 0.881 | −0.008 | +0.0026 → +0.0059 |
| **hf-noise m=0.5** | 0.916 | 0.911 | 0.882 | **0.861** | **−0.055** | +0.0015 → **+0.0070** (4.7×) |
| **hf-noise m=1.0** | 0.878 | 0.872 | 0.838 | **0.802** | **−0.076** | +0.0031 → **+0.0111** (3.6×) |
| **hf-noise m=2.0** | 0.745 | 0.768 | 0.757 | 0.738 | −0.007 | +0.0094 → +0.0168 |
| wrong-physics | −0.048 | −0.080 | −0.107 | −0.084 | −0.036 | +0.0536 → +0.0804 |

Every systematic degradation is essentially K-invariant. Only the arm whose
error is drawn **independently per candidate** degrades monotonically with the
number of candidates, and it does so exactly as the winner's-curse mechanism
predicts: more candidates means more chances for an independent error to
promote a mediocre action.

**Failed extrapolation, reported because it failed.** The plan proposed fitting
a bivariate-normal (Gaussian-copula) model per arm — correlation ρ between
predicted and true dV within a slate — and extrapolating past K=31 to K=10²–10³.
It does not survive validation on the measured range: at K=4 it predicts 0.894
for the linear operator against a measured 0.958, and for the noise arms it
predicts the *wrong direction* (hf-noise m=0.5 rising to 0.913 at K=31 against a
measured 0.861). A ρ-only model cannot represent skewed within-slate `dv_true`
or heteroscedastic model error, and both are present. **No extrapolated number
from it is quoted, and the "hundreds to thousands of candidates" question is
not answered by this record** — it needs the dense slates of EXP-B.

goal=center: degenerate again (`dv_true` sd 0.0012, 0% helpful pushes),
consistent with C-040/EXP-0012/EXP-0024. Not used.

## What this means

**The K=4 objection is dead on this domain, for these candidates.** Asking a
model to pick the best of 31 instead of the best of 4 costs it nothing in
captured fraction: the ridge operator holds 0.958 at both ends and the UNet
0.974/0.975, and the gap between them is flat at ~1.6 points. C-045's reading —
the UNet is measurably better and it hardly matters, because a ridge operator
is already near the ceiling — survives a test designed to break it, and now
rests on top-1-of-31 rather than top-1-of-4.

**But "flat capture" is not "K does not matter".** The absolute value left on
the table grows 2.7× over the same range. Both facts have to be quoted
together, and the register's habit of quoting only the bounded metric would
have hidden the second one.

**The real finding is the interaction.** Selection pressure does not amplify
prediction error in general; it amplifies exactly the error type that is
independent across candidates, and leaves systematic error alone at every K
tested. That sharpens C-035/C-039 instead of overturning them: "blur and
amplitude are nearly free" is not a K=4 artifact (blur s=1.0 is, if anything,
*freer* at K=31), while "high-frequency noise destroys ranking" is
**understated** at K=4 by a factor approaching 5 in excess regret.

It also explains the flat model curve mechanistically. The UNet−linear
difference behaves like a *systematic* difference, not a noise difference —
which is what C-044 says it is (a frequency-band difference in what each model
gets right, not extra randomness). A model class whose advantage were noise-
shaped would have shown a widening gap here, and none did.

**Consequence for the programme.** EXP-A closes the first of the three
weaknesses in `docs/plan_selection_pressure_validation.md` §1 and leaves the
other two untouched, in a way that raises their value rather than lowering it:
these candidates are 32 sampler draws, so at K=31 all models choose among *the
same* actions, and nothing here exposes a model to candidates chosen *because*
its own prediction likes them. The winner's curse measured in the hf-noise arms
is the passive version; EXP-C's optimizer is the active one, and this record
gives it a calibrated expectation — the arm-by-arm K-dependence is a direct
measure of how much independent-error a search can exploit.

## What would change the verdict

- **Dense candidates (EXP-B).** These 32 actions are sampler draws, not
  neighbours. Discriminating between actions 2–3 mm apart is a fine-band
  problem in a way that discriminating between 32 spread-out pushes is not,
  and it is the regime a real optimizer works in. Cost: one overnight
  collection.
- **Candidates selected by the model itself (EXP-C).** Nothing here measures
  the exploitation gap.
- **A second UNet seed and a second fit (EXP-F).** The gap's stability across
  slates is now well measured at every K; its stability across *training seeds*
  is still one sample.

## Threats

- `imprecision`: one UNet training seed, one linear fit — inherited from
  EXP-0024 unchanged. The slate-level floor is now much better measured (paired
  sem at six values of K, plus a 10k bootstrap), but that is the wrong axis for
  a model-class claim; EXP-F is the fix.
- `untested-dependency`: `settled-state` is `unchecked` for the rigid-cube
  path, as in EXP-0012/0024/0025.
- Considered and dismissed: `provenance` for the headline. The model
  comparison runs entirely through EXP-0024's own code path, with its own
  checkpoint and fit; the *only* new code is the sampler and the metrics.
  The reproduction check makes this concrete — running `rank_metrics`
  unchanged (with replacement) on this record's cached vectors returns
  EXP-0024's published numbers to four decimals (0.9503 / 0.9688 / 0.7956).
  The **degradation arms** are a weaker case: their recipes are EXP-0025's but
  the fields are built in this record's path, so their absolute values are not
  interchangeable with EXP-0025's table. Every claim made from them here is
  about the shape across K *within* this run, which that difference does not
  touch.
- Considered and dismissed: `selection` — the K grid, the model set and the
  refute thresholds were fixed in `622c495f` before the run; both goals were
  computed and both are reported.
- Considered and dismissed: `inconsistency` — the flat-capture result holds
  for both models, at every K, on both the 50-slate and the 49-slate
  (EXP-0024 subset) evaluations. Largest disagreement between them: 0.0006
  (linear) and 0.0012 (UNet) across all six K; mean-delta moves by up to
  0.0036, which is the one extra slate being an easy one for it.

## Unrelated findings

- `configs/dataset/genesis_cube_spectrum_n20_slates.yaml`'s comment claims
  `test_pct: 99` "rounds to num_groups all test". It does not, for any value:
  `_assign_group_splits` takes `round(n·test_pct/100)` and then clamps the
  total to `n − 1`, so at least one group is always withheld from `test`. Any
  record that evaluated "all of" a slate collection through the registry path
  has silently dropped one file. EXP-0024 is the known case; not audited
  further.
- EXP-0024's "Unrelated findings" attributes its 49-of-50 slate count to the
  `min_push_length_m` filter. That is wrong: a direct length filter on the raw
  files keeps 1597 of 1600 candidates and drops no whole file (47 slates keep
  32, 3 keep 31). The missing slate is `_4_data.pt`, withheld by the split
  hash above.
- `control_utility_test.rank_metrics` computes `slate16` and every caller in
  the repo discards it, keeping only `slate4` — including
  `same_state_degradation.per_slate_metrics`, which is where the register's
  control numbers come from. A 16-candidate figure was available (if
  with-replacement) for the whole life of the register and was never looked at.
