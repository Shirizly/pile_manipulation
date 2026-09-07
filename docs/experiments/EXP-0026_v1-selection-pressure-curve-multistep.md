---
id: EXP-0026_v1
title: >
  C-046 (selection pressure amplifies only independent-per-candidate error)
  re-tested at K=128 -- 4x EXP-0026's K=31 ceiling -- on REAL 128-candidate
  same-state slates (not with-replacement resampling of ~32) across three
  push lengths. Survives cleanly at L20mm/L40mm; mean-delta becomes the
  first model in the register whose OWN capture fraction falls with K.
  L10mm's degradation arms are confounded by the warp-limited regime
  EXP-0024_v1 diagnoses and are reported but not used for the verdict
tier: T1
mode: confirmatory
date: 2026-09-07
hypothesis: C-046
claim: >
  EXP-0026 established, at K<=31 on 32 sampler-drawn candidates, that
  selection pressure amplifies a prediction error's control cost only when
  the error is INDEPENDENT across candidates (hf-noise), leaving systematic
  errors (displacement, amplitude, blur) K-invariant. On genuinely 128-deep,
  REAL (not resampled) same-state candidate slates, collected independently
  at three push lengths, this pattern holds through K=128 wherever the
  underlying model comparison is itself valid (L20mm, L40mm): systematic
  arms stay flat, hf-noise and mean-delta (whose only source of
  across-candidate variation is the warp/state interaction, effectively
  independent per candidate) degrade monotonically.
prediction:
  supports: >
    at L20mm and L40mm, every systematic degradation arm's slateK_exact
    moves by less than 3 points from K=4 to K=128, while every hf-noise arm
    and mean-delta itself decline monotonically over the same range by more
    than that.
  refutes: >
    at L20mm and L40mm, a systematic arm (displacement/amplitude/blur)
    degrades monotonically by more than 3 points, or an independent-error
    arm (hf-noise/mean-delta) stays flat -- either would mean the K<=31
    finding was itself an artifact of the shallower sweep or the resampled
    candidate pool, not a real mechanism.
  discriminating: true
provenance:
  commit: 4afea641
  dirty: false
  dirty_note: >
    Identical to EXP-0024_v1's dirty_note -- L10mm's dv cache and first K-sweep
    were produced at 46e2db69 (clean); L20mm/L40mm and all degradation-arm
    analysis ran at 4afea641 (clean), which adds only an unrelated warp
    diagnostic script. Every number here reconstructs from a clean commit.
  data_commit: >
    Identical to EXP-0024_v1 -- Genesis/data/slates_multistep/n20_L{10,20,40}mm,
    commit 8c006d88, verified per-cell by scripts/probes/verify_slate_cell.py.
  script: >
    scripts/probes/expB_multistep_eval.py (--degradations flag, rebuilding
    EXP-0025/EXP-0026's 13 arms via exp0026_selection_pressure._degradation_arms,
    imported unchanged) writes the dv cache; scripts/probes/exp0026_kcurve.py
    and scripts/probes/exp0026_kcurve_exact.py (new, this session -- see
    EXP-0024_v1) sweep K = 2,4,8,16,32,64,128 against it, both UNCHANGED from
    how EXP-0026 used the sampled script and how EXP-0024_v1 introduces the
    closed-form one.
  data: ["runs_expB/n20_L{10,20,40}mm_dv_cache.pt (step-0 subset, 20 slates x 128 candidates, 13 degradation arms each)"]
  code_path: >
    Identical to EXP-0026's own note: the model comparison (linear, UNet,
    mean-delta) runs through the SAME code path as EXP-0024_v1's checkpoints
    and fits. The degradation arms are EXP-0025/EXP-0026's recipes
    (`_gaussian_blur2d`, `build_pyramid`, `finest_band_noise`, noise seed
    1234) applied inside THIS record's pipeline, as EXP-0026 did relative to
    EXP-0025 -- shape-comparable within this run, not absolute-value
    comparable to EXP-0025/EXP-0026's own degradation numbers.
  seed: >
    Identical slate-level split as EXP-0024_v1 (seed 0). Degradation-arm
    noise seed 1234 (EXP-0025/EXP-0026's own). K-sweep bootstrap seed 0.
  split: >
    Same 30 train / 20 eval slate-level split as EXP-0024_v1; this record
    uses only the 20 eval slates' STEP-0 batches (128 real candidates each,
    2560 transitions/cell) -- steps 1-2 are diverged rollouts, not slates,
    and are not used here at all (unlike EXP-0024_v1, which also scores
    image accuracy on all 3 steps).
  runtime: "dv cache + degradation arms: ~1-2 min CPU/cell (shared with EXP-0024_v1's run). K-sweep, sampled + exact, all arms: ~10-15s CPU/cell."
budget:
  declared: "~4h wall-clock, ~250k tokens (shared with EXP-0024_v1 -- one collection pass across both records)"
  spent: "~3h45min wall-clock, ~230k tokens (see EXP-0024_v1; the GPU training is this record's cost too, since it reuses EXP-0024_v1's checkpoints)"
  outcome: within
design:
  varied:
    push_length_mm: [10, 20, 40]
    K: [2, 4, 8, 16, 32, 64, 128]
    model: [persistence, mean-delta, "linear (ridge->identity)", UNetFilm, oracle]
    degradation: ["displacement k=1/2/4", "amplitude a=0.5/0.75/1.25/1.5",
                  "hf-noise m=0.5/1.0/2.0", "blur s=1.0/2.0", wrong-physics]
  held_fixed:
    predictions: "identical to EXP-0024_v1's fit/checkpoint per cell -- nothing about the models changes; only K, the candidate pool (real, not resampled), and push length vary"
    sampler: "K-subsets drawn WITHOUT replacement (torch.randperm) for the sampled slateK; slateK_exact/worstK/rank_profile are the CLOSED FORM over the model's actual ranking, no subset sampling at all"
    candidate_pool: "128 REAL, distinct actions per slate (the collection's own env count) -- not 32 sampler draws, not a resampled/replicated pool. This is the deepest real (non-resampled) same-state slate this register has scored"
    goal: "corner (center degenerate again, not used)"
    noise_seed: 1234
  baselines: [persistence, mean-delta, oracle]
  metric: "slateK_exact, worstK, rank_profile (docs/experiments/METRICS.md, closed form), with sampled slateK/regret_dv/pick_pctile and slate4 (K=4) as the bridge to EXP-0026's published values"
noise_floor: >
  Paired across the 20 shared eval slates per cell (not the across-slate sd
  -- same correction EXP-0024's reviewer amendment made for C-045). With 20
  slates instead of EXP-0026's 49-50, sem is mechanically wider at matched
  effect size; L10mm's ~10x smaller dv_true spread widens it further there
  specifically (measured UNet-linear sem at K=4: 0.0192 L10, 0.0032 L20,
  0.0010 L40 -- see EXP-0024_v1).
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  PREDICTION SUPPORTED at L20mm and L40mm; L10mm excluded from the verdict
  (confounded, not contradictory -- see below). Systematic arms stay within
  1-3 points from K=4 to K=128 at both clean cells: L20mm amplitude
  a=0.5/1.5 0.9532/0.9421 (K=4) -> 0.9666/0.9657 (K=128); blur s=1.0/2.0
  0.9373/0.9053 -> 0.9554/0.9117; displacement k=1/2/4 0.9370/0.8930/0.7945
  -> 0.9220/0.8871/0.8589. L40mm: amplitude a=0.5/1.5 0.9819/0.9756 ->
  0.9818/0.9820; blur s=1.0/2.0 0.9772/0.9726 -> 0.9765/0.9782; displacement
  k=1/2/4 0.9775/0.9691/0.9529 -> 0.9854/0.9710/0.9550. Independent-error
  arms decline monotonically at both: L20mm hf-noise m=0.5/1.0/2.0
  0.8982/0.8072/0.5989 (K=4) -> 0.8671/0.7236/0.4853 (K=128); L40mm
  0.9750/0.9628/0.9170 -> 0.9579/0.9250/0.8673. mean-delta ITSELF declines
  monotonically at both -- L20mm 0.5515 (K=4) -> 0.2871 (K=128), a near-halving;
  L40mm 0.8555 -> 0.7166 -- the FIRST model in the register whose own capture
  fraction falls with K, exactly the mechanism C-046 predicts for a model
  whose across-candidate variation is warp/state-driven (effectively
  independent per candidate) rather than reflecting real per-action
  discrimination. This confirms C-046 on independent data, 4x the K depth,
  in TWO cells rather than one. L10mm's arms do NOT cleanly separate:
  its own undegraded linear curve RISES with K (0.7934 -> 0.9132, EXP-0024_v1)
  rather than staying flat, so amplitude/blur inherit that rise (e.g.
  amplitude a=0.5 0.7931 -> 0.9132) and hf-noise is nearly flat rather than
  declining (m=1.0: 0.5454 -> 0.5589) -- confounded by the warp-limited,
  small-signal regime EXP-0024_v1 diagnoses, not a counterexample to C-046.
  mean-delta still declines clearly at L10 (0.1830 -> -0.0426), the one
  consistent signal across all three cells.
verdict: supported
downgrades: [provenance, imprecision, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

C-046 was established on 50 slates of 32 sampler-drawn candidates, K<=31 (the
whole slate). Two objections survive that design specifically: (1) K=31 is
still shallow next to a real MPC's hundreds-to-thousands, and (2) 32
sampler draws are not dense, real candidates -- `rank_metrics`' own `slate4`
draws WITH replacement, so even EXP-0026's "without replacement" K-sweep was
still sampling a resampled pool rather than scoring a genuinely deep,
real-action slate. This design changes both at once, independently, on three
push lengths: K reaches 128 (4x deeper), and every one of those 128
candidates is a REAL, distinct, physically executed action -- the collection
itself produced 128 envs per slate at step 0, so there is no with/without
replacement question at all for the closed-form metrics. If the systematic/
independent split held only because K<=31 or because candidates were sparse,
it should break here; if it holds at K=128 on real candidates across three
independent push lengths, C-046 is confirmed well past its original scope.

## What was actually run

Reuses EXP-0024_v1's dv caches unchanged (`runs_expB/n20_L{10,20,40}mm_dv_cache.pt`,
written by `scripts/probes/expB_multistep_eval.py --degradations`, which
rebuilds EXP-0025/EXP-0026's 13-arm spectrum via
`exp0026_selection_pressure._degradation_arms`, imported not reimplemented,
applied to the step-0 linear predictions). Sweeps K = 2,4,8,16,32,64,128
against each cache with BOTH `exp0026_kcurve.py` (sampled, unchanged from
EXP-0026 -- the `slate4`/`slateK`/`regret_dv`/`pick_pctile` bridge) and
`scripts/probes/exp0026_kcurve_exact.py` (closed-form `slateK_exact`/
`worstK`/`rank_profile`, new this session, self-test-validated against
EXP-0026's published numbers -- see EXP-0024_v1's "What was actually run").
No new training, no new fit -- every model here is EXP-0024_v1's checkpoint
and fit, unmodified.

## Numbers

**`slateK_exact` by degradation arm, K=4 -> K=128** (goal=corner):

| arm | L10mm K=4 | L10mm K=128 | L20mm K=4 | L20mm K=128 | L40mm K=4 | L40mm K=128 |
|---|---|---|---|---|---|---|
| linear (undegraded) | 0.7934 | 0.9132 | 0.9533 | 0.9670 | 0.9820 | 0.9790 |
| displacement k=1 | 0.6823 | 0.8841 | 0.9370 | 0.9220 | 0.9775 | 0.9854 |
| displacement k=4 | 0.0788 | 0.1016 | 0.7945 | 0.8589 | 0.9529 | 0.9550 |
| amplitude a=0.5 | 0.7931 | 0.9132 | 0.9532 | 0.9666 | 0.9819 | 0.9818 |
| amplitude a=1.5 | 0.7511 | 0.8980 | 0.9421 | 0.9657 | 0.9756 | 0.9820 |
| blur s=1.0 | 0.7047 | 0.8841 | 0.9373 | 0.9554 | 0.9772 | 0.9765 |
| blur s=2.0 | 0.5449 | 0.8226 | 0.9053 | 0.9117 | 0.9726 | 0.9782 |
| **hf-noise m=0.5** | 0.7218 | 0.7306 | **0.8982** | **0.8671** | **0.9750** | **0.9579** |
| **hf-noise m=1.0** | 0.5454 | 0.5589 | **0.8072** | **0.7236** | **0.9628** | **0.9250** |
| **hf-noise m=2.0** | 0.3081 | 0.3299 | **0.5989** | **0.4853** | **0.9170** | **0.8673** |
| **mean-delta** | 0.1830 | -0.0426 | **0.5515** | **0.2871** | **0.8555** | **0.7166** |
| wrong-physics | 0.0078 | -0.0240 | -0.0018 | 0.0966 | -0.0004 | 0.1003 |

At L20mm/L40mm the split is clean: every systematic arm (displacement,
amplitude, blur) moves by 1-3 points; every independent-error arm (hf-noise,
mean-delta) declines monotonically by 5-27 points. At L10mm the undegraded
`linear` curve itself rises 12 points (0.79->0.91, EXP-0024_v1's warp-limited
finding), so amplitude/blur inherit that rise and hf-noise is confounded to
near-flat -- reported, not used for the verdict. `wrong-physics` stays near
zero everywhere (its undegraded capture is already near-random), too small a
signal to classify either way at any cell.

**`worstK`, K=4 -> K=128** (adversarial pool; L20mm/L40mm shown, the clean
cells): linear 1.0299->-0.0236 / 0.6150->0.0185; UNet 0.7406->-0.0472 /
0.4847->0.0005; mean-delta 1.9016->0.7129 / 1.5865->0.2834; hf-noise m=1.0
1.1783->0.2621 / (L40 not separately tabulated, tracks the slateK_exact
decline). worstK's approach to ~0 at K=128 for every model is the same
degeneracy noted in EXP-0024_v1 (only one candidate subset exists at
K=n_slate, so "adversarial" loses meaning) -- read the K<64 columns as the
informative range for this metric.

## What this means

**C-046 survives at K=128, on real candidates, on independent data, in the
two cells where the underlying comparison is not itself broken.** The
mechanism EXP-0026 proposed -- selection pressure only punishes error that is
independent across candidates, because a shared/systematic perturbation
moves every candidate the same way and cancels under comparison, while an
independent one gives an adversary (or an optimizer) more chances to promote
a mediocre action as the pool grows -- reproduces with a 4x deeper K sweep
and a genuinely dense (not resampled) candidate pool, at TWO push lengths
collected under a different sampler than EXP-0026's original data. This is
the strongest form of C-046 evidence in the register.

**mean-delta's own decline is new evidence, not just a replication.**
EXP-0026 tested degradation ARMS applied to the linear operator; it never
had a model whose OWN capture fraction fell with K, because mean-delta at
K<=31 in the original data happened to stay roughly flat. Here, at K=128, it
clearly does not (L20mm: 0.55->0.29; L40mm: 0.86->0.72; L10mm: 0.18-> -0.04).
The reading is mechanistic, not just empirical: mean-delta predicts ONE
canonical-frame delta for every candidate in a slate, so its across-candidate
variation comes entirely from the warp/state interaction (how that one fixed
delta lands differently depending on each candidate's push geometry) --
which behaves like an independent-per-candidate quantity in exactly the sense
C-046 is about, not like the systematic model-quality difference that keeps
linear/UNet flat. This is the first real test of "does C-046 predict which
MODELS degrade with K, not just which perturbations do," and it says yes.

**L10mm is reported, not smoothed into either side of the verdict.** Its
degradation arms inherit the warp-limited regime's anomalous rising baseline
(EXP-0024_v1), so neither the systematic arms' near-flatness nor hf-noise's
near-flatness at L10mm can be read as evidence for or against C-046 there --
the baseline they would be compared to is itself moving for an unrelated
reason. mean-delta's decline is the one L10mm number kept as supporting,
because it is a comparison to ITS OWN K=4 value, not to the confounded
linear baseline.

## What would change the verdict

- **A push length between 10 and 20mm**, to locate where the warp-limited
  regime ends and find whether C-046's degradation-arm test becomes
  interpretable exactly where EXP-0024_v1's warp-floor check says it should.
  Cost: one more collection cell (~48-71 min sim, scaling with push length
  per `docs/plan_selection_pressure_validation.md`'s measured cost model)
  plus one ~37 min UNet training.
- **K beyond 128** needs a deeper same-state collection (more envs/slate);
  this design's ceiling is the collection's own env count, not the metric.
- **A seed sweep** (EXP-F), as EXP-0024_v1 -- the between-seed sd of the
  UNet-linear gap under degradation is unmeasured.

## Threats

- `provenance`: (1) not drop-in comparable with `Genesis/data/slates/n20_heap_5mm`
  -- same caveats as EXP-0024_v1 (placement-aware, no contact-triggered early
  stop, different sampler/training-set size). (2) The degradation arms here
  are EXP-0025/EXP-0026's recipes applied inside a different collection's
  pipeline -- shape-comparable across K WITHIN this run, not absolute-value
  comparable to EXP-0025/EXP-0026's own numbers, exactly as EXP-0026 itself
  flagged relative to EXP-0025. (3) L10mm's degradation numbers are reported
  but explicitly excluded from the verdict due to the warp-limited confound
  -- stated here again because it is the single most important caveat in
  this record and bears repeating past the frontmatter.
- `imprecision`: one UNet training seed, one linear fit per cell (inherited
  from EXP-0024_v1). 20 eval slates per cell (vs EXP-0026's 49-50) widens
  every paired sem; L10mm's ~10x smaller `dv_true` spread widens its sems
  further still (see EXP-0024_v1's noise_floor).
- `untested-dependency`: `settled-state` unchecked for the rigid-cube path,
  inherited unchanged from EXP-0012/EXP-0024/EXP-0026.
- Considered and dismissed: `selection` -- the degradation arm set, noise
  seed (1234), and K grid were EXP-0026's own, fixed before this session;
  all three cells and all 13 arms are reported, including the one cell
  (L10mm) and the one arm (`wrong-physics`) that do not cleanly support the
  claim.
- Considered and dismissed: `inconsistency` -- the systematic/independent
  split holds at BOTH clean cells (L20mm, L40mm) at every K tested, with
  mean-delta's decline appearing in all three cells including the confounded
  one. The one real inconsistency (L10mm's degradation-arm shapes) is
  explained by a diagnosed, independently-evidenced confound (EXP-0024_v1's
  warp-only baseline), not left unexplained.

## Unrelated findings

- Identical to EXP-0024_v1's: the `PileSweepData._filter_split` per-file
  fallback has no slate concept (relevant here too, since this record's
  step-0-only cache depends on the same manifest-based split); METRICS.md
  documented `slateK_exact`/`worstK`/`rank_profile` with no implementing code
  anywhere in the repo before this session.
- `worstK` at K=n_slate (128 here) is not just numerically small but can be
  slightly NEGATIVE for every model including the oracle (e.g. L10mm oracle
  -0.0754 at K=128) -- a property of the formula's single-valid-rank
  degeneracy at K=n, not a bug: confirmed by checking the oracle case
  algebraically (its rank-by-true-value ordering is exact, so the
  `worstK` inner term is <=0 by construction at every K, and the "adversarial
  pool" freedom that makes the metric meaningful at K<n disappears entirely
  when K=n). Worth a one-line guard in `exp0026_kcurve_exact.py`'s docstring
  for the next reader; not fixed here since it does not change any number
  used in either record's verdict (K=128's worstK column was never load-bearing).
