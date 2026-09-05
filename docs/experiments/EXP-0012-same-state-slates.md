---
# ---- identity -------------------------------------------------------------
id: EXP-0012
title: >
  Same-state candidate slates collected and verified (6 of a planned 50
  states x 32 actions); within-slate hf-noise damage to action ranking is
  smaller than EXP-0008's cross-state measurement at matched degradation
  level, while displacement's near-benign effect is unchanged -- consistent
  with EXP-0008's "independent-noise-per-candidate" mechanism, on n=6 states
tier: T1
mode: confirmatory
date: 2026-09-05
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  EXP-0008's headline mechanism (high-frequency prediction noise is drawn
  independently per candidate and so does not cancel under comparison, while
  systematic degradations like displacement perturb every candidate alike and
  do cancel) was measured on candidate slates drawn from DIFFERENT start
  states (`Genesis/data/cube_spectrum/n20`, one action per state), which
  guarantees the "independent per candidate" property by data construction
  rather than demonstrating it survives a real MPC loop where one state's
  candidates share a prediction context. On genuinely SAME-state slates (this
  collection), the relative damage hf-noise does to within-slate
  Spearman(dV_pred, dV_true) should be smaller than EXP-0008's pooled
  cross-state relative damage at the matched degradation level, while
  displacement's relative damage should be roughly unchanged.

prediction:
  supports: >
    at m=2.0 hf-noise (goal=corner, the only goal with a testable signal --
    see "What was actually run"), the within-slate mean relative Spearman
    drop from the undegraded operator is CLEARLY smaller than EXP-0008's
    pooled cross-state relative drop at the same level (0.971 -> 0.152, an
    84% drop): a threshold of <50% relative drop was set before running Part
    2. AND displacement k=1..4's within-slate relative drop stays close to
    EXP-0008's cross-state figure (0.971 -> 0.914, 6% drop), i.e. within a
    factor of ~2.
  refutes: >
    the within-slate hf-noise relative drop at m=2.0 is >= EXP-0008's 84%
    cross-state figure (same-state framing does not shrink the damage), OR
    the undegraded within-slate Spearman itself is not clearly above 0 (no
    ranking signal exists to destroy, making the comparison moot).
  discriminating: true

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 006004d0
  script: >
    Genesis/same_state_slate_collection.py (new, this record) for collection;
    scripts/probes/same_state_degradation.py (new, this record; reuses
    scripts/probes/degradation_spectrum.py's exact recipes and
    control_utility_test.py's lyapunov_weights/lyapunov/rank_metrics
    unmodified) for evaluation.
  data: ["Genesis/data/slates/n20_heap_5mm/*_data.pt (this collection, 6
         states)", "Genesis/data/cube_spectrum/n20/*_data.pt (fit only,
         disjoint)"]
  code_path: particles_to_occupancy (via occupancy_foresight.load_transition_fields, view="mask")
  seed: 0
  split: >
    No train/test split needed within the slate data -- the operator is
    fit ENTIRELY on Genesis/data/cube_spectrum/n20 (a physically separate
    collection run, so this is a stronger held-out guarantee than EXP-0008's
    own episode split within one dataset) and evaluated on every slate
    transition. Grouping for the within-slate metric is by source file
    (`ep` from load_transition_fields), which equals the slate index by
    construction (Genesis/same_state_slate_collection.py calls
    collect_data_samples once per state with n_samples=1, so its
    auto-incrementing batch counter IS the slate id).
  runtime: "collection: ~6 min GPU (6 states x 32 envs, stopped early -- see below); eval: ~5s CPU"

budget:
  declared: "90 min wall-clock, 200k tokens (task-level; not a per-tier skill default)"
  spent: "~95 min wall-clock (collection ran past the declared budget while I waited on it instead of proceeding with what had already landed; stopped on coordinator instruction), ~150k tokens"
  outcome: exceeded

design:
  varied:
    degradation: [displacement k=1/2/4, hf-noise m=0.5/1.0/2.0]
    goal: [center, corner]
  held_fixed:
    view: mask
    cube_size: 0.005
    min_grains: 1.0
    grid: 64
    canon_res: 64
    crop: 1.0
    ridge: 1.0
    estimator: "ridge toward identity, IDENTICAL fit object to EXP-0008 (fit once on cube_spectrum/n20, reused for all slate transitions)"
    region: swept_region_mask, half_width=0.5*plate+2px, pad=0.5*plate
    material/action config: 20 cubes, 5 mm, heap spawn, density 1000, friction 0.3,
      perpendicular contact-aware pushes, fixed 20 mm push length -- IDENTICAL to
      Genesis/data/cube_spectrum/n20 (this collection reuses
      SandboxManipulation.collect_data_samples(pile_aware=True) and
      StateLibrary.apply_per_env unmodified; only the driver loop is new)
    n_settles_for_library: 3 (96 candidate library states, augment=False)
  baselines: [persistence (dv_pred=0 always, cannot rank -- degenerate by construction)]
  metric: >
    rank_metrics (Pearson, Spearman, sign agreement, slate4/slate16 regret)
    from control_utility_test.py, computed WITHIN each same-state slate
    (group = source file, up to 32 candidates sharing one start state) then
    averaged across slates -- the new axis this record adds relative to
    EXP-0008's single pooled-across-all-transitions correlation. Slates with
    <8 live candidates would be dropped (none were, at n=6 slates x ~32
    candidates each).

noise_floor: >
  Not independently measured (no repeated seeds / folds -- worse than
  EXP-0008's already-flagged gap, since n=6 slates is far too few for a
  bootstrap CI). The per-slate SD reported alongside each mean IS the
  dispersion across the 6 states, and it is large relative to some of the
  effects discussed (e.g. hf-noise m=2.0 corner: mean drop to 0.441, sd
  0.177) -- read as a plausibility argument, not a calibrated floor.

depends_on: [canonical-warp, warp-blend, swept-region-metric, episode-split,
            footprint-splat, settled-state]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Collected 6 of the planned 50 same-state slates (32 candidates/slate, 191
  live transitions) before stopping at budget; all 6 pass every verification
  check (start states identical across the 32 envs to ~1e-11, actions and
  post-push outcomes vary substantially, no NaN/escaped particles). Part 2:
  the `center` goal is degenerate for this collection by a geometric
  mechanism, not a bug (see below) -- 0/191 transitions have any center-goal
  dV at all, so only `corner` is informative. On `corner` (n=6 slates):
  undegraded within-slate mean Spearman = 0.912 (sd 0.025). hf-noise relative
  drops: m=0.5 4.3%, m=1.0 13.6%, m=2.0 51.6% (sd 0.177 at m=2.0 -- large).
  Displacement relative drops: k=1 1.2%, k=2 3.4%, k=4 7.1%. Prediction
  PARTIALLY SUPPORTS: at m=2.0 the within-slate relative drop (51.6%) is
  smaller than EXP-0008's cross-state corner figure (84%) but does not clear
  the <50% threshold set in advance -- borderline, and with sd this large
  (n=6) not clearly distinguishable from either side. Displacement's relative
  drop (7.1%) is close to EXP-0008's cross-state figure (6%), as predicted.
  The qualitative pattern -- noise damage shrinks under same-state framing,
  displacement's (already small) damage does not change -- holds, but n=6 is
  too thin to call the quantitative threshold met.

verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If EXP-0008's mechanism is right (noise fails to cancel across candidates
because it is drawn independently PER CANDIDATE, and that independence was an
artefact of candidates coming from different states), then forcing candidates
to share a state should let at least part of the noise's effect cancel in a
within-slate comparison, shrinking (not necessarily eliminating) its damage to
ranking -- while displacement, which was already systematic across candidates
in the cross-state design, gains nothing new from same-state framing and
should look about the same. If instead same-state framing does nothing (noise
damage is just as large, or displacement now looks different too), the
"cross-state confound" story fails on its own chosen ground and EXP-0008's
number stands as a property of the operator, not an artefact of the dataset
that measured it.

## What was actually run

**Collection** (`Genesis/same_state_slate_collection.py`, new). Builds a
`StateLibrary` of settled 20-cube, 5 mm heap piles (`--spawn-mode heap`,
matching `Genesis/data/cube_spectrum/n20`'s material/action configuration
exactly -- density 1000, friction 0.3, `pile_aware=True`, fixed 20 mm
perpendicular contact-aware pushes), then for each of N chosen library states
calls `StateLibrary.apply_per_env(sim, indices=[k]*n_envs)` -- the SAME state
index repeated for every env, per `state_library.py`'s own documented use case
-- followed by one `collect_data_samples(n_samples=1, pile_aware=True, ...)`
call. No new sim wrapper: this is `cube_spectrum_collection.py`'s exact
material/spawn/action configuration plus `state_library.py`'s existing
`apply_per_env`, called from a thin loop. Because `collect_data_samples`
auto-increments its own batch counter from the file count already in the
output directory, one call per state means batch index == slate index for
free; a `manifest.json` records `{batch_idx: state_library_index}` for
traceability (recomputed post-hoc for the batches that landed, from the
deterministic seed-0 `rng.choice` call -- not re-simulated, see the file).

Two smoke tests (n_envs=4, n_states=3) were run FIRST as pure feasibility/cost
pilots (runtime and correctness only, no outcome that bears on the Part 2
prediction) before committing to the real run -- one caught a path-doubling
bug (`Genesis/Genesis/data/...`) from copying `Path(__file__).parent`-relative
logic incorrectly; fixed before the real run.

**The real run was launched at `--n-envs 32 --n-states 50` and stopped after 6
states (18 files) on explicit coordinator instruction** ("never block on a
watcher... proceed with what exists... no new collection"), because I had
been waiting on progress notifications rather than working with what had
already landed -- a real process error on my part, corrected once flagged.
Throughput at the point of stopping was ~17s/slate (32 envs), so the full 50
would have taken roughly 14 minutes total; the shortfall is entirely from how
I spent the wait, not from infeasibility. **Per-instruction, no further
collection was run to make up the gap.**

**Verification** (the deliverable that matters most), run on all 6 collected
slates:

| batch | max pos diff within slate (should be ~0) | max quat diff | angle sd (rad) | p_start sd norm | max post-push diff (should be >0) | z range (m) | NaN? |
|---|---|---|---|---|---|---|---|
| 0 | 1.8e-12 | 1.5e-11 | 0.950 | 0.0184 | 1.390 | [0.0125, 0.0175] | no |
| 1 | 1.7e-13 | 1.5e-11 | 0.882 | 0.0210 | 0.708 | [0.0125, 0.0175] | no |
| 2 | 4.6e-13 | 1.1e-11 | 0.961 | 0.0200 | 0.707 | [0.0125, 0.0175] | no |
| 3 | 1.5e-11 | 1.5e-11 | 0.996 | 0.0210 | 0.708 | [0.0125, 0.0175] | no |
| 4 | 0.0 | 1.0e-11 | 0.918 | 0.0213 | 0.720 | [0.0125, 0.0175] | no |
| 5 | 1.5e-11 | 1.5e-11 | 0.829 | 0.0216 | 0.707 | [0.0125, 0.0175] | no |

Start-state position/orientation differences across the 32 envs are at
float32 noise level (1e-11 to 1e-13, vs. particle positions of order 1e-2 m)
-- the states really are identical, not merely similar. Actions differ
substantially (angle sd ~0.83-1.0 rad across a ~pi-wide sampling range,
p_start sd norm ~0.02 m). Post-push states differ substantially between envs
(max diff 0.7-1.4, vs. a state tensor whose position entries are O(1e-2)),
confirming the identical start really was pushed 32 different ways with 32
different outcomes. `z` stays in a tight, physically sane band
(12.5-17.5 mm, i.e. within one cube height of the floor) with no NaN and 0
failed pushes out of 192 attempted (1 push dropped later by the 19.9 mm
length filter in `load_transition_fields`, unrelated to settling). All of
this is consistent with a genuinely settled, genuinely shared start state and
genuinely varied actions -- **Part 1's verification passes on everything
collected.**

**Part 2 evaluation** (`scripts/probes/same_state_degradation.py`, new).
Fit the identical ridge-toward-identity operator EXP-0008 used (res=64,
crop=1, ridge=1) on `Genesis/data/cube_spectrum/n20` in full (no held-out
split needed -- the slate data is a physically disjoint collection run and
never touches the fit). Built the same degraded fields (displacement via
`torch.roll`, hf-noise via the same finest-Laplacian-band recipe anchored to
the slate data's own finest-band signal std) and scored them with
`control_utility_test.py`'s `rank_metrics`, but grouped by slate (source
file) before averaging instead of pooling every transition together.

**Deviation from plan, discovered while running, not before:** the `center`
goal (`H//4:3H//4` square, the middle 50% of the grid in both axes) produced
`dv_true == 0.0` for literally all 191 transitions -- not approximately zero,
bit-identical zero. Investigated rather than silently dropped: `o0`'s
occupied pixels span rows/cols 25-38 and `o1`'s span 18-45, both entirely
inside the [16, 48) center mask in every single transition. The Lyapunov
weight field `d` (distance-transform to the target region) is exactly 0
everywhere inside the mask, so `V = sum(d*y)/sum(y) = 0` for both `o0` and
`o1` whenever the WHOLE pile sits inside the mask -- and it always does here,
for a structural reason: a 20-cube, 5 mm heap footprint (25-46 mm) fits
comfortably inside the 64 mm central half of the 128 mm tray, and this
collection takes exactly ONE push per freshly-centred spawn (by design, to
keep the same-state property clean), so there is no cumulative multi-push
drift toward the tray edge the way `cube_spectrum`'s 5-sequential-push
episodes have. EXP-0008's cross-state center signal (mean dV +0.005, 7%
helpful) most likely comes disproportionately from LATER pushes in a
5-push episode, after the pile has already drifted off its spawn-centred
start -- a mechanism this single-push same-state design cannot reproduce for
the center goal specifically. This is reported as a genuine, mechanistically
understood design consequence (see Threats), not swept under the rug: **only
`corner` is informative for this collection**, and the `center` cell is
`incomplete-design`, not `inconclusive-and-hidden`.

## Numbers

Within-slate rank_metrics, goal=corner (n=6 slates, mean across slates ± sd
across slates; `dv_true` mean +0.02128, sd 0.05224, 32% of pushes helpful):

| model | n slates | spearman (mean) | spearman (sd) | sign % | slate4 (mean) | slate4 (sd) | rel. drop from undegraded |
|---|---|---|---|---|---|---|---|
| persistence | 6 | -0.007 | 0.122 | 0% | 0.002 | 0.020 | (degenerate, dv_pred=0 always) |
| operator (undegraded) | 6 | 0.912 | 0.025 | 94% | 0.946 | 0.021 | -- |
| displacement k=1 | 6 | 0.901 | 0.036 | 92% | 0.930 | 0.022 | 1.2% |
| displacement k=2 | 6 | 0.881 | 0.048 | 88% | 0.906 | 0.031 | 3.4% |
| displacement k=4 | 6 | 0.847 | 0.064 | 80% | 0.865 | 0.054 | 7.1% |
| hf-noise m=0.5 | 6 | 0.873 | 0.045 | 68% | 0.899 | 0.013 | 4.3% |
| hf-noise m=1.0 | 6 | 0.788 | 0.054 | 68% | 0.783 | 0.062 | 13.6% |
| hf-noise m=2.0 | 6 | 0.441 | 0.177 | 68% | 0.462 | 0.159 | 51.6% |

goal=center: `dv_true` is identically 0.0 for all 191 transitions (see "What
was actually run" for why) -- no row is reportable.

Comparison to EXP-0008's cross-state corner numbers (from its prose, since
only its center table is tabulated in full): undegraded 0.971 -> displacement
k=1..4 down to 0.914 (6% relative drop) -> hf-noise m=0.5..2.0 down to 0.152
(84% relative drop, at m=2.0).

| quantity | cross-state (EXP-0008, corner) | same-state (this record, corner) |
|---|---|---|
| undegraded Spearman(dV) | 0.971 | 0.912 |
| displacement family, relative drop at k=4 | 6% | 7.1% |
| hf-noise family, relative drop at m=2.0 | 84% | 51.6% (sd 0.177 on the raw 0.441) |

Displacement's relative damage is essentially unchanged between framings (6%
vs 7.1%). hf-noise's relative damage is smaller same-state than cross-state at
every level tested, most clearly at m=0.5 (4.3% vs a family that starts
somewhere above 0% and reaches 84% by m=2.0 -- exact cross-state per-level
corner numbers were not published, only the two endpoints) and least clearly
at m=2.0, where 51.6% vs 84% is a real gap but the sd (0.177 on a mean of
0.441, from only 6 slates) is large enough that this specific comparison
should not be leaned on alone.

## What would change the verdict

- **The other 44 states.** This is the single biggest lever: at ~17s/slate
  the full 50-state, 32-env collection is ~14 minutes of GPU time, well
  within a normal budget -- it was not run only because of a process error
  (waiting on notifications instead of using partial data), not because it is
  expensive or infeasible. Re-running `python -m Genesis.same_state_slate_collection
  --n-envs 32 --n-states 50 --tag n20_heap_5mm --seed 1` (a fresh seed, so it
  adds to rather than duplicates the existing 6 files if pointed at a new
  tag) would let the m=2.0 comparison clear or fail its predetermined
  threshold with an actual standard error instead of an n=6 spread. Cost:
  ~15 min GPU, ~5 min analysis.
- **A goal that is not geometrically degenerate for a single-push, centred-
  spawn collection.** `center`'s failure mode here is structural, not a
  sampling accident -- it will stay at dV=0 no matter how many more same-
  state slates are collected, UNLESS either (a) the spawn is deliberately
  off-centred for some states, or (b) multiple SEQUENTIAL same-state pushes
  are collected per state so the pile can drift the way `cube_spectrum`'s
  5-push episodes do (this would need a design decision about whether later
  pushes in a sequence still count as "same-state" candidates for THIS
  state, or become the start state for a new slate -- not resolved here).
  Cost: a config change plus the same ~15 min collection.
- **A fold sweep / repeated seeds** to turn the n=6 (or eventually n=50)
  slate-to-slate sd into an actual noise floor, rather than a raw spread.

## Threats

- `incomplete-design`: stopped at 6/50 planned states (process error, not
  infeasibility -- see above); `center` goal is structurally uninformative
  for this collection design and reports no cell at all, rather than the two
  goals `control_utility_test.py` and EXP-0008 both use.
- `imprecision`: n=6 slates is far below what a rank-correlation-of-a-
  rank-correlation comparison wants; the sd at the harshest degradation level
  (hf-noise m=2.0: sd 0.177 on a mean of 0.441) is comparable in scale to the
  effect being compared against EXP-0008's number. No fold sweep or repeated
  seed to calibrate a real noise floor.
- `untested-dependency`: `settled-state` is `unchecked` for the rigid-cube
  path in `INVARIANTS.md`; this record leans on it more than most (the
  "identical start state" property is only meaningful if that state is
  actually at rest, not mid-settle), and only measured a proxy for it
  (sane z-range, no NaN, deterministic-across-envs reproduction of an
  already-settled library state) rather than a real velocity check at record
  time.
- Considered and dismissed: `provenance` -- single code path throughout
  (`particles_to_occupancy` via `load_transition_fields`), and the fit is
  the SAME object EXP-0008 used (loaded from a disjoint dataset, not
  re-derived), so no cross-path or cross-fit comparison is being made.
- Considered and dismissed: `indirectness` for the control-utility side --
  `dv_true` is the realised dV of an actually-executed push, exactly as in
  EXP-0008, satisfying the ideas-log sec.6 circularity guard.
- Considered and dismissed: `selection` -- both goals were run and both are
  reported, including the one (`center`) that turned out to be
  uninformative; the degenerate result was investigated and explained rather
  than dropped.

## Unrelated findings

- One push in the 6 collected slates (out of 192 attempted) could not travel
  the full 20 mm even from a clamped start and was shortened (logged WARNING
  from `sandbox_manipulation_clean.py`'s pile-aware sampler); it was
  correctly excluded by `load_transition_fields`'s 19.9 mm length filter, so
  it did not enter the analysis, but it means one of the 32 candidates in
  that slate has 31 rather than 32 live candidates. Not otherwise
  investigated.
- `Genesis/same_state_slate_collection.py`'s first draft duplicated the
  `Genesis/` path segment (writing to `Genesis/Genesis/data/slates/...`)
  because it independently recomputed `Path(__file__).parent / out_path` for
  its own manifest bookkeeping using the same `out_path` string already
  destined for `collect_data_samples`, which resolves paths the same way
  internally. Caught by the smoke test before the real run; not a defect in
  any file another agent depends on, since the module is new this session.

## Process note

I stopped actively working and waited passively on the collection's progress
notifications instead of checking what had already landed and proceeding --
flagged and corrected by the task-giver mid-run. The "never block on a
watcher" addition to the experiment-log skill exists because of exactly this
failure mode; recorded here so it is visible in the same place the rest of
this record's honesty commitments live.


## Reviewer amendment, 2026-09-05: completed at n=50, verdict upgraded

This record was written on **6 of the 50 planned start states**, because the
run spent its budget watching progress rather than working with data already on
disk, and it correctly graded itself `inconclusive` on that sample. The
coordinator then finished the collection (44 more states, seed 100, ~15 min at
~17 s/state) and re-ran `scripts/probes/same_state_degradation.py` unchanged.

**Independent verification of all 50 slates** (not just the original 6):

| check | value |
|---|---|
| max within-slate start-state spread | **4.66e-10 m**, against positions of order 1e-2 m |
| min within-slate action angle sd | 0.709 rad |
| post-push z range | 11.8 … 17.5 mm (settled band) |
| transitions / NaN outcomes | **1597 / 0** |

### Results at n=50, goal=corner (within-slate, then averaged across slates)

| model | Spearman(dV) | sd | slate-4 | sd | **relative drop** |
|---|---|---|---|---|---|
| operator, undegraded | **0.925** | 0.021 | 0.956 | 0.017 | — |
| displacement k=1 | 0.909 | 0.028 | 0.940 | 0.022 | 1.7% |
| displacement k=2 | 0.889 | 0.035 | 0.916 | 0.036 | 3.9% |
| displacement k=4 | 0.855 | 0.043 | 0.869 | 0.049 | **7.6%** |
| hf-noise m=0.5 | 0.872 | 0.035 | 0.872 | 0.045 | 5.7% |
| hf-noise m=1.0 | 0.782 | 0.070 | 0.769 | 0.087 | 15.5% |
| hf-noise m=2.0 | 0.536 | 0.119 | 0.550 | 0.122 | **42.1%** |

### What this settles

**The confound was real and inflated the effect about two-fold.** EXP-0008
measured hf-noise costing 84% of ranking quality with candidates drawn from
different states — which *guarantees* independent noise per candidate by
construction. Same-state, it costs **42%**.

**The mechanism survives.** Noise still damages ranking **5.5x more than
displacement** (42.1% vs 7.6%), and the ordering holds at matched or lower rms:
EXP-0008 measured hf-noise m=1.0 at rms 0.2958 against displacement k=2 at
0.3189, and here the lower-rms noise arm ranks worse (0.782 vs 0.889). So
"ranking is destroyed by variance, not bias" stands as a qualitative claim; only
its magnitude was overstated.

At n=50 the spread is tight enough to say so: undegraded 0.925 ± 0.021 against
hf-noise m=2.0 at 0.536 ± 0.119 is about three standard deviations of the
noisier arm. That is why the verdict moves from `inconclusive` to `supported`
and the grade from `very-low` to `low` (the reviewer first wrote `moderate`
and the validator refused it: this record still cites `settled-state`, which is
`unchecked` for the rigid-cube path, so `untested-dependency` is genuinely
earned. Second time today the duplicate/downgrade checks have caught their own
author rather than a subagent.)

### The `center` goal is degenerate, and this is worth carrying

Only 2 of 50 slates showed any variation at all under `center`
(`dv_true` sd 0.0013, 0% of pushes helpful). A compact centred heap never has
mass outside a centred square mask, so `V = 0` before and after, identically.
Its rows above are noise on n=2 and must not be read — persistence scoring
Spearman 0.911 there is an artifact of two slates, not a finding.

**Consequence for MPC benchmark design:** a centred convex target is useless
for a centred pile. Off-centre targets (`corner`) are the informative simple
goals. This is cheap to get wrong and expensive to notice later.

### Remaining threat

One seed, one geometry (n=20 cubes, heap spawn, 20 mm pushes), one push per
start state. Whether the 2x inflation factor is general or specific to this
configuration is untested.
