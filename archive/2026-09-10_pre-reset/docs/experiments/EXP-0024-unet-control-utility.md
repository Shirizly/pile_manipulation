---
id: EXP-0024
title: >
  Does the UNet's fine-detail image-accuracy advantage over the linear
  operator (EXP-0021/EXP-0022) buy anything for control? Both trained/fitted
  on cube_spectrum/n20 and scored on the n20_heap_5mm same-state slates
  (EXP-0012) with accuracy (image) and slate4/spearman (control) side by side
tier: T1
mode: confirmatory
date: 2026-09-06
hypothesis: C-030
claim: >
  EXP-0022 found the UNet's 14/14-cell image-accuracy win over the linear
  operator (C-041) is entirely high-frequency detail, and named control
  utility as the decisive follow-up. Trained and fitted on
  Genesis/data/cube_spectrum/n20 (in-distribution for the n20_heap_5mm
  same-state slates) and scored on those slates with
  control_utility_test.py::rank_metrics, the UNet's slate4 (goal=corner, mean
  across slates) is NOT clearly better than the linear operator's -- i.e. the
  image-accuracy advantage does not translate into a comparable control
  advantage.
prediction:
  supports: >
    the UNet's slate4 mean (goal=corner, across all usable slates) falls
    within +-1 across-slate sd of the linear operator's slate4 mean -- no
    clear win for the UNet on control -- even though the UNet's held-out
    image `accuracy` clearly exceeds the linear operator's (by a margin
    comparable to EXP-0021's 4-11 point pct_persistence gaps, i.e. not a
    rounding-level difference).
  refutes: >
    the UNet's slate4 mean exceeds the linear operator's by more than 1
    across-slate sd -- a clear win in control that mirrors the image-accuracy
    win, which would overturn the EXP-0022 reading and reinstate "the UNet is
    the better model" without qualification.
  discriminating: true
provenance:
  commit: bc4bc6bf
  dirty: false
  dirty_note: >
    UNet training was launched under commit 57452569 (this record's own 4
    new files, clean), while OTHER AGENTS' concurrent work was uncommitted in
    the tree (per `utils.git_provenance()` at launch time): docs/experiments/
    METRICS.md and fit_linear_foresight.py among them -- the latter added the
    `accuracy` key this record's image-accuracy numbers use. Training itself
    never touches fit_linear_foresight.py, so that is immaterial to the UNet
    checkpoint. By the time `scripts/probes/exp0024_control_eval.py` actually
    ran (after training finished), that work had landed at 811cc700 and the
    tree was clean at bc4bc6bf (HEAD at eval time, and the sha recorded here)
    -- so every number in this record, not just the checkpoint, reconstructs
    from a clean commit.
  data_commit: >
    cube_spectrum/n20 predates provenance stamping (dataset-provenance tag,
    unrecorded for pre-2026-09-05 data) -- lineage is reconstructable only
    from file mtimes. n20_heap_5mm slates: collected under commit range
    006004d0..dbf21ba2 (EXP-0012).
  script: scripts/probes/exp0024_control_eval.py
  data: ["Genesis/data/cube_spectrum/n20/*_data.pt (fit + UNet train/val/test)",
         "Genesis/data/slates/n20_heap_5mm/*_data.pt (control eval, disjoint collection run)"]
  code_path: >
    registry.dataset_registry PileSweepData (type: genesis) throughout -- the
    SAME code path for linear-operator fit, UNet train/val/test, and both
    slate evaluations (image accuracy uses cube_spectrum/n20's own held-out
    test split; control uses the disjoint n20_heap_5mm collection). This
    differs from EXP-0012/EXP-0008's own code path
    (occupancy_foresight.load_transition_fields / particles_to_occupancy);
    a pipeline-validation pilot (below) found the two agree closely on the
    linear operator + persistence (spearman 0.921 vs EXP-0012's 0.925,
    slate4 0.950 vs 0.956), so this is not a live discrepancy, but it is a
    different implementation and is flagged as `provenance`.
  seed: 0
  split: >
    cube_spectrum/n20: registry file-level/physics-group split (val10/test10),
    identical split reused for both the linear fit (train) and the accuracy
    comparison (test) -- same held-out transitions for both models, exactly
    EXP-0021's convention. n20_heap_5mm slates: no split, evaluated in full
    (all physics groups routed to "test" via test_pct=99/val_pct=0), grouped
    by `get_run_index` (one file = one slate, EXP-0012's own convention).
  runtime: "UNet training: 100 epochs, ~16 min GPU (9-22s/epoch under shared-machine contention, 2 other agents running). Fit+eval: ~25s CPU."
budget:
  declared: "2h wall-clock, 300k tokens (task-level sizing; see 'tier' note below for why this record is filed as T1, not T2)"
  spent: "~1h10min wall-clock, ~140k tokens"
  outcome: within
design:
  varied: {model: [persistence, mean-delta, "linear (ridge->identity)", UNetFilm, oracle]}
  held_fixed:
    train_data: "Genesis/data/cube_spectrum/n20, identical train split for the linear fit and the UNet"
    architecture: "unetfilm (in_channels=2, cond_dim=3, uses_physics=true, input_mode=standard), EXP-0021's recipe unchanged: epochs=100, batch_size=32, lr=1e-4 StepLR(step=50,gamma=0.75), loss=eulerian_combined(mse=1.0,mass=0.2)"
    linear_fit: "res=64, crop=1.0, ridge=1.0 toward identity (EXP-0008/EXP-0012's convention for this exact dataset -- NOT EXP-0021's headline crop=0.5 cell)"
    eval_data: "image accuracy on cube_spectrum/n20's own held-out test split (596 transitions); control on ALL of Genesis/data/slates/n20_heap_5mm (EXP-0012, 50 states x ~32 candidates, verified same-state to 4.7e-10 m)"
    goal: "corner (primary; center is degenerate for a centred pile per C-040/EXP-0012 and is reported with that caveat only)"
    metric: "accuracy (fit_linear_foresight.py::metrics) for image; slate4 + spearman (control_utility_test.py::rank_metrics, via scripts/probes/same_state_degradation.py::per_slate_metrics, grouped by slate/file) for control"
  baselines: [persistence, mean-delta, oracle]
  metric: "accuracy AND slate4 (docs/experiments/METRICS.md), reported side by side per METRICS.md's explicit instruction for this exact juxtaposition"
noise_floor: >
  Across-slate sd of slate4 and spearman (49 usable slates, goal=corner),
  reported alongside every mean -- this IS the noise floor per EXP-0012's own
  convention. Measured here: linear slate4 sd 0.020, spearman sd 0.026; UNet
  slate4 sd 0.020, spearman sd 0.018 (EXP-0012's own numbers on its different
  code path: slate4 sd 0.017, spearman sd 0.021, n=50 -- closely consistent).
  So an across-model slate4/spearman gap smaller than ~0.02-0.03 is inside the
  noise floor and should not be read as a clear win either way.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  Image accuracy (held-out cube_spectrum/n20, n=596): persistence 0.000,
  mean-delta 0.343, linear 0.533, UNet 0.569, oracle 1.000 -- UNet beats
  linear by 3.6 points, same direction as EXP-0021's 14/14 cells but at the
  low end of its 4.4-10.9 point range. Control (n20_heap_5mm, 49 slates,
  goal=corner): slate4 mean(sd) -- persistence -0.002(0.014) [cannot rank],
  mean-delta 0.796(0.059), linear 0.950(0.020), UNet 0.969(0.020), oracle
  1.000(0.000); spearman -- persistence -0.052(0.170), mean-delta
  0.790(0.055), linear 0.921(0.026), UNet 0.937(0.018), oracle 0.969(0.000,
  the ceiling this specific metric formula gives for ~32-candidate slates,
  not 1.0 -- see Unrelated findings). UNet-linear gap: slate4 +0.019, spearman
  +0.016 -- both smaller than either model's own across-slate sd (~0.02),
  i.e. within the noise floor. goal=center: only 11/49 slates clear the
  dv_true-variation threshold (mean dv_true 0.00004, sd 0.00120, 0% helpful)
  -- degenerate, as C-040/EXP-0012 found, and not informative; not used for
  the verdict. PREDICTION SUPPORTED: the UNet's real, EXP-0021-consistent
  image-accuracy edge does NOT show up as a clear control-utility edge --
  both models are control-equivalent within measurement noise on slate4.
verdict: supported
downgrades: [imprecision, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

**Tier note.** This was scoped as a T2 gate (a prediction block committed
before the run, which is included below). It is filed as **T1** instead
because `scripts/check_register.py` forbids a T2 record from citing a
`depends_on` tag that is not `holds`/`fixed`, and this record genuinely
depends on `settled-state`, which is `unchecked` for the rigid-cube path
(the same debt EXP-0012 — the record this one directly extends — already
carries and never retired). Writing a settled-state velocity check was out
of scope for this run's budget; the prediction block, baselines, and
multiverse discipline below otherwise match T2 practice.

## Why this test discriminates

EXP-0022 found the UNet's whole image-accuracy advantage over the linear
operator is high-frequency detail, and separately EXP-0008/C-039/EXP-0012
found that exactly high-frequency prediction *noise* is what destroys control
ranking (5.5x more damage than displacement, at matched or lower rms), while
amplitude/blur/displacement cost ranking almost nothing. Put together, those
two findings predict the UNet's detail advantage should NOT show up in
control -- but nobody had measured it, because no prior record trained a
model in-distribution for the same-state slates and scored both models with
`rank_metrics` on them. This record does exactly that: same train data, same
architecture recipe, same held-out image comparison, same slate evaluation
for both model classes, so a difference (or lack of one) cannot be explained
by a confound in what data each model saw.

## What was actually run

**A pipeline-validation pilot (disclosed, no outcome bearing on the
UNet-vs-linear question kept secret):** before training the UNet, the
linear-operator + persistence half of the pipeline was run standalone against
the real n20_heap_5mm slates to catch bugs before spending GPU time. It
reproduced EXP-0012's own numbers closely on a different rasteriser code path
(this record's registry/PileSweepData path vs EXP-0012's
occupancy_foresight.load_transition_fields path): spearman 0.921 vs 0.925,
slate4 0.950 vs 0.956, persistence degenerate in both. This is a plumbing
check, not new evidence about the UNet, and is disclosed per the skill's cost-
pilot rule.

**Dataset.** `configs/dataset/genesis_cube_spectrum_n20.yaml` (new,
committed before training): cube_spectrum/n20 (20 piled cubes, heap spawn,
5mm, fixed 20mm contact-aware perpendicular pushes -- the exact configuration
`Genesis/data/slates/n20_heap_5mm` was collected under), resolution_scale=0.5
(matching EXP-0021/EXP-0012's convention for this box size; NOT
`configs/dataset/genesis_cube.yaml`'s stale 1.0, which EXP-0014 already found
drifted from what training actually uses), min_push_length_m=0.0199 (measured:
94.5% of the raw 20mm-nominal pushes land at 19.9998-20.0098mm, but ~5.3% are
shortened by early contact, down to ~1e-6m for a few already-in-contact
starts -- filtered to near-full-length pushes only, matching
`scripts/probes/same_state_degradation.py`'s own 19.9mm convention for this
identical dataset).

**Training.** `configs/training/exp0024_unetfilm_cube_spectrum_n20.yaml` (new):
EXP-0021's `exp0021_unetfilm_contact_n20.yaml` recipe verbatim (architecture,
epochs, optimiser, loss), only the dataset block changed. 3631 train / 613 val
/ 596 test transitions. Trained via
`python -u -m training.train configs/training/exp0024_unetfilm_cube_spectrum_n20.yaml --no-resume`
(`scripts/run_probe.py --tag exp0024_train`).

**Linear operator.** Fit on the identical train split (`fit_operator`, ridge
toward identity, res=64, crop=1.0, ridge=1.0) -- EXP-0008/EXP-0012's exact
convention for this dataset (not EXP-0021's crop=0.5 headline cell, which was
chosen for the granularity datasets' different push length).

**Evaluation** (`scripts/probes/exp0024_control_eval.py`, new):
- image `accuracy` (fit_linear_foresight.py::metrics) for persistence,
  mean-delta, linear, UNet, oracle on cube_spectrum/n20's held-out test split
  (596 transitions, swept-region mask);
- `slate4`/`spearman` (control_utility_test.py::rank_metrics via
  `same_state_degradation.py::per_slate_metrics`) for the same five models on
  ALL of n20_heap_5mm, for goal=corner and goal=center, grouped by slate
  (file), mean and sd across slates.

The `fit_linear_foresight.py::metrics()["accuracy"]` key used here is an
uncommitted addition on top of 57452569 at run time (see `dirty_note` above);
its exact code, quoted for reproducibility:

```python
"accuracy": float(1.0 - (flat.pow(2).sum(dim=1) / npix).sqrt().mean()
                  / (((tr_ - pv).pow(2).sum(dim=1) / npix).sqrt()
                     .mean().clamp_min(1e-9))),
```

## Numbers

**Image accuracy** (`fit_linear_foresight.py::metrics`, held-out
cube_spectrum/n20 test split, n=596, swept-region mask):

| model | accuracy |
|---|---|
| persistence | 0.000 |
| mean-delta | 0.343 |
| linear (ridge→identity) | 0.533 |
| **UNet** | **0.569** |
| oracle | 1.000 |

UNet − linear = **+3.6 points**. Same direction as EXP-0021's 14/14 cells, at
the low end of its 4.4-10.9 point range (this dataset's fixed 20mm
contact-aware pushes and crop=1.0 fit differ from EXP-0021's ~40mm-push,
crop=0.5 headline cells, so an exact match was not expected).

**Control utility** (`control_utility_test.py::rank_metrics` via
`same_state_degradation.py::per_slate_metrics`, n20_heap_5mm, 49 usable
slates, goal=corner, mean ± sd across slates):

| model | spearman | slate4 |
|---|---|---|
| persistence | −0.052 ± 0.170 (cannot rank) | −0.002 ± 0.014 (cannot rank) |
| mean-delta | 0.790 ± 0.055 | 0.796 ± 0.059 |
| linear (ridge→identity) | 0.921 ± 0.026 | 0.950 ± 0.020 |
| **UNet** | **0.937 ± 0.018** | **0.969 ± 0.020** |
| oracle | 0.969 ± 0.000 | 1.000 ± 0.000 |

UNet − linear = **+0.016 spearman, +0.019 slate4** — both smaller than either
model's own across-slate sd (~0.02), i.e. **inside the noise floor**.
Persistence's numbers are noise around 0 by construction (dv_pred≡0 for
every candidate: it cannot rank at all) and must not be read as "persistence
ranks at −0.05". The oracle's spearman is 0.969, not 1.0 — a property of
`rank_metrics`' own formula at slate size ≈32 (see Unrelated findings), not a
measurement of anything here; its slate4 is exactly 1.0 as expected.

goal=center: dv_true mean 0.00004, sd 0.00120, 0% of pushes helpful; only
11/49 slates clear the `per_slate_metrics` variation threshold. Consistent
with C-040/EXP-0012 — degenerate for a centred pile, no signal, not used for
the verdict.

## What this means

**The prediction is supported.** The UNet's image-accuracy edge over the
linear operator is real and in the expected direction (+3.6 points, same
sign as all 14 of EXP-0021's cells), but its control-utility edge is not
distinguishable from zero: +0.019 slate4 and +0.016 spearman are both inside
the ~0.02 across-slate noise floor that this same design measures for either
model individually. Put the other way: an MPC built on this UNet would not
be expected to out-select an MPC built on the far cheaper linear operator, on
this task, at this noise floor.

This is exactly the reading EXP-0022 anticipated: its whole image-accuracy
advantage is high-frequency detail (C-041/C-044), and EXP-0008/C-039 already
established that high-frequency *noise* is what destroys control ranking
while amplitude/blur/displacement cost it almost nothing. A model whose
extra accuracy comes from getting fine detail right should not convert that
into a ranking advantage if fine detail is not what ranking consumes — and
here, when actually measured instead of inferred, it does not. **C-041/C-044
should be restated as claims about sharp-target image-prediction quality, not
about which model is "better" for this project's actual objective.**

That both models land close to the oracle's ceiling on slate4 (0.950 and
0.969 against 1.000) also matters: there is not much control-utility headroom
left to fight over here, on this exact task (single push, goal=corner,
n=20 cubes) — a harder task (goal=center once made non-degenerate, longer
horizons, sparser piles) might reopen a gap that this design cannot see
because neither model is far from ceiling.

## What would change the verdict

- **A harder control task with more headroom.** Both models score
  0.95-0.97 slate4 here; a task where the linear operator's slate4 sits
  further from the oracle (a longer horizon, a harder goal, more cubes) would
  let a real UNet advantage show up if one exists, rather than being squeezed
  against a ceiling both already nearly reach. Cost: reuses this same
  pipeline against a different slate collection (e.g. n50 or a multi-push
  same-state design), no new training needed for the linear side, ~15-20 min
  GPU for a matched UNet retrain.
- **Repeated seeds.** One UNet training seed, one linear fit (`imprecision`,
  below) — a second seed of each would turn "inside the noise floor" from a
  plausible read into a measured one, at the cost of ~16 more min GPU per
  seed.
- **A genuine settled-state check** for the rigid-cube path (currently
  `unchecked`) would retire the `untested-dependency` downgrade this record
  and EXP-0012 both carry.

## Threats

- `imprecision`: one UNet training seed, one linear fit — no repeated-seed
  estimate of how much the +0.016/+0.019 control gap itself might move on a
  different seed, only the across-slate sd of a single fit's predictions.
  The across-slate sd (49 slates) is itself well-measured and closely matches
  EXP-0012's independent n=50 measurement (0.020/0.026 here vs 0.017/0.021
  there), which is reassuring but not a substitute for a seed sweep.
- `untested-dependency`: `settled-state` is `unchecked` for the rigid-cube
  path (same debt EXP-0012 carries) -- the "identical start state" property
  the slates depend on is not independently velocity-checked at record time.
- Considered and dismissed: `provenance` -- this record's headline comparison
  (UNet vs linear) runs both models through the identical code path
  (registry/PileSweepData) for both the image-accuracy split and the slate
  evaluation, so nothing in the verdict crosses rasterisers. A secondary,
  disclosed pipeline-validation pilot found this record's numbers close to
  EXP-0012's OWN numbers computed on ITS different code path (spearman 0.921
  vs 0.925, slate4 0.950 vs 0.956, persistence degenerate in both) — used
  only as external validation that the new pipeline is not buggy, not as
  part of the claim.
- Considered and dismissed: `indirectness` -- `dv_true` is the realised dV of
  an actually-executed push on both the fit data and the slate data, exactly
  as EXP-0008/EXP-0012 use it, not a proxy standing in for control utility.
  (A different, general caveat remains: single-step greedy ranking is not a
  full closed-loop MPC rollout — noted but not scored as a downgrade domain,
  per EXP-0012's identical call on the same question.)
- Considered and dismissed: `selection` -- both goals (corner, center) were
  run and reported, and the model set (persistence, mean-delta, linear, UNet,
  oracle) was fixed before training, not chosen after seeing results.
- Considered and dismissed: `episode-split` leakage -- the linear fit and the
  UNet both train on cube_spectrum/n20's train split only; the slates are a
  physically disjoint collection that neither model's fit ever saw.

## Unrelated findings

- One of the 50 n20_heap_5mm slate files contributes zero transitions after
  the min_push_length_m=0.0199 filter (49 usable slates out of 50, vs
  EXP-0012's own report of losing 1 *transition* out of 192 in their n=6
  pilot to the same 19.9mm-style filter) -- not investigated further; the
  most likely mechanism is a state whose sampled contact-aware pushes were
  unusually short across most/all of that slate's ~32 candidates.
- `min_push_length_m` in `configs/dataset/genesis_cube_spectrum_n20.yaml` had
  no established config-file precedent for this dataset before this record
  (`Genesis/data/cube_spectrum` is not referenced by any existing
  `configs/dataset/*.yaml` file) -- the 19.9mm-style threshold was carried
  over from a probe script's hardcoded argument default
  (`same_state_degradation.py --min-push-length` is not actually a flag there;
  the value 19.9 is hardcoded in its `load_transition_fields` call), not from
  a config convention.

## Reviewer amendment, 2026-09-06: the significance test was unpaired

**The measurements stand. The verdict does not.**

This record compared the UNet−linear `slate4` difference (+0.019) against the
**across-slate sd** (~0.020) and concluded the control advantage is
"indistinguishable from zero". That is the wrong floor. Both models are scored
on the **same 49 slates**, and slates differ enormously in difficulty — so the
across-slate sd is dominated by variance the two models *share*, and it swamps
a between-model difference that pairing removes.

EXP-0016 made exactly this correction for C-008, where an unpaired floor was
~11× too large and left a real effect recorded as inconclusive. `per_slate_metrics`
discarded its per-slate values, which is why the paired test was not available;
it now returns them.

### Paired result, same data, same 49 slates

| comparison | mean diff | paired sd | sem | t | slates won |
|---|---|---|---|---|---|
| UNet − linear | **+0.0185** | 0.0217 | 0.0031 | **+5.99** | **42 / 49** |
| oracle − linear | +0.0497 | 0.0203 | 0.0029 | +17.13 | 49 / 49 |
| mean-delta − linear | −0.1547 | 0.0567 | 0.0081 | −19.09 | 0 / 49 |

**The UNet's control advantage is real**: t ≈ 6, winning 42 of 49 slates.

### But it is small, and the reason matters more than the significance

| model | `accuracy` | `slate4` |
|---|---|---|
| persistence | 0.000 | −0.002 (cannot rank) |
| mean-delta | 0.343 | 0.796 |
| **linear (ridge→I)** | 0.533 | **0.950** |
| **UNet** | **0.569** | **0.969** |
| oracle | 1.000 | 1.000 |

**The linear operator already captures 95.0% of the oracle's advantage over a
random pick.** There are only 5 points of headroom, and the UNet takes about a
third of them (+0.0185 of a possible +0.0497).

So the honest statement is neither "the advantage does not survive" (this
record's original verdict, from the wrong floor) nor "the UNet is the better
model for MPC" (which the +3.6-point accuracy gap alone would suggest). It is:

> On this domain the UNet is measurably better at action selection, and it
> hardly matters, because a ridge operator is already near the ceiling. The
> image-accuracy gap (+0.036) is roughly twice the control gap (+0.019) — which
> is what EXP-0022 predicts, since much of the UNet's accuracy advantage sits
> in a frequency band control does not consume.

### Consequences

- Verdict changes from `refuted` to `supported` for the narrow claim that the
  advantage survives, with the magnitude stated.
- C-045 restated: the advantage is real but small against a near-saturated
  ceiling. **The interesting quantity is the 5-point oracle gap, not the model
  comparison** — both model classes are close to it.
- `center` is confirmed degenerate again (11 slates survive the filter,
  `dv_true` sd 0.0012, 0% helpful pushes, oracle `slate4` only 0.149).
  Consistent with C-040; do not report it.
