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
  commit: 57452569
  dirty: true
  dirty_note: >
    Working tree carried OTHER AGENTS' concurrent uncommitted edits at run
    time (per `utils.git_provenance()`): .claude/skills/experiment-log/SKILL.md,
    docs/experiments/{EXP-0004,EXP-0009,EXP-0014,EXP-0015}*.md,
    docs/experiments/METRICS.md, and fit_linear_foresight.py. The last one is
    load-bearing: it adds the `accuracy` key to `metrics()` (the exact diff is
    quoted in "What was actually run" below) that this record's image-accuracy
    numbers use. None of this experiment's own new files (the four committed
    at 57452569) were touched by any other session. Re-running from 57452569
    alone will NOT reproduce the `accuracy` numbers until that key lands in a
    commit -- reproduce instead from the diff quoted below, or from whatever
    later commit adds it.
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
  runtime: "UNet training: 100 epochs, ~<FILL> min GPU. Fit+eval: <FILL> s CPU."
budget:
  declared: "2h wall-clock, 300k tokens (task-level T2-gate sizing)"
  spent: "<FILL>"
  outcome: "<FILL>"
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
  Across-slate sd of slate4 and spearman (49-50 slates), reported alongside
  every mean -- this IS the noise floor per EXP-0012's own convention. Pilot
  value observed on the linear operator (goal=corner, this code path): slate4
  sd 0.020, spearman sd 0.026 (n=49 slates; EXP-0012's own numbers: slate4 sd
  0.017, spearman sd 0.021, n=50) -- so an across-model slate4 gap smaller
  than ~0.02-0.03 should not be read as a clear win either way.
depends_on: [grid-convention, rasteriser-identity, canonical-warp, warp-blend,
             swept-region-metric, episode-split, settled-state]
establishes: []
result: >
  <FILL AFTER RUN>
verdict: "<FILL>"
downgrades: [provenance, indirectness, untested-dependency]
grade: low
supersedes: []
invalidated_by: null
---

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

<FILL>

## What this means

<FILL>

## What would change the verdict

<FILL>

## Threats

- `provenance`: this record's rasteriser code path (registry/PileSweepData)
  differs from EXP-0008/EXP-0012's (occupancy_foresight.load_transition_fields);
  the pipeline-validation pilot found close agreement on the linear operator +
  persistence (spearman 0.921 vs 0.925, slate4 0.950 vs 0.956), but it is a
  different implementation, not a re-run of the same one. Internally, though,
  every number in THIS record's headline comparison (linear vs UNet) goes
  through the identical code path, so the UNet-vs-linear comparison itself is
  not cross-path.
- `indirectness`: still a proxy in one direction -- image `accuracy` is a
  one-step pixel metric, and while `slate4`/`spearman` are the real control
  quantities this record cares about, neither is a closed-loop MPC rollout;
  a model that ranks single-step actions well is not guaranteed to control
  well over a rollout (a caveat this whole register carries throughout, not
  specific to this record).
- `untested-dependency`: `settled-state` is `unchecked` for the rigid-cube
  path (same debt EXP-0012 carries) -- the "identical start state" property
  the slates depend on is not independently velocity-checked at record time.
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
