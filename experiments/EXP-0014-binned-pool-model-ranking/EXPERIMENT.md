---
# ---- identity -------------------------------------------------------------
id: EXP-0014
title: >
  Three-way action-ranking comparison on DS-0001 (visual-switched vs NFD vs
  descriptor-only), and a diagnosed saturation mechanism that makes
  MODEL-0002's point-mass readout indistinguishable from random
tier: T1
mode: exploratory
date: 2026-09-15
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On DS-0001 (`slates_binned` n20 scatter, 20 slates x 1000 candidates),
  under the `corner` goal and the `lyapunov` value function at step 0,
  `slateN` ranks weights/MODEL-0001 (switched-linear visual, +0.740 sem
  0.055) above weights/MODEL-0003 (NFD fine-tuned, +0.516 sem 0.066), both
  far above weights/MODEL-0002 (descriptor-only via its point-mass readout,
  -0.090 sem 0.030), which is indistinguishable from the `random` baseline
  (-0.141 sem 0.101) and from `persistence` (-0.004 sem 0.089). MODEL-0002's
  failure has a diagnosed mechanism that is a property of the READOUT, not of
  the corpus: 26% of its predicted values are exactly 0 because the predicted
  centre of mass lands inside the target region, where the `corner` distance
  field is identically zero -- so ~26% of every 1000-candidate pool ties at
  the minimum and the top-1 pick is decided by row order.

prediction: null

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 6ea03278
  dirty: true
  data_commit: unrecorded
  script: scripts/probes/binned_pool_cache.py
  data: ["DS-0001"]
  code_path: scripts/probes/
  seed: 0
  split: >
    no train/test split -- DS-0001 is used as a fixed evaluation pool, as
    EXP-0011/0012/0013 do. All three models reused unmodified from
    weights/MODEL-0001, MODEL-0002, MODEL-0003; none was fitted here. Note
    none of the three was trained on DS-0001, but MODEL-0001 and MODEL-0002
    were fitted on `overnight_randlen` and MODEL-0003 on `slates_multistep`,
    so "held out" is not equally true of all three.
  runtime: "~5 min cache build (GPU) + ~2 min survey/figures, CPU"
  runs: [RUN-0001]

budget:
  declared: "~30 min / ~60k tokens (coordinator-run, not delegated)"
  spent: "~35 min / ~60k tokens"
  outcome: within

design:
  varied: {model: [random, persistence, descriptor, nfd, visual-switched]}
  held_fixed: {goal: corner, value_fn: lyapunov, step: 0, dataset: DS-0001,
               grid: 64, bounds: "+/-0.064 m", pool_size: 1000, n_slates: 20}
  baselines: [persistence, random]
  metric: "slateN"

noise_floor: >
  Between-slate sem over the 20 slates, per model, reported in every row
  above. The decisive comparisons: MODEL-0001 vs MODEL-0003 is +0.224 against
  a combined sem of ~0.086 (~2.6 sem, resolved); MODEL-0002 vs `random` is
  +0.051 against a combined sem of ~0.105 (well inside noise, UNRESOLVED and
  reported as indistinguishable, not as a ranking). `random`'s own sem
  (0.101) is large because a random ranker's per-slate capture is itself
  high-variance; this limits how tightly any near-zero model can be placed
  against it, and is not improved by more candidates, only by more slates.

depends_on: [occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x,
             push-frame-warp-roundtrip, slates-binned-uniform-difficulty]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  slateN (mean, sem, n=20 slates), corner/lyapunov, step 0 of DS-0001:
  visual-switched (MODEL-0001) +0.7399 (0.0551); nfd (MODEL-0003) +0.5163
  (0.0658); persistence -0.0042 (0.0892); descriptor (MODEL-0002) -0.0904
  (0.0301); random -0.1409 (0.1007). Oracle top-1 agreement: 2/20 for
  MODEL-0001, 0/20 for every other model. `top1_regret` (Lyapunov units,
  lower better) orders the same way: 0.0155 (sem 0.0034), 0.0308 (0.0045),
  0.0704 (0.0050), 0.0718 (0.0065) for visual-switched / nfd / descriptor /
  random. Saturation mechanism measured directly on the cached predictions:
  MODEL-0002 emits 14857 distinct values over 20000 rows with 25.7% exactly
  0; within one slate (slate 17) 26.3% of candidates sit at the pool minimum.
  MODEL-0001 and MODEL-0003 show no such degeneracy (19949 and 19958 distinct
  values, 0.0% exactly zero).
verdict: supported
downgrades: [provenance, incomplete-design, imprecision, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## What was uncommitted (dirty tree)

The working tree was dirty at run time and at this record's commit time:
`Genesis/binned_slate_collection.py`, `Genesis/binned_slate_dataset.py`,
`scripts/probes/binned_pool_cache.py`, the edits to
`scripts/probes/pool_inspect.py` and `pool_survey.py`, and this record's own
files. No previously-committed project module changed behaviour except
`Genesis/sandbox_manipulation_clean.py::_pile_aware_stops` (per-env
`push_length` tensor support), which affects data collection only, not
scoring.

## Why this comparison is only exploratory

No prediction was registered before the run. The comparison was made to
decide what to plot, and the numbers were computed before any claim was
written — so `mode: exploratory` and no `prediction` block. Anything leaning
on the MODEL-0001 > MODEL-0003 ordering should re-run it pre-registered and
across more than one goal and value function.

## What was actually run

**RUN-0001.** `scripts/probes/binned_pool_cache.py` built a dV cache over all
20000 rows of DS-0001 step 0: `dv = value(after) - value(before)` under
`lyapunov`/`corner` at 64x64, plus one predicted dV per model. Each model's
prediction path was IMPORTED from the code already validated against it
(`experiments/temp/multistep-rollout/rollout.py`'s image-space step functions
for MODEL-0001 and MODEL-0003; `experiments/temp/desc-mlp/eval_control.py`'s
descriptor pipeline — the one EXP-0012 ran on this same corpus — for
MODEL-0002), not reimplemented.

`scripts/probes/pool_survey.py` then computed `slateN` via the canonical
`Baselines/common/goals.py::slate_n_capture`, plus `top1_regret`, `|R_32|`,
agreement rates and the near-tie/large-loss split, and named the typical and
worst pool per model. `scripts/probes/pool_inspect.py` produced six per-pool
figures (typical and worst for each of the three models).

## Unrelated findings

**`persistence` is degenerate as a RANKER and should not be used as the
ranking baseline anywhere in this repo.** It predicts `dv = 0` for every
candidate, so `argmin` always returns row 0 and the induced ordering is row
order, not a prediction. Its `slateN` here (-0.004 sem 0.089) is therefore a
measurement of row order, which happens to be uninformative, not of a model.
`random` is the honest ranking floor. This affects how the archived
`reports/action_pool_diagnostics.md` should be read — its persistence curves
are row-order curves — and `scripts/probes/binned_pool_cache.py` and
`docs/CODEMAP.md` now say so explicitly.

## What would change the verdict

- **One goal, one value function.** `slateN`'s known weakness is power, and
  the standard here is breadth: this ran `corner`/`lyapunov` only. EXP-0013
  already scored MODEL-0001 on this corpus across 9 goal x value-fn cells and
  found +0.389 to +0.87, consistent with this record's +0.740, so the
  MODEL-0001 result is corroborated; MODEL-0003's and MODEL-0002's are not.
  Cost to close: ~15 min, the cache already holds the occupancies.
- **The MODEL-0002 vs `random` comparison is inside the noise floor** and is
  reported as unresolved. What IS resolved, and does not depend on that
  comparison, is the saturation mechanism, measured directly on the
  predictions.
- **Cross-code-path comparison** (`provenance`): the two image-space models
  are scored by integrating a value function over a predicted occupancy,
  while MODEL-0002 is scored by sampling a distance field at a predicted
  point. These are different readouts, not just different models, so this
  record cannot separate "MODEL-0002's operator is worse" from "MODEL-0002's
  readout is worse". EXP-0015 attacks exactly that separation.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- DS-0001 was simulated with friction 0.3 / density 1000, not the randlen models' training physics (`benchmark-physics-matches-training`, broken).
- EXP-0065 / ISS-013: 46 % of DS-0001's candidate pushes put the blade on a cube at touchdown (pre-fix pile-aware sampler); this record's pools were not re-scored on legal-only candidates.
- The pre-registered 9-cell repeat (EXP-0021) was never run; C-020 remains a one-goal exploratory ordering.
