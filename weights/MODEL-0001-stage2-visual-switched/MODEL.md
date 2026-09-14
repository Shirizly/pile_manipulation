# MODEL-0001 — stage-2 switched-linear visual operator (32x32, 6 push-length bins)

**Status:** active

## What this is

A set of 6 per-push-length-bin ridge-toward-identity linear operators (plus
one unswitched global operator, in the same file) mapping 32x32 canonical
push-frame occupancy at time t to the same representation at t+1. `D=1024`
(pure visual, `desc_dim=0` — this is the `visual` cell, not one of the
hybrid/descriptor variants). Fit on `Genesis/data/overnight_randlen`'s
170-file (87040-transition) train split from `experiments/temp/hybrid-vis-
desc/cache/train_cache.pt`.

This is the operator EXP-0005 reports headline `accuracy` numbers for
(holdout switched 0.1697, global 0.1010) and EXP-0006 scores under `slateN`
on `slates_multistep` real same-state pools (pooled corner switched +0.919).
It stopped being purely internal to `experiments/temp/hybrid-vis-desc/` the
moment EXP-0006 compared it against the stage-3 latent/descriptor family and
against its own global counterpart on a second dataset — hence promotion
here per the `experiment-log` skill's "the moment it's compared against
anything else" rule.

## Provenance

- **commit:** 0ddab20f (dirty tree, unrelated pre-existing changes — see
  EXP-0005's provenance note)
- **script:** `experiments/temp/hybrid-vis-desc/fit_hybrid.py::fit_cell`
  (cell_name="visual", lam=1.0)
- **data:** `experiments/temp/hybrid-vis-desc/cache/train_cache.pt` — 32x32
  canonical push-frame occupancy, 170-file / 87040-transition train split of
  `Genesis/data/overnight_randlen` (80/20, stratified by spawn_mode, seed 0)
- **recipe:** `fit_operator(..., ridge=1.0, toward_identity=True)`, 6
  EXP-0003-scheme push-length bins over `[0, 0.0800]` m
- **file copied from:** `experiments/temp/stage2-slaten/operators/visual_lam1.0.pt`
  (persisted there during EXP-0006's Run 2; not persisted by EXP-0005's own
  original run, which saved only metrics)

## Contents

`checkpoint.pt` (torch, dict): `cell_name`, `cell_spec` (`use_visual`,
`desc_dim=0`), `lam=1.0`, `bin_edges` (7 values), `switched_ops` (6 x
[1024,1024] tensors), `global_op` ([1024,1024]), `note` (recipe string).

## Regeneration

Not directly regenerable by a single documented command yet — the source
was `experiments/temp/hybrid-vis-desc/fit_hybrid.py`'s `fit_cell` called
from `experiments/temp/stage2-slaten/eval_slaten.py` (Run 2), both scratch
scripts. A future caller wanting to regenerate this exact operator should
call `fit_hybrid.fit_cell(cell_name="visual", lam=1.0, ...)` against the
same cached train tensors.

## Test history

See `tests.md`.
