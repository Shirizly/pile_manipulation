# MODEL-0002 — descriptor-only switched-linear operator (D-all-local, 94-dim)

**Status:** active

## What this is

A set of 6 per-push-length-bin ridge-toward-identity linear operators
(94x94 each) mapping the D-all-local 94-dim push-frame analytic-descriptor
vector (`global_mass`, `global_length`, `mass`, `com`, `moments2`,
`mass_ahead`, `mass_behind`, `band_mass`, `dft_real`, `dft_imag` slices,
`lam=1e-4`) at time t to the same vector at t+1. It has **no decoder** and
scores **exactly 0.0 image-space `accuracy` by construction** (EXP-0004's
own definition of the model). It is scored in `slates_multistep`/holdout
control experiments via a **point-mass value readout**: the model's
predicted world centre-of-mass (from the `com` slice) and, for
mass-based value functions, its predicted total mass (`global_mass` x H*W)
are used to construct a value estimate without ever reconstructing an
image.

## Why this is promoted, not left in `experiments/temp/`

Per the `experiment-log` skill's rule ("the moment it's compared against
anything else"): this operator has now been compared, under `slateN`,
against the stage-2 visual operator (`MODEL-0001`) and the stage-3
`latent+desc94` operator in both EXP-0006 and EXP-0008, on two different
corpora (`slates_multistep` and the overnight holdout) — it is no longer
purely internal to its own producing directory.

## Corpus-specificity finding (must be read before reusing this model)

**This model's control-ranking usefulness is corpus-specific, not a
general property of the point-mass readout.** EXP-0006 established it
scores a real, well-powered positive `slateN` (pooled corner/lyapunov
+0.461) on `slates_multistep`, the corpus its stage-1 descriptor fit was
built from. EXP-0008 found this weakens under `mass_in_region`/
`signed_mass_in_region` on the same in-corpus data (still positive, e.g.
corner +0.239/+0.240) and **degrades sharply, reversing sign, on the
43-file overnight holdout it was never trained on** — most sharply on
out-of-distribution annular/thin target shapes (`ring_O`/`lyapunov`:
-0.313, worse than persistence; `ring_O`/`mass_in_region`: -0.003).
The image-based models compared alongside it (visual, latent+desc94) show
no comparable out-of-corpus failure. See EXP-0006 and EXP-0008 for the
full numbers; see `experiments/REGISTER.md`'s `C-008` for the narrowed
claim this finding drives.

## Provenance

- **commit:** 0ddab20f (dirty tree, same pre-existing unrelated changes as
  EXP-0004..EXP-0008)
- **script:** `experiments/temp/dmdc-lenbins/{descriptors.py,fit.py}`
  (`fit_cell`/`fit_operator(ridge=1e-4, toward_identity=True)`, per stage-1's
  reported-best lam), assembled into a standalone switched-operator file by
  `experiments/temp/stage3-slaten/eval_slaten_latent.py::fit_desc_operator`
- **data:** `experiments/temp/dmdc-lenbins/cache/descD_train.npz` — the
  94-dim D-all-local descriptor basis (EXP-0004's own-basis-pooled best
  common-target ranking, D-all-local == D-all excluding a duplicate/
  redundant slice per that record's own basis-selection writeup), built
  from `Genesis/data/overnight_randlen`'s train split
- **recipe:** `fit_operator(..., ridge=1e-4, toward_identity=True)`, 6
  EXP-0003-scheme push-length bins over `[0, 0.0800]` m (identical bin
  edges to MODEL-0001)
- **file copied from:**
  `experiments/temp/stage3-slaten/desc_operator.pt`

## Contents

`checkpoint.pt` (torch, dict): `ops` (6 x `[94,94]` tensors, one per
push-length bin), `bin_edges` (7 values, identical scheme to MODEL-0001),
`slices` (dict naming each descriptor sub-block's index range within the
94 dims: `global_mass`, `global_length`, `mass`, `com`, `moments2`,
`mass_ahead`, `mass_behind`, `band_mass`, `dft_real`, `dft_imag`, `_total`),
`note` (recipe string).

## Regeneration

Not directly regenerable by a single documented command yet — the source
was `experiments/temp/stage3-slaten/eval_slaten_latent.py`'s
`fit_desc_operator`, a scratch script, calling into
`experiments/temp/dmdc-lenbins/fit.py::fit_cell`/`fit_operator` against the
cached `descD_train.npz` descriptor tensors. A future caller wanting to
regenerate this exact operator should call the same `fit_cell` against the
same cache.

## Test history

See `tests.md`.
