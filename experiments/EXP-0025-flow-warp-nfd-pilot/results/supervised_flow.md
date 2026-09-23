# EXP-0025 extension — direct particle-correspondence supervision of the flow head

NOTE (post-run refactor, pure code move, no behaviour/retrain change): the
`Baselines/NFD/supervised_flow_lib.py` / `Baselines/NFD/flow_nfd_lib.py`
paths cited below moved to `model/flow_nfd/supervised.py` / `model/flow_nfd/
lib.py` after these runs -- registered type names and every number below are
unchanged.

Extends EXP-0025 (same subset `slates_multistep/n20_L20mm_train`, same
40-epoch pilot scale, scored on the same L20mm eval cell through
`Baselines/common/eval_report.py`). Runs: RUN-0008 (train, cell
`supervised`, UNaugmented), RUN-0009 (train, cell `supervised_masked`,
UNaugmented), RUN-0010 (score both), RUN-0011 (STEP 1 ceiling gate,
model-free), RUN-0012/RUN-0013 (re-run of RUN-0008/RUN-0009 WITH the x8
augmentation restored, per coordinator request below), RUN-0014 (score
RUN-0012/RUN-0013).

**2026-09-23 update (coordinator-requested re-run):** RUN-0008/RUN-0009
dropped the x8 flip/rotation augmentation EXP-0025's photometric cells all
used, leaving only 768 effective training samples vs ~6,144 — a sufficient
alternative explanation for underperformance on its own, independent of
whether direct flow supervision helps. RUN-0012/RUN-0013 redo those two
cells with `augmentation: true` restored, everything else unchanged
(`max_disp_px=24`, same architecture/subset/epochs/optimizer). RUN-0008/
RUN-0009 are kept on disk, unaugmented, and reported below as an
(unintentional) data-ablation, not deleted or overwritten. See "STEP 2
(re-run with augmentation restored)" below for the vector-field-equivariant
augmentation implementation, its verification, and the new numbers.

## Idea under test

EXP-0025's flow head only ever saw a photometric loss on the warped
occupancy, so the flow itself was never directly supervised: it drifts
wherever `occ0` is locally flat, and gradient descent on a photometric loss
can only correct displacements already within about one feature width,
while true per-particle displacements go far outside that basin. `states`/
`states_` share particle indexing, so a target displacement field can be
built directly from particle correspondence and supervised with
`||f - f_target||^2` — convex, no basin to escape.

## STEP 1 — target-field convention and the mandatory gate

**Convention** (`Baselines/NFD/supervised_flow_lib.py`, verified below):
`_extract_sample_in_pxl` converts `states`/`states_` to pixel space via
`particles[:, :3] * to_pxl + ctr_in_PXL`; the resulting `particles[:, 0]` IS
the final occupancy grid's dim0 (row, world x) index and `particles[:, 1]`
IS dim1 (col, world y) — no further axis swap needed. The warp is
**backward** (`grid_sample`'s own contract, matching `flow_nfd_lib.py`): for
destination pixel `x`, `f(x) = x0 - x`, i.e. **channel 0 = dcol (source_col
− dest_col), channel 1 = drow (source_row − dest_row)**, indexed AT THE
DESTINATION pixel, matching `FlowWarpWrapper`'s own [disp_x(col), disp_y(row)]
channel order.

**Sparsity / collisions**: 20 particles for a 64×64 field. Chose a **masked
loss at particle footprints**, not scatter-and-diffuse: the target is
defined only where the rasteriser marks a destination pixel as foreground
(`occ1 > 0.5`); every such pixel is assigned the **rigid, whole-particle**
(translation-only) displacement of whichever particle's destination center
is nearest (**nearest-destination-center** collision resolution for
stacked/overlapping cubes). This does not model rotation — a translation
field cannot express it regardless of how the target is built, so this is
also a ceiling limitation, not just a target-construction choice.

**Verification of the convention** (before trusting anything downstream):
in a real sample, 18/20 particle destination centers land exactly on an
`occ1 == 1` pixel (the other 2 miss by rounding at a box corner) —
confirms the pixel mapping is right, not transposed or sign-flipped.

**GATE result (RUN-0011, model-free — warp `occ0` by the ground-truth field,
score against `occ1`)**:

| quantity | value |
|---|---|
| swept-region `accuracy` (ceiling) | **0.1668** |
| slateN lyapunov (ceiling) | **0.8129** |
| slateN mass_in_region (ceiling) | **0.7695** |
| slateN signed_mass (ceiling) | **0.5122** |

Displacement stats among supervised (foreground) pixels: mean 1.78px,
median 0.22px, p95 7.02px, max 39.6px — consistent with the brief's
independent measurement (p95 6.5px, max 20px was reported at a different
resolution/subset; both agree the true transport reaches far past a
photometric loss's basin of attraction).

**Reading the gate: split verdict, not a single ceiling.**
- On `accuracy`, the ceiling (0.1668) is **below both** the direct-prediction
  control (0.294) and the best existing photometric flow cell,
  `flow_coarse16` (0.213). So on `accuracy`, this parameterisation cannot
  win regardless of supervision — the residual error is not a training
  problem, it is what nearest-rigid-particle correspondence plus bilinear
  `grid_sample` costs on a near-binary occupancy field (see mechanism note
  below).
- On `slateN` — the metric this project treats as deciding — the ceiling
  (0.813 / 0.770 / 0.512) **beats every existing EXP-0025 cell on all three
  value functions**, including the direct control (0.608 / 0.738 / 0.487)
  and `flow_coarse16` (0.754 / 0.517 / 0.447). So there IS headroom on the
  metric that leads, and the gate does not, on its own, rule out a
  supervised flow head — hence STEP 2 was run rather than stopping.

**Diagnosed mechanism for the low accuracy ceiling** (checked, not asserted):
stratifying the ceiling's per-transition error by each row's max supervised
displacement shows accuracy is *worst* (−0.08, i.e. below persistence) in
the near-static bin (max displacement < 1px, 2037/7680 rows) and
*improves* monotonically with displacement (0.08 → 0.15 → 0.19 → 0.19 across
increasing bins). This is the opposite of "large displacement breaks a
capture radius" — because this is ground truth, there is no basin to
escape. Instead: bilinear `grid_sample` sampling a near-binary occupancy
field at a non-integer location necessarily blends 0/1 values into gray,
and this blur cost is paid on every foreground pixel regardless of how
small its true displacement is; it hurts most exactly where persistence's
own error is already tiny (denominator near zero), which is the accuracy
ratio's known sensitive regime. So: the backward-warp-field
parameterisation pays a structural resampling cost that a direct model
(which just emits a probability per pixel, no resampling) does not pay —
this is a real, previously undocumented mechanism for why flow-style heads
underperform direct prediction on `accuracy` in this project, distinct from
the drift/capture-radius argument EXP-0025 already made.

## STEP 2 — training

Two cells, `max_disp_px=24` (comparable to EXP-0025's `flow_largedisp24`,
which was catastrophic under photometric-only training — the natural
test of whether direct supervision fixes that), same UNet backbone
(`features=[4,8,16]`, `final_kernel_size=1`, `kernel_size=3`, `residual=False`),
same subset (`n20_L20mm_train_pilotsubset`), 40 epochs, batch 32, Adam
lr 1e-4, grad clip 1.0 (`experiments/EXP-0025-flow-warp-nfd-pilot/code/train_supervised_flow.py`).

**Deviation from EXP-0025's recipe, stated up front**: this script does NOT
apply the ×8 flip/rotation augmentation the original six cells used (a
standalone script was written because the target-field construction needs
raw `states`/`states_`, which `EulerianDatasetWrapper` does not expose
downstream — plumbing that through the registered dataset/Trainer pipeline
was out of budget). Training set is 768 unaugmented samples (vs ~6,144
after ×8 augmentation for the original cells). **This confounds the
comparison below** — a loss of accuracy relative to the photometric-only
cells is not attributable to "direct supervision doesn't help" without also
controlling for this 8x reduction in effective training data.

- `supervised`: photometric MSE + masked direct flow-target MSE (foreground
  only, normalised by pixel count and by `max_disp_px^2`).
- `supervised_masked`: additionally, a magnitude penalty shrinking `disp_px`
  toward zero **weighted by `(1 - occ0)`** (the task brief's literal
  suggestion — chosen over an occ1-based weight for exactly that reason;
  not independently re-derived), λ=0.05.

**Did the magnitude penalty collapse the model?** Close to it, but not
total collapse to persistence. `supervised_masked`'s background loss term
fell steadily (0.048 → 0.022 over training) while `flow`/`photo` losses
stayed roughly flat after epoch ~30 — i.e. the penalty did suppress
background flow magnitude as intended without visibly fighting the
foreground terms in the training curves. But its DOWNSTREAM scores are
**worse than persistence-adjacent**: `accuracy = -0.0004` (persistence's own
accuracy is exactly 0 by construction — this is indistinguishable from it)
and slateN averaged over goals is **negative on all three value functions**
(−0.018 / −0.088 / −0.079), i.e. worse than the `random` ranking floor on
two of them. So while the loss curves don't show a collapse, the held-out
behaviour does: this cell is not usefully different from predicting no
change.

## STEP 2b — re-run with the x8 augmentation restored (RUN-0012/RUN-0013)

**Is the vector-field augmentation valid? Checked explicitly before relying
on it — this was the one thing that could silently corrupt every augmented
sample while still training to a plausible loss.**
`training/trainer.py::_augment_eulerian_batch` transforms `input`/`target`/
`physics`/`push_px` only and drops any other batch key, and even for a key
it does carry (`push_px`, a POINT, not a vector) it only permutes/reflects
coordinates — it has no notion of rotating a vector's own components.
`flow_target` is a genuine 2-vector field (dcol, drow), so riding it through
that function unmodified, or writing a custom augmenter that only spatially
permutes the array (as `torch.rot90`/`torch.flip` do to any image), would
silently corrupt it: the vector's magnitude/direction must be rotated by
the same transform's linear part, not just moved to a new pixel.

Because of this, this task does NOT use `_augment_eulerian_batch` at all —
`Baselines/NFD/supervised_flow_lib.py::augment_flow_batch_x8` is a new,
purpose-built augmenter that (a) spatially permutes `inputs`/`occ1`/`mask`
exactly as `_augment_eulerian_batch` does (`rot90(k)` then optional
`flip(dims=[-1])`, same `k` order), and (b) ADDITIONALLY rotates
`flow_target`'s two components by that same transform's linear part
(`_rotate_flow_components`, derived from the point map already verified in
`Baselines/NFD/WARPED_NFD_NOTES.md` section 2 by reading off its
translation-independent/linear part: `(dc_o,dr_o) = (dc,dr)`,`(dr,-dc)`,
`(-dc,-dr)`,`(-dr,dc)` for `k=0..3`, with an extra `dc_o <- -dc_o` under the
horizontal flip).

**Verified, not just derived** (`code/verify_flow_augmentation.py`, log in
`runs/RUN-0011-gt-flow-ceiling-gate/verify_flow_augmentation.log`): warping
is EQUIVARIANT under a correct rigid transform, so
`warp(rot_flip(occ0), rotate_vector(rot_flip(flow_target)))` must equal
`rot_flip(warp(occ0, flow_target))` for every `k`/flip combination — checked
against the RUN-0011 ground-truth-ceiling cache (real `occ0`/`flow_target`
pairs, not synthetic) at 5 sample indices x 4 rotations x 2 flips = 40
checks. **Max abs diff 3.8e-6** (float32 precision) — PASS. A **negative
control** (skip the component rotation, i.e. exactly the "carry a vector
like a scalar image" bug being guarded against) gives max abs diff **1.0**
on the same checks, confirming the test has power to catch that class of
bug, not just that it doesn't fire on a working implementation by chance.

Recipe otherwise identical to RUN-0008/RUN-0009 (`max_disp_px=24`, same
architecture/subset/epochs/optimizer/loss weights); only the train split is
augmented (matching `Trainer`'s own behaviour — validation is never
augmented there either), expanding 768 -> 6,144 effective training samples,
matching EXP-0025's own count. One structural difference from the real
`Trainer` worth naming: `Trainer` augments each already-drawn mini-batch on
the fly (so a gradient step's 32 augmented views come from only 4 distinct
underlying pushes), while this script precomputes the full 6,144-sample
augmented set once and shuffles freely across it (so a step's 32 samples
are generally 32 distinct underlying pushes, each contributing one of its 8
views) — same total 192 steps/epoch, slightly different batch composition;
not expected to matter at this scale but stated for the record.

**Results**: both cells improve sharply once real data volume is restored.

| cell | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| `flow_supervised` (RUN-0008, UNaugmented, 768 samples) | 0.352 | 0.016 | 0.185 | −0.067 |
| **`flow_supervised_augmented`** (RUN-0012, augmented, 6,144 samples) | 0.576 | 0.454 | 0.256 | 0.150 |
| `flow_supervised_masked` (RUN-0009, UNaugmented, 768 samples) | −0.018 | −0.088 | −0.079 | −0.0004 |
| **`flow_supervised_masked_augmented`** (RUN-0013, augmented, 6,144 samples) | 0.707 | 0.559 | 0.411 | 0.180 |

Every value function and `accuracy` improves substantially for BOTH cells
once augmentation is restored — the data-starvation confound was real and
large. The masked-penalty cell, which looked collapse-adjacent when
data-starved, is now the BETTER of the two supervised cells on every
metric — with real data, the background magnitude penalty helps rather
than pushing the model toward persistence.

## STEP 3 — comparison table

`slateN` (lyapunov / mass_in_region / signed_mass, averaged over 3 goals)
leads; `accuracy` beside it, flagged suspect per `experiments/METRICS.md`.

| cell | lyapunov | mass_in_region | signed_mass | accuracy |
|---|---|---|---|---|
| `random` (ranking floor) | −0.008 | −0.005 | −0.012 | n/a |
| **ground-truth flow ceiling** (RUN-0011, no model) | **0.813** | **0.770** | **0.512** | 0.167 |
| `flow_direct_control` (direct prediction, EXP-0025) | 0.608 | **0.738** | 0.487 | **0.294** |
| `flow_coarse16` (photometric-only, EXP-0025) | **0.754** | 0.517 | 0.447 | 0.213 |
| `flow_baseline` (photometric-only, EXP-0025) | 0.586 | 0.430 | 0.472 | 0.141 |
| `flow_largedisp24` (photometric-only, EXP-0025 — catastrophic) | 0.374 | 0.151 | 0.232 | −0.027 |
| `flow_supervised` (this extension, RUN-0008, UNaugmented/768 samples) | 0.352 | 0.016 | 0.185 | −0.067 |
| `flow_supervised_masked` (this extension, RUN-0009, UNaugmented/768 samples) | −0.018 | −0.088 | −0.079 | −0.0004 |
| **`flow_supervised_augmented`** (this extension, RUN-0012, x8 augmented/6,144 samples) | 0.576 | 0.454 | 0.256 | 0.150 |
| **`flow_supervised_masked_augmented`** (this extension, RUN-0013, x8 augmented/6,144 samples) | 0.707 | 0.559 | 0.411 | 0.180 |

(Full existing-cell table with every EXP-0025 variant is in `results/RESULTS.md`;
only the cells relevant to this comparison are repeated here.)

## Verdict

**With the data confound removed, direct particle-correspondence
supervision still loses to both `flow_coarse16` and the direct-prediction
control — but the margin shrank enormously, and the ranking among the
supervised variants reversed.**

Augmented vs unaugmented, the improvement is large and monotone on every
metric (`flow_supervised`: lyapunov 0.352→0.576, accuracy −0.067→0.150;
`flow_supervised_masked`: lyapunov −0.018→0.707, accuracy −0.0004→0.180) —
**the 8x data-starvation confound the coordinator flagged was real, and
large enough on its own to explain most of the earlier gap.** It is not,
however, the whole story: even fully augmented,
- `flow_supervised_masked_augmented` (best supervised cell: lyapunov 0.707,
  mass_in_region 0.559, signed_mass 0.411, accuracy 0.180) still loses to
  `flow_coarse16` (0.754 / 0.517 / 0.447, accuracy 0.213) on lyapunov,
  signed_mass and accuracy — it only wins on mass_in_region (0.559 vs
  0.517).
- Both augmented supervised cells still lose to the direct-prediction
  control (0.608 / 0.738 / 0.487, accuracy 0.294) on every metric, by a
  clear margin.
- So **direct flow supervision, at this pilot scale, does not beat the
  existing best cells even once given the same effective data volume** —
  the earlier verdict survives the confound's removal, just by a much
  smaller margin than the unaugmented numbers implied.

**The masked-penalty cell flips from worst to best once real data is
available** — with only 768 samples it looked collapse-adjacent
(indistinguishable from persistence); with 6,144 it is clearly the better
of the two supervised cells on every metric. This says the earlier
"does the magnitude penalty collapse the model" finding was itself
confounded by data starvation, not a property of the penalty design: a
background-magnitude regulariser is not obviously harmful once the model
has enough signal to place its foreground/background boundary correctly in
the first place.

Independent of the confound, both points from the unaugmented write-up
still stand:
- The **model-free ceiling** for this exact parameterisation (backward-warp,
  translation-only, nearest-rigid-particle target) is 0.167 accuracy /
  0.81-0.77-0.51 slateN — a real number any future attempt at this
  approach should be checked against.
- The **accuracy ceiling (0.167) is below the direct control's actual
  accuracy (0.294)**, so no amount of supervision on this exact
  target-field construction can make a flow head beat the direct control
  on `accuracy` — consistent with both augmented cells landing at 0.150/0.180,
  under the ceiling and under the control.

**Does this change EXP-0025's verdict?** Not on its own, but it changes HOW
CONFIDENTLY the earlier negative should be read: EXP-0025's verdict ("the
flow head does NOT beat direct prediction... `flow_coarse16` is the one
exception on slateN/lyapunov") still stands — the best supervised cell here
loses to both the control and `flow_coarse16` on the deciding lyapunov
value function. But direct supervision is now a much closer second than
the unaugmented numbers suggested, and the `slateN` ceiling (0.813
lyapunov) remains clearly above every existing cell including
`flow_coarse16` — so there is real headroom this specific implementation
did not reach, most plausibly because of the remaining gaps named in "What
would change this outcome" below (the training-loop structural difference
from `Trainer`, and the rigid/no-rotation target-field construction) rather
than because direct supervision is fundamentally the wrong idea. Whether
this warrants revisiting EXP-0025's `verdict`/`downgrades` frontmatter is
the owner's call, not made here.

## What would change this outcome

- ~~Re-run with the ×8 flip/rotation augmentation~~ — **done** (RUN-0012/
  RUN-0013): closed most, not all, of the gap. See STEP 2b/Verdict above.
- Match `Trainer`'s own on-the-fly-per-minibatch augmentation structure
  (this script precomputes and shuffles the full 8x set instead — noted as
  a minor, likely-inconsequential difference in STEP 2b, but untested).
- A smoother/anti-aliased target field (e.g. a soft assignment/kernel
  around each particle instead of a hard nearest-center rigid label) to see
  if it raises the accuracy ceiling above 0.167 — the augmented cells now
  sit close enough to `flow_coarse16` that this could plausibly close the
  remaining gap.
- Seeds — one run per cell here, same as EXP-0025's own caveat.

## Threats

- **incomplete-design**: loss weights (`lambda_flow=1.0`, `lambda_bg=0.05`)
  were picked once, not swept; the augmented training loop's batch
  composition differs structurally from `Trainer`'s (STEP 2b).
- **imprecision**: single run per cell, no seed repetition.
- **untested-dependency**: `occ-rasteriser-consistency` is `unchecked`
  (inherited from EXP-0025).
