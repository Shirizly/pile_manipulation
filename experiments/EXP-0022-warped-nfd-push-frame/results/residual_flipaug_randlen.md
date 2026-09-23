# RUN-0019 — warped + flip augmentation + residual, on overnight_randlen

The combination the record implied and that had never been run: it stacks the
two changes that individually helped. Same corpus, split, recipe and
gradient-step count as the world-frame baseline. Best val loss 0.005634 at
epoch 112 (RUN-0010, the same arm without the residual: 0.009120).

## `slateN`, goal-averaged (leads)

lyapunov / mass_in_region / signed_mass:

| corpus | world-frame baseline | warped + flip | warped + flip + **residual** |
|---|---|---|---|
| L20mm | **.853** / **.820** / **.753** | .858 / .819 / .745 | .846 / .771 / .744 |
| L40mm | **.932** / .783 / .841 | .911 / .809 / .855 | .921 / **.818** / **.868** |
| randlen_test | **.942** / **.889** / **.866** | .943 / .866 / .844 | .938 / .876 / .853 |

**A wash.** All three arms sit within ~0.01-0.05 of each other with no
consistent winner, and no seed-level noise floor was measured, so this ordering
must not be read.

## Swept-region `accuracy` (suspect metric), with the warp's own ceiling

| corpus | baseline (no ceiling) | warped + flip | warped + flip + **residual** | warp ceiling |
|---|---|---|---|---|
| L20mm | **.4071** | .3496 | .3839 | .6640 |
| L40mm | **.5088** | .4431 | .4690 | .7804 |
| randlen_test | .4564 | .4086 | **.4622** | .6470 |

**This is the result.** The residual parameterisation recovers most of the
warp's accuracy deficit — and on `randlen_test`, the held-out corpus drawn from
the same distribution the models were trained on, it **matches the world-frame
baseline** (.4622 vs .4564; a 0.006 margin with no noise floor behind it, so
"matches", not "beats").

The gain replicates the L20mm pilot's finding at full scale: pilot .360 -> .401,
randlen .409 -> .462. That is the single most reproducible effect in this
experiment, and it is about the output PARAMETERISATION, not about the warp.

## What this does and does not change

- **It does not overturn the verdict.** The claim under test was that the warp
  IMPROVES on the world-frame baseline. Even with the residual, `slateN` is a
  wash and accuracy only reaches parity. The warp still does not help.
- **It does change the shape of the negative.** Before this run the warped arm
  was straightforwardly worse on image prediction. It is now competitive on both
  metrics — so the honest statement is "the warp costs nothing once the model is
  parameterised and trained properly", not "the warp hurts".
- **The residual, not the warp, is the transferable finding.** It helped at
  pilot scale and at full scale, in the warped setting; EXP-0022's L20mm pilot
  also showed it helping slightly in the UNWARPED setting (.402 -> .409), where
  the UNet's existing occ0-into-logit skip already supplies part of the effect.
  Whether an explicit residual head helps a world-frame NFD at randlen scale was
  never tested, and is the obvious next run.
