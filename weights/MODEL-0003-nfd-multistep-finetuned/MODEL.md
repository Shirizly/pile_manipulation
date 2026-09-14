# MODEL-0003 — NFD fine-tuned through closed-loop 3-step rollout (lambda=0.9)

**Status:** active

## What this is

`Baselines/NFD/runs/nfd_3ch/unet_best.pth` (a ~30k-parameter 3-channel
UNet), fine-tuned by EXP-0010/RUN-0003 through the closed-loop 3-step
rollout objective `L = lam*L1 + lam^2*L2 + lam^3*L3` (full-image, unmasked
per-step MSE, gradients through all 3 rollout steps), lambda=0.9, 100
epochs, batch=256, Adam lr=3e-4, on `slates_multistep` (n20_L20mm +
n20_L40mm, 35/15 slate split seed 0). This is the checkpoint that showed
the largest (nominal, unresolvable at n=1 seed) closed-loop step-3 accuracy
improvement among the 4 lambdas tested (+0.216 vs untrained +0.092-0.099).

## Why this is promoted, not left in `experiments/temp/`

Per the `experiment-log` skill's rule that a fitted object which will be
compared against in future work belongs in `weights/`, not `temp/`: this is
the best available fine-tuned rollout checkpoint and the natural object a
future experiment (e.g. testing whether the accuracy gain transfers to a
harder, non-ceilinged control corpus — see EXP-0010's "What would change
the verdict") would want to reuse rather than re-derive.

## Read this before reusing this checkpoint

**Lambda is UNRESOLVED.** All four lambdas trained (0.3, 0.5, 0.7, 0.9)
land within ~0.01 of each other on every metric measured (step-3 accuracy:
0.209-0.216); this is a single-seed-per-lambda comparison, so "0.9 is
best" is a nominal ordering, not a resolved claim — see EXP-0010's Threats
(`imprecision`). Do not cite this checkpoint's specific lambda as
meaningfully better than 0.7's.

**The control benefit of this fine-tuning is UNMEASURED, not established.**
EXP-0010/RUN-0003 found terminal `slateN` (lyapunov capture) was ALREADY at
ceiling for the UNTRAINED NFD model on the corpus tested here (0.967,
15/0/0 wins over the untrained init at every lambda) — the metric had no
headroom left to detect whether the ~2.2x accuracy gain also improved
control ranking. Only `mass_in_region` moved at all (0.821 -> ~0.91), and
that is a single, non-primary value function. **Do not cite this model as
"control-improved" or "slateN-validated"** — it is validated on `accuracy`
only. See EXP-0010's "What would change the verdict" for the specific
follow-up (re-score this exact checkpoint against a harder, non-ceilinged
corpus, e.g. EXP-0008's `slaten-broad` overnight holdout) that would
resolve this.

## Provenance

- **commit:** 0ddab20f (dirty tree, same pre-existing unrelated docs/skills
  reorganisation as other EXP-0004..EXP-0010 records)
- **script:** `experiments/temp/multistep-nfd/train_nfd_multistep.py`
  (copied to `experiments/EXP-0010-multistep-horizon3-rollout/code/
  multistep-nfd__train_nfd_multistep.py`)
- **data:** raw `Genesis/data/slates_multistep/{n20_L20mm,n20_L40mm}/
  *_{batch}_data.pt` files, 35/15 slate split, seed 0 (see EXP-0010's
  RUN-0003 for the exact split)
- **init:** `Baselines/NFD/runs/nfd_3ch/unet_best.pth`
- **recipe:** `L = 0.9*L1 + 0.81*L2 + 0.729*L3`, full-image unmasked MSE
  per step, Adam lr=3e-4, batch=256, 100 epochs (3500 iterations)
- **file copied from:** `experiments/temp/multistep-nfd/nfd_lam0.9.pth`

## Contents

`checkpoint.pth` (torch state dict, same format as
`Baselines/NFD/runs/nfd_3ch/unet_best.pth`) — the fine-tuned UNet weights.

## Regeneration

`python experiments/EXP-0010-multistep-horizon3-rollout/code/
multistep-nfd__train_nfd_multistep.py` with `lam=0.9` (see the script's own
argument handling), starting from `Baselines/NFD/runs/nfd_3ch/unet_best.pth`
and the same 35/15 seed-0 split. Not yet wrapped in a single documented
one-command entry point with a fixed `--lambda` flag; the script's own
`__main__` sweeps all 4 lambdas in one run (see EXP-0010/RUN-0003's
provenance for the exact invocation).

## Test history

See `tests.md`.
