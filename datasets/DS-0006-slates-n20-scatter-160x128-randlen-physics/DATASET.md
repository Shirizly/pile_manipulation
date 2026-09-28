# DS-0006 — n20 scatter same-state pools, 160 states x 128 pushes, TRAINING-MATCHED physics

**Status:** active
**Payload:** `Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/` (`step0.pt`, `manifest.json`; gitignored)
**Collector:** `Genesis/binned_slate_collection.py` (checkpointed every 5 batches); `runs/ds0006_collect.*`
**Reader:** `Genesis/binned_slate_dataset.py::BinnedSlateCorpus`

## What this is
The benchmark test corpus in the unified shape (data-collection skill): 160 settled
n20 scatter (`drop`) start states, 128 pile-aware candidate pushes each, push length
drawn in 5 balanced bins over 20-70 mm (100% on target, no underflow, no no-ops),
one push per chain. Designed to be split into TWO DISJOINT 64-push POOLS per state
(any random halving -- actions are i.i.d. per state), so a model's per-state
advantage can be tested for repeatability across pools (EXP-0029/0030).

## Physics (matches overnight_randlen, the models' training corpus)
particle friction 0.7, box friction 0.5, density 450, safety_margin 0.005,
settle cap 3000, 5 mm cubes, 128 mm tray, seed 1. (DS-0001 and the stopped
DS-0005 used 0.3 / 1000 -- invariant `benchmark-physics-matches-training`.)

## Validation
`python -m Genesis.binned_slate_dataset <dir>`: chain continuity OK; slate start
spread 0.0 m; bins [4160, 4160, 4160, 4000, 4000] requested = realized; push length
20.0 / 44.8 / 70.0 mm. 4727 s on 128 envs.

## Regenerate
    python -m Genesis.binned_slate_collection --n-cubes 20 --n-envs 128 --n-states 160 \
      --n-actions 128 --n-steps 1 --spawn-mode drop --seed 1 --friction 0.7 \
      --box-friction 0.5 --density 450 --settle-steps 3000 --safety-margin 0.005 \
      --tag n20_scatter_s160a128_L20-70mm_randlenphys
