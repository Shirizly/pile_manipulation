# MODEL-0012 -- dyn-res GNN on FleX DS-0021, <=30 true-particle nodes, CORRECTED action frame (EXP-0064 RUN-0005)

**Status:** active

## What this is

MODEL-0011's training repeated with ONE change: the adapter reads the stored action as (x, -z)
(`train.action_z_sign: -1`, invariant `flex-action-frame-neg-z`). Same source-repo training loop
(`experiments/EXP-0064-*/code/train_gnn_dyn_grouped.py`, loop body identical to the source
`train/train_gnn_dyn.py`), same `GroupedParticleDataset` adapter otherwise (Gaussian-tube
`build_action_delta` encoding, sigma 1.2/24; particle_den 1000; <= 30 FPS nodes; frames 0-5 of each
trajectory; 1800 / 200 windows), seed 42, batch 16, Adam 1e-3, n_rollout 5, 100 epochs (as-run
MODEL-0011 was stopped by hand at 100). `checkpoint.pth` = `net_best.pth` (lowest val loss).

## Read this before reusing

- Escaped-particle rows of DS-0021 (3.5-14.6 % by group) are NOT filtered (kept identical to MODEL-0011).
- Action encoding is not the baseline `ParticleDataset`'s (EXP-0064 issues.md I-3).
- One seed. Native FleX units /24; particle input, not images -- not comparable to MODEL-0010 (visual).
- Feed actions with z sign -1: `code/eval_extended.py --models gnn_fixed:-1:<ckpt>`.

## Provenance

commit d72bb304 (dirty; this port's files uncommitted), RTX 4070 Laptop, 2026-10-03 21:19 -> 22:50 CEST, 100 epochs, best val loss 0.0453 (MODEL-0011: 0.0615). sha256 570224e9...
Exact argv + git provenance: `experiments/EXP-0064-*/artifacts/RUN-0005-train-fixed-frame/{COMMAND.txt,*.json}`;
resolved config: `config.yaml` here (= RUN-0005 `config.yaml`). Promotion: `code/finalize_model0012.sh`.

## Contents

`checkpoint.pth`, `config.yaml`, `epochs/` (all epoch checkpoints + `train_log.txt`, gitignored).

## Test history

See `tests.md`.
