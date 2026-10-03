# MODEL-0011 -- dyn-res GNN (PropNetDiffDenModel) trained on FleX DS-0021 from TRUE particles, <=30 FPS nodes, AS RUN with a z-MIRRORED action input

**Status:** active (kept as the as-run object of EXP-0064; **do not use as a GNN-quality reference** -- see below)

## What this is

EXP-0064 RUN-0003 (run in the external source repo ~/Code/dyn-res-pile-manip, ported 2026-10-03).
`PropNetDiffDenModel` (`Baselines/GNN/model/gnn_dyn.py`, byte-identical to the source and to `model/gnn_dyn.py`)
trained with the source repo's own loop (`train_gnn_dyn.py`, copied as EXP-0064 `code/train_gnn_dyn_grouped.py`;
loop body verified identical by diff) on DS-0021 through `GroupedParticleDataset` (EXP-0064 `code/dataset_grouped_particles.py`):
input = <= 30 FPS-sampled TRUE simulator particles (min(30, target carrot count)), normalised /24, (x, z, y) order;
action input = `transforms.functional.build_action_delta` Gaussian-tube field (sigma 1.2/24) x full push vector
(NOT the baseline ParticleDataset's distance-to-end, gated encoding); particle_den fixed 1000;
one 6-frame window per state (frames 0-5), 1800 train / 200 val windows; batch 16, Adam 1e-3, n_rollout 5, seed 42.
Configured 300 epochs, stopped by hand after epoch 100; `checkpoint.pth` = `net_best.pth`, epoch 87, val loss 0.0615.

## Read this before reusing

- **The action was read in the wrong frame.** The dataset stores actions as (x, -z) (`flex-action-frame-neg-z`);
  the adapter used them as (x, z), so every push the model saw in training and test is mirrored in z.
  cos(true node displacement, action input) = 0.05 as run vs 0.68 corrected (EXP-0064 RUN-0007).
  This model learned dynamics with a nearly uninformative action channel. Corrected sibling: MODEL-0012.
- ~3.5-14.6 % of training rows (rising with object count) contain escaped particles; not filtered.
- Units: native FleX /24. Meaningful only on DS-0021/DS-0022-style particle input.

## Provenance

- source repo commit 5c9eca9 (dirty: collector, adapter, train/eval scripts uncommitted there), 2026-10-03 ~04:53-06:15 (source report: ~1h20m), GPU.
- command (RECONSTRUCTED from the source report): `python -m train.train_gnn_dyn_grouped` with `config/train/gnn_dyn_grouped.yaml`
  (copy: EXP-0064 `code/configs/gnn_dyn_grouped.yaml`; as-run resolved copy: `train_config_as_run.yaml` here).
- training log: `epochs/train_log.txt`; console log EXP-0064 `artifacts/RUN-0003-train/`.

## Contents

`checkpoint.pth` (sha256 83198dc4...), `epochs/` (all net_epoch_*_iter_0.pth + net_best.pth, gitignored), `train_config_as_run.yaml`, `config.yaml`.
Load: `PropNetDiffDenModel(yaml.safe_load(open(<config>)), use_gpu)` + `load_state_dict`; feed actions with z sign +1
to reproduce the as-run numbers (EXP-0064 `code/eval_extended.py --models gnn_asrun:+1:<ckpt>`).

## Test history

See `tests.md`.
