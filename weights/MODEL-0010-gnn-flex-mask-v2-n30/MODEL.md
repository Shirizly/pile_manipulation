# MODEL-0010 -- dyn-res-pile-manip GNN architecture trained FROM SCRATCH on FleX DS-0020 v2, constant 30 nodes, visual target

**Status:** active

## What this is

EXP-0062 RUN-0003. `PropNetDiffDenModel` (Wang et al. RSS 2023; `model/gnn_dyn.py`, byte-identical to the
source) trained with `Baselines/GNN/flex_train.py` on DS-0020 v2 (train 1707 / val 190 trajectories,
5-step windows: 8,289 / 911). **Input = visual only**: 30 nodes FPS-sampled from the colour image's
foreground point cloud back-projected onto the constant plane y = 0.24 (`flex_predictor.perceive`, the
same function and per-state seed as at test time). **Target = visual** (`chamfer_carry`): the input
voxel cloud carried by each voxel's nearest node's predicted displacement, compared with the true next
frame's voxel cloud by a symmetric squared Chamfer distance. No simulator particle is used in input or
loss. Source recipe kept: Adam 1e-3, batch 4, n_rollout 5 (chained), StepLR(1000). Changes from the
source (variable node count from random density; particle-correspondence targets; s_delta from true
tracked positions) are listed in `Baselines/GNN/flex_train.py`'s docstring and the RUN record.

## Read this before reusing

- **Plateau-stopped at epoch 22** (RUN-0003 reports 23 epochs run -- most likely 0- vs 1-based counting; not re-checked) (patience 10 epochs without a > 0.5 % relative val improvement);
  best epoch 12, val loss 2.98e-4 vs persistence 2.16e-3 on the same windows. Val loss is noisy
  (epoch-to-epoch swings up to 2x), so "plateau" is coarse. Trained on CPU (GPU failed).
- Node count 30 is FIXED: the checkpoint has only seen 30-node graphs and N=30 densities (den 174-10,430).
- The mask output is capped by the nearest-node renderer: at N = 30, carrying pixels with the TRUE node
  motion scores accuracy 0.363 on DS-0019 and 0.312 on DS-0020 v2 val (EXP-0062 RUN-0003).
- Native FleX units; meaningful only on the FleX grid (DS-0019 / DS-0020 instance configs).
- Score with `eval_report.py --corpora flex_ds0019_mask --models gnn_flex_v2_n30 --truth-scoring image`.

## Provenance

- commit d72bb304 (dirty: EXP-0061/0062 uncommitted files + this run's `Baselines/GNN/flex_train.py`,
  `flex_predictor.py` additions), 2026-10-02 16:55-18:20 CEST, CPU (4 threads), concurrent with the
  particle-target sensitivity run.
- `env PYTHONPATH=. OMP_NUM_THREADS=4 python -u -m Baselines.GNN.flex_train train --out Baselines/GNN/runs/flex_v2_n30_chamfer_carry_s0 --target chamfer_carry --rollout 5 --batch 4 --lr 1e-3 --seed 0 --max-epochs 300 --patience 10 --min-rel-delta 0.005 --max-wall-min 25 --save-min 10 --device cpu`
  then the same with `--max-wall-min 150 --resume` (the first 25 min were the target-choice pilot).
- node cache: `python -u -m Baselines.GNN.flex_train cache --particle-num 30 --workers 12`
- run dir: `Baselines/GNN/runs/flex_v2_n30_chamfer_carry_s0/` (net_best.pth, last_state.pt, history.json, train.log)

## Contents

`checkpoint.pth` = `net_best.pth` (epoch 12; sha256 a8342a85...), a plain `state_dict` of
`PropNetDiffDenModel(MODEL_CFG)`; `config.yaml`. Load with `Baselines/GNN/flex_predictor.py::FlexGNNPredictor(ckpt_path=..., particle_num=30)`
or eval_report spec `gnn_flex_v2_n30`.
Sensitivity sibling (not promoted): particle-supervised target, same everything else,
`experiments/EXP-0062-*/artifacts/RUN-0003-gnn/alt_particle_target_best.pth` (run dir `Baselines/GNN/runs/flex_v2_n30_particle_s0/`).

## Regeneration

The commands above (~25 + 57 min on 4 CPU threads under contention). Not bitwise reproducible across devices.

## Inference cost

CUDA (RTX 4070 Laptop): 77.2 ms per DS-0019 slate incl. its own perception, 43.6 ms with perception cached;
CPU 259 / 207 ms. Launch-bound by per-candidate render and per-sample graph-build loops (EXP-0062 RUN-0003).

## Test history

See `tests.md`.
