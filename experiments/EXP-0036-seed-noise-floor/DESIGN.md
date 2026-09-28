# EXP-0036 — training-seed noise floor (TODO H1), and an ensemble of seeds

Written 2026-09-24 BEFORE training (plan gate).

## Why
Every model comparison in the register is between single training runs, so no
margin has a training-noise floor. EXP-0035 also found an ensemble of seven NFD
models (6 variants + the weak finetuned one) beats the best single model; whether any averaging (e.g. of seeds of one
config) does the same is unknown.

## Design
- Train 3 more seeds (1, 2, 3; `train_nfd.py --seed k`) of the world-frame NFD
  baseline, config `Baselines/NFD/configs/nfd_train_3ch_randlen.yaml` unchanged
  except `output.log_dir` (configs in configs/). With the existing unseeded run
  (`nfd_3ch_randlen`) this gives 4 runs. ~1.8 h each, sequential.
- Score all 4 + their prediction-average ("seed ensemble") on: randlen_test (eval
  harness, soft truth, 30 goals, `--device cuda`) and DS-0007 scattered_n20/n50
  (EXP-0030 analysis code, adding the 3 seeds as OCC_ADAPTERS entries).
- Quantities: per-corpus sd over the 4 seeds of lyapunov slateN and accuracy (the
  seed floor); seed-ensemble minus mean single seed (paired over states).

## Predictions
P1: the seed sd of lyapunov slateN on randlen_test (30 goals) is >= 0.005 -- i.e.
    comparable to many margins the register reads (0.01-0.03), so several close
    calls become unresolvable. Refuted if < 0.002.
P2: the seed ensemble beats the mean single seed on DS-0007 (state-bootstrap CI
    excluding 0), but by LESS than the seven-model NFD ensemble beat the best single
    model in EXP-0035 (diversity of architectures adds beyond seed averaging).
Both discriminating: seeds could be near-identical (tiny sd, no ensemble gain).

## Scope limits written in advance
One architecture (world-frame NFD). The expensive residual-warped arm (5.4 h/run)
is not seeded here.

## Addendum (2026-09-24, after the per-seed slateN numbers, before this comparison ran)
P2's reference number (EXP-0035's 7-model gain) was measured on HALF pools, while the
seed scoring here uses whole pools, so the two gains are not comparable. P2 is
re-scored like for like by code/ensemble_compare.py on the same whole pools:
seed_ens4 (4 seeds) vs arch_ens4 (4 architectures: nfd_3ch_randlen, nfd_warped_randlen,
nfd_residual_warped_flipaug_randlen, nfd_residual_worldframe_noaug_ep43; member count
matched) and vs the 7-model ensemble, each minus the best single architecture.
P2 as pre-registered predicts seed_ens4 - best < arch_ens4 - best (diversity adds).
Also: P1's randlen_test slateN comes from eval_report (accuracy_seeds.sh).
