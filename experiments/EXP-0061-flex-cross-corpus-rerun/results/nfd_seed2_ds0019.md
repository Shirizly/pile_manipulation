# NFD seed 2 on DS-0019 -- binary image-mask truth (EXP-0061, 2026-10-01)

Checkpoint `Baselines/NFD/runs/nfd_3ch_flex_mask_seed2/unet_best.pth` (= weights/MODEL-0007). Corpus `flex_ds0019_mask`: 16583 rows, 100 same-state slates (pool 86-191). Input occ0 AND truth = binary top-down image mask (frac > 0); slateN via `eval_report._capture_report(truth_s0=None)` (= `eval_report.py --corpora flex_ds0019_mask --truth-scoring image`). Plate 2.4 units = 10.67 px. Script `code/score_nfd_flex.py ds0019`. Sanity pass by the training agent, not the final eval.

## slateN (lead metric), per goal x value function

| row | goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| nfd_flex_mask_seed2 | random_quadrant | 0.939 | 0.831 | 0.902 |
| nfd_flex_mask_seed2 | ring_O | 0.950 | 0.877 | 0.883 |
| nfd_flex_mask_seed2 | T | 0.969 | 0.925 | 0.927 |
| **nfd_flex_mask_seed2** | **avg** | **0.953** | **0.878** | **0.904** |
| random | random_quadrant | 0.008 | 0.001 | 0.011 |
| random | ring_O | 0.006 | 0.009 | 0.001 |
| random | T | 0.003 | -0.004 | -0.004 |
| **random** | **avg** | **0.006** | **0.002** | **0.003** |
| persistence | random_quadrant | 0.184 | -0.031 | 0.004 |
| persistence | ring_O | -0.117 | -0.093 | 0.061 |
| persistence | T | 0.303 | 0.158 | 0.215 |
| **persistence** | **avg** | **0.123** | **0.011** | **0.093** |

Per-slate sem of the model's slateN (n=100 slates): random_quadrant/lyapunov 0.009, random_quadrant/mass_in_region 0.017, random_quadrant/signed_mass 0.012, ring_O/lyapunov 0.008, ring_O/mass_in_region 0.016, ring_O/signed_mass 0.015, T/lyapunov 0.007, T/mass_in_region 0.013, T/signed_mass 0.012

persistence's slateN is DEGENERATE (constant prediction per slate -> argmax tie -> first candidate); not a ranking baseline. random = 20-seed Monte-Carlo of an exact 0.

## Goal degeneracy (binary-mask truth): frac(dv_true == 0) rows / frac slates with a flat pool

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| random_quadrant | 0.001 / 0.00 | 0.187 / 0.00 | 0.006 / 0.00 |
| ring_O | 0.001 / 0.00 | 0.047 / 0.00 | 0.008 / 0.00 |
| T | 0.001 / 0.00 | 0.067 / 0.00 | 0.008 / 0.00 |

## Swept-region accuracy (secondary; within-model-type only)

| row | accuracy |
|---|---|
| nfd_flex_mask_seed2 | 0.4842 |
| persistence | 0.0000 |
| random | n/a |

## For comparison only: slateN vs the SOFT PARTICLE splat (eval_report default `--truth-scoring soft`; NOT the user-decided truth)

| row | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| persistence | 0.119 | 0.007 | 0.007 |
| nfd_flex_mask_seed2 | 0.955 | 0.830 | 0.899 |

Predictor parity (eval-harness `predict_occ` vs training-path tensors, 16 rows): max |diff| 3.2e-05.

Known train/test mismatch (DATASET.md): DS-0020 = one uniform full-workspace spread (occupied 0.42), DS-0019 = compact piles (0.12), obj-biased actions.
