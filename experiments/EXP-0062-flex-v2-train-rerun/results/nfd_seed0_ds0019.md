# EXP-0062 -- nfd_v2_s0 (DS-0020 v2-trained NFD, seed 0) on DS-0019

Truth: binary image mask of the true after-state (cell.occ1, occ_source=image_mask); eval_report._capture_report(truth_s0=None) == --truth-scoring image. Truth check: values [0.0, 1.0].
100 slates, 16583 rows. v1 = MODEL-0005 (EXP-0061 per-slate file, reproduced: max |per-slate diff| 0, accuracy 0.4895 -> 0.4895).

## slateN, mean of 3 goals [slate-bootstrap 95% CI]

| model | lyapunov | mass_in_region | signed_mass | all 9 |
|---|---|---|---|---|
| nfd_v2_s0 | 0.961 [0.953, 0.968] | 0.925 [0.914, 0.935] | 0.933 [0.922, 0.944] | 0.940 |
| nfd_v1_s0 (EXP-0061 file) | 0.948 [0.937, 0.958] | 0.878 [0.861, 0.893] | 0.896 [0.878, 0.913] | 0.907 |
| random | 0.006 [-0.007, 0.018] | 0.002 [-0.008, 0.011] | 0.003 [-0.007, 0.012] | 0.004 |
| persistence | 0.123 [0.077, 0.168] | 0.011 [-0.033, 0.057] | 0.093 [0.054, 0.133] | 0.076 |

## slateN per goal x vf (mean +- slate sem)

| model | random/lyapun | random/mass_i | random/signed | ring_O/lyapun | ring_O/mass_i | ring_O/signed | T/lyapun | T/mass_i | T/signed |
|---|---|---|---|---|---|---|---|---|---|
| nfd_v2_s0 | 0.948+-0.007 | 0.875+-0.015 | 0.922+-0.011 | 0.957+-0.008 | 0.937+-0.008 | 0.930+-0.010 | 0.977+-0.006 | 0.963+-0.008 | 0.947+-0.010 |
| nfd_v1_s0 (EXP-0061 file) | 0.926+-0.010 | 0.832+-0.018 | 0.898+-0.012 | 0.952+-0.008 | 0.871+-0.015 | 0.885+-0.014 | 0.967+-0.008 | 0.929+-0.012 | 0.906+-0.014 |
| random | 0.008+-0.013 | 0.001+-0.009 | 0.011+-0.008 | 0.006+-0.010 | 0.009+-0.009 | 0.001+-0.010 | 0.003+-0.012 | -0.004+-0.008 | -0.004+-0.007 |
| persistence | 0.184+-0.047 | -0.031+-0.028 | 0.004+-0.029 | -0.117+-0.038 | -0.093+-0.037 | 0.061+-0.032 | 0.303+-0.042 | 0.158+-0.034 | 0.215+-0.030 |

## accuracy (swept region, ratio of population means; slate-cluster bootstrap CI)

| model | DS-0019 | slate CI | DS-0020 v2 val | traj CI |
|---|---|---|---|---|
| nfd_v2_s0 | 0.5490 | 0.536-0.562 | 0.5052 | 0.496-0.514 |
| nfd_v1_s0 (MODEL-0005) | 0.4895 | 0.471-0.508 | 0.4290 | 0.419-0.439 |
| persistence | 0 | | 0 | |

Paired accuracy delta v2 - v1 on DS-0019: +0.0595 [+0.0524, +0.0674] (slate-cluster).
DS-0020 v2 val rows: 1665.

## paired slateN v2 - v1 (paired_stats.paired_comparison; per-slate mean of the cells; 100 slates)

| cells | diff [95% CI] | wins/losses | Holm p |
|---|---|---|---|
| all_9_cells: nfd_v2_s0 - nfd_v1_s0 | +0.0323 [+0.0220, +0.0428] | 68/31 | 0 |
| all_9_cells: nfd_v2_s0 - random | +0.9360 [+0.9273, +0.9451] | 100/0 | 0 |
| all_9_cells: nfd_v1_s0 - random | +0.9037 [+0.8920, +0.9157] | 100/0 | 0 |
| lyapunov: nfd_v2_s0 - nfd_v1_s0 | +0.0129 [+0.0035, +0.0227] | 50/30 | 0.011 |
| lyapunov: nfd_v2_s0 - random | +0.9552 [+0.9407, +0.9698] | 100/0 | 0 |
| lyapunov: nfd_v1_s0 - random | +0.9423 [+0.9268, +0.9581] | 100/0 | 0 |
| mass_in_region: nfd_v2_s0 - nfd_v1_s0 | +0.0473 [+0.0301, +0.0647] | 62/30 | 0 |
| mass_in_region: nfd_v2_s0 - random | +0.9228 [+0.9088, +0.9369] | 100/0 | 0 |
| mass_in_region: nfd_v1_s0 - random | +0.8755 [+0.8572, +0.8932] | 100/0 | 0 |
| signed_mass: nfd_v2_s0 - nfd_v1_s0 | +0.0368 [+0.0201, +0.0538] | 57/24 | 0 |
| signed_mass: nfd_v2_s0 - random | +0.9301 [+0.9169, +0.9435] | 100/0 | 0 |
| signed_mass: nfd_v1_s0 - random | +0.8934 [+0.8740, +0.9127] | 100/0 | 0 |
| goal random_quadrant (3 vf): v2 - v1 | +0.0299 [+0.0110, +0.0493] | 52/30 | 0.0035 |
| goal ring_O (3 vf): v2 - v1 | +0.0387 [+0.0258, +0.0522] | 64/23 | 0 |
| goal T (3 vf): v2 - v1 | +0.0284 [+0.0152, +0.0423] | 47/16 | 5e-05 |

## strata (piece-count tercile edges [190.0, 430.0])

| stratum (slates) | slateN v2 | slateN v1 | delta [CI] | acc v2 | acc v1 | acc delta CI |
|---|---|---|---|---|---|---|
| rand_blob (50) | 0.942 | 0.906 | +0.036 [+0.023, +0.051] | 0.535 | 0.467 | [+0.063, +0.072] |
| rand_spread (50) | 0.937 | 0.909 | +0.028 [+0.012, +0.044] | 0.561 | 0.508 | [+0.040, +0.065] |
| pieces_small (21) | 0.931 | 0.888 | +0.044 [+0.020, +0.070] | 0.482 | 0.425 | [+0.052, +0.063] |
| pieces_mid (39) | 0.945 | 0.918 | +0.026 [+0.011, +0.043] | 0.559 | 0.489 | [+0.062, +0.079] |
| pieces_large (40) | 0.939 | 0.907 | +0.032 [+0.017, +0.048] | 0.570 | 0.519 | [+0.037, +0.066] |

## goal degeneracy frac(dv_true == 0) (binary-mask truth; from EXP-0061, same rows)

| goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| random_quadrant | 0.0010 | 0.1874 | 0.0061 |
| ring_O | 0.0005 | 0.0471 | 0.0083 |
| T | 0.0005 | 0.0674 | 0.0076 |
