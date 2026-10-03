# EXP-0062 RUN-0003 -- dyn-res GNN trained on DS-0020 v2, constant 30 nodes (MODEL-0010), on DS-0019

Truth: binary image mask of the true after-state (cell.occ1, occ_source=image_mask); eval_report._capture_report(truth_s0=None) == --truth-scoring image (values [0.0, 1.0]). 100 slates, 16583 rows.
Reuse check (original ckpt N 200 re-scored vs EXP-0061 `gnn_flex_drp_samp0`): max |per-slate diff| 0, max |row rms diff| 0.00667, accuracy 0.1908 -> 0.1908.
Small-pile fallback (states with <= N voxels): gnn_old_n200_rescored 43/100, gnn_old_n30 0/100, gnn_v2_n30_alt_target 0/100, gnn_v2_n30 0/100; cap 0/100.

## slateN, mean of 3 goals [slate-bootstrap 95% CI]

| model | lyapunov | mass_in_region | signed_mass | all 9 [CI] |
|---|---|---|---|---|
| gnn_v2_n30 (MODEL-0010) | 0.900 [0.887, 0.914] | 0.813 [0.786, 0.838] | 0.835 [0.814, 0.856] | 0.850 [0.835, 0.863] |
| gnn_v2_n30_alt_target | 0.895 [0.879, 0.910] | 0.813 [0.789, 0.834] | 0.833 [0.809, 0.855] | 0.847 [0.832, 0.861] |
| gnn_old_n30 (orig ckpt, N 30) | 0.841 [0.821, 0.860] | 0.692 [0.660, 0.722] | 0.705 [0.667, 0.740] | 0.746 [0.726, 0.766] |
| gnn_old_n200 (orig ckpt, EXP-0061 file) | 0.764 [0.719, 0.803] | 0.644 [0.602, 0.684] | 0.636 [0.591, 0.680] | 0.681 [0.646, 0.715] |
| cap_n30 (true node motion) | 0.945 [0.934, 0.956] | 0.888 [0.873, 0.903] | 0.906 [0.885, 0.925] | 0.913 [0.901, 0.925] |
| nfd_v2_s0 (MODEL-0008, RUN-0001 file) | 0.961 [0.953, 0.968] | 0.925 [0.914, 0.935] | 0.933 [0.922, 0.944] | 0.940 [0.933, 0.946] |
| lf_v2_switched (MODEL-0009, RUN-0002 file) | 0.935 [0.925, 0.945] | 0.884 [0.868, 0.901] | 0.897 [0.882, 0.912] | 0.906 [0.895, 0.916] |
| random | 0.006 [-0.007, 0.019] | 0.002 [-0.007, 0.012] | 0.003 [-0.007, 0.012] | 0.004 [-0.003, 0.010] |
| persistence | 0.123 [0.076, 0.167] | 0.011 [-0.031, 0.055] | 0.093 [0.051, 0.135] | 0.076 [0.039, 0.111] |

## slateN per goal x vf (mean +- slate sem)

| model | random/lyapun | random/mass_i | random/signed | ring_O/lyapun | ring_O/mass_i | ring_O/signed | T/lyapun | T/mass_i | T/signed |
|---|---|---|---|---|---|---|---|---|---|
| gnn_v2_n30 (MODEL-0010) | 0.941+-0.008 | 0.858+-0.016 | 0.886+-0.014 | 0.816+-0.019 | 0.752+-0.025 | 0.782+-0.020 | 0.944+-0.009 | 0.829+-0.021 | 0.838+-0.022 |
| gnn_v2_n30_alt_target | 0.946+-0.007 | 0.838+-0.016 | 0.895+-0.012 | 0.807+-0.020 | 0.814+-0.021 | 0.814+-0.022 | 0.930+-0.011 | 0.785+-0.023 | 0.789+-0.025 |
| gnn_old_n30 (orig ckpt, N 30) | 0.905+-0.012 | 0.777+-0.019 | 0.852+-0.015 | 0.742+-0.021 | 0.598+-0.031 | 0.588+-0.037 | 0.877+-0.016 | 0.701+-0.028 | 0.675+-0.035 |
| gnn_old_n200 (orig ckpt, EXP-0061 file) | 0.863+-0.024 | 0.719+-0.025 | 0.818+-0.023 | 0.589+-0.037 | 0.562+-0.035 | 0.509+-0.037 | 0.839+-0.025 | 0.650+-0.028 | 0.582+-0.037 |
| cap_n30 (true node motion) | 0.942+-0.008 | 0.876+-0.013 | 0.908+-0.011 | 0.924+-0.012 | 0.864+-0.013 | 0.884+-0.021 | 0.969+-0.007 | 0.924+-0.012 | 0.926+-0.013 |
| nfd_v2_s0 (MODEL-0008, RUN-0001 file) | 0.948+-0.007 | 0.875+-0.015 | 0.922+-0.011 | 0.957+-0.008 | 0.937+-0.008 | 0.930+-0.010 | 0.977+-0.006 | 0.963+-0.008 | 0.947+-0.010 |
| lf_v2_switched (MODEL-0009, RUN-0002 file) | 0.925+-0.009 | 0.867+-0.015 | 0.891+-0.013 | 0.926+-0.010 | 0.875+-0.015 | 0.876+-0.012 | 0.955+-0.008 | 0.912+-0.016 | 0.925+-0.012 |
| random | 0.008+-0.013 | 0.001+-0.009 | 0.011+-0.008 | 0.006+-0.010 | 0.009+-0.009 | 0.001+-0.010 | 0.003+-0.012 | -0.004+-0.008 | -0.004+-0.007 |
| persistence | 0.184+-0.047 | -0.031+-0.028 | 0.004+-0.029 | -0.117+-0.038 | -0.093+-0.037 | 0.061+-0.032 | 0.303+-0.042 | 0.158+-0.034 | 0.215+-0.030 |

## accuracy (swept region, ratio of population means) -- SUSPECT across model types

| model | DS-0019 | slate CI | DS-0020 v2 val | val traj CI |
|---|---|---|---|---|
| gnn_v2_n30 | 0.2541 | 0.240-0.267 | 0.1903 | 0.180-0.200 |
| gnn_old_n30 | 0.1883 | 0.172-0.205 | 0.1080 | 0.097-0.119 |
| gnn_old_n200 | 0.1908 | 0.168-0.212 | -0.0037 | -0.026-0.018 |
| cap_n30 | 0.3627 | 0.345-0.380 | 0.3124 | 0.297-0.329 |
| nfd_v2_s0 | 0.5490 | 0.534-0.563 | 0.5052 |  |
| lf_v2_switched | 0.4694 | 0.454-0.483 | 0.3903 |  |
| gnn_v2_n30_alt_target | 0.2459 | 0.231-0.261 | 0.1926 | 0.181-0.204 |
| persistence | 0 | | 0 | |

- delta gnn_v2_n30 - gnn_old_n30: +0.0657 [+0.0604, +0.0717] (DS-0019, slate-cluster)
- delta gnn_v2_n30 - gnn_old_n200: +0.0633 [+0.0451, +0.0818] (DS-0019, slate-cluster)
- delta gnn_v2_n30 - nfd_v2_s0: -0.2949 [-0.3011, -0.2886] (DS-0019, slate-cluster)
- delta gnn_v2_n30 - lf_v2_switched: -0.2153 [-0.2219, -0.2083] (DS-0019, slate-cluster)
- delta cap_n30 - gnn_v2_n30: +0.1086 [+0.0953, +0.1226] (DS-0019, slate-cluster)
- delta gnn_v2_n30 - gnn_v2_n30_alt_target: +0.0081 [+0.0036, +0.0129] (DS-0019, slate-cluster)
- GNN v2 accuracy / rendering cap (true node motion, N 30) = 0.701
- slateN cap - GNN v2 (all 9): +0.0635 [0.04890838734921328, 0.0784126026963015]
- DS-0020 v2 val rows: 1665; val fallback states: {'gnn_old_n200_rescored': 224, 'gnn_old_n30': 0, 'cap_n30': 0, 'gnn_v2_n30_alt_target': 0, 'gnn_v2_n30': 0}

## paired slateN (paired_stats.paired_comparison; per-slate mean of the cells; Holm over all pairs of the listed models, GNN-v2 pairs shown)

| cells | a - b | diff [95% CI] | wins/losses | Holm p |
|---|---|---|---|---|
| all_9_cells | gnn_v2_n30 - gnn_old_n30 | +0.1033 [+0.0869, +0.1195] | 91/9 | 0 |
| all_9_cells | gnn_v2_n30 - gnn_old_n200 | +0.1684 [+0.1369, +0.2044] | 91/9 | 0 |
| all_9_cells | gnn_v2_n30 - nfd_v2_s0 | -0.0901 [-0.1038, -0.0768] | 3/97 | 0 |
| all_9_cells | gnn_v2_n30 - lf_v2_switched | -0.0562 [-0.0703, -0.0427] | 21/79 | 0 |
| all_9_cells | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0029 [-0.0110, +0.0165] | 57/43 | 0.68 |
| lyapunov | gnn_v2_n30 - gnn_old_n30 | +0.0589 [+0.0406, +0.0766] | 71/21 | 0 |
| lyapunov | gnn_v2_n30 - gnn_old_n200 | +0.1366 [+0.0988, +0.1793] | 81/15 | 0 |
| lyapunov | gnn_v2_n30 - nfd_v2_s0 | -0.0607 [-0.0761, -0.0461] | 18/71 | 0 |
| lyapunov | gnn_v2_n30 - lf_v2_switched | -0.0351 [-0.0513, -0.0188] | 30/67 | 0 |
| lyapunov | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0058 [-0.0101, +0.0215] | 48/33 | 0.49 |
| mass_in_region | gnn_v2_n30 - gnn_old_n30 | +0.1209 [+0.0919, +0.1499] | 78/19 | 0 |
| mass_in_region | gnn_v2_n30 - gnn_old_n200 | +0.1695 [+0.1281, +0.2131] | 80/19 | 0 |
| mass_in_region | gnn_v2_n30 - nfd_v2_s0 | -0.1118 [-0.1380, -0.0868] | 14/83 | 0 |
| mass_in_region | gnn_v2_n30 - lf_v2_switched | -0.0714 [-0.0939, -0.0496] | 27/73 | 0 |
| mass_in_region | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0005 [-0.0220, +0.0231] | 51/44 | 0.96 |
| signed_mass | gnn_v2_n30 - gnn_old_n30 | +0.1301 [+0.1003, +0.1609] | 80/15 | 0 |
| signed_mass | gnn_v2_n30 - gnn_old_n200 | +0.1990 [+0.1539, +0.2477] | 82/17 | 0 |
| signed_mass | gnn_v2_n30 - nfd_v2_s0 | -0.0978 [-0.1183, -0.0782] | 13/81 | 0 |
| signed_mass | gnn_v2_n30 - lf_v2_switched | -0.0622 [-0.0863, -0.0389] | 27/70 | 0 |
| signed_mass | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0023 [-0.0205, +0.0249] | 45/43 | 0.84 |
| goal random_quadrant | gnn_v2_n30 - gnn_old_n30 | +0.0502 [+0.0316, +0.0692] | 62/24 | 0 |
| goal random_quadrant | gnn_v2_n30 - gnn_old_n200 | +0.0949 [+0.0579, +0.1350] | 63/29 | 0 |
| goal random_quadrant | gnn_v2_n30 - nfd_v2_s0 | -0.0202 [-0.0342, -0.0061] | 31/54 | 0.036 |
| goal random_quadrant | gnn_v2_n30 - lf_v2_switched | +0.0007 [-0.0172, +0.0185] | 43/48 | 1 |
| goal random_quadrant | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0018 [-0.0109, +0.0150] | 41/37 | 1 |
| goal ring_O | gnn_v2_n30 - gnn_old_n30 | +0.1405 [+0.1037, +0.1785] | 77/22 | 0 |
| goal ring_O | gnn_v2_n30 - gnn_old_n200 | +0.2301 [+0.1770, +0.2864] | 76/24 | 0 |
| goal ring_O | gnn_v2_n30 - nfd_v2_s0 | -0.1581 [-0.1861, -0.1315] | 8/88 | 0 |
| goal ring_O | gnn_v2_n30 - lf_v2_switched | -0.1090 [-0.1379, -0.0807] | 18/80 | 0 |
| goal ring_O | gnn_v2_n30 - gnn_v2_n30_alt_target | -0.0284 [-0.0563, -0.0022] | 41/50 | 0.038 |
| goal T | gnn_v2_n30 - gnn_old_n30 | +0.1191 [+0.0841, +0.1559] | 65/17 | 0 |
| goal T | gnn_v2_n30 - gnn_old_n200 | +0.1800 [+0.1362, +0.2275] | 74/17 | 0 |
| goal T | gnn_v2_n30 - nfd_v2_s0 | -0.0920 [-0.1157, -0.0689] | 9/73 | 0 |
| goal T | gnn_v2_n30 - lf_v2_switched | -0.0603 [-0.0856, -0.0359] | 21/64 | 0 |
| goal T | gnn_v2_n30 - gnn_v2_n30_alt_target | +0.0353 [+0.0095, +0.0618] | 46/27 | 0.014 |

## strata (piece-count tercile edges [190.0, 430.0]; slateN = per-slate mean of 9 cells)

| stratum (slates) | slateN GNN v2 / old N30 / old N200 / NFD v2 / LF v2 / cap | d(v2 - old N30) | d(v2 - old N200) | d(v2 - NFD v2) | d(v2 - LF v2) | acc GNN v2 / cap |
|---|---|---|---|---|---|---|
| rand_blob (50) | 0.840 / 0.721 / 0.629 / 0.942 / 0.900 / 0.928 | +0.119 [+0.095, +0.140] | +0.211 [+0.155, +0.270] | -0.102 [-0.124, -0.083] | -0.060 [-0.083, -0.037] | 0.226 / 0.400 |
| rand_spread (50) | 0.859 / 0.771 / 0.733 / 0.937 / 0.912 / 0.898 | +0.088 [+0.067, +0.111] | +0.126 [+0.097, +0.158] | -0.078 [-0.095, -0.062] | -0.053 [-0.068, -0.036] | 0.277 / 0.332 |
| pieces_small (21) | 0.826 / 0.695 / 0.688 / 0.931 / 0.899 / 0.947 | +0.131 [+0.096, +0.164] | +0.138 [+0.096, +0.177] | -0.105 [-0.139, -0.072] | -0.073 [-0.112, -0.035] | 0.187 / 0.401 |
| pieces_mid (39) | 0.851 / 0.747 / 0.596 / 0.945 / 0.905 / 0.913 | +0.104 [+0.077, +0.130] | +0.254 [+0.190, +0.325] | -0.094 [-0.116, -0.074] | -0.054 [-0.075, -0.035] | 0.258 / 0.370 |
| pieces_large (40) | 0.861 / 0.772 / 0.760 / 0.939 / 0.910 / 0.895 | +0.088 [+0.065, +0.112] | +0.100 [+0.071, +0.132] | -0.078 [-0.097, -0.060] | -0.049 [-0.068, -0.031] | 0.281 / 0.339 |
