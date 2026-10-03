# EXP-0062 combined summary (DS-0019, 100 slates; derived by code/combined_summary.py)

| model | slateN lyapunov | slateN mass_in_region | slateN signed_mass | slateN all 9 | accuracy (suspect across families) |
|---|---|---|---|---|---|
| NFD v2 (MODEL-0008) | 0.961 [0.953, 0.968] | 0.925 [0.914, 0.935] | 0.933 [0.922, 0.944] | 0.940 [0.932, 0.946] | 0.549 [0.536, 0.562] |
| NFD v1 seed 0 (MODEL-0005) | 0.948 [0.937, 0.958] | 0.878 [0.861, 0.893] | 0.896 [0.878, 0.913] | 0.907 [0.896, 0.918] | 0.489 [0.471, 0.508] |
| LF switched v2 (MODEL-0009) | 0.935 [0.925, 0.946] | 0.884 [0.867, 0.900] | 0.897 [0.882, 0.912] | 0.906 [0.894, 0.916] | 0.469 [0.455, 0.482] |
| LF switched v1 (MODEL-0004) | 0.843 [0.817, 0.868] | 0.711 [0.685, 0.736] | 0.626 [0.585, 0.667] | 0.726 [0.701, 0.750] | 0.328 [0.312, 0.345] |
| LF single v2 (MODEL-0009) | 0.676 [0.642, 0.709] | 0.683 [0.650, 0.717] | 0.674 [0.644, 0.703] | 0.678 [0.654, 0.699] | 0.305 [0.295, 0.314] |
| GNN v2 N30 (MODEL-0010) | 0.900 [0.887, 0.914] | 0.813 [0.786, 0.838] | 0.835 [0.814, 0.856] | 0.850 [0.835, 0.863] | 0.254 [0.240, 0.267] |
| GNN original ckpt N200 (EXP-0061) | 0.764 [0.719, 0.803] | 0.644 [0.602, 0.684] | 0.636 [0.591, 0.680] | 0.681 [0.646, 0.715] | 0.191 [0.168, 0.212] |
| GNN original ckpt N30 | 0.841 [0.821, 0.860] | 0.692 [0.660, 0.722] | 0.705 [0.667, 0.740] | 0.746 [0.726, 0.766] | 0.188 [0.172, 0.205] |
| GNN render cap N30 (true node motion) | 0.945 [0.934, 0.956] | 0.888 [0.873, 0.903] | 0.906 [0.885, 0.925] | 0.913 [0.901, 0.925] | 0.363 [0.345, 0.380] |

Paired slateN, better model first (diff [95 % slate-bootstrap CI], wins/losses, Holm p in its source file):

| pair | all 9 | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| nfd_v2 - lf_v2_switched | +0.034 [+0.025, +0.044] 79/21 | +0.026 [+0.015, +0.036] 59/32 | +0.040 [+0.025, +0.057] 61/22 | +0.036 [+0.021, +0.051] 64/25 |
| lf_v2_switched - gnn_v2 | +0.056 [+0.043, +0.070] 79/21 | +0.035 [+0.019, +0.051] 67/30 | +0.071 [+0.050, +0.094] 73/27 | +0.062 [+0.039, +0.086] 70/27 |
| nfd_v2 - gnn_v2 | +0.090 [+0.077, +0.104] 97/3 | +0.061 [+0.046, +0.076] 71/18 | +0.112 [+0.087, +0.138] 83/14 | +0.098 [+0.078, +0.118] 81/13 |

Inference, ms per DS-0019 slate (whole pool, 86-191 candidates, one batch; median of 3 processes). Perception (PNG -> binary mask, CPU) adds 10.3 ms per image for NFD/LF; the GNN's own perception is inside its 'incl.' row.

| model | CUDA (RTX 4070 Laptop) | CPU (4 threads) |
|---|---|---|
| NFD v2 | 4.9 | 80.7 |
| LF switched v2 | 10.4 | 55.7 |
| LF single v2 | 4.2 | 39.7 |
| GNN v2 incl. perception | 77.2 | 258.7 |
| GNN v2 perception cached | 43.6 | 206.8 |
