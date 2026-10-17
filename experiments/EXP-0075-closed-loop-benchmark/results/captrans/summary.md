# EXP-0075 capacity-transport objective / proposals on thin letters (O/T/S/X w15 and w10, DS-0006 starts 40-47 = 32 paired episodes per cell, ens2bal, 10 pushes, full push range, legal scoring, fetch/captrans extras n_extra 64, GD 20 steps, 1,280 evaluations / decision, planner seed 0; H=2 cells use cumspill)

Cells ct_w<W>_H<H>_<obj>_<prop>: obj cur = lyapunov - in-goal fraction; captrans = + crowding-transport term (captrans.CrowdTransport, m_cube = pi r^2 = 4.91 raster units, d_free from the true current state, 3.5 mm slot occupancy radius). prop fetch = fetch extras (control); captrans = capacity-transport extras (pushes along the Hungarian displacement to the capacity slots and +-10 / +-20 deg; H=2: distinct-cube suffixes from the same rows, pool_mode flot). rel uses the width's own normaliser (EXP-0067 vstar_width.json). transport = mean cube distance (mm) to its rectangular-Hungarian capacity slot on the TRUE state.

Capacity slots S per goal (5 mm lattice, 5 x 5 mm footprint inside the mask): w20: O 113, T 69, S 112, X 99; w15: O 75, T 55, S 71, X 61; w10: O 48, T 23, S 40, X 38 (all >= 20 cubes: no goal is over capacity).


## w15, H=1

| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | cov>=0.8 | transport mm @10 | transport mean 1-10 | fallbacks | shifted | optimism exec | plan s/dec | GD s/dec | proposals ms | crowd term exec: excess / off-slot (x1e-3) | crowd now | free slots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ct_w15_H1_cur_captrans | 32 | 0.934 | 0.772 | 26/32 [0.647, 0.911] | 19.4 | 16.9 / 0.4 | 73 % | 0.638 | 3/32 | 2.9 | 6.4 | 0 | 0 | +0.0193 | 1.47 | 0.48 | 1.7 | nan / nan | nan | nan |
| ct_w15_H1_cur_fetch | 32 | 0.947 | 0.781 | 29/32 [0.758, 0.968] | 19.7 | 17.1 / 0.3 | 69 % | 0.643 | 2/32 | 2.6 | 6.2 | 0 | 0 | +0.0212 | 1.45 | 0.46 | 1.9 | nan / nan | nan | nan |
| ct_w15_H1_captrans_captrans | 32 | 0.940 | 0.777 | 29/32 [0.758, 0.968] | 19.5 | 17.0 / 0.4 | 74 % | 0.650 | 1/32 | 2.8 | 6.2 | 0 | 0 | +0.0226 | 1.28 | 0.28 | 1.0 | 0.016 / 10.6 | 10.4 | 47.3 |

## w15, H=2

| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | cov>=0.8 | transport mm @10 | transport mean 1-10 | fallbacks | shifted | optimism exec | plan s/dec | GD s/dec | proposals ms | crowd term exec: excess / off-slot (x1e-3) | crowd now | free slots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ct_w15_H2_cur_fetch | 32 | 0.933 | 0.763 | 30/32 [0.799, 0.983] | 19.5 | 17.2 / 0.6 | 68 % | 0.618 | 2/32 | 3.0 | 6.6 | 0 | 0 | +0.0281 | 2.61 | 0.51 | 2.6 | nan / nan | nan | nan |
| ct_w15_H2_captrans_captrans | 32 | 0.938 | 0.767 | 29/32 [0.758, 0.968] | 19.4 | 16.8 / 0.3 | 73 % | 0.645 | 3/32 | 2.7 | 6.4 | 0 | 0 | +0.0278 | 2.45 | 0.49 | 0.9 | 0.010 / 10.9 | 10.6 | 47.9 |

## w10, H=1

| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | cov>=0.8 | transport mm @10 | transport mean 1-10 | fallbacks | shifted | optimism exec | plan s/dec | GD s/dec | proposals ms | crowd term exec: excess / off-slot (x1e-3) | crowd now | free slots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ct_w10_H1_cur_captrans | 32 | 0.858 | 0.690 | 6/32 [0.089, 0.353] | 18.4 | 17.6 / 0.9 | 59 % | 0.698 | 3/32 | 5.0 | 9.0 | 0 | 0 | +0.0237 | 1.25 | 0.27 | 1.1 | nan / nan | nan | nan |
| ct_w10_H1_cur_fetch | 32 | 0.865 | 0.691 | 10/32 [0.18, 0.486] | 18.5 | 18.2 / 1.4 | 63 % | 0.691 | 6/32 | 5.1 | 9.2 | 0 | 0 | +0.0231 | 1.41 | 0.42 | 1.8 | nan / nan | nan | nan |
| ct_w10_H1_captrans_captrans | 32 | 0.858 | 0.689 | 3/32 [0.032, 0.242] | 18.3 | 17.8 / 1.2 | 60 % | 0.691 | 5/32 | 5.2 | 9.1 | 0 | 0 | +0.0231 | 1.24 | 0.25 | 1.0 | 0.014 / 25.9 | 24.9 | 24.3 |

## w10, H=2

| cell | n | rel@10 | mean rel 1-10 | done@0.9 | cubes in @10 | entering / leaving per ep | wasted 7-10 | coverage@10 | cov>=0.8 | transport mm @10 | transport mean 1-10 | fallbacks | shifted | optimism exec | plan s/dec | GD s/dec | proposals ms | crowd term exec: excess / off-slot (x1e-3) | crowd now | free slots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ct_w10_H2_cur_fetch | 32 | 0.831 | 0.659 | 1/32 [0.006, 0.157] | 17.8 | 17.2 / 1.2 | 58 % | 0.664 | 3/32 | 5.6 | 9.3 | 0 | 0 | +0.0273 | 2.45 | 0.49 | 1.2 | nan / nan | nan | nan |
| ct_w10_H2_captrans_captrans | 32 | 0.840 | 0.672 | 4/32 [0.05, 0.281] | 18.2 | 17.8 / 1.4 | 62 % | 0.673 | 4/32 | 5.1 | 8.9 | 0 | 0 | +0.0278 | 2.45 | 0.49 | 1.0 | 0.024 / 25.4 | 24.6 | 24.9 |

## Paired differences (b - a), bootstrap 95 % CI over the common (goal, start) episodes (coverage / rel: higher is better; transport / leaving: lower is better)

| effect | a | b | n | d rel@10 [CI] | d mean rel 1-10 [CI] | d coverage@10 [CI] | d transport@10 mm [CI] | d transport mean 1-10 [CI] | d leaving [CI] | cov better / worse |
|---|---|---|---|---|---|---|---|---|---|---|
| proposals (H1) | ct_w15_H1_cur_fetch | ct_w15_H1_cur_captrans | 32 | -0.013 [-0.028, -0.003] | -0.009 [-0.017, -0.002] | -0.005 [-0.027, 0.017] | +0.29 [0.077, 0.562] | +0.20 [-0.021, 0.418] | +0.031 [-0.251, 0.312] | 16 / 15 |
| objective + proposals (H1) | ct_w15_H1_cur_fetch | ct_w15_H1_captrans_captrans | 32 | -0.007 [-0.019, 0.004] | -0.004 [-0.013, 0.005] | +0.007 [-0.015, 0.03] | +0.20 [-0.026, 0.467] | +0.01 [-0.259, 0.274] | +0.062 [-0.188, 0.344] | 15 / 14 |
| objective given captrans proposals (H1) | ct_w15_H1_cur_captrans | ct_w15_H1_captrans_captrans | 32 | +0.007 [-0.002, 0.015] | +0.005 [-0.001, 0.012] | +0.012 [-0.014, 0.041] | -0.09 [-0.179, -0.008] | -0.19 [-0.345, -0.042] | +0.031 [-0.281, 0.312] | 13 / 15 |
| objective + proposals (H2) | ct_w15_H2_cur_fetch | ct_w15_H2_captrans_captrans | 32 | +0.005 [-0.006, 0.016] | +0.004 [-0.007, 0.015] | +0.027 [-0.003, 0.058] | -0.30 [-0.557, -0.061] | -0.17 [-0.455, 0.099] | -0.312 [-0.688, 0.031] | 19 / 10 |
| horizon, captrans | ct_w15_H1_captrans_captrans | ct_w15_H2_captrans_captrans | 32 | -0.002 [-0.012, 0.008] | -0.011 [-0.019, -0.002] | -0.005 [-0.035, 0.024] | -0.12 [-0.336, 0.061] | +0.24 [-0.012, 0.469] | -0.125 [-0.344, 0.094] | 17 / 15 |
| horizon, control | ct_w15_H1_cur_fetch | ct_w15_H2_cur_fetch | 32 | -0.014 [-0.022, -0.006] | -0.018 [-0.027, -0.01] | -0.025 [-0.052, 0.0] | +0.37 [0.149, 0.622] | +0.42 [0.158, 0.679] | +0.250 [-0.094, 0.594] | 12 / 17 |
| proposals (H1) | ct_w10_H1_cur_fetch | ct_w10_H1_cur_captrans | 32 | -0.007 [-0.023, 0.009] | -0.002 [-0.012, 0.009] | +0.007 [-0.019, 0.034] | -0.07 [-0.974, 0.615] | -0.23 [-1.058, 0.329] | -0.500 [-1.344, 0.219] | 15 / 15 |
| objective + proposals (H1) | ct_w10_H1_cur_fetch | ct_w10_H1_captrans_captrans | 32 | -0.007 [-0.023, 0.01] | -0.002 [-0.011, 0.008] | +0.000 [-0.028, 0.029] | +0.12 [-0.509, 0.748] | -0.15 [-0.807, 0.329] | -0.219 [-1.157, 0.625] | 12 / 15 |
| objective given captrans proposals (H1) | ct_w10_H1_cur_captrans | ct_w10_H1_captrans_captrans | 32 | +0.001 [-0.014, 0.013] | -0.000 [-0.009, 0.008] | -0.007 [-0.031, 0.014] | +0.20 [-0.162, 0.591] | +0.08 [-0.228, 0.437] | +0.281 [-0.344, 1.0] | 13 / 14 |
| objective + proposals (H2) | ct_w10_H2_cur_fetch | ct_w10_H2_captrans_captrans | 32 | +0.009 [-0.01, 0.028] | +0.012 [-0.0, 0.024] | +0.009 [-0.02, 0.038] | -0.51 [-0.919, -0.103] | -0.37 [-0.655, -0.12] | +0.281 [-0.312, 0.876] | 17 / 13 |
| horizon, captrans | ct_w10_H1_captrans_captrans | ct_w10_H2_captrans_captrans | 32 | -0.018 [-0.033, -0.004] | -0.018 [-0.028, -0.007] | -0.018 [-0.044, 0.006] | -0.14 [-0.765, 0.508] | -0.17 [-0.775, 0.369] | +0.250 [-0.281, 0.906] | 13 / 16 |
| horizon, control | ct_w10_H1_cur_fetch | ct_w10_H2_cur_fetch | 32 | -0.034 [-0.057, -0.011] | -0.032 [-0.047, -0.017] | -0.027 [-0.054, -0.0] | +0.49 [-0.287, 1.183] | +0.05 [-0.691, 0.708] | -0.250 [-1.188, 0.532] | 13 / 16 |

## Per goal, H=1 (means over the 8 starts)

| width | goal | S | cov@10 cur_fetch | cov@10 captrans_captrans | transport@10 cur_fetch | transport@10 captrans_captrans | rel@10 cur_fetch | rel@10 captrans_captrans |
|---|---|---|---|---|---|---|---|---|
| w15 | O | 75 | 0.613 | 0.628 | 2.8 | 3.1 | 0.946 | 0.928 |
| w15 | T | 55 | 0.764 | 0.759 | 2.3 | 2.7 | 0.953 | 0.952 |
| w15 | S | 71 | 0.576 | 0.612 | 2.7 | 2.7 | 0.931 | 0.931 |
| w15 | X | 61 | 0.618 | 0.601 | 2.7 | 2.7 | 0.959 | 0.949 |
| w10 | O | 48 | 0.690 | 0.680 | 3.4 | 4.2 | 0.881 | 0.843 |
| w10 | T | 23 | 0.800 | 0.802 | 10.3 | 9.8 | 0.873 | 0.880 |
| w10 | S | 40 | 0.620 | 0.655 | 3.7 | 3.8 | 0.821 | 0.842 |
| w10 | X | 38 | 0.653 | 0.627 | 3.0 | 3.0 | 0.883 | 0.867 |

## Candidate-level diagnostic (captrans_diag.py: visited states of the captrans_captrans H=1 cells, pushes 3 / 6 / 9, up to 512 legal captrans + fetch proposals scored under both objectives)

| width | states | std base cost | std crowd term | crowd range (max - min) | Spearman(base, captrans) | same argmin | top-8 overlap | max excess part | mean off-slot part |
|---|---|---|---|---|---|---|---|---|---|
| w10 | 48 | 0.1190 | 0.0059 | 0.0241 | 0.9989 | 98 % | 96 % | 0.0000 | 0.0202 |
| w15 | 33 | 0.0863 | 0.0019 | 0.0084 | 0.9994 | 94 % | 95 % | 0.0000 | 0.0106 |
