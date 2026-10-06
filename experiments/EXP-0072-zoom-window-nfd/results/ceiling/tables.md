## T1 raster perceptibility (per-cube iid xy gaussian + yaw 2 deg/mm; 200 rows x 2 draws)
Changed-pixel fraction is linear in sigma (boundary pixels flip at a rate proportional to perimeter x shift), so there is no threshold: sigma_1% = sigma where the hard raster changes by 1% of its occupied pixels; sigma_95 = sigma where >=95% of states give a bit-identical raster, extrapolated from the 0.01 mm point with P(identical)=exp(-lam sigma).

| model input raster | px/mm | changed px @0.01mm | frac occupied changed @0.01mm | P(identical) @0.01mm | sigma_1% (mm) | sigma_95 identical (mm) |
|---|---|---|---|---|---|---|
| world64(2mm/px) | | 0.7 | 0.0076 | 0.49 | 0.016 | 0.0007 |
| world128(1mm/px) | | 3.7 | 0.0054 | 0.35 | 0.018 | 0.0005 |
| zoom64(1mm/px) | | 1.5 | 0.0058 | 0.63 | 0.018 | 0.0011 |
| zoom128(0.5mm/px) | | 6.4 | 0.0065 | 0.39 | 0.017 | 0.0005 |

## T2 ceiling: perfect-physics re-simulation scored as a model prediction of the recorded DS-0016 outcome (accuracy_1, pooled; n=896 rows)
| level / rep | w64_native | w128_native | window64 | window128 | paste_zoom64 | paste_zoom128 | paste_world128 | mm_rms | frac cubes >1mm |
|---|---|---|---|---|---|---|---|---|---|
| TRUE label rendered in frame (raster-style cap) | 1.000 | 1.000 | 1.000 | 1.000 | 0.651 | 0.681 | 0.633 | 0 | 0 |
| 0mm_r0 | 0.807 | 0.782 | 0.899 | 0.900 | 0.641 | 0.670 | 0.622 | 0.32 | 0.013 |
| 0mm_r1 | 0.804 | 0.780 | 0.898 | 0.900 | 0.642 | 0.670 | 0.622 | 0.34 | 0.014 |
| 0.01mm_r0 | 0.789 | 0.765 | 0.889 | 0.890 | 0.639 | 0.666 | 0.621 | 0.49 | 0.019 |
| 0.05mm_r0 | 0.760 | 0.740 | 0.881 | 0.881 | 0.636 | 0.662 | 0.615 | 0.66 | 0.025 |
| 0.25mm_r0 | 0.659 | 0.648 | 0.839 | 0.840 | 0.622 | 0.647 | 0.596 | 0.78 | 0.051 |
| 0.5mm_r0 | 0.579 | 0.569 | 0.813 | 0.812 | 0.607 | 0.629 | 0.571 | 1.12 | 0.171 |
| 1mm_r0 | 0.436 | 0.439 | 0.749 | 0.751 | 0.561 | 0.576 | 0.488 | 1.80 | 0.566 |

Scatter / clump split (w64_native | paste_zoom64):
- 0mm_r0: scatter 0.806 | 0.619; clump 0.807 | 0.654
- 0mm_r1: scatter 0.794 | 0.618; clump 0.809 | 0.655
- 0.01mm_r0: scatter 0.769 | 0.614; clump 0.801 | 0.654
- 0.05mm_r0: scatter 0.725 | 0.611; clump 0.781 | 0.650
- 0.25mm_r0: scatter 0.558 | 0.587; clump 0.730 | 0.643
- 0.5mm_r0: scatter 0.442 | 0.579; clump 0.677 | 0.624
- 1mm_r0: scatter 0.237 | 0.509; clump 0.581 | 0.593

## T3 ceiling slateN (32 pools x 64 pushes; one perturbed start per pool; 13 goals)
| level / rep | w64_native | paste_zoom64 | paste_zoom128 |
|---|---|---|---|
| TRUE label in frame (no noise) | 0.896 | 0.778 | 0.824 |
| 0mm_r0 | 0.889 | 0.786 | 0.825 |
| 0mm_r1 | 0.889 | 0.785 | 0.825 |
| 0.01mm_r0 | 0.876 | 0.777 | 0.821 |
| 0.05mm_r0 | 0.897 | 0.790 | 0.825 |
| 0.25mm_r0 | 0.901 | 0.777 | 0.819 |
| 0.5mm_r0 | 0.897 | 0.785 | 0.836 |
| 1mm_r0 | 0.838 | 0.768 | 0.807 |

## T4 multi-push divergence (chains, perturbed ONCE at chain start, pushes re-run in sequence; accuracy of push k's swept region, truth = recorded)
| level | metric | push 1 | push 2 | push 3 | push 4 | push 5 | push 6 | push 7 | push 8 |
|---|---|---|---|---|---|---|---|---|---|
| 0mm_r0 | acc w64_native | 0.792 | 0.794 | 0.765 | 0.723 | 0.700 | 0.677 | 0.629 | 0.651 |
| 0mm_r0 | acc paste_zoom64 | 0.655 | 0.652 | 0.625 | 0.605 | 0.623 | 0.600 | 0.588 | 0.575 |
| 0mm_r0 | cube mm_rms | 0.28 | 0.60 | 1.00 | 1.32 | 1.58 | 1.86 | 1.94 | 2.19 |
| 0mm_r0 | frac cubes >2.5mm | 0.002 | 0.014 | 0.033 | 0.053 | 0.078 | 0.090 | 0.104 | 0.129 |
| 0mm_r1 | acc w64_native | 0.791 | 0.778 | 0.780 | 0.709 | 0.672 | 0.636 | 0.605 | 0.618 |
| 0mm_r1 | acc paste_zoom64 | 0.655 | 0.648 | 0.633 | 0.606 | 0.615 | 0.589 | 0.571 | 0.560 |
| 0mm_r1 | cube mm_rms | 0.28 | 0.62 | 1.04 | 1.41 | 1.69 | 1.88 | 2.10 | 2.34 |
| 0mm_r1 | frac cubes >2.5mm | 0.002 | 0.014 | 0.034 | 0.061 | 0.083 | 0.098 | 0.119 | 0.135 |
| 0.01mm_r0 | acc w64_native | 0.800 | 0.800 | 0.755 | 0.668 | 0.664 | 0.627 | 0.577 | 0.598 |
| 0.01mm_r0 | acc paste_zoom64 | 0.654 | 0.649 | 0.633 | 0.596 | 0.614 | 0.591 | 0.562 | 0.562 |
| 0.01mm_r0 | cube mm_rms | 0.42 | 0.69 | 1.05 | 1.34 | 1.69 | 2.08 | 2.27 | 2.57 |
| 0.01mm_r0 | frac cubes >2.5mm | 0.004 | 0.013 | 0.029 | 0.054 | 0.077 | 0.100 | 0.122 | 0.144 |
| 0.05mm_r0 | acc w64_native | 0.754 | 0.772 | 0.714 | 0.648 | 0.629 | 0.629 | 0.562 | 0.601 |
| 0.05mm_r0 | acc paste_zoom64 | 0.651 | 0.653 | 0.629 | 0.588 | 0.610 | 0.601 | 0.568 | 0.564 |
| 0.05mm_r0 | cube mm_rms | 0.86 | 1.02 | 1.35 | 1.61 | 1.82 | 2.15 | 2.40 | 2.64 |
| 0.05mm_r0 | frac cubes >2.5mm | 0.007 | 0.017 | 0.039 | 0.064 | 0.079 | 0.107 | 0.121 | 0.146 |
| 0.25mm_r0 | acc w64_native | 0.667 | 0.653 | 0.648 | 0.574 | 0.565 | 0.540 | 0.487 | 0.516 |
| 0.25mm_r0 | acc paste_zoom64 | 0.649 | 0.636 | 0.619 | 0.581 | 0.587 | 0.577 | 0.532 | 0.535 |
| 0.25mm_r0 | cube mm_rms | 0.78 | 1.03 | 1.50 | 1.81 | 2.21 | 2.48 | 2.76 | 3.08 |
| 0.25mm_r0 | frac cubes >2.5mm | 0.015 | 0.032 | 0.062 | 0.092 | 0.119 | 0.144 | 0.157 | 0.181 |
| 0.5mm_r0 | acc w64_native | 0.565 | 0.558 | 0.564 | 0.511 | 0.487 | 0.498 | 0.457 | 0.475 |
| 0.5mm_r0 | acc paste_zoom64 | 0.621 | 0.624 | 0.602 | 0.562 | 0.572 | 0.564 | 0.537 | 0.525 |
| 0.5mm_r0 | cube mm_rms | 1.16 | 1.42 | 1.79 | 2.21 | 2.49 | 2.75 | 2.90 | 3.18 |
| 0.5mm_r0 | frac cubes >2.5mm | 0.031 | 0.054 | 0.081 | 0.113 | 0.145 | 0.163 | 0.184 | 0.200 |
| 1mm_r0 | acc w64_native | 0.416 | 0.444 | 0.462 | 0.406 | 0.397 | 0.395 | 0.373 | 0.422 |
| 1mm_r0 | acc paste_zoom64 | 0.588 | 0.574 | 0.566 | 0.525 | 0.544 | 0.520 | 0.513 | 0.515 |
| 1mm_r0 | cube mm_rms | 1.75 | 1.99 | 2.39 | 2.80 | 2.94 | 3.23 | 3.47 | 3.76 |
| 1mm_r0 | frac cubes >2.5mm | 0.089 | 0.108 | 0.136 | 0.175 | 0.197 | 0.223 | 0.242 | 0.258 |

## T5 models vs ceiling (accuracy_1 on DS-0016 chains, same frame; ceiling = mean of two 0 mm re-sims; slateN in pasted frame; window slateN not given)
| model | frame | model acc | ceiling@0mm | true-label cap | model/ceiling | equivalent xy noise (mm, interpolated) | model slateN | slateN ceiling@0mm same frame | true-label slateN cap |
|---|---|---|---|---|---|---|---|---|---|
| nfd64 | w64_native | 0.559 | 0.805 | 1.000 | 0.69 | 0.57 | 0.710 | 0.889 | 0.896 |
| world128 (300ep) | window64 | 0.807 | 0.898 | 1.000 | 0.90 | 0.54 | 0.728 | nan | nan |
| world128 (300ep) | paste_world128 | 0.591 | 0.622 | 0.633 | 0.95 | 0.30 | 0.728 | nan | nan |
| zoom64 ft300 | window64 | 0.846 | 0.898 | 1.000 | 0.94 | 0.22 | 0.765 | nan | nan |
| zoom64 ft300 | paste_zoom64 | 0.622 | 0.641 | 0.651 | 0.97 | 0.26 | 0.765 | 0.786 | 0.778 |
| zoom128 | window128 | 0.859 | 0.900 | 1.000 | 0.95 | 0.16 | 0.809 | nan | nan |
| zoom128 | paste_zoom128 | 0.640 | 0.670 | 0.681 | 0.96 | 0.35 | 0.809 | 0.825 | 0.824 |

## T6 metric vs slateN across models (model-level Kendall tau / Spearman rho; boot = tau mean+-sd over 200 pool-bootstraps; per-pool = mean over models of within-model Spearman across the 32 pools; xm = pool-demeaned across-model Spearman)

**core7** (n=7 models)

| metric | pools tau | rho | boot tau | chains tau | within-model per-pool rho | within-pool across-model rho |
|---|---|---|---|---|---|---|
| acc_hard | +0.90 | +0.96 | +0.74+-0.20 | +0.90 | +0.66 | +0.29 |
| acc_blur1 | +0.90 | +0.96 | +0.70+-0.20 | +0.90 | +0.72 | +0.24 |
| acc_blur2 | +0.90 | +0.96 | +0.70+-0.20 | +0.90 | +0.77 | +0.15 |
| acc_blur4 | +0.90 | +0.96 | +0.63+-0.22 | +0.43 | +0.76 | +0.16 |
| acc_hard_fullframe | +0.43 | +0.43 | +0.35+-0.19 | +0.71 | -0.67 | +0.18 |
| changed_cell_IoU | +1.00 | +1.00 | +0.75+-0.20 | +0.90 | +0.60 | +0.28 |
| mass_in_region | +0.90 | +0.96 | +0.70+-0.20 | +0.90 | +0.75 | +0.21 |
| region_centroid_mm | +0.81 | +0.93 | +0.65+-0.20 | +0.71 | +0.26 | +0.23 |
| sliced_EMD_region | +0.90 | +0.96 | +0.70+-0.21 | +0.90 | +0.75 | +0.22 |
| dv_err_allgoals | -0.24 | -0.32 | -0.12+-0.20 | -0.33 | +0.84 | +0.03 |
| in_goal_mass_err | +0.81 | +0.89 | +0.44+-0.26 | +0.43 | +0.68 | +0.05 |
| abs_dv_bias(optimism) | -0.24 | -0.32 | -0.20+-0.18 | -0.33 | +0.80 | -0.05 |

**all15** (n=15 models)

| metric | pools tau | rho | boot tau | chains tau | within-model per-pool rho | within-pool across-model rho |
|---|---|---|---|---|---|---|
| acc_hard | +0.79 | +0.91 | +0.72+-0.08 | +0.81 | +0.65 | +0.41 |
| acc_blur1 | +0.70 | +0.84 | +0.64+-0.08 | +0.77 | +0.72 | +0.41 |
| acc_blur2 | +0.64 | +0.80 | +0.58+-0.07 | +0.68 | +0.76 | +0.33 |
| acc_blur4 | +0.35 | +0.44 | +0.31+-0.14 | +0.07 | +0.77 | +0.17 |
| acc_hard_fullframe | +0.58 | +0.72 | +0.59+-0.09 | +0.79 | -0.14 | +0.37 |
| changed_cell_IoU | +0.58 | +0.71 | +0.53+-0.09 | +0.58 | +0.57 | +0.35 |
| mass_in_region | +0.45 | +0.49 | +0.40+-0.08 | +0.39 | +0.73 | +0.20 |
| region_centroid_mm | +0.62 | +0.71 | +0.55+-0.08 | +0.60 | +0.13 | +0.31 |
| sliced_EMD_region | +0.45 | +0.49 | +0.41+-0.08 | +0.43 | +0.74 | +0.22 |
| dv_err_allgoals | +0.35 | +0.52 | +0.31+-0.13 | +0.12 | +0.82 | +0.19 |
| in_goal_mass_err | +0.87 | +0.96 | +0.72+-0.08 | +0.79 | +0.67 | +0.32 |
| abs_dv_bias(optimism) | +0.09 | +0.16 | +0.10+-0.09 | +0.07 | +0.77 | +0.02 |

## T7 ceiling in every metric (perfect-physics re-sim, pool rows; columns frame@level)
| metric | model range (15 models) | w64_n@0 | paste@0 | w64_n@0.25 | paste@0.25 | w64_n@0.5 | paste@0.5 | w64_n@1 | paste@1 |
|---|---|---|---|---|---|---|---|---|---|
| acc_hard | +0.43..+0.66 | +0.81 | +0.66 | +0.71 | +0.65 | +0.63 | +0.63 | +0.49 | +0.59 |
| acc_blur1 | +0.47..+0.80 | +0.90 | +0.81 | +0.85 | +0.80 | +0.81 | +0.78 | +0.71 | +0.72 |
| acc_blur2 | +0.47..+0.82 | +0.93 | +0.80 | +0.89 | +0.80 | +0.86 | +0.79 | +0.81 | +0.75 |
| acc_blur4 | +0.32..+0.74 | +0.92 | +0.70 | +0.88 | +0.70 | +0.86 | +0.69 | +0.82 | +0.66 |
| acc_hard_fullframe | -0.04..+0.36 | +0.63 | +0.37 | +0.32 | +0.34 | +0.14 | +0.32 | -0.15 | +0.22 |
| changed_cell_IoU | +0.68..+0.86 | +0.96 | +0.86 | +0.92 | +0.85 | +0.88 | +0.84 | +0.79 | +0.81 |
| mass_in_region | +0.33..+0.85 | +0.97 | +0.84 | +0.94 | +0.83 | +0.93 | +0.83 | +0.90 | +0.81 |
| region_centroid_mm | -0.06..+0.41 | +0.85 | +0.42 | +0.74 | +0.41 | +0.65 | +0.40 | +0.48 | +0.36 |
| sliced_EMD_region | +0.33..+0.85 | +0.97 | +0.84 | +0.94 | +0.83 | +0.92 | +0.83 | +0.89 | +0.81 |
| dv_err_allgoals | +0.02..+0.54 | +0.84 | +0.47 | +0.71 | +0.42 | +0.67 | +0.37 | +0.62 | +0.33 |
| in_goal_mass_err | +0.23..+0.65 | +0.78 | +0.63 | +0.71 | +0.61 | +0.66 | +0.59 | +0.60 | +0.54 |
| abs_dv_bias(optimism) | -2.67..-0.25 | +0.52 | -0.91 | +0.16 | -1.17 | +0.02 | -1.43 | -0.08 | -1.62 |

## T8 models (pool rows, pasted world-64 frame): slateN and metrics
| model | slateN | acc_hard | acc_blur1 | acc_blur2 | acc_blur4 | acc_hard_fullframe | changed_cell_IoU | mass_in_region | region_centroid_mm | sliced_EMD_region | dv_err_allgoals | in_goal_mass_err | abs_dv_bias(optimism) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nfd64 | 0.710 | +0.59 | +0.66 | +0.69 | +0.66 | +0.34 | +0.78 | +0.75 | +0.25 | +0.74 | +0.54 | +0.57 | -0.25 |
| scratch100 | 0.761 | +0.64 | +0.78 | +0.78 | +0.67 | +0.32 | +0.84 | +0.79 | +0.31 | +0.79 | +0.35 | +0.58 | -1.47 |
| ft100 | 0.746 | +0.64 | +0.78 | +0.79 | +0.69 | +0.33 | +0.84 | +0.83 | +0.40 | +0.83 | +0.44 | +0.60 | -1.04 |
| ft300 | 0.765 | +0.65 | +0.79 | +0.80 | +0.70 | +0.33 | +0.84 | +0.84 | +0.41 | +0.84 | +0.45 | +0.60 | -1.02 |
| zoom128 | 0.809 | +0.66 | +0.80 | +0.82 | +0.74 | +0.36 | +0.86 | +0.84 | +0.41 | +0.84 | +0.53 | +0.65 | -0.64 |
| world128 | 0.726 | +0.61 | +0.74 | +0.76 | +0.66 | +0.31 | +0.81 | +0.78 | +0.23 | +0.78 | +0.46 | +0.58 | -0.82 |
| world128_300 | 0.728 | +0.61 | +0.75 | +0.77 | +0.67 | +0.32 | +0.81 | +0.78 | +0.29 | +0.78 | +0.48 | +0.59 | -0.79 |
| nfd_3ch_narrow_l20_v2_seed1 | 0.711 | +0.59 | +0.67 | +0.71 | +0.68 | +0.33 | +0.79 | +0.79 | +0.14 | +0.78 | +0.53 | +0.58 | -0.56 |
| nfd_3ch_narrow_l20_v2_seed2 | 0.695 | +0.60 | +0.68 | +0.71 | +0.67 | +0.33 | +0.80 | +0.72 | +0.07 | +0.72 | +0.26 | +0.53 | -1.56 |
| nfd_3ch_narrow_l20_v2_sharp_w03 | 0.680 | +0.60 | +0.68 | +0.71 | +0.69 | +0.30 | +0.80 | +0.85 | +0.36 | +0.84 | +0.44 | +0.47 | -0.72 |
| nfd_3ch_narrow_l20_v2_sharp_w07 | 0.670 | +0.57 | +0.68 | +0.72 | +0.69 | +0.26 | +0.81 | +0.85 | +0.32 | +0.85 | +0.43 | +0.45 | -0.79 |
| nfd_3ch_narrow_l20_v2_soft_s1 | 0.700 | +0.53 | +0.61 | +0.62 | +0.50 | +0.18 | +0.78 | +0.60 | +0.21 | +0.60 | +0.43 | +0.50 | -0.43 |
| nfd_3ch_narrow_l20_v2_soft_s2 | 0.587 | +0.43 | +0.47 | +0.47 | +0.32 | -0.04 | +0.68 | +0.33 | -0.06 | +0.33 | +0.05 | +0.23 | -2.12 |
| linear_narrow_l20_v2_res64 | 0.599 | +0.55 | +0.66 | +0.69 | +0.62 | +0.17 | +0.82 | +0.66 | +0.02 | +0.66 | +0.02 | +0.37 | -2.67 |
| linear_narrow_l20_v2_res32 | 0.641 | +0.52 | +0.67 | +0.74 | +0.73 | +0.04 | +0.75 | +0.77 | +0.19 | +0.76 | +0.48 | +0.50 | -0.60 |