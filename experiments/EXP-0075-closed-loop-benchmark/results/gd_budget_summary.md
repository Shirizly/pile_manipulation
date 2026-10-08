# gd_budget: 10 tasks, H=4, predicted terminal value change (lower = better)

| task | ref | gd_ref | gd5 | gd10 |
|---|---|---|---|---|
| O/40 | -0.673 | -0.969 | -0.903 | -0.895 |
| O/41 | -0.477 | -0.669 | -0.657 | -0.701 |
| O/45 | -0.452 | -0.611 | -0.606 | -0.629 |
| S/42 | -0.576 | -0.830 | -0.859 | -0.826 |
| S/46 | +nan | -0.744 | -0.664 | -0.722 |
| T/40 | -0.697 | -0.991 | -0.915 | -0.952 |
| T/44 | -0.606 | -0.867 | -0.763 | -0.850 |
| X/41 | -0.450 | -0.685 | -0.674 | -0.639 |
| X/43 | -0.635 | -0.967 | -0.949 | -0.961 |
| X/47 | -0.701 | -0.855 | -0.841 | -0.851 |
| **mean** | **-0.585** | **-0.819** | **-0.783** | **-0.803** |
| sd/sqrt(n) | 0.034 | 0.043 | 0.040 | 0.039 |

NOTE: the benchmark CEM (ref) returned a NaN cost on S/46 (degenerate plan, true +0.244); it is excluded from predicted-value statistics of ref and kept in the simulator rows.

evaluations before GD: ref: 1,280, gd_ref: 1,280 + GD, gd5: 10,000 + GD, gd10: 20,000 + GD; GD = 150 Adam steps x 24 sequences (3,600 differentiable evaluations, ~26 s) in every gd_* column.

Paired differences (mean [min, max] over tasks; negative = first is better):
* gd_ref - ref: -0.242 [-0.332, -0.154]  (better in 9/9)
* gd5 - ref: -0.211 [-0.314, -0.140]  (better in 9/9)
* gd10 - ref: -0.226 [-0.326, -0.150]  (better in 9/9)
* gd5 - gd_ref: +0.036 [-0.029, +0.104]  (better in 1/10)
* gd10 - gd_ref: +0.016 [-0.032, +0.075]  (better in 2/10)
* gd10 - gd5: -0.020 [-0.088, +0.035]  (better in 7/10)

Fraction of the best plan found per task (value / min over the four plans): ref: 0.70, gd_ref: 0.99, gd5: 0.95, gd10: 0.97

## Simulator replay (open loop, pushes legalised; one run per plan)

| plan | predicted | simulator | optimism (pred - sim) | sd of optimism | shifted pushes /plan |
|---|---|---|---|---|---|
| ref | -0.585 | -0.439 | -0.070 | 0.061 | 0.70 |
| gd_ref | -0.819 | -0.697 | -0.122 | 0.175 | 0.90 |
| gd5 | -0.783 | -0.676 | -0.107 | 0.112 | 1.00 |
| gd10 | -0.803 | -0.667 | -0.136 | 0.085 | 0.80 |

Simulator paired differences (true terminal value change; negative = first is better):
* gd_ref - ref: -0.257 [-0.870, +0.165]  (better in 9/10)
* gd5 - ref: -0.236 [-0.851, +0.100]  (better in 9/10)
* gd10 - ref: -0.227 [-0.871, -0.030]  (better in 10/10)
* gd5 - gd_ref: +0.021 [-0.362, +0.216]  (better in 4/10)
* gd10 - gd_ref: +0.030 [-0.272, +0.246]  (better in 5/10)
* gd10 - gd5: +0.009 [-0.130, +0.090]  (better in 3/10)