# EXP-0018 results — metric `value_readout_r2` everywhere

All numbers are `value_readout_r2` (METRICS.md). Baselines are quoted at their OWN
best capacity (max over the capacities run in that cell family). `incr` = 
`value_readout_r2` minus best-capacity `goal_only`. `+-` is the sd over split repeats;
for the corpus family it is the spread ACROSS held-out corpora, i.e. between-corpus
variation, not a noise floor.

## Aggregated by split family and capacity

| split | capacity | value fn | n | value_readout_r2 | goal_only* | incr over goal_only | state_only* |
|---|---|---|---|---|---|---|---|
| held-out goal | linear | lyapunov | 3 | +0.649+-0.016 | +0.138 | **+0.511+-0.021** | +0.558 |
| held-out goal | linear | mass_in_region | 3 | +0.447+-0.005 | +0.248 | **+0.199+-0.012** | +0.351 |
| held-out goal | linear | signed_mass_in_region | 3 | +0.618+-0.015 | +0.181 | **+0.437+-0.023** | +0.553 |
| held-out goal | poly2 | lyapunov | 1 | +0.697+-0.000 | +0.147 | **+0.550+-0.000** | +0.543 |
| held-out goal | mlp | lyapunov | 3 | +0.776+-0.021 | +0.138 | **+0.637+-0.019** | +0.558 |
| held-out goal | mlp | mass_in_region | 3 | +0.671+-0.002 | +0.248 | **+0.422+-0.011** | +0.351 |
| held-out goal | mlp | signed_mass_in_region | 3 | +0.779+-0.006 | +0.181 | **+0.598+-0.014** | +0.553 |
| held-out state | linear | lyapunov | 2 | +0.642+-0.001 | +0.132 | **+0.510+-0.000** | +0.552 |
| held-out state | linear | mass_in_region | 2 | +0.438+-0.003 | +0.253 | **+0.185+-0.011** | +0.344 |
| held-out state | linear | signed_mass_in_region | 2 | +0.617+-0.003 | +0.182 | **+0.435+-0.003** | +0.555 |
| held-out state | mlp | lyapunov | 2 | +0.693+-0.040 | +0.132 | **+0.562+-0.041** | +0.552 |
| held-out state | mlp | mass_in_region | 2 | +0.564+-0.011 | +0.253 | **+0.311+-0.003** | +0.344 |
| held-out state | mlp | signed_mass_in_region | 2 | +0.746+-0.003 | +0.182 | **+0.564+-0.003** | +0.555 |
| held-out corpus | linear | lyapunov | 5 | +0.653+-0.195 | +0.131 | **+0.522+-0.220** | +0.550 |
| held-out corpus | linear | mass_in_region | 5 | +0.290+-0.182 | +0.169 | **+0.121+-0.095** | +0.419 |
| held-out corpus | linear | signed_mass_in_region | 5 | +0.443+-0.394 | +0.114 | **+0.329+-0.408** | +0.590 |
| held-out corpus | mlp | lyapunov | 5 | +0.573+-0.245 | +0.131 | **+0.441+-0.299** | +0.550 |

## Held-out SOURCE CORPUS, per held-out corpus

| held-out corpus | capacity | value fn | value_readout_r2 | goal_only* | incr over goal_only |
|---|---|---|---|---|---|
| randlen | linear | lyapunov | +0.833 | +0.128 | +0.706 |
| randlen | linear | mass_in_region | +0.083 | -0.018 | +0.100 |
| randlen | linear | signed_mass_in_region | -0.249 | +0.028 | -0.277 |
| randlen | mlp | lyapunov | +0.771 | +0.128 | +0.643 |
| sean_inbetween | linear | lyapunov | +0.545 | +0.240 | +0.305 |
| sean_inbetween | linear | mass_in_region | +0.488 | +0.220 | +0.268 |
| sean_inbetween | linear | signed_mass_in_region | +0.585 | +0.175 | +0.410 |
| sean_inbetween | mlp | lyapunov | +0.192 | +0.240 | -0.048 |
| sean_piled | linear | lyapunov | +0.324 | +0.119 | +0.205 |
| sean_piled | linear | mass_in_region | +0.406 | +0.231 | +0.176 |
| sean_piled | linear | signed_mass_in_region | +0.289 | +0.279 | +0.010 |
| sean_piled | mlp | lyapunov | +0.380 | +0.119 | +0.261 |
| sean_scattered | linear | lyapunov | +0.824 | +0.133 | +0.691 |
| sean_scattered | linear | mass_in_region | +0.415 | +0.425 | -0.011 |
| sean_scattered | linear | signed_mass_in_region | +0.856 | +0.100 | +0.757 |
| sean_scattered | mlp | lyapunov | +0.697 | +0.133 | +0.563 |
| slates_multistep | linear | lyapunov | +0.738 | +0.037 | +0.702 |
| slates_multistep | linear | mass_in_region | +0.058 | -0.012 | +0.069 |
| slates_multistep | linear | signed_mass_in_region | +0.734 | -0.010 | +0.744 |
| slates_multistep | mlp | lyapunov | +0.825 | +0.037 | +0.788 |

## Selected ridge lambda (grouped inner CV)

linear/poly2 cells: min=61.6 max=2976.4, n=31. Every linear/poly2 cell
used `GroupKFold` inner folds (`cv_group_by='state'` for the state/corpus families,
`'goal'` for the goal family), so near-duplicate states cannot straddle a fold boundary.

## mean_only baseline

exactly 0.0e+00 in the worst cell — confirms R2 is normalised against the TRAINING mean.
