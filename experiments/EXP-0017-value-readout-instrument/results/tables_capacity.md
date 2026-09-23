
### held-out GOAL

| value_fn | capacity | mode | R2 mean | spread (sd) | n splits | goal_only (same capacity) | increment |
|---|---|---|---|---|---|---|---|
| lyapunov | linear | diff | +0.420 | 0.051 | 4 | +0.327+-0.064 (n=4) | +0.093 |
| lyapunov | linear | concat | +0.451 | 0.061 | 4 | +0.327+-0.064 (n=4) | +0.124 |
| lyapunov | linear | state_only | +0.122 | 0.017 | 4 | +0.327+-0.064 (n=4) | -0.205 |
| lyapunov | linear | mean_only | +0.000 | 0.000 | 4 | +0.327+-0.064 (n=4) | -0.327 |
| lyapunov | linear | goal_only | +0.327 | 0.064 | 4 | — | 0 |
| lyapunov | poly2 | diff | +0.794 | 0.016 | 3 | +0.280+-0.000 (n=1) | +0.514 |
| lyapunov | poly2 | goal_only | +0.280 | 0.000 | 1 | — | 0 |
| lyapunov | mlp | diff | +0.871 | 0.010 | 3 | -3.010+-4.612 (n=3) | +3.881 |
| lyapunov | mlp | goal_only | -3.010 | 4.612 | 3 | — | 0 |
| mass_in_region | linear | diff | +0.285 | 0.051 | 4 | +0.364+-0.064 (n=4) | -0.080 |
| mass_in_region | linear | concat | +0.378 | 0.063 | 4 | +0.364+-0.064 (n=4) | +0.013 |
| mass_in_region | linear | state_only | +0.010 | 0.002 | 4 | +0.364+-0.064 (n=4) | -0.355 |
| mass_in_region | linear | mean_only | +0.000 | 0.000 | 4 | +0.364+-0.064 (n=4) | -0.364 |
| mass_in_region | linear | goal_only | +0.364 | 0.064 | 4 | — | 0 |
| mass_in_region | poly2 | diff | +0.585 | 0.061 | 3 | +0.332+-0.000 (n=1) | +0.254 |
| mass_in_region | poly2 | goal_only | +0.332 | 0.000 | 1 | — | 0 |
| mass_in_region | mlp | diff | +0.644 | 0.033 | 3 | +0.312+-0.080 (n=3) | +0.332 |
| mass_in_region | mlp | goal_only | +0.312 | 0.080 | 3 | — | 0 |
| signed_mass_in_region | linear | diff | +0.278 | 0.050 | 4 | +0.361+-0.063 (n=4) | -0.083 |
| signed_mass_in_region | linear | concat | +0.384 | 0.061 | 4 | +0.361+-0.063 (n=4) | +0.023 |
| signed_mass_in_region | linear | state_only | +0.020 | 0.004 | 4 | +0.361+-0.063 (n=4) | -0.341 |
| signed_mass_in_region | linear | mean_only | +0.000 | 0.000 | 4 | +0.361+-0.063 (n=4) | -0.361 |
| signed_mass_in_region | linear | goal_only | +0.361 | 0.063 | 4 | — | 0 |
| signed_mass_in_region | poly2 | diff | +0.597 | 0.055 | 3 | +0.326+-0.000 (n=1) | +0.271 |
| signed_mass_in_region | poly2 | goal_only | +0.326 | 0.000 | 1 | — | 0 |
| signed_mass_in_region | mlp | diff | +0.703 | 0.026 | 3 | +0.363+-0.057 (n=3) | +0.339 |
| signed_mass_in_region | mlp | goal_only | +0.363 | 0.057 | 3 | — | 0 |

### held-out STATE

| value_fn | capacity | mode | R2 mean | spread (sd) | n splits | goal_only (same capacity) | increment |
|---|---|---|---|---|---|---|---|
| lyapunov | linear | diff | +0.129 | 0.074 | 3 | +0.329+-0.042 (n=3) | -0.200 |
| lyapunov | linear | state_only | +0.105 | 0.021 | 3 | +0.329+-0.042 (n=3) | -0.225 |
| lyapunov | linear | mean_only | +0.000 | 0.000 | 3 | +0.329+-0.042 (n=3) | -0.329 |
| lyapunov | linear | goal_only | +0.329 | 0.042 | 3 | — | 0 |
| lyapunov | mlp | diff | +0.585 | 0.000 | 1 | +0.411+-0.000 (n=1) | +0.174 |
| lyapunov | mlp | goal_only | +0.411 | 0.000 | 1 | — | 0 |
| mass_in_region | linear | diff | -0.075 | 0.237 | 3 | +0.381+-0.021 (n=3) | -0.456 |
| mass_in_region | linear | state_only | +0.005 | 0.003 | 3 | +0.381+-0.021 (n=3) | -0.376 |
| mass_in_region | linear | mean_only | +0.000 | 0.000 | 3 | +0.381+-0.021 (n=3) | -0.381 |
| mass_in_region | linear | goal_only | +0.381 | 0.021 | 3 | — | 0 |
| mass_in_region | mlp | diff | +0.332 | 0.000 | 1 | +0.424+-0.000 (n=1) | -0.092 |
| mass_in_region | mlp | goal_only | +0.424 | 0.000 | 1 | — | 0 |
| signed_mass_in_region | linear | diff | -0.134 | 0.239 | 3 | +0.377+-0.021 (n=3) | -0.511 |
| signed_mass_in_region | linear | state_only | +0.009 | 0.011 | 3 | +0.377+-0.021 (n=3) | -0.368 |
| signed_mass_in_region | linear | mean_only | +0.000 | 0.000 | 3 | +0.377+-0.021 (n=3) | -0.377 |
| signed_mass_in_region | linear | goal_only | +0.377 | 0.021 | 3 | — | 0 |
| signed_mass_in_region | mlp | diff | +0.108 | 0.000 | 1 | +0.425+-0.000 (n=1) | -0.318 |
| signed_mass_in_region | mlp | goal_only | +0.425 | 0.000 | 1 | — | 0 |
