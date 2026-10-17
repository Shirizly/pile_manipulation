# Total time (physical + planning) comparisons, v64 - ens (positive = v64 slower)

## Bulk / whole episode, 270 paired n20 scenarios (ens f20_all median plan 1.21 s / decision, v64 F1_v64_all 0.38 s)

| target | time | v64 - ens censored [95 % CI] | mean ens | mean v64 |
|---|---|---|---|---|
| mass 0.8 | physical only | +0.58 [+0.40, +0.77] s | 5.03 | 5.61 |
| mass 0.8 | physical + recorded planning | -3.27 [-3.67, -2.87] s | 10.55 | 7.28 |
| mass 0.8 | physical + median planning | -2.64 [-2.99, -2.31] s | 9.91 | 7.28 |
| mass 0.95 | physical only | +1.10 [+0.80, +1.39] s | 8.65 | 9.75 |
| mass 0.95 | physical + recorded planning | -5.89 [-6.57, -5.19] s | 18.78 | 12.89 |
| mass 0.95 | physical + median planning | -5.03 [-5.65, -4.43] s | 17.90 | 12.88 |

## Tail switch: 96 recorded 0.8-mass states, each finished by ens and by v64 (2 seeds, <= 8 pushes; censored = all 8 pushes)
median plan s / decision: ens s0 1.29, ens s1 1.23, v64 s0 0.38, v64 s1 0.38

| bulk states from | time | v64 - ens censored, seed-averaged [95 % CI] | mean ens | mean v64 | ens seed floor |
|---|---|---|---|---|---|
| ens (n 64) | physical only | +0.76 [+0.37, +1.18] s | 4.22 | 4.99 | -0.01 [-0.35, +0.34] |
| ens (n 64) | physical + recorded planning | -2.93 [-3.80, -2.00] s | 9.75 | 6.82 | -0.53 [-1.34, +0.32] |
| ens (n 64) | physical + median planning | -2.81 [-3.70, -1.91] s | 9.60 | 6.78 | -0.24 [-1.05, +0.58] |
| v64 (n 32) | physical only | +0.31 [-0.22, +0.85] s | 3.95 | 4.26 | -0.25 [-0.54, -0.00] |
| v64 (n 32) | physical + recorded planning | -3.13 [-4.33, -1.98] s | 8.88 | 5.75 | -0.96 [-1.73, -0.32] |
| v64 (n 32) | physical + median planning | -3.07 [-4.23, -1.98] s | 8.82 | 5.74 | -0.84 [-1.62, -0.22] |
| all (n 96) | physical only | +0.61 [+0.29, +0.93] s | 4.13 | 4.74 | -0.09 [-0.33, +0.15] |
| all (n 96) | physical + recorded planning | -3.00 [-3.67, -2.28] s | 9.46 | 6.47 | -0.67 [-1.27, -0.07] |
| all (n 96) | physical + median planning | -2.90 [-3.60, -2.18] s | 9.34 | 6.44 | -0.44 [-1.06, +0.17] |

## Composite pipelines (approximate): mean total time = bulk part (270 scenarios, start -> 0.8, physical + median planning)
+ tail part (tail-switch, states from that bulk model, physical + median planning, censored). Tail starts from v64 bulk come from 36 scenarios only.

| bulk model | tail model | bulk part s | tail part s | total s |
|---|---|---|---|---|
| ens | ens | 9.91 | 9.60 | 19.51 |
| ens | v64 | 9.91 | 6.78 | 16.70 |
| v64 | ens | 7.28 | 8.82 | 16.09 |
| v64 | v64 | 7.28 | 5.74 | 13.02 |

## Censoring check: pairs where both models reach the target (total = physical + median planning)

| comparison | n both | v64 - ens physical | v64 - ens total | solved ens / v64 |
|---|---|---|---|---|
| bulk/episode to 0.8 (270) | 254 | +0.44 [+0.29, +0.59] | -2.56 [-2.85, -2.26] | 266 / 255 |
| bulk/episode to 0.95 (270) | 152 | +0.86 [+0.60, +1.11] | -3.46 [-3.99, -2.95] | 196 / 166 |
| tail seed 0 (96) | 58 | +0.22 [-0.07, +0.52] | -2.02 [-2.66, -1.43] | 67 / 65 |
| tail seed 1 (96) | 60 | +0.18 [-0.06, +0.43] | -1.94 [-2.52, -1.40] | 71 / 62 |
