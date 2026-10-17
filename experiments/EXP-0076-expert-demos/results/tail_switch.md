# Tail-switch experiment (96 tail starts complete in every run; runs present: [('ens', 0), ('ens', 1), ('v64', 0), ('v64', 1)])

## bulk phase by ens (64 tail starts; entry mass 0.847)

| tail model, seed | solved | time to 0.95 on solved | censored time (unsolved = all 8 pushes) |
|---|---|---|---|
| ens s0 | 45/64 | 2.90 | 4.23 |
| ens s1 | 45/64 | 2.94 | 4.22 |
| v64 s0 | 42/64 | 3.48 | 5.00 |
| v64 s1 | 39/64 | 3.14 | 4.97 |

paired differences (b - a) over tail starts: solved-rate difference, censored time, time on starts both solve [95 % bootstrap CI]
- noise floor ens: solved +0.000 [-0.094, +0.078]; censored time -0.01 s [-0.32, +0.34]; time on both-solved (n 41) -0.05 s [-0.24, +0.12]
- noise floor v64: solved -0.047 [-0.125, +0.031]; censored time -0.04 s [-0.44, +0.34]; time on both-solved (n 37) -0.29 s [-0.68, +0.03]
- v64 - ens seed 0: solved -0.047 [-0.156, +0.062]; censored time +0.77 s [+0.22, +1.33]; time on both-solved (n 37) +0.32 s [-0.10, +0.77]
- v64 - ens seed 1: solved -0.094 [-0.172, -0.016]; censored time +0.75 s [+0.37, +1.14]; time on both-solved (n 38) +0.27 s [-0.05, +0.61]
- seed-averaged v64 - ens: solved -0.070 [-0.148, +0.000]; censored time +0.76 s [+0.36, +1.17]

## bulk phase by v64 (32 tail starts; entry mass 0.850)

| tail model, seed | solved | time to 0.95 on solved | censored time (unsolved = all 8 pushes) |
|---|---|---|---|
| ens s0 | 22/32 | 2.55 | 4.07 |
| ens s1 | 26/32 | 2.98 | 3.83 |
| v64 s0 | 23/32 | 2.73 | 4.26 |
| v64 s1 | 23/32 | 2.86 | 4.25 |

paired differences (b - a) over tail starts: solved-rate difference, censored time, time on starts both solve [95 % bootstrap CI]
- noise floor ens: solved +0.125 [+0.031, +0.250]; censored time -0.25 s [-0.54, +0.00]; time on both-solved (n 22) -0.02 s [-0.17, +0.12]
- noise floor v64: solved +0.000 [+0.000, +0.000]; censored time -0.02 s [-0.19, +0.16]; time on both-solved (n 23) +0.13 s [+0.00, +0.29]
- v64 - ens seed 0: solved +0.031 [-0.062, +0.126]; censored time +0.19 s [-0.36, +0.74]; time on both-solved (n 21) +0.04 s [-0.21, +0.28]
- v64 - ens seed 1: solved -0.094 [-0.219, +0.031]; censored time +0.42 s [-0.12, +0.99]; time on both-solved (n 22) +0.03 s [-0.35, +0.35]
- seed-averaged v64 - ens: solved -0.031 [-0.141, +0.078]; censored time +0.31 s [-0.25, +0.86]

## bulk phase by all (96 tail starts; entry mass 0.848)

| tail model, seed | solved | time to 0.95 on solved | censored time (unsolved = all 8 pushes) |
|---|---|---|---|
| ens s0 | 67/96 | 2.79 | 4.18 |
| ens s1 | 71/96 | 2.96 | 4.09 |
| v64 s0 | 65/96 | 3.21 | 4.76 |
| v64 s1 | 62/96 | 3.04 | 4.73 |

paired differences (b - a) over tail starts: solved-rate difference, censored time, time on starts both solve [95 % bootstrap CI]
- noise floor ens: solved +0.042 [-0.031, +0.115]; censored time -0.09 s [-0.34, +0.16]; time on both-solved (n 63) -0.04 s [-0.17, +0.08]
- noise floor v64: solved -0.031 [-0.083, +0.021]; censored time -0.03 s [-0.30, +0.24]; time on both-solved (n 60) -0.13 s [-0.37, +0.09]
- v64 - ens seed 0: solved -0.021 [-0.104, +0.062]; censored time +0.58 s [+0.17, +1.00]; time on both-solved (n 58) +0.22 s [-0.07, +0.54]
- v64 - ens seed 1: solved -0.094 [-0.167, -0.021]; censored time +0.64 s [+0.32, +0.96]; time on both-solved (n 60) +0.18 s [-0.06, +0.43]
- seed-averaged v64 - ens: solved -0.057 [-0.115, +0.000]; censored time +0.61 s [+0.28, +0.94]

