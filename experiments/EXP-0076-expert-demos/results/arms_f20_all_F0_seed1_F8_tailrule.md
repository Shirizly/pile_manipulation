# arms over 36 common scenarios

| arm | solved | pushes (solved) mean | time s (solved) mean | final mass mean | sims / scenario | wall s / scenario | plan s / decision |
|---|---|---|---|---|---|---|---|
| f20_all | 27/36 | 5.93 | 7.12 | 0.962 | 7 | 788 | 1.34 |
| F0_seed1 | 28/36 | 6.18 | 7.27 | 0.941 | 7 | 332 | 1.22 |
| F8_tailrule | 24/36 | 5.50 | 6.49 | 0.943 | 8 | 343 | 1.37 |

paired on scenarios solved by both (b - a): pushes, time s
- F0_seed1 - f20_all: n 26, pushes +0.23, time +0.11 s [-0.34, +0.58], only-a solved 1, only-b solved 2
- F8_tailrule - f20_all: n 23, pushes +0.04, time -0.04 s [-0.51, +0.41], only-a solved 4, only-b solved 1
- F8_tailrule - F0_seed1: n 24, pushes -0.25, time -0.18 s [-0.66, +0.29], only-a solved 4, only-b solved 0

| scenario | f20_all | F0_seed1 | F8_tailrule |
|---|---|---|---|
| n20-inbetween-disc-002 (rho 0.49) | 10p 11.4s | 9p 11.0s | x 0.92 |
| n20-inbetween-disc-003 (rho 0.42) | 6p 7.1s | 7p 7.9s | 7p 7.2s |
| n20-inbetween-disc-004 (rho 0.37) | 8p 9.4s | 8p 9.4s | 7p 8.0s |
| n20-inbetween-disc-005 (rho 0.24) | 4p 5.0s | 4p 5.0s | 5p 6.0s |
| n20-inbetween-letter-002 (rho 0.22) | 6p 7.1s | 9p 9.8s | 9p 10.6s |
| n20-inbetween-letter-003 (rho 0.20) | x 0.91 | x 0.89 | x 0.93 |
| n20-inbetween-letter-004 (rho 0.14) | x 0.93 | x 0.95 | x 0.93 |
| n20-inbetween-letter-005 (rho 0.22) | 6p 7.8s | 5p 6.7s | 4p 5.7s |
| n20-inbetween-square-002 (rho 0.49) | 8p 9.2s | 9p 9.4s | 6p 6.8s |
| n20-inbetween-square-003 (rho 0.47) | x 0.93 | 10p 11.4s | 11p 12.3s |
| n20-inbetween-square-004 (rho 0.25) | x 0.95 | x 0.95 | x 0.95 |
| n20-inbetween-square-005 (rho 0.32) | 11p 12.2s | 10p 11.1s | x 0.94 |
| n20-piled-disc-002 (rho 0.50) | 3p 2.7s | 4p 3.7s | 3p 2.8s |
| n20-piled-disc-003 (rho 0.17) | 1p 0.8s | 1p 0.8s | 1p 0.8s |
| n20-piled-disc-004 (rho 0.43) | 3p 2.8s | 4p 3.7s | 3p 2.8s |
| n20-piled-disc-005 (rho 0.28) | 1p 1.3s | 2p 2.1s | 1p 1.3s |
| n20-piled-letter-002 (rho 0.14) | 2p 2.1s | 2p 2.1s | 2p 2.1s |
| n20-piled-letter-003 (rho 0.17) | 2p 1.7s | 2p 1.6s | 2p 1.6s |
| n20-piled-letter-004 (rho 0.22) | 3p 3.1s | 4p 4.3s | 3p 3.3s |
| n20-piled-letter-005 (rho 0.20) | x 0.94 | x 0.94 | x 0.93 |
| n20-piled-square-002 (rho 0.43) | 3p 3.5s | 3p 3.7s | 3p 3.5s |
| n20-piled-square-003 (rho 0.17) | 4p 4.4s | 4p 4.1s | 5p 5.2s |
| n20-piled-square-004 (rho 0.48) | 5p 4.7s | 2p 1.8s | 3p 3.0s |
| n20-piled-square-005 (rho 0.56) | x 0.95 | x 0.44 | x 0.82 |
| n20-scattered-disc-002 (rho 0.54) | 8p 10.6s | 12p 13.3s | 9p 10.8s |
| n20-scattered-disc-003 (rho 0.47) | x 0.87 | x 0.89 | x 0.82 |
| n20-scattered-disc-004 (rho 0.33) | x 0.94 | x 0.69 | x 0.64 |
| n20-scattered-disc-005 (rho 0.32) | 7p 9.2s | 7p 8.8s | 7p 8.9s |
| n20-scattered-letter-002 (rho 0.22) | 9p 11.8s | 8p 10.0s | 10p 12.8s |
| n20-scattered-letter-003 (rho 0.14) | 6p 7.6s | 6p 8.3s | 6p 7.6s |
| n20-scattered-letter-004 (rho 0.20) | 8p 10.7s | 8p 10.7s | x 0.94 |
| n20-scattered-letter-005 (rho 0.22) | x 0.94 | 8p 10.8s | x 0.93 |
| n20-scattered-square-002 (rho 0.17) | 7p 9.5s | 7p 9.1s | 7p 9.4s |
| n20-scattered-square-003 (rho 0.35) | 9p 11.4s | 9p 11.2s | 10p 12.3s |
| n20-scattered-square-004 (rho 0.20) | 9p 11.5s | 9p 11.7s | 8p 11.0s |
| n20-scattered-square-005 (rho 0.45) | 11p 13.6s | x 0.90 | x 0.89 |
