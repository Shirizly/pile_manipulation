# arms over 36 common scenarios

| arm | solved | pushes (solved) mean | time s (solved) mean | final mass mean | sims / scenario | wall s / scenario | plan s / decision |
|---|---|---|---|---|---|---|---|
| f20_all | 27/36 | 5.93 | 7.12 | 0.962 | 7 | 788 | 1.34 |
| F1_v64big | 24/36 | 5.79 | 7.29 | 0.932 | 8 | 179 | 0.38 |
| F2_ensH2ot | 26/36 | 5.62 | 6.60 | 0.943 | 7 | 544 | 2.39 |

paired on scenarios solved by both (b - a): pushes, time s
- F1_v64big - f20_all: n 23, pushes +0.00, time +0.35 s [-0.20, +0.85], only-a solved 4, only-b solved 1
- F2_ensH2ot - f20_all: n 22, pushes -0.05, time -0.10 s [-0.93, +0.83], only-a solved 5, only-b solved 4
- F2_ensH2ot - F1_v64big: n 22, pushes -0.18, time -0.66 s [-1.41, +0.21], only-a solved 2, only-b solved 4

| scenario | f20_all | F1_v64big | F2_ensH2ot |
|---|---|---|---|
| n20-inbetween-disc-002 (rho 0.49) | 10p 11.4s | 9p 11.6s | 10p 11.7s |
| n20-inbetween-disc-003 (rho 0.42) | 6p 7.1s | 6p 7.5s | 11p 11.8s |
| n20-inbetween-disc-004 (rho 0.37) | 8p 9.4s | 5p 6.1s | 5p 6.1s |
| n20-inbetween-disc-005 (rho 0.24) | 4p 5.0s | 4p 5.5s | 4p 5.1s |
| n20-inbetween-letter-002 (rho 0.22) | 6p 7.1s | 7p 8.6s | 11p 12.6s |
| n20-inbetween-letter-003 (rho 0.20) | x 0.91 | x 0.93 | x 0.93 |
| n20-inbetween-letter-004 (rho 0.14) | x 0.93 | x 0.94 | x 0.94 |
| n20-inbetween-letter-005 (rho 0.22) | 6p 7.8s | 8p 8.7s | x 0.94 |
| n20-inbetween-square-002 (rho 0.49) | 8p 9.2s | x 0.92 | x 0.89 |
| n20-inbetween-square-003 (rho 0.47) | x 0.93 | x 0.90 | 12p 13.9s |
| n20-inbetween-square-004 (rho 0.25) | x 0.95 | x 0.95 | 7p 8.2s |
| n20-inbetween-square-005 (rho 0.32) | 11p 12.2s | 8p 9.9s | 8p 9.2s |
| n20-piled-disc-002 (rho 0.50) | 3p 2.7s | 3p 2.9s | 3p 3.0s |
| n20-piled-disc-003 (rho 0.17) | 1p 0.8s | 1p 1.1s | 1p 0.8s |
| n20-piled-disc-004 (rho 0.43) | 3p 2.8s | 3p 3.1s | 2p 2.0s |
| n20-piled-disc-005 (rho 0.28) | 1p 1.3s | 2p 2.2s | 2p 2.1s |
| n20-piled-letter-002 (rho 0.14) | 2p 2.1s | 2p 2.2s | 2p 2.4s |
| n20-piled-letter-003 (rho 0.17) | 2p 1.7s | 3p 3.2s | 2p 1.9s |
| n20-piled-letter-004 (rho 0.22) | 3p 3.1s | x 0.95 | 5p 5.2s |
| n20-piled-letter-005 (rho 0.20) | x 0.94 | x 0.90 | 3p 3.4s |
| n20-piled-square-002 (rho 0.43) | 3p 3.5s | 3p 3.5s | 3p 3.3s |
| n20-piled-square-003 (rho 0.17) | 4p 4.4s | 5p 6.2s | 3p 3.5s |
| n20-piled-square-004 (rho 0.48) | 5p 4.7s | 3p 2.9s | 4p 3.3s |
| n20-piled-square-005 (rho 0.56) | x 0.95 | x 0.25 | x 0.55 |
| n20-scattered-disc-002 (rho 0.54) | 8p 10.6s | 8p 10.8s | x 0.95 |
| n20-scattered-disc-003 (rho 0.47) | x 0.87 | x 0.83 | x 0.80 |
| n20-scattered-disc-004 (rho 0.33) | x 0.94 | x 0.65 | x 0.70 |
| n20-scattered-disc-005 (rho 0.32) | 7p 9.2s | 7p 9.1s | 6p 7.7s |
| n20-scattered-letter-002 (rho 0.22) | 9p 11.8s | 10p 12.9s | 8p 10.0s |
| n20-scattered-letter-003 (rho 0.14) | 6p 7.6s | 7p 9.6s | 6p 8.2s |
| n20-scattered-letter-004 (rho 0.20) | 8p 10.7s | x 0.91 | x 0.94 |
| n20-scattered-letter-005 (rho 0.22) | x 0.94 | 9p 11.3s | 6p 8.0s |
| n20-scattered-square-002 (rho 0.17) | 7p 9.5s | 9p 12.1s | 7p 9.2s |
| n20-scattered-square-003 (rho 0.35) | 9p 11.4s | 8p 11.2s | 7p 8.9s |
| n20-scattered-square-004 (rho 0.20) | 9p 11.5s | 9p 12.7s | 8p 10.4s |
| n20-scattered-square-005 (rho 0.45) | 11p 13.6s | x 0.93 | x 0.91 |
