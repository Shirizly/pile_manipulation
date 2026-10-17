# arms over 18 common scenarios

| arm | solved | pushes (solved) mean | time s (solved) mean | final mass mean | sims / scenario | wall s / scenario | plan s / decision |
|---|---|---|---|---|---|---|---|
| p1A_feas | 14/18 | 5.71 | 6.67 | 0.959 | 7 | 202 | 1.33 |
| p1B_greedy16 | 16/18 | 5.81 | 7.14 | 0.964 | 104 | 493 | 2.47 |
| p1C_beam4x8 | 18/18 | 5.78 | 6.89 | 0.965 | 185 | 420 | 2.12 |

paired on scenarios solved by both (b - a): pushes, time s
- p1B_greedy16 - p1A_feas: n 13, pushes -0.62, time -0.48 s [-1.11, +0.06], only-a solved 1, only-b solved 3
- p1C_beam4x8 - p1A_feas: n 14, pushes -0.79, time -0.86 s [-1.73, -0.05], only-a solved 0, only-b solved 4
- p1C_beam4x8 - p1B_greedy16: n 16, pushes -0.44, time -0.66 s [-1.02, -0.32], only-a solved 0, only-b solved 2

| scenario | p1A_feas | p1B_greedy16 | p1C_beam4x8 |
|---|---|---|---|
| n20-inbetween-disc-000 (rho 0.24) | 5p 6.1s | 4p 5.3s | 4p 5.2s |
| n20-inbetween-disc-001 (rho 0.57) | x 0.84 | x 0.93 | 9p 10.7s |
| n20-inbetween-letter-000 (rho 0.31) | 6p 7.3s | x 0.93 | 9p 9.5s |
| n20-inbetween-letter-001 (rho 0.17) | 5p 6.8s | 6p 7.5s | 5p 6.7s |
| n20-inbetween-square-000 (rho 0.19) | 4p 4.3s | 4p 4.4s | 4p 4.4s |
| n20-inbetween-square-001 (rho 0.35) | 9p 11.2s | 9p 11.4s | 8p 10.1s |
| n20-piled-disc-000 (rho 0.38) | 6p 7.0s | 4p 4.9s | 4p 4.4s |
| n20-piled-disc-001 (rho 0.23) | 2p 1.8s | 2p 1.7s | 2p 1.7s |
| n20-piled-letter-000 (rho 0.17) | 2p 1.6s | 2p 1.7s | 2p 1.7s |
| n20-piled-letter-001 (rho 0.14) | 3p 2.8s | 3p 2.7s | 2p 1.7s |
| n20-piled-square-000 (rho 0.36) | 3p 2.8s | 3p 2.8s | 3p 2.7s |
| n20-piled-square-001 (rho 0.39) | 8p 8.5s | 4p 5.1s | 3p 3.7s |
| n20-scattered-disc-000 (rho 0.55) | x 0.91 | 11p 13.6s | 10p 11.7s |
| n20-scattered-disc-001 (rho 0.44) | 10p 11.8s | 9p 11.2s | 7p 9.0s |
| n20-scattered-letter-000 (rho 0.20) | x 0.94 | 7p 9.4s | 6p 8.4s |
| n20-scattered-letter-001 (rho 0.31) | 9p 11.2s | 8p 10.1s | 8p 10.1s |
| n20-scattered-square-000 (rho 0.49) | x 0.88 | 9p 11.4s | 10p 11.7s |
| n20-scattered-square-001 (rho 0.31) | 8p 10.2s | 8p 11.0s | 8p 10.4s |
