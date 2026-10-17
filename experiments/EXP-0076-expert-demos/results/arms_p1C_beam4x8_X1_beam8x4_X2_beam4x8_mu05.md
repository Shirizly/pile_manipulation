# arms over 12 common scenarios

| arm | solved | pushes (solved) mean | time s (solved) mean | final mass mean | sims / scenario | wall s / scenario | plan s / decision |
|---|---|---|---|---|---|---|---|
| p1C_beam4x8 | 12/12 | 7.33 | 9.01 | 0.964 | 235 | 574 | 2.22 |
| X1_beam8x4 | 12/12 | 7.33 | 9.15 | 0.962 | 235 | 505 | 1.54 |
| X2_beam4x8_mu05 | 12/12 | 7.08 | 8.78 | 0.964 | 227 | 340 | 1.54 |

paired on scenarios solved by both (b - a): pushes, time s
- X1_beam8x4 - p1C_beam4x8: n 12, pushes +0.00, time +0.14 s [-0.55, +0.74], only-a solved 0, only-b solved 0
- X2_beam4x8_mu05 - p1C_beam4x8: n 12, pushes -0.25, time -0.23 s [-1.06, +0.48], only-a solved 0, only-b solved 0
- X2_beam4x8_mu05 - X1_beam8x4: n 12, pushes -0.25, time -0.38 s [-0.93, +0.12], only-a solved 0, only-b solved 0

| scenario | p1C_beam4x8 | X1_beam8x4 | X2_beam4x8_mu05 |
|---|---|---|---|
| n20-inbetween-disc-000 (rho 0.24) | 4p 5.2s | 4p 5.1s | 5p 5.8s |
| n20-inbetween-disc-001 (rho 0.57) | 9p 10.7s | 10p 11.8s | 10p 12.3s |
| n20-inbetween-letter-000 (rho 0.31) | 9p 9.5s | 5p 6.8s | 5p 5.8s |
| n20-inbetween-letter-001 (rho 0.17) | 5p 6.7s | 5p 6.9s | 5p 6.5s |
| n20-inbetween-square-000 (rho 0.19) | 4p 4.4s | 4p 4.3s | 3p 3.2s |
| n20-inbetween-square-001 (rho 0.35) | 8p 10.1s | 7p 9.7s | 7p 9.8s |
| n20-scattered-disc-000 (rho 0.55) | 10p 11.7s | 11p 13.0s | 8p 10.3s |
| n20-scattered-disc-001 (rho 0.44) | 7p 9.0s | 7p 8.7s | 8p 9.5s |
| n20-scattered-letter-000 (rho 0.20) | 6p 8.4s | 8p 10.5s | 8p 10.0s |
| n20-scattered-letter-001 (rho 0.31) | 8p 10.1s | 9p 10.7s | 9p 10.8s |
| n20-scattered-square-000 (rho 0.49) | 10p 11.7s | 10p 11.8s | 9p 10.9s |
| n20-scattered-square-001 (rho 0.31) | 8p 10.4s | 8p 10.5s | 8p 10.5s |
