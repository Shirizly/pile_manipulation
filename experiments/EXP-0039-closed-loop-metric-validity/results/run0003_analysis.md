# EXP-0039 RUN-0003 analysis (code/analyse_run0003.py)

| model | scatter | clump |
|---|---|---|
| nfd_3ch_randlen | 0.355 | 0.554 |
| linear_switched_soft | 0.348 | 0.516 |
| nfd_3ch_narrow_l20_v2 | 0.331 | 0.510 |
| nfd_3ch_narrow_l20_v2_soft_s2 | 0.307 | 0.547 |
| nfd_3ch_narrow_l20_v2_seed2 | 0.272 | 0.499 |
| nfd_3ch_narrow_l20_v2_seed1 | 0.245 | 0.514 |
| nfd_3ch_narrow_l20_v2_epoch10 | 0.213 | 0.465 |
| linear_narrow_l20_v2_res64 | 0.211 | 0.471 |

Kendall tau (scatter order vs clump order): +0.64
S1 clump - scatter (mean over models): +0.224 [+0.187, +0.261]
resolved pairs: scatter 19, clump 8
reversals (resolved in >= 1 regime): nfd_3ch_narrow_l20_v2 vs nfd_3ch_narrow_l20_v2_seed1: scatter +0.086 (res), clump -0.004 (n.s.); linear_switched_soft vs nfd_3ch_narrow_l20_v2_soft_s2: scatter +0.042 (res), clump -0.031 (n.s.)
reversals resolved in BOTH: none
S2 verdict (pre-registered rule): inconclusive
