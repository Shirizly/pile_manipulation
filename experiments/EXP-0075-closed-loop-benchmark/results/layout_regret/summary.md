# EXP-0075 layout regret decomposition (states.json; 50 states, lam=2 H=1 ens2bal; gains in mm of Hungarian transport to slots, positive = better)

| subset | n | executed (replayed) | model pick (S0) | best in S0 | best overall (+refine) | best in S0 - executed | best overall - executed | pick - executed | share best-in-S0 > exec+1 mm | share best-overall > exec+1 mm | share pick >= best-in-S0 - 0.5 | Spearman(model cost, true) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 50 | +0.69 [+0.45, +0.95] | +0.70 [+0.46, +0.97] | +1.20 [+0.96, +1.46] | +1.32 [+1.09, +1.56] | +0.51 [+0.36, +0.68] | +0.63 [+0.47, +0.80] | +0.01 [-0.11, +0.13] | 0.20 [0.10, 0.32] | 0.22 [0.12, 0.34] | 0.58 [0.44, 0.72] | +0.89 |
| w15 | 26 | +0.81 [+0.45, +1.20] | +0.75 [+0.41, +1.14] | +1.17 [+0.82, +1.56] | +1.32 [+0.98, +1.70] | +0.36 [+0.22, +0.50] | +0.51 [+0.37, +0.66] | -0.06 [-0.22, +0.06] | 0.08 [0.00, 0.19] | 0.08 [0.00, 0.19] | 0.62 [0.42, 0.81] | +0.91 |
| w10 | 24 | +0.55 [+0.23, +0.91] | +0.65 [+0.34, +0.99] | +1.22 [+0.90, +1.55] | +1.31 [+0.97, +1.65] | +0.67 [+0.39, +0.99] | +0.76 [+0.47, +1.07] | +0.10 [-0.08, +0.29] | 0.33 [0.17, 0.54] | 0.38 [0.21, 0.54] | 0.54 [0.33, 0.75] | +0.88 |
| k=4 | 17 | +1.74 [+1.34, +2.16] | +1.69 [+1.32, +2.06] | +2.18 [+1.94, +2.47] | +2.27 [+2.01, +2.57] | +0.44 [+0.18, +0.73] | +0.52 [+0.25, +0.85] | -0.06 [-0.25, +0.10] | 0.18 [0.00, 0.35] | 0.18 [0.00, 0.35] | 0.65 [0.41, 0.88] | +0.88 |
| k=8 | 17 | +0.29 [+0.12, +0.49] | +0.35 [+0.15, +0.57] | +0.85 [+0.64, +1.11] | +1.05 [+0.78, +1.35] | +0.56 [+0.31, +0.88] | +0.76 [+0.53, +1.06] | +0.06 [-0.22, +0.34] | 0.18 [0.00, 0.35] | 0.24 [0.06, 0.47] | 0.47 [0.24, 0.71] | +0.90 |
| k=12 | 16 | -0.01 [-0.12, +0.08] | +0.03 [+0.00, +0.07] | +0.52 [+0.28, +0.79] | +0.59 [+0.36, +0.84] | +0.52 [+0.26, +0.82] | +0.60 [+0.35, +0.90] | +0.04 [-0.03, +0.14] | 0.25 [0.06, 0.50] | 0.25 [0.06, 0.50] | 0.62 [0.38, 0.88] | +0.90 |

## Best candidate by source (overall best, incl. refinement) and by group (mean over states of the best gain within the group)

best-in-S0 source: {np.str_('fetch'): 12, np.str_('ot'): 13, np.str_('refined_fetch'): 9, np.str_('refined_bank'): 16}
best-overall group: {np.str_('refine'): 35, np.str_('top'): 9, np.str_('ofTop'): 5, np.str_('exec'): 1}
best-overall source: {np.str_('refine'): 35, np.str_('refined_bank'): 5, np.str_('fetch'): 5, np.str_('exec'): 1, np.str_('refined_fetch'): 3, np.str_('ot'): 1}

| group | mean best gain mm |
|---|---|
| ofRand | -0.78 [-1.37, -0.22] |
| ofTop | +0.84 [+0.52, +1.16] |
| randbank | -0.48 [-0.74, -0.21] |
| refine | +1.29 [+1.04, +1.54] |
| tailOT | -1.38 [-2.27, -0.50] |
| top | +0.98 [+0.73, +1.25] |
| random bank (mean over 16) | -6.69 [-7.34, -6.02] |

cubes moved > 2 mm: executed 1.46, model pick 1.58, best-in-S0 3.18, best-overall 3.12; share of best-overall with 1 cube moved 0.18, >= 3 cubes 0.58
share of S0 candidates with positive true gain: 0.31

## Execution / model-vs-sim (c)
replay fidelity (replayed exec gain - recorded next-state gain): +0.01 [-0.02, +0.03] mm, SD 0.09 (sim re-run noise + legaliser; n states)
executed push Sinkhorn-layout gain (lam=1 units): pred +0.01 [+0.01, +0.02], true +0.01 [+0.01, +0.02]; gap (pred-true change) exec +0.00 [-0.00, +0.00], pick -0.00 [-0.00, +0.00]
within-state corr(pred layout change, true layout change) over S0: mean 0.99; Spearman(true Sinkhorn change, true Hungarian) 0.94
skeleton coverage change: executed +0.03 [+0.02, +0.04], model pick +0.03 [+0.02, +0.04], candidate with best Hungarian gain +0.01 [-0.01, +0.02]

## Tail cubes (> 10 mm from assigned slot): 251 cubes in 50 of 50 states

- some simulated candidate moves it >= 1 mm closer with no other cube worse by > 1 mm ('clean'): 0.31 [0.25, 0.37]; ... and net Hungarian gain > 0: 0.58 [0.52, 0.65]
- the clean push exists only among refinement rows: 0.12 of clean cases
- best single-cube improvement over all candidates: +16.16 [+14.88, +17.52] mm; improvement by the executed push: +2.27 [+1.37, +3.24] mm; executed push is clean for it: 0.12 [0.08, 0.16]
- tail rank 0-2 (0 = worst; the worst 3 had targeted OT rows added): n 139, clean candidate exists 0.37 [0.28, 0.45]
- tail rank 3-98 (0 = worst; the worst 3 had targeted OT rows added): n 112, clean candidate exists 0.23 [0.16, 0.31]
- group providing a clean push: {np.str_('top'): 50, np.str_('ofTop'): 9, np.str_('refine'): 14, np.str_('tailOT'): 2, np.str_('ofRand'): 2}
