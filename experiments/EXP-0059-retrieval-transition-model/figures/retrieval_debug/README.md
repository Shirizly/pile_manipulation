# Retrieval debug figures (EXP-0059)

Bank: `experiments/EXP-0059-retrieval-transition-model/artifacts/bank.pt` (11921 transitions). Predictor: `retrieval_k5_cube_median` (k=5, cube_median, cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0). Query source: DS-0016 `test_chains_v2_clean` (ISS-010-fix sampler; valid rows only).

| file | query idx | start kind | why picked | moved-mm | top1_dist | knn_disagreement |
|---|---|---|---|---|---|---|
| `q0102_scatter_typical.png` | 102 | scatter | scatter_typical | 1.68 | 0.0056 | 0.0005 |
| `q0815_scatter_typical.png` | 815 | scatter | scatter_typical | 2.42 | 0.0020 | 0.0001 |
| `q0518_clump_typical.png` | 518 | clump | clump_typical | 4.23 | 0.0035 | 0.0005 |
| `q0017_clump_typical.png` | 17 | clump | clump_typical | 3.65 | 0.0050 | 0.0013 |
| `q0409_near_wall.png` | 409 | clump | near_wall | 6.16 | 0.0057 | 0.0030 |
| `q0805_near_wall.png` | 805 | clump | near_wall | 5.66 | 0.0054 | 0.0035 |
| `q0147_best_mm.png` | 147 | scatter | best_mm | 0.55 | 0.0067 | 0.0001 |
| `q0268_best_mm.png` | 268 | scatter | best_mm | 0.57 | 0.0022 | 0.0001 |
| `q0263_worst_mm.png` | 263 | scatter | worst_mm | 10.61 | 0.0058 | 0.0005 |
| `q0679_worst_mm.png` | 679 | clump | worst_mm | 10.48 | 0.0070 | 0.0012 |
| `q0696_high_disagreement.png` | 696 | clump | high_disagreement | 3.99 | 0.0047 | 0.0059 |
| `q0601_high_disagreement.png` | 601 | clump | high_disagreement | 5.52 | 0.0052 | 0.0057 |
