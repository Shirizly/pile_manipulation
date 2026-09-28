# Retrieval debug figures (EXP-0059)

Bank: `experiments/EXP-0059-retrieval-transition-model/artifacts/bank_curated.pt` (9198 transitions). Predictor: `retrieval_k5_cube_median` (k=5, cube_median, cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0). Query source: DS-0009 `test_chains` (valid rows only).

| file | query idx | start kind | why picked | moved-mm | top1_dist | knn_disagreement |
|---|---|---|---|---|---|---|
| `q0111_scatter_typical.png` | 111 | scatter | scatter_typical | 5.12 | 0.0007 | 0.0001 |
| `q0928_scatter_typical.png` | 928 | scatter | scatter_typical | 1.95 | 0.0035 | 0.0001 |
| `q1009_clump_typical.png` | 1009 | clump | clump_typical | nan | 0.0000 | 0.0000 |
| `q0696_clump_typical.png` | 696 | clump | clump_typical | 5.42 | 0.0058 | 0.0030 |
| `q0908_near_wall.png` | 908 | scatter | near_wall | 4.97 | 0.0116 | 0.0004 |
| `q0064_near_wall.png` | 64 | scatter | near_wall | 2.43 | 0.0030 | 0.0002 |
| `q0705_best_mm.png` | 705 | scatter | best_mm | 0.53 | 0.0006 | 0.0002 |
| `q0835_best_mm.png` | 835 | scatter | best_mm | 0.54 | 0.0008 | 0.0001 |
| `q0044_worst_mm.png` | 44 | scatter | worst_mm | 16.35 | 0.0093 | 0.0001 |
| `q0448_worst_mm.png` | 448 | scatter | worst_mm | 15.61 | 0.0109 | 0.0002 |
| `q0601_high_disagreement.png` | 601 | clump | high_disagreement | 4.56 | 0.0050 | 0.0066 |
| `q0574_high_disagreement.png` | 574 | clump | high_disagreement | 6.34 | 0.0034 | 0.0061 |
