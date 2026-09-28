# Retrieval debug figures (EXP-0059)

Bank: `experiments/EXP-0059-retrieval-transition-model/artifacts/bank.pt` (11921 transitions). Predictor: `retrieval_k5_cube_median` (k=5, cube_median, cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0). Query source: DS-0009 `test_chains` (valid rows only).

| file | query idx | start kind | why picked | moved-mm | top1_dist | knn_disagreement |
|---|---|---|---|---|---|---|
| `q0111_scatter_typical.png` | 111 | scatter | scatter_typical | 2.99 | 0.0093 | 0.0032 |
| `q0928_scatter_typical.png` | 928 | scatter | scatter_typical | 0.85 | 0.0122 | 0.0003 |
| `q1009_clump_typical.png` | 1009 | clump | clump_typical | nan | 0.0095 | 0.0005 |
| `q0696_clump_typical.png` | 696 | clump | clump_typical | 5.63 | 0.0066 | 0.0052 |
| `q0908_near_wall.png` | 908 | scatter | near_wall | 4.97 | 0.0151 | 0.0023 |
| `q0064_near_wall.png` | 64 | scatter | near_wall | 2.66 | 0.0123 | 0.0048 |
| `q0240_best_mm.png` | 240 | clump | best_mm | 0.53 | 0.0073 | 0.0028 |
| `q0864_best_mm.png` | 864 | scatter | best_mm | 0.54 | 0.0105 | 0.0024 |
| `q0364_worst_mm.png` | 364 | scatter | worst_mm | 22.07 | 0.0118 | 0.0008 |
| `q0775_worst_mm.png` | 775 | scatter | worst_mm | 22.05 | 0.0129 | 0.0012 |
| `q0626_high_disagreement.png` | 626 | clump | high_disagreement | 5.24 | 0.0069 | 0.0076 |
| `q0815_high_disagreement.png` | 815 | scatter | high_disagreement | 9.15 | 0.0090 | 0.0074 |
