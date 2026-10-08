# EXP-0075 timing of basic units (RTX 4070 Laptop 8 GB, one process, chunk 256; measured 2026-10-08 with code/time_units*.py)

Forward cost of ONE candidate sequence through `SeqObjective.cost` (model rollout + 64x64 paste + mass balance + objective), ms:

| model | H=1 | H=2 | H=4 | H=4 vs ens128 | evals/s at H=4 |
|---|---|---|---|---|---|
| zoom128 | 0.477 | 0.956 | 1.914 | 0.63x | 522 |
| vanilla128 | 0.273 | 0.551 | 1.102 | 0.37x | 907 |
| ens128 (zoom+vanilla) | 0.751 | 1.503 | 3.016 | 1.00x | 332 |
| zoom64 (f8) | 0.207 | 0.415 | 0.832 | 0.28x | 1203 |
| vanilla64 (f4) | 0.037 | 0.061 | 0.130 | 0.04x | 7682 |
| vanilla64 (f8) | 0.061 | 0.120 | 0.247 | 0.08x | 4041 |

GD step (Adam step through the differentiable model, forward+backward), ms per step by batch of sequences:

| model | H=1: 4 / 8 / 24 / 64 seqs | H=4: 4 / 8 / 24 / 64 seqs |
|---|---|---|
| ens128 (zoom+vanilla) | 10 / 13 / 27 / 87 | 35 / 38 / 115 / 372 |
| zoom128 | 7 / 8 / 19 / 55 | 22 / 22 / 76 / 245 |
| vanilla128 | 5 / 6 / 12 / 34 | 15 / 16 / 39 / 129 |
| zoom64 (f8) | 6 / 6 / 8 / 21 | 21 / 21 / 32 / 103 |
| vanilla64 (f4) | 7 / 5 / 5 / 7 | 15 / 18 / 17 / 19 |

Other units: pile-aware proposals on GPU: ~1 ms per 10,000 (free); **legalisation of candidate pushes on CPU (legalize_pushes): 2.2 ms per push** (2,000 pushes 4.4 s; 60,000 pushes 127 s) -- the dominant cost of a legal sampler, ~ as expensive as the ensemble model per candidate; state encoding per decision (canvas/raster): 3 ms; simulator: ~11 s per batched push of 32 envs (4-push replay of 32 plans ~ 46 s, +~2 min start-up).

Checks: ens128 H=4 3.0 ms/seq matches the earlier CEM-10k x 20 run (210,000 evals, 638 s = 3.04 ms). GD of 150 steps on 24 sequences at H=4 with ens128 = 17 s (+ final cost). Small models are launch-bound: GD step time of vanilla64 is flat (15-19 ms) from 4 to 64 sequences.

Consequence: ens128 H=4 sequences cost 3.0 ms -> 1 s = 330, 3 s = 1,000, 10 s = 3,300 evaluations; vanilla64 (f4) 0.13 ms -> 7,700 / 23,000 / 77,000 (23x). H=1 evaluations are 4x cheaper than H=4 for every model.