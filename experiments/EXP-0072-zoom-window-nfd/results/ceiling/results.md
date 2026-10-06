# EXP-0072 ceiling study -- how good can ANY predictor be under simulator noise (2026-10-06; pilot-grade: narrow n20 exact-20 mm corpus, DS-0016 test, 1 perturbation draw per level)

Full tables: `tables.md` (T1-T8). Raw: `raster_perceptibility.json`, `ceiling_scores.json`, `metrics_vs_slate.json`. Re-sim states: `experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/resim_*.pt`, cached model predictions `preds_*.pt` (artifacts, gitignored). Code: `code/ceiling_*.py`.
Method: "perfect-physics predictor" = Genesis re-simulation (TRAINING_PHYSICS, same seam as chain_collection / EXP-0059 chaos_floor) of the recorded push from the recorded start state with xy gaussian noise sigma (+ yaw jitter 2 deg/mm), scored as if it were a model prediction of the RECORDED outcome. Three setups per level: 896 one-step rows (independent noise per row), 112 chains x 8 pushes (noise once at chain start, no re-set between pushes), 32 pools x 64 pushes (one noisy start per pool). Levels 0 (2 reps), 0.01, 0.05, 0.25, 0.5, 1 mm. Models' DS-0016 numbers were reproduced exactly (zoom128 0.640/0.809, ft300 0.622/0.765, world128_300 0.591/0.728, nfd64 0.559/0.710).

## 1a. Imperceptible perturbation (T1)
Changed-pixel fraction of a hard raster is LINEAR in sigma (no threshold: boundary pixels flip in proportion to the shift). Hence the criterion is a choice:
- sigma at which the hard raster changes by 1% of occupied pixels: **0.016 (world64), 0.018 (world128), 0.018 (zoom64), 0.017 mm (zoom128)** -- the same in mm for all four (pixel pitch cancels: more pixels but proportionally more boundary).
- sigma at which >=95% of states give a bit-identical hard raster: **0.0005-0.001 mm** for all four (extrapolated from the 0.01 mm point, where only 35-63% are identical) -- below float32/simulator reproducibility, so unusable.
So "imperceptible" = ~0.01-0.02 mm for every model; I simulated 0.01 mm (changes 0.5-0.8% of pixels) and 0.05 mm (2.7-3.2%). The soft-raster criterion was not separately run (the models consume hard rasters).

## 1b. Ceiling (one-step accuracy_1; T2) -- mean-of-reps at 0 mm unless stated
| frame | true-label cap | re-sim 0 mm | 0.01 | 0.05 | 0.25 | 0.5 | 1 mm |
|---|---|---|---|---|---|---|---|
| w64 native (nfd64's frame) | 1.0 | 0.805 | 0.789 | 0.760 | 0.659 | 0.579 | 0.436 |
| w128 native | 1.0 | 0.781 | 0.765 | 0.740 | 0.648 | 0.569 | 0.439 |
| window64 (zoom64 / world128 resampled) | 1.0 | 0.898 | 0.889 | 0.881 | 0.839 | 0.813 | 0.749 |
| window128 (zoom128) | 1.0 | 0.900 | 0.890 | 0.881 | 0.840 | 0.812 | 0.751 |
| PASTED zoom64 | 0.651 | 0.641 | 0.639 | 0.636 | 0.622 | 0.607 | 0.561 |
| PASTED zoom128 | 0.681 | 0.670 | 0.666 | 0.662 | 0.647 | 0.629 | 0.576 |
| PASTED world128 | 0.633 | 0.622 | 0.621 | 0.615 | 0.596 | 0.571 | 0.488 |
- Zero perturbation is NOT a noiseless ceiling: re-simulating the identical start gives cube error rms 0.32 mm/push vs the recorded outcome, and two identical re-runs differ from each other (one-step rms 0.18 mm, 8-push chain rms 1.25 mm; 33% of cubes not bit-identical) -- the simulator is non-deterministic on this setup. I did not isolate the cause (GPU non-associativity vs reset-vs-continuous sim state; untested). The 0.01-0.05 mm "imperceptible" levels are therefore indistinguishable from re-run noise in cube position (rms 0.32 -> 0.49 -> 0.66 mm) and cost 0.016/0.045 accuracy in w64.
- Scatter is far more chaotic than clump (w64 @0.5 mm: 0.44 vs 0.68; @1 mm 0.24 vs 0.58).
- Multi-push (T4, noise once at chain start): cube rms error vs recorded grows ~linearly, 0.28 mm (push 1) -> 2.2-2.3 mm (push 8) at 0 mm; 0.78 -> 3.1 at 0.25 mm; 1.75 -> 3.8 at 1 mm. Fraction of cubes off by >2.5 mm at push 8: 13% (0 mm), 18% (0.25), 26% (1 mm). w64 push-k accuracy at 0 mm falls 0.79 -> ~0.62-0.65 by push 8; paste frame 0.655 -> 0.56-0.575. Noise levels converge at late pushes (0.01-0.05 mm ~ 0 mm at push >=4): the chaos is amplified to the same level regardless of the tiny initial perturbation.
- slateN ceiling (T3; pools share one noisy start, so ranking is robust): w64 hard-raster 0.84-0.90 at all levels (0.889 at 0 mm), PASTED zoom64 0.768-0.790 (0.786), PASTED zoom128 0.807-0.836 (0.825). Noise barely moves it (pool sd 0.01-0.04); the cap is the raster style: true label in frame gives 0.896 / 0.778 / 0.824.

## Where the four models sit (T5; accuracy pooled over 896 chain rows)
| model | frame | acc | ceiling@0 | model/ceiling | ~equiv. xy noise | slateN (pasted) | slateN ceiling same frame |
|---|---|---|---|---|---|---|---|
| nfd64 | w64 native | 0.559 | 0.805 | 0.69 | ~0.57 mm | 0.710 | 0.889 (headroom 0.18) |
| world128_300 | window64 / pasted | 0.807 / 0.591 | 0.898 / 0.622 | 0.90 / 0.95 | 0.54 / 0.30 mm | 0.728 | not computed (~0.78 expected, untested) |
| zoom64 ft300 | window64 / pasted | 0.846 / 0.622 | 0.898 / 0.641 | 0.94 / 0.97 | 0.22 / 0.26 mm | 0.765 | 0.786 (cap 0.778) |
| zoom128 | window128 / pasted | 0.859 / 0.640 | 0.900 / 0.670 | 0.95 / 0.96 | 0.16 / 0.35 mm | 0.809 | 0.825 (cap 0.824) |
(Equivalent noise = the xy sigma at which the perfect-physics predictor's score falls to the model's, linear interpolation over 6 levels; the model's error is not of that kind, so read as a ruler, not a diagnosis.) Ordering by window-frame equivalent noise matches slateN ordering (zoom128 < zoom64 < world128 ~ nfd64).

## The 0.651 paste cap vs the window frame
0.651 is exactly reproduced (true label rendered in window style, change pasted on the disc-style world raster; 0.681 for 128 px, 0.633 for world128, so it is not one number). It is a METRIC artefact: even a re-simulated predictor with 0.3 mm rms position error scores 0.641 pasted but 0.80-0.90 in-frame, and moves only 0.03 between 0 and 0.5 mm of noise -- pasted accuracy has ~0.01-0.03 of usable range and models already sit at 96-97% of it. Same for pasted slateN (caps 0.778/0.824; zoom64/zoom128 are within 0.02 of them, below the pool sd). The window frame has no style cap (1.0), spans 0.90 -> 0.75 over 0 -> 1 mm, and is where the models can be told apart, so it is the more trustworthy frame for the zoom-vs-zoom comparison -- with caveats: window frames differ per model (resolution, window-local goal; world128 is bilinearly resampled into the window, adding a small blur penalty), so window numbers rank within-window models only; and window-frame slateN ceiling was NOT computed here (only the pasted frame).

## 2. Accuracy-style metrics vs slateN (T6-T8; pasted world-64 frame; 15 models = 7 zoom/world/nfd64 + nfd seeds/sharp/soft variants + 2 linear)
n is models, not independent experiments; tau on 7 models has bootstrap sd ~0.2, on 15 ~0.08 (pool bootstrap; model fits fixed, one seed each).
- Raw hard accuracy is already good here: tau 0.90 (7 models, pools), 0.79 (15). Nothing clearly beats it. changed-cell IoU 1.00 (7) but 0.58 (15); in-goal mass-change skill 0.81 (7) / **0.87 (15)** (boot 0.72 vs 0.72 for acc: not different); blur sigma 1 ~ same, blur 2/4 worse on the 15 (0.64 / 0.35); mass-in-region and sliced-EMD 0.90 (7) but 0.45 (15); region centroid 0.81 / 0.62; goal-dv error skill is NEGATIVE on the 7 (-0.24) and 0.35 on the 15; the "dv bias/optimism" skill is uncorrelated (0.09). Full-frame accuracy 0.43-0.58.
- Per-pool: within-model across-pool Spearman is 0.6-0.8 for most metrics (dominated by pool difficulty); pool-demeaned across-model correlation is only 0.0-0.4 (best acc_hard / blur1 0.41 on 15) -- accuracy-style metrics explain model-level differences far better than which model wins a given pool.
- Ceiling in each metric (T7, w64 native | pasted zoom64 at 0 mm): acc 0.81|0.66, blur1 0.90|0.81, blur2 0.93|0.80, changed-cell IoU 0.96|0.86, mass-in-region 0.97|0.84, centroid skill 0.85|0.42, EMD 0.97|0.84, dv-err skill 0.84|0.47, in-goal mass 0.78|0.63. Pasted-frame ceilings of centroid/dv metrics are low (0.4-0.5) because of the style cap, while models reach 0.41 (centroid) and 0.45-0.53 (dv) -- i.e. model metrics are AT the pasted ceiling for those metrics; w64-native leaves headroom only for nfd64-type models.
- Caveats: models differ in family/training, correlation is on 7-15 models; the in-goal and dv metrics use the same 13 goals and the soft truth as slateN (partly circular); degenerate slateN pool/goal cells are skipped; metric definitions are my own (ceiling_metrics.py), centroid/EMD restricted to the swept region.

## Honest limits
One perturbation draw per level (two at 0 mm); 896 rows / 112 chains / 32 pools; no sim determinism fix; cause of 0 mm non-reproducibility not isolated; ceilings assume the recorded outcome is "the" truth although it is itself one noisy draw (so true achievable accuracy for a mean-predictor can differ -- a blurry/mean predictor can beat a physics re-run in expectation, EXP-0059 zoo); window-frame slateN ceiling, soft-raster perceptibility criterion and world128 slateN ceiling not run. Raised ISS-015 in experiments/OPEN_ISSUES.md.
