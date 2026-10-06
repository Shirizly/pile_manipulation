# EXP-0072 zoom-window NFD -- first results (2026-10-06, pilot-grade: narrow data, 1 seed, no EXPERIMENT.md/register entry yet)
Window 64 mm (plate 40 + 12 mm each side), 1 mm/px, back margin 1 mm. Init: EXP-0022 RUN-0010 UNet. Train DS-0015 train_v2 (10653 rows), 100 epochs, lr 2e-4 cosine, flip aug.
Test DS-0016 (test_chains_v2_clean 896 rows, test_pools_v2 32 pools x 64), `code/score_zoom.py`.
| | accuracy_1 | slateN |
|---|---|---|
| ft100 PASTED (world, comparable) | 0.618 | 0.746 |
| ft100 WINDOW (ceiling-style) | 0.845 | 0.949 |
| scratch100 PASTED / WINDOW | 0.615 / 0.836 | 0.761 / 0.925 |
| zero-shot warped NFD PASTED / WINDOW | 0.294 / 0.453 | 0.567 / 0.743 |
| nfd_3ch_narrow_l20_v2 (standard NFD, same code path) | 0.559 | 0.710 |
| paste ceiling (true label pasted) | 0.651 | - |
Caveats: window input rasterised from particles (reference adapters get the 2 mm raster); pasted = predicted CHANGE added to the world raster (box-vs-disc raster styles cap accuracy at 0.651); WINDOW slateN uses a window-local goal/mass so it is not comparable to PASTED.

## No-crop 128x128 control (same pixel pitch, 1 mm/px), added 2026-10-06
`model/zoom_nfd/world128.py`, `code/make_cache128.py`, `train_zoom.py --cache world128_cache.pt --aug d4 --scratch --epochs 100`, `score_zoom.py --world128`.
Same UNet [4,8,16], same recipe/data/epochs (scratch, lr 2e-4 cosine, MSE), D4 augmentation (zoom model: flip only). Weights `runs/world128/`.
| | PASTED acc1 / slateN | WINDOW acc1 / slateN |
|---|---|---|
| zoom window (ft100) | 0.618 / 0.746 | 0.845 / 0.949 |
| world 128x128 | 0.581 / 0.726 | 0.794 / 0.871 |
| standard 64x64 NFD (nfd_3ch_narrow_l20_v2) | 0.559 / 0.710 | n/a |
128-model's window score = its prediction bilinearly resampled into the same window. One seed each; NFD slateN seed sd is 0.01-0.04 (EXP-0036), accuracy 0.003.

## Visual-input fidelity, convergence, timing (2026-10-06)
**Visual pipeline** (`model/zoom_nfd/window_gpu.py`): whole-tray binary raster HxH -> rotation+zoom as ONE grid_sample (3x3 antialiased) -> 64x64 window, binarised at 0.2.
IoU vs the direct (training) window raster, 200 test rows: H=128 0.86 (thr .5), 192 0.89 (.3), 256 0.90 (.2), 320 0.90 (.2), 400 0.90 (.1-.2). Ceiling ~0.90 = the training raster inflates cubes ~0.5 px/side (fillPoly boundary) + jagged edges; not a resolution limit. H>=256 suffices (1 mm/px window from 0.5 mm/px source).
Scores on DS-0016 (pasted acc1/slateN | window acc1/slateN), ft100 weights: direct 0.618/0.746 | 0.845/0.949; HR256 thr.2 0.617/0.760 | 0.834/0.947; HR400 thr.2 0.618/0.785 | 0.831/0.945; HR128 soft 0.599/0.769 | 0.816/0.936; HR400 soft 0.599/0.786 | 0.804/0.940. -> binarised H>=256 = same up to noise (window acc -0.01); soft input costs ~0.02.
**Convergence**: ft100 best val MSE 0.01167 (ep 84), ft300 0.01156 (ep 119; flat after, train 0.0104 < val 0.0117). world128 100 ep 0.00393 (still falling), 300 ep 0.00355 (train 0.00346).
300-epoch retest (pasted | window): ft300 0.622/0.765 | 0.846/0.947 (via HR256 thr.2: 0.621/0.763 | 0.835/0.950); world128_300 0.591/0.728 | 0.807/0.905. (100-ep: ft 0.618/0.746|0.845/0.949, world128 0.581/0.726|0.794/0.871.)
**Timing** (`code/time_zoom.py`, benchmark_time.py method, idle RTX 4070 laptop, ms per call of K candidates from one state; K=1/32/128/1024):
nfd_unet3ch (orig, 64) 1.54/2.2/4.6/37.9 | zoom window-out 1.3/1.6-2.0/6.2-6.7/64 | zoom world-out (+paste) 1.6/2.0-2.6/10.5/101 | world128 native 1.2/3.3/16.3/165 (world-out 1.4/3.9/17.7/170) | narrow nfd64 adapter 1.7/2.1/4.5/39.8 | gnn 3.4/18/56/436 | linear (CPU) 3.5/10.7/31/230 | mean-delta (CPU) 0.5/2.1/4.7/76.
Zoom stages at K=1024: extract 25, plates 5.8, UNet 34, paste 36.7 ms (HR 256 vs 400 identical cost).

## 128x128 zoom window (same 64 mm crop at 0.5 mm/px) + 300x300 source raster (2026-10-06)
Model `runs/zoom128/` (UNet [4,8,16] as before, init ft300, 150 ep, flip aug, lr 2e-4 cosine; val MSE 0.01189@100 -> 0.01181@150, train 0.01134: converged).
Source-raster fidelity (`code/hr_vs_poserendered.py`): reference = window rendered from cube poses at 1024 px (true footprints) area-averaged to model res.
Hard cv2-filled full-frame source INFLATES cubes (area x1.116 at 300 px); anti-aliased (aa=4, soft coverage) source: area x1.014, soft MAE 0.0022 (64 win) / 0.0038 (128 win), IoU@0.5 0.976 / 0.971 at H=300; H=400 0.982/0.982; H=256 0.971/0.962; H=128 0.931/0.918. The training-style direct raster itself is x1.275 (64 px) / x1.138 (128 px) too big vs truth -> models were trained on slightly inflated cubes; threshold 0.2-0.3 of true coverage reproduces that.
Scores (pasted acc1/slateN | window acc1/slateN; window-frame numbers at different grid res are NOT comparable):
zoom128 direct 0.640/0.809 | 0.859/0.946; zoom128 via HR300 aa4 thr.3 0.639/0.820 | 0.850/0.952; zoom64(ft300) direct 0.622/0.765 | 0.846/0.947; via HR300 aa4 thr.2 0.621/0.778 | 0.827/0.948 (thr.3 0.618/0.788 | 0.819/0.947). world128_300 0.591/0.728. Pasted accuracy ceiling (true label pasted) = 0.651.
Timing (idle, ms for K=1/32/128/1024; HR300 source): zoom128 window-out 1.3/8.4/29/263, world-out 1.5/7.4/34/301; stages @1024: extract 101, plates 23, UNet 143, paste 37. (zoom64: 1.3/1.6-2.0/6.2-6.7/64; world128 16.3 @128, 165 @1024.)

## Multi-step (rollout) prediction, 2026-10-06 (pilot-grade: narrow n20, 1 seed, DS-0016 test_chains_v2_clean, 111 chains x 8 pushes)
Code: `model/zoom_nfd/rollout.py` (300x300 canvas -> window -> UNet -> predicted CHANGE pasted back to canvas; differentiable), `code/ms_eval.py` (eval_narrow-style rollout_accuracy_k: own previous output fed back, truth = recorded state after k pushes, persistence = start, region = union of k swept rectangles, pasted 64x64 world frame), `code/ms_train.py` (unrolled training on DS-0015 train_v2 8-push chains, 1353 chains, random 90-degree world rotation aug, lr 1e-4 cosine, init from the single-step model), `code/ms_train_world128.py` (same recipe, plain 128x128 world model).
rollout accuracy k=1 / 2 / 3 / 4 / 5 / 8:
- nfd_3ch_narrow_l20_v2 (64 world): .578 .472 .396 .329 .275 .115
- linear_narrow_l20_v2_res64:       .521 .452 .414 .381 .361 .322
- world128 (single-step trained):   .614 .493 .403 .325 .251 .087
- zoom64 ft300 (single-step):       .627 .551 .510 .481 .461 .409
- zoom128 (single-step):            .650 .564 .526 .507 .486 .442
- zoom64, T=1 control on canvas inputs: .623 .506 .415 .328 .244 .015
- zoom64 unrolled T=4:              .631 .562 .525 .497 .478 .415
- zoom64 unrolled T=8:              .629 .562 .533 .507 .496 .454
- world128 unrolled T=4 (control):  .625 .549 .514 .492 .472 .439
- zoom128 unrolled T=4:             .657 .582 .554 .532 .520 .485
- zoom128 unrolled T=8 (from T4):   .654 .582 .556 .535 .524 .495
Architecture (zoom64 single-step, scratch): 2000-row seed subset, best val MSE [4,8,16] 0.0145 vs [8,16,32] 0.0154, [16,32,64] 0.0156, [4,8,16,32] 0.0168, [8,16,32,64] 0.0165 (wider/deeper overfit: train 0.001-0.006). Full data 150 ep: [4,8,16] 0.01216 (ep110), [8,16,32] 0.0117 (ep30, then overfits to 0.0134), [4,8,16,32] 0.0127 (ep33); ft300 reference 0.01156 -> no capacity gain.
Not done: unrolled fine-tune of the plain 64x64 NFD (EXP-0010 did this for the old NFD), slateN of the multi-step models, a cumulative-rollout perfect-physics ceiling, seeds, ms training of the wider nets.
