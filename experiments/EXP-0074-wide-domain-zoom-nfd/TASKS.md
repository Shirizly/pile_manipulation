# EXP-0074 overnight task list (started 2026-10-06 22:50; hard deadline for the summary: 2026-10-07 07:50, keep running tasks after)
Rules: work top to bottom; the coordinator or any agent may add/edit tasks. Mark [x] done with a one-line result and a pointer to LOG.md;
[~] running; [!] blocked (say why). Every finished task appends an entry to LOG.md (living summary, newest at the BOTTOM of the log,
the headline table at the TOP kept current). Narrow-pilot lessons to REUSE: fine-tune/init from narrow models, unrolled multi-step training,
300x300 canvas + thr 0.2 input path, [4,8,16] UNet (wider/deeper overfit: skip). Skip: wider nets, T=1 canvas control, blur metrics.

## Phase 0 -- setup
- [~] T0.1 docs: Sean = canonical wider-domain training dataset (CODEMAP Datasets, DS-0007 DATASET.md, memory) -- CODEMAP done; DS-0007 DATASET.md + memory note pending
- [x] T0.2 audit Sean: chains (successor rows, chain length), legality of touchdowns per shard, push-length and n distributions, failed rows -- results/sean_splits.json: chains <=4 linked pushes, ~21k train rows/shard, piled_n100 too small to train on
- [x] T0.3 split Sean by FILE (train/val/test, grouped, no leakage), record in datasets/DS-0007-.../ or EXP-0074/artifacts split file -- code/sean_data.py (file-level, per shard 10/5/85)
- [x] T0.4 test sets: held-out Sean files (one-step rows + chains) per shard n20/n50/n100; same-state pools from held-out files for slateN -- artifacts/test_sets.pt: 1000 rows, <=400 4-push chains, 60 pools per shard; harness code/eval_wide.py
## Phase 1 -- infrastructure for variable push length / object count
- [x] T1.1 window geometry: side = max(64 mm, L + 44 mm), 64x64 and 128x128 variants; plates for variable L; extraction from 300x300 canvas (check raster res needed for 1.8 mm/px) -- model/zoom_nfd/window_var.py, verified vs narrow path at L=20
- [x] T1.2 caches (windows/canvases, train+val+test) for variable L -- artifacts/*.pt (80k-row single-step, 15k-chain multi-step)
- [~] T1.3 baselines trained on the SAME Sean train split: plain NFD 64x64 world, vanilla 128x128 world (so we can compare 'normal NFD on this data') -- w64_s0 done (acc1 .531); w128_s0, ms versions running
## Phase 2 -- models (best first: zoom128, then vanilla128, zoom64)
- [x] T2.1 zoom128 default geometry: single-step train (init from narrow zoom128), eval per shard/length bin -- z128_s0: acc1 .580 slateN .903 roll4 .429; z64_s0 .539/.911/.441
- [~] T2.2 zoom128 multi-step (unrolled) if chains usable -- T=4 (chains are <=4 long, not 8): z64_ms4 trained, z128_ms4 / w64_ms4 running
- [ ] T2.3 vanilla128, zoom64 same
- [ ] T2.4 decision: does the move to wide domain hurt (gap to normal NFD collapses)? if yes -> T3
## Phase 3 -- fallbacks (only if T2.4 says so)
- [ ] T3.1 fallback A: separate fixed windows per push-length bin
- [ ] T3.2 fallback B: 128x128 models separated by bins (window constant per bin = bin max sweep length); first train single-bin models on each split (object count n20/n50/n100 vs length bins) and compare which separation helps more
## Phase 4 -- cover the whole domain, then seeds
- [ ] T4.1 complete models covering the whole wide domain, final eval table (accuracy, slateN, rollout k=1..5)
- [ ] T4.2 extra seeds of the same
- [ ] T4.3 timing of the final models (benchmark_time method)
