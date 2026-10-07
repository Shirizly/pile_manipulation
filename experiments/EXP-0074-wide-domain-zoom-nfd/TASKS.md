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

## RESET 2026-10-07 04:00 (data leak found, see LOG): all trained models invalid; clean re-run queue
- [~] R1 caches rebuilt with the pinned split (results/split_files.json): zoom/world 128 single-step done (overlap with test 0.2 % = duplicate transitions in the corpus); ms caches + 64-res caches building
- [~] R2 single-step: z128_f8, w128_f8 (+ seeds s1) training; then w64 (normal NFD baseline) and z64_f8
- [ ] R3 unrolled T=4 fine-tunes from the single-step models (z128_f8_ms4, w128_f8_ms4, + w64/z64), evals via code/eval_wide.py (eval_when_done.sh)
- [ ] R4 seeds: s1 (running), s2 if time; ms for the seeds
- [ ] R5 timing with code/time_wide.py on an idle GPU (stop other jobs ~07:00)
- [ ] R6 re-check the earlier conclusions on clean models: capacity (f4 vs f8), zoom vs vanilla gap, specialists/bins fallbacks (only if the main gap still looks small)
- [ ] R7 summary at 07:50 (LOG headline table + results/table.md)

## Status 2026-10-07 06:35 and follow-ups (add / tick here)
- [x] R1 caches (pinned split); [x] R2 single-step z64_f8, w64_s0, z128_f8, w128_f8; [x] R3 ms4 for those four (results/table.md); [x] LF wide baseline (agent); [x] slateN diagnosis (agent): zoom pasted prediction gains spurious mass (grows with push length)
- [~] R4 seed 1 pipelines (z128_f8_s1, w128_f8_s1 ms4 stage, eval ~07:25); [~] data scaling 168k rows (z64_f8_full, w64_f8_full; chain.sh evals + ms4); [~] mass-conservation loss fine-tune at 64 px vs no-mass control (z64_f8_massA/C, w64_s0_massA/C)
- [ ] R5 timing on an IDLE GPU: `PYTHONPATH=. python code/time_wide.py --zoom64 CKPT 8,16,32 --zoom128 CKPT 8,16,32 --world64 CKPT 4,8,16 --world128 CKPT 8,16,32` (runs/z128_f8_ms4, w128_f8_ms4, z64_f8_ms4, w64_s0_ms4 unet_best.pth); code verified on a contended GPU only
- [x] F1 DONE 07:50 (agent): val-tuned plain mass-balance post-hoc fix works for all four models (LOG 07:50; code/posthoc_fix.py). Remaining: integrate it into the predictor / MPC adapter and retrain WITH balance in the loop (zero-sum delta head) -- if mass loss helped little, test a zero-sum delta parameterisation.
- [ ] (old F1) if mass loss helps: add it to ms_wide.py (T=4 unrolled) for z128_f8 / w128_f8 and re-evaluate; else adopt the post-hoc zero-sum delta (results/slateN_diagnosis.md) tuned on VAL pools (the agent tuned on test: redo on val before reporting)
- [x] F2 DONE 08:55 seed 2 for the 128 pipelines (both models) so the zoom-vs-vanilla gap (+.010-.017 acc1, +.011-.023 roll4) has 3-seed error bars
- [ ] F3 'zoom as an extra channel' (the user's second approach): vanilla 128 world NFD + extra input channels carrying a high-res crop around the push start; or two-branch; compare to z128_f8_ms4 / w128_f8_ms4
- [x] F4 DONE 2026-10-07 (LOG 10:00, 13:00): domain-specialist probes clean on the pinned split; specialists lose to the generalist; zoom-factor ablation. (old F4:) re-run the fallback probes (bins windows, n100-only / long-push specialists) CLEAN on the pinned split (earlier runs were contaminated: indicative 'no gain'); only if the zoom-vs-vanilla gap survives F2
- [ ] F5 more chains / longer T: Sean chains are <=4 pushes; the narrow pilot used 8
- [ ] F6 write proper EXPERIMENT.md / register entries for EXP-0072/0073/0074 (experiment-log, register-validator skills) once exploration settles; add DS-0007 DATASET note (done) and a pinned-split note
