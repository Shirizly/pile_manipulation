---
id: EXP-0074
title: >
  Wide Sean domain (8 held-out shards, pinned split): 128-px NFDs [8,16,32] beat the plain 64-px NFD by +0.04-0.05
  one-step accuracy; the zoom crop adds only +0.010 accuracy / +0.010 4-push accuracy over a capacity-matched vanilla
  128 (3 seeds) while ranking worse (slateN -0.022); a post-hoc mass balance lifts every model and the soft
  zoom128 + vanilla128 ensemble is the best predictor tested (acc1 0.613 / slateN 0.962 / roll4 0.490)
tier: T1
mode: exploratory
date: 2026-10-07
hypothesis: null
claim: >
  On the wide Sean corpus (DS-0007 lineage; scattered / inbetween / piled x n20 / n50 / n100, piled_n100 excluded;
  pushes 0-70 mm; test = the 8 held-out shards of the pinned file split results/split_files.json; pasted 64x64 world
  frame; acc1 = swept-region accuracy, roll_k = rollout accuracy after k own-prediction pushes, slateN on ~80 same-state
  pools per shard), one training run per configuration unless stated: (1) 128-px UNets [8,16,32] (zoom-window or whole
  tray) are more accurate than the plain 64-px NFD [4,8,16] (one-step +0.04-0.05, 4-push +0.05-0.08); (2) at matched
  capacity and unrolled T=4 training the zoom crop gains +0.010 acc1 and +0.010 roll4 over vanilla128 and loses 0.022
  slateN, stable over 3 training seeds (spread <= 0.002); (3) a val-selected post-hoc mass balance of each predicted
  change raises acc1 by +0.02-0.07 and slateN by +0.035-0.045 for every NFD; (4) the mass-balanced mean of zoom128 and
  vanilla128 beats either member on every metric. Offline only; no closed loop here (see EXP-0075).
provenance:
  commit: a7ff5cf3
  dirty: true
  script: "code/train_wide.py, ms_wide.py, chain.sh (training); eval_wide.py, eval_lf_wide.py, eval_variants.py (scoring); posthoc_fix.py + posthoc_report.py (mass balance); fit_lf_wide.py (LF); gate_*.py, divisions.py, paste_cap.py, zoomfactor.sh, longpush.sh, specialize.sh (analyses)"
  data: ["Genesis/data/Sean (canonical wide corpus, user 2026-10-06), audited by code/sean_audit.py", "results/split_files.json (pinned file split, 2026-10-07 04:00)", "artifacts/test_sets.pt, artifacts/val_sets.pt (cached held-out rows / chains / pools)"]
  data_commit: "Sean corpus as supplied by the user; no collection in this record"
  code_path: "sean_data.load_split -> build_cache / build_ms_cache (window / world rasters) -> train_wide (single step) -> ms_wide (unrolled T=4) -> eval_wide (pasted 64x64 frame vs occ_from_particles truth)"
  seed: "training seeds 0 (all), 1 and 2 (128-px unrolled pipelines only); eval row sampling rng 0"
  split: "file-level, pinned in results/split_files.json (test = files behind artifacts/test_sets.pt). Runs before 2026-10-07 04:00 used a per-process-salted hash() split and leaked 35-55 % of test rows into training; those runs are quarantined in runs/ and results/contaminated_pre_split_fix/ and NOT reported here. LF (MODEL-0024) used a fingerprint-recovered split with the same test files (code/lf_split.py)."
  runs: ["runs/w64_s0_ms4", "runs/z64_f8_ms4", "runs/w128_f8{,_s1,_s2}_ms4", "runs/z128_f8{,_s1,_s2}_ms4", "runs/lf_wide"]
  runtime: "overnight 2026-10-06 22:50 -> 2026-10-07 ~13:00 plus analyses to 2026-10-08; shared RTX 4070 laptop"
budget:
  declared: "overnight run, summary due 2026-10-07 07:50 (LOG.md header)"
  spent: "core table by 08:55 2026-10-07; follow-up analyses (ensembles, gating, LF-zoom) to 2026-10-08"
  outcome: exceeded
design:
  varied:
    model: "vanilla64 [4,8,16] / zoom64 [8,16,32] / vanilla128 [8,16,32] / zoom128 [8,16,32] / switched-linear foresight (wide fit) / narrow-trained zero-shot references"
    training: "single-step vs unrolled T=4; seeds 0/1/2 (128 px); post-hoc mass balance none / balance (val-selected)"
  held_fixed:
    data: "10k random non-null train rows per shard (80k); a 168k-row check at 64 px showed no gain (LOG 07:25)"
    window: "side = max(64 mm, L + 44 mm) (model/zoom_nfd/window_var.py), not swept except in the zoom-factor ablation"
  baselines: "persistence (acc1 0 by construction; slateN .25-.29 on the 2 shards run before it was stopped); plain 64-px NFD w64_s0; narrow-trained zoom64 and LF zero-shot"
  metric: "accuracy (swept-region acc1 in the pasted 64x64 frame), roll_k, slateN (lyapunov, same-state pools)"
noise_floor: "training-seed spread after unrolled training <= 0.002 acc1 / roll4 and <= 0.005 slateN (3 seeds, 128 px); single test set, row-sampling noise not bootstrapped; slateN is near-saturated (~0.90) for every model"
depends_on: [score-occupancy-subpixel-stable]
establishes: []
result: >
  Clean runs, mean over 8 test shards, acc1 / slateN / roll4 (results/table.md, headline_table.md, eval_*.json):
  vanilla64 unrolled .534/.907/.401; zoom64 unrolled .534/.896/.428; vanilla128 unrolled .572/.924/.453 (seeds 1, 2
  .572/.925/.454, .572/.924/.453); zoom128 unrolled .582/.899/.464 (seeds 1, 2 .582/.904/.463, .583/.902/.463);
  LF wide .539/.896/.410; narrow zoom64 zero-shot .297/.703/.260. Post-hoc balance (chosen on val, results/posthoc_fix_val.md):
  zoom128 .613/.940/.482, vanilla128 .595/.957/.475, zoom64 .602/.941/.464, vanilla64 .573/.942/.442. Ensemble zoom128 +
  vanilla128 (seed 0, balanced) .613/.962/.490 (results/eval_ens_zoom_vanilla_s0_balance.json); no hard gate by push length,
  scene or object count beats it (gating_bal.json). Zoom edge is largest on scattered scenes and short pushes and shrinks as
  the window grows with push length (zoom-factor ablation: 64-79 / 114 / 150 mm window .538 / .481 / .440 vs vanilla .483).
  Pasted-frame accuracy cap on these rows 0.677 (paste_cap.py).
verdict: supported
downgrades: [imprecision, provenance]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The narrow pilot (EXP-0072) found zoom128 > zoom64 ~ vanilla128 > plain NFD on n20 / 20 mm data. The wide corpus mixes
object counts, scene types and 0-70 mm pushes; if the pilot edge were a narrow-domain artefact it would vanish here, and
the capacity-matched vanilla128 control separates "crop" from "resolution + capacity".

## What was actually run
LOG.md is the authoritative running log. Clean pipeline (code/chain.sh): single-step training (128 px cut at epoch 20,
best-val checkpoint; 64 px 40 epochs), unrolled T=4 fine-tune (14 / 20 epochs), eval_wide on the pinned test set.
Registered checkpoints: MODEL-0016 (vanilla64), 0017 (zoom64), 0018-0020 (vanilla128 s0-2), 0021-0023 (zoom128 s0-2),
0024 (LF wide), 0025 (ens128 composite). Training commands are reconstructed from chain.sh + LOG (no run_config written).
Dirty tree throughout (overnight run; code committed afterwards in a7ff5cf3 / 5a242b72 / c86153e4 / 992594da).

**Leak incident (2026-10-07 04:00):** `sean_data.split_files()` seeded its RNG with python `hash(str)` (salted per process),
so every process drew a different file split and 35-55 % of cached test rows were in training caches. All runs before the
fix are quarantined; the fix pinned `results/split_files.json`. The `hash()` fallback was removed on 2026-10-15 (a missing
map now raises).

## Numbers
See `result`. Per-shard / per-push-length breakdowns: results/divisions.md, slateN_diagnosis.md (zoom slateN deficit =
spurious mass gain of the pasted prediction, growing with push length), specialists.md (no specialisation gain).

## What would change the verdict
A bootstrap over test pools/rows (the +0.010 zoom-vs-vanilla gap is outside seed noise but row-sampling noise is unmeasured);
a second Sean split; closed-loop confirmation (EXP-0075 found the model effect small next to planner effects).

## Threats
One test split; slateN near-saturated; window models are scored in the pasted frame, which is capped (0.677, ISS-015-type);
LF used different train/val files than the NFDs (same test files).

## Unrelated findings
- The per-process `hash()` seeding pattern is a general trap; the pinned-split rule is now in the LOG and sean_data.py.
- Multi-step (unrolled) training removes most single-step seed variance at 128 px (roll4 spread .387-.448 -> <= .001).
