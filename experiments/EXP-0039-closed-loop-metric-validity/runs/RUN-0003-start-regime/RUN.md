# RUN-0003 — clump vs scatter start regime in legal closed loop (pre-registered in EXPERIMENT.md)

- Pre-registration committed before running: dd27e84e (2026-10-04 12:42, also adds the driver's `--starts-file`). Run
  commit dd27e84e (first units clean, later units dirty: result files accumulating in the tree); results committed in a5ccb680.
- Queue: `code/run0003_queue.sh` (one `scripts/run_probe.py` unit per model; provenance per unit in
  `results/run0003/rp_hm_cem_<model>.{json,log}`). 2026-10-04 12:42-16:19 CEST (`results/run0003/full.done`), 8/8 units,
  every episode complete.
- Cells: RUN-0002's CEM arm (pop 1024 / elite 0.25, 0.5 s, 8 letters, 20 mm, 16 pushes, `--legalize --record-states`) with
  starts from `Genesis/data/narrow_l20_n20/test_pools_v2/pools_0.pt`: scatter pools 0-7 and single-layer clump pools 16-23
  -> 64 paired episodes per model per regime.
- Analysis (pre-registered quantities): `CUDA_VISIBLE_DEVICES= python -u code/analyse_run0003.py` ->
  `results/run0003_analysis.{json,md}`; S1 / S2 verdicts in EXPERIMENT.md.

## Addendum (2026-10-04, noise null; zero simulation; EXPLORATORY)

`CUDA_VISIBLE_DEVICES= python -u code/analyse_run0003_noise_null.py` -> `results/run0003_noise_null.{json,md}`.

- **Is tau +0.64 between regimes distinguishable from within-regime noise?** No. Over all 35 balanced 4/4 splits of the
  8 starts, the split-half tau WITHIN clump has mean +0.57 (min +0.29, max +0.79; 83 % of splits at or below the observed
  +0.64), within scatter +0.80 (+0.64..+1.00). An episode bootstrap of the between-regime tau gives +0.59 [+0.36, +0.79]:
  it spans both registered thresholds (P(tau <= 0.43) 0.12, P(tau >= 0.71) 0.22). With 8 starts per regime S2 was
  underpowered; "order ~stable" is not a supported reading, nor is "order changes".
- The scatter order replicates RUN-0002's CEM arm on DS-0006 starts (tau +0.93, independent start set); clump vs RUN-0002
  +0.71.
- **Regime effect per model**: clump - scatter +0.17 (linear_switched_soft) .. +0.27 (narrow seed 1); every CI excludes 0;
  spread over models 0.10 -- the three worst scatter models gain the most, i.e. clumps compress the gaps.
- **Executed un-legalisable pushes** (`legal_ok == False`, executed unchanged): scatter 195 / 8192 = 2.4 % (soft_s2 64 / 1024
  = 6.3 %), clump 11 / 8192 = 0.1 %; total 206 / 16384 = 1.3 %. Dropping each model's affected episodes: order tau vs full
  +0.79 (scatter), +0.93 (clump); between-regime tau then +0.79 (paired drop, 9 scatter episodes left: +0.50). ISS-014.
- Per goal: clump - scatter +0.20..+0.27 on every letter; per-letter clump orders agree poorly with the pooled clump order
  (tau +0.14..+0.93) -- the clump regime's 8-model order is noisy at the per-goal level.
