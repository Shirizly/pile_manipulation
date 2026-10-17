---
id: EXP-0073
title: >
  Retrospective active-data mining for the zoom64 NFD (narrow pilot, 3 retrain seeds, one selection draw): 500 rows mined
  by the model's own p(1-p) or ensemble variance cut swept-region error ~9-10 % vs base (random 500: -4 %), about as much
  as 1,000+ random rows, but slateN does not move for any arm
tier: T1
mode: exploratory
date: 2026-10-06
hypothesis: null
claim: >
  Pilot-grade (narrow n20 / 20 mm; zoom64 NFD; seed set 2,000 DS-0015 train_v2 rows, reservoir 8,653 rows with known
  outcomes; +500 rows per arm; 3 retrain seeds; one scoring ensemble; one selection draw; scored on DS-0017 and DS-0016
  in the window frame): adding 500 rows chosen by an uncertainty proxy (model p(1-p), independent-seed ensemble variance)
  reduces swept-region error more than 500 or 1,000 random rows, while window and pasted slateN stay within seed noise
  for every arm.
provenance:
  commit: ded30444
  dirty: true
  script: "code/active_common.py, active_reducible.py, active_retro.py, active_retro_report.py, active_rowcurves.py, active_optimism.py (+ v1 active_*.py proxies)"
  data: ["DS-0015 train_v2 (seed + reservoir)", "DS-0016, DS-0017 (evaluation)", "EXP-0072 runs/window_cache.pt and artifacts/ceiling/resim_* (re-sim replicates)"]
  data_commit: "no new simulation (retrospective)"
  code_path: "active_common.train_unet (pretrained-init fine-tune, 150 ep) on seed + selected rows -> EvalSet window-frame e_sw / acc1 / slateN; pasted frame via EXP-0072 score_zoom.py for 5 arms"
  seed: "retrain seeds 0/1/2 per arm; selection fixed per arm (one draw), random arms redrawn"
  split: "seed / reservoir from DS-0015 train_v2; DS-0017 used for the headline, DS-0016 for stratification"
  runs: ["runs/active_pilot/retro/"]
  runtime: "2026-10-06; paused by the user the same day"
budget:
  declared: "not recorded (exploration session)"
  spent: "~1 day"
  outcome: stopped-early
design:
  varied:
    arm: "base 2000, random +500, random +1000, ens-var +500, p(1-p) +500, epistemic ratio +500, decision score +500, diversity (FPS) +500, oracle true-error +500"
  held_fixed:
    model: "zoom64 [4,8,16], pretrained init, 150 ep, identical recipe per arm"
  baselines: "base (no added rows = do-nothing); random +500 / +1000"
  metric: "swept-region squared error e_sw (window frame), window accuracy, slateN (window and pasted)"
noise_floor: "retrain sd of e_sw 0.01-0.05 (0.5-2.5 %), of slateN 0.002-0.008; selection variance NOT measured (one draw)"
depends_on: [score-occupancy-subpixel-stable]
establishes: []
result: >
  DS-0017 e_sw (mean +- sd over 3 retrain seeds; results/active_collection_plan.md v2.2, active_retro.json): base 2.171,
  random +500 2.088, random +1000 2.025, ens-var +500 1.971, p(1-p) +500 1.952 (paired vs random +500: -0.117 / -0.136,
  2-4 seed sd); oracle-by-true-error +500 2.003 (not better than the proxies); decision score 2.051, diversity 2.063 (~random).
  Window slateN 0.945-0.954 and pasted slateN 0.741-0.747 for every arm (differences <= 0.006). ~42 % of the raw error is a
  simulator noise floor (re-sim replicates, active_reducible.json). Live collection loop NOT run (NO-GO for the 15k-sim study).
verdict: supported
downgrades: [imprecision, indirectness]
grade: low
supersedes: []
invalidated_by: null
---

## Why this test discriminates
The random +500 / +1000 arms give a data-efficiency curve; a mined arm below random +1000 means the proxy picks rows worth
more than twice their count. The oracle arm (true error) tests whether noise rows dilute raw-error mining.

## What was actually run
Retrospective (zero sim cost) loop only; `results/active_collection_plan.md` ("v2 -- READ THIS FIRST") is the running record.
The README said no EXPERIMENT.md until a later validation round; written 2026-10-15 from those files (RAL run plan T0d), no new runs.

## What would change the verdict
A second selection draw (selection variance unmeasured); a live round with new simulations; any slateN or closed-loop gain
(none seen: the claim is about prediction error, an indirect proxy for control).

## Threats
Pilot domain; window-frame metrics; ensemble members share data with the scored model family.

## Unrelated findings
Plain 3-seed ensembling removes about as much reducible error (-13 %) as 2-3x more random data, at no simulation cost (v2.1).
