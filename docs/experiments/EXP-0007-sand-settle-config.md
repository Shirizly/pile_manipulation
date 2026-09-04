---
id: EXP-0007
title: A duplicate YAML key capped the sand settle at 100 steps, biasing every recorded state by ~1.3 mm
tier: T1
mode: exploratory
date: 2026-09-04
hypothesis: null
claim: >
  Genesis/configs/sand.yaml carried two top-level `simulation:` blocks, so
  settle_steps ran at the code default of 100 instead of the declared 2500, and
  every sand transition collected before 2026-09-04 was recorded before the pile
  finished moving -- by 1.3 mm of mean grain displacement, ~11% of the 10-15 mm
  a push moves.
prediction: null
provenance:
  commit: 901ba35e
  script: "scripts/sand_physicality_video.py, scripts/sand_settle_decay.py"
  data: ["Genesis/configs/sand.yaml", "live sim, 1 env, 1912 grains"]
  code_path: sand_to_mask
  seed: 0
  split: n/a
  runtime: "~6 min video (3 eps x 5 pushes, 2 cameras) + ~90 s decay probe, RTX 4070"
design:
  varied: {settle_steps: [100, 2500, 3000], phase: [post-spawn, post-push]}
  held_fixed: {config: configs/sand.yaml, n_envs: 1, particle_size: 0.002, substeps_mpm: 30, dt: 0.004, push_length: 0.02, E: 100000.0, settle_velocity_threshold: 0.001, settle_rest_quantile: 0.995}
  baselines: [persistence, mean-delta]
  metric: "q=0.995 grain speed (mm/s) after the settle, and mean grain displacement between step 100 and step N"
noise_floor: "not applicable to the config defect, which is exact; for the 1.3 mm bias, one pile and one push were measured, so treat it as an order of magnitude and not a calibrated number"
depends_on: [sand-projection]
establishes: [config-keys-reach-sim, settled-state]
result: "settle_steps 2500 -> 100 silently; post-push q0.995 speed 1.905 mm/s at step 100 vs 0.811 at 3000; pile moves a further 1.314 mm mean after step 100; mass 1.0000 and floor containment hold throughout"
verdict: supported
downgrades: [imprecision]
grade: moderate
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The config defect is not a statistical claim: either the parsed `simulation`
block contains `settle_steps` or it does not, and it did not — the parsed block
was `{'substeps_mpm': 30}` and nothing else. What needed measuring was the
consequence, and that has two possible shapes. If the pile were nearly settled
at step 100, the collected data would be fine and this would be a tidiness fix.
If the pile were still rearranging, every recorded `s'` is early by a knowable
amount, and because each transition's `s` is the previous `s'`, the error
compounds along an episode.

## What was actually run

Both `simulation:` blocks are visible in the file; `yaml.safe_load` keeps the
second. `dt` (4e-3) and `substeps` survived by coincidence, because the code
defaults match what the discarded block declared — `settle_steps` is the only
consequential loss.

The decay probe pushes once, then steps the scene with no further action,
recording speed quantiles and cumulative displacement against step 100 (where
the collected data was recorded).

Physicality was checked separately over 3 episodes x 5 pushes with the cap
fixed: mass, floor containment, rest speed, displacement and launching, plus
side-by-side oblique/overhead video (`outputs/sand_physicality/`).

## Numbers

Post-push settle decay, 1912 grains, one pile:

| step | q0.50 | q0.90 | q0.995 | max | drift (um/step) | moved since 100 (mm) |
|---|---|---|---|---|---|---|
| 0 | 0.376 | 1.502 | 2.149 | 2.561 | — | — |
| **100** | 0.058 | 0.238 | **1.905** | 2.419 | 0.99 | **0.000** |
| 500 | 0.047 | 0.257 | 1.619 | 2.341 | 0.54 | 0.211 |
| 1000 | 0.046 | 0.242 | 1.275 | 1.778 | 0.48 | 0.456 |
| 2000 | 0.046 | 0.274 | 1.044 | 1.644 | 0.45 | 0.908 |
| **3000** | 0.049 | 0.249 | **0.811** | 1.134 | 0.40 | **1.314** |

Physicality with the cap fixed, 15 pushes across 3 episodes:

| check | result |
|---|---|
| mass in tray | 1.0000 on every push |
| min grain z | 10.0 mm, exactly the floor — no leakage |
| post-settle q0.995 | 0.98-1.00 mm/s, i.e. it now converges |
| highest grain | 14.5 mm (floor 10.0) — nothing launched |
| mean displacement | 0.25-14.84 mm |

Two readings that matter beyond the bug:

1. **Sand creeps and does not stop.** The median grain is at rest (0.046 mm/s)
   while the top 0.5% keeps moving, and drift/step flattens at ~0.4 um/step
   rather than reaching zero. The `q=0.995 < 1 mm/s` criterion was inherited
   from the rigid path ("matches Genesis' hibernation default") and is a test of
   the moving tail, not of the pile. Post-*spawn* settles never pass it at all
   (2.12-2.22 mm/s across three episodes), because a sampled cylinder collapsing
   is a much larger event than a push.
2. **Late-episode sand pushes do almost nothing.** Mean displacement per push
   fell to 0.25 mm and 0.59 mm (episode 2, pushes 4-5) with pile extent at
   63 mm, and 1.42 mm (episode 1, push 5) at 82 mm extent. The pile spreads into
   a thin sheet the blade barely engages. Combined with the separately measured
   flattening to ~1.2 grain layers by push 5, a 5-push sand episode yields
   roughly 2-3 informative transitions and 2-3 near-no-ops.

## What would change the verdict

The 1.3 mm figure is one pile and one push. Repeating it over ~20 pushes from
varied starts would turn it into a calibrated bias with a spread; ~20 min.

Whether the bias matters for the fitted operators is a separate question this
does not answer: re-collect the 48 000-transition set with the fix (3.5 h) and
re-run `sand_full_analysis.sh`. If the margins move by less than the ~0.05 noise
floor, the existing sand conclusions stand as measured and this becomes a
caveat rather than a retraction.

## Threats

- `imprecision`: the bias is a single measurement, and the drift is not
  monotone step-to-step (q0.995 wanders 1.58-1.66 between steps 300 and 700),
  so 1.3 mm should be read as "about a millimetre", not as 1.314.
- Considered and dismissed: **that `dt` or `substeps` were also lost.** They
  were in the discarded block, but the code defaults are identical to the
  declared values, so the simulation ran as intended in those respects.
- Considered and NOT dismissed: the once-per-process `_settle_cap_warned` guard.
  The 48 000-transition run warned once, during build, and then went silent for
  all 300 episodes. The warning is correct and its suppression is reasonable for
  log volume, but it means "no warning in the log" does not mean "the settle
  converged" — only that it converged or already complained. Any future check
  should read the residual per transition rather than trust the absence of a
  warning.
