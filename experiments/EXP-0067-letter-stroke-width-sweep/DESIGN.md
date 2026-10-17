# EXP-0067 — Are thin letter goals a tool-geometry limit? Stroke-width sweep in legal closed loop

Written 2026-10-05 09:10 CEST, BEFORE any scored cell ran. Done before this: `code/calibrate_width.py` (CPU, mask
geometry only), `code/vstar_width.py` (CPU placement optimum per widened goal), and a cost pilot (`results/pilot/`,
1 chunk x 2 pushes on letter_O_w10, timing + "driver accepts the new goal name" only; no outcome read). Hard budget:
7 h wall (ends 15:58 CEST), <= 5 GPU-h, one GPU job at a time (`experiments/temp/gpu.lock`).

## Question
No planner, including the simulator-as-model CEM (EXP-0051, EXP-0057), has ever completed a `helvetica_thin` letter
(0.9 x the EXP-0046 placement optimum). The strokes are ~4 px = 8.1-8.6 mm on the 64 px / 128 mm scoring grid
(realised; the brief's "7.5 mm" is nominal), cubes are 5 mm, the blade 40 mm. If completion rises with stroke width
at fixed planner/model/objective, the letters are a geometry limit; if it stays near 0 at 20 mm strokes (4 cube
widths, half a blade), the planner/objective is at fault.

## Goals (new names, existing names untouched)
`letter_<L>_w<mm>` for L in O, T, S, X and nominal width 10 / 15 / 20 mm: the 720 px glyph's ink is dilated by d asset
px BEFORE the INTER_AREA resize (`Baselines/common/goals.py::widened_letter_mask`; the asset is a chamfer distance
field, so dilation = `max(field - d, 0)`; d = 0 is byte-identical to `letter_mask`). d per nominal width was chosen by
the rule fixed in `code/calibrate_width.py` (argmin over d of |mean over O/T/S/X of the realised median-skeleton
stroke width - target|): **d = 6 / 18 / 31 -> realised mean 10.8 / 15.1 / 19.8 mm** (`results/width_calibration.json`).
Realised per letter (median skeleton width, mm; EXP-0046 vstar.py's measure):
| letter | base (w7.5) | w10 | w15 | w20 |
|---|---|---|---|---|
| O | 8.1 | 10.3 | 14.7 | 20.3 |
| T | 8.1 | 12.2 | 16.3 | 20.3 |
| S | 8.1 | 9.1 | 14.7 | 18.2 |
| X | 8.6 | 11.5 | 14.7 | 20.3 |
The 64 px grid quantises width in ~2 mm steps, so "10 mm" is 1-2 px wider than the base, and S_w10 is the smallest step
(+1 px median, +35 % mask area). Placement optimum per goal recomputed with EXP-0046's solver
(`results/vstar_width.json`, identity check vs EXP-0046 exact for the 4 base letters): `mass_frac_best_placement`
base 0.956 / 0.965 / 0.938 / 0.949 (O/T/S/X) -> w10 0.990 / 0.985 / 0.986 / 0.989 -> w15 >= 0.9994 -> w20 1.000.
Capacity (non-overlapping 5 mm cubes inside the mask) base 32/22/30/21 -> w10 43/34/38/27 -> w15 70/57/68/57 -> w20 106/82/105/87.

## Cells (learned-model sweep; one driver call per width = 4 letters x 32 starts = 128 episodes = 4 chunks)
Settings identical to EXP-0066 (`code/queue.sh` there): driver `experiments/EXP-0043-batched-closed-loop/code/
batched_closed_loop.py`, model `nfd_3ch_randlen` (broad NFD, EXP-0066's best context cell), CEM pop 1024 / elite
0.25 / 0.5 s / n_cand 1024 / bank 16384, 16 pushes of 20 mm, DS-0006 scatter starts 40-71, TRAINING_PHYSICS executor,
`--legalize --legalize-fallback --planner-seed 0 --record-states`.
| order | tag | width | note |
|---|---|---|---|
| (reused) | EXP-0066 `broad_s0_ps0` | base 7.5 (realised 8.1-8.6) | same model, settings, starts, planner seed (config verified field by field); env-slot assignment differs (4 goals x 32 starts there too, so identical) |
| 1 | w10 | 10 | smallest step first |
| 2 | w15 | 15 | |
| 3 | w20 | 20 | |
**n = 32 starts per (letter, width) is FIXED from the cost pilot** (25 s per push per chunk of 32 envs -> ~8 min per
chunk incl. start-up, ~32 min per width, ~1.6 GPU-h for the sweep).

## Anchor: perfect-model (simulator) CEM, EXP-0057 `oracle_A_default` settings
`experiments/EXP-0050-oracle-ceiling/code/oracle.py --mode cem --cem-pop 64 --cem-iters 4 --cem-elite 16 --n-envs 128
--seed-base 0 --stop-solved` (256 sims per push, push length 20-70 mm as in EXP-0057, 16 pushes here vs 20 there --
the per-(task, push, iteration) seeding makes pushes 1-16 identical to a 20-push run). Baseline width: REUSED from
EXP-0057 `oracle_A_default` (letter_O, letter_T, starts 40, 41; read at k = 16). Budget: ~1.5 h per anchor cell
(8 tasks x 16 pushes x 256 sims / 6 sims/s), run only after the learned sweep:
| priority | tag | goals | starts | run if |
|---|---|---|---|---|
| A1 | oracle_w15 | letter_O_w15, letter_T_w15 | 40, 41, 42, 43 | sweep done and started before 12:30 CEST |
| A2 | oracle_w10 | letter_O_w10, letter_T_w10 | 40, 41, 42, 43 | A1 done, started before 13:30 CEST and cumulative GPU time < 3.5 h |
Deviation from the brief (8 starts): 8 starts x 2 widths is ~6 GPU-h on its own; 4 starts per cell, paired with the
EXP-0057 baseline on starts 40/41, is what fits. Skipped cells are stated as skipped.

## Metrics per cell (learned sweep; `code/analyse.py`, written before the sweep finished)
rel_k = in-goal mass fraction at push k (`occ_for_scoring` of the recorded states, `mass_in_region`) / the goal's
`mass_frac_best_placement` at THAT width (`results/vstar_width.json`; EXP-0046 `vstar.json` for the base letters).
- rel at k = 8 and k = 16: median and 5 / 95 % quantiles over the 128 episodes (and per letter); the EXP-0066 score
  (mean of rel over pushes 4-16) beside it for continuity.
- completion rate at theta = 0.9 x optimum: fraction of episodes with max_{k <= 16} rel_k >= 0.9, with a Wilson 95 %
  CI; also at k <= 8.
- pushes to completion: first k with rel_k >= 0.9, censored at 16 (Kaplan-Meier median; reported as "> 16" if the
  survival curve never crosses 0.5).
- fallback-fire rate and executed-illegal count (must be 0).
- Secondary, unregistered in the hypotheses: the same at theta = 0.8, and the start's own rel_0.

## Hypotheses (pre-registered)
- **H1** completion rate at 0.9 rises monotonically with width over the 4 cells (base, w10, w15, w20): every
  successive difference >= 0 and the base-to-w20 difference's Wilson/Newcombe 95 % CI excludes 0.
- **H2** the smallest width whose completion rate exceeds 50 % lies in [10, 20] mm nominal (realised 10.8-19.8 mm),
  i.e. 2-4 cube widths, well below the 40 mm blade.
- **Embarrassing result:** completion stays below 10 % even at w20 (4 cube widths, optimum 1.0, capacity > 80 cubes).
  Then the letters are NOT a geometry limit at this planner/model/objective and the fault is in the planner or the
  objective; it is reported as such.
- **Opposite result:** w10 already completes > 50 %: the thin letters were a near-miss and the goal library needs
  only a 1-2 px widening.
- The anchor (if run) is read as: does the perfect model complete w15 where the learned model does / does not; it is
  descriptive (4 tasks per cell) and carries no hypothesis.

## Noise floor
EXP-0066: planner-seed replicate sd of a 128-episode cell mean 0.0065; per-episode score sd ~0.11; broad model cell
means 0.364 / 0.390 (two training seeds). A completion-rate difference between two 128-episode cells needs ~0.12
to clear a Newcombe CI near 50 %, ~0.05 near 0-10 %. The widths are not paired by start in the ordinary sense (same
starts, different goals), but the same 32 starts are used in every cell, so start identity (37 % of episode variance
in EXP-0066) is matched across cells.

## Budget rule
Learned sweep cells run in order w10, w15, w20; any cell not started by 13:00 CEST is skipped. Oracle anchors as above.
Everything is checkpointed per push (driver) and resumable by tag. Analysis runs on whatever completed.
