---
name: visualization
description: "How figures are made and what the user prefers in this repo: native model resolution, display orientation, goal outline, prediction-error rows, optimisation-trace plots, where the plotting helpers live. Use whenever you produce a figure of rasters, pushes, model predictions, rollouts or optimisation progress, or before regenerating one."
user-invocable: true
---

# Visualisation conventions (collected from EXP-0072..0075 reviews)

## Preferences (user-stated)
* **Show model input/output at the models' NATIVE resolution.** The zoom128 / vanilla128 NFDs work at 128 px; do not display their predictions on the 64x64 pasted frame unless the figure is about the planner's objective (which does use 64x64). Prefer the higher resolution wherever it represents the model input/output correctly. Action sequences are saved with each result (`seq` in results JSON), so re-predicting at 128 is cheap (`EXP-0075.../code/fig128.py: predict128, truth128`).
* **Prediction error = predicted state - true state at every step** (accumulating over a sequence is fine). Do NOT mix prediction error with "change" panels; the bottom row of a trajectory figure is `pred_k - true_hard_k`. Red = model has mass the truth lacks, blue = reverse. Compare against truth rendered in the MODELS' raster convention (`truth128(inflated=True)`: training-style cubes inflated ~28 % in area); exact footprints (`raster_res`) make every unmoved cube show up as a blue rim, which is a rendering artefact, not model error. Say which truth was used.
* **Optimisation-process figures**: predicted objective of the best-so-far sequence vs sequence evaluations (log x), one line per method, ceiling and 90 % lines marked (`code/figures_traces.py`).
* Always state in a title/caption what was executed (sequence, simulator vs model) and the numbers (predicted vs true value change).

## Conventions
* Display orientation: world x right, world y DOWN (letters read upright): `imshow(img.T, origin='lower', extent=[LO,HI,LO,HI])`, `set_ylim(HI, LO)`, extents in mm from `OCC_BOUNDS`.
* Goal = blue contour of `goal_mask(goal)`; pushes = arrows start->end with a dot at the start, colour per push index, number at the start.
* Gray_r colormap, vmin 0 vmax 1 for occupancies; red/blue `err_rgb(d)` for signed differences (alpha ~ 1.5|d|).
* Truth after a simulator push: `raster_res(particles, SIZES(n), 128)` (hard) or `occ_for_scoring` (soft, what the objective scores). Save particle states (not only rasters) from simulator replays so any resolution can be rendered later.
* matplotlib Agg, dpi ~90, `tight_layout`, one PNG per sequence plus one summary; keep titles short enough not to overlap (shorten or wrap).

## Code pointers
| what | where |
|---|---|
| trajectory figures (rows: true / predicted / error), helpers `show`, `goal_outline`, `arrow`, `err_rgb` | `experiments/EXP-0075-closed-loop-benchmark/code/figures_diag.py` |
| intensive-plan simulator figures (prediction-error row) | `.../code/figures_intensive_sim.py`, `figures_intensive_sim_128.py` |
| optimisation traces | `.../code/figures_traces.py` |
| native-128 predictions and truth | `.../code/fig128.py` |
| per-method plan figures of the intensive search | figure section of `.../code/intensive.py` |
| rasterisers | `model/zoom_nfd/world_res.py` (`raster_res`), `simple_mpc/adapters.py` (`occ_from_particles`, `occ_for_scoring`, `OCC_BOUNDS`) |
