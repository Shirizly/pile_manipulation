# EXP-0072 zoom-window NFD (higher-resolution inputs)

Status: **ACTIVE**, exploration mode, pilot-grade (narrow n20 exact-20 mm data only). There is deliberately **no EXPERIMENT.md and no REGISTER entry** yet;
the proper record comes later (experiments/TODO.md, MID, EXP-0073 item (h)).

What it is: NFD whose input is a high-resolution zoom window around the push (64 / 128 px windows, soft/hard thresholds, pasted back to the world for slateN) instead of the 1 mm/px world raster.
Read `results/NOTES.md` first (scores, timings, narrative) and `results/ceiling/results.md` (accuracy ceiling and simulator noise-floor study; code `code/ceiling_*.py`, artifacts `artifacts/ceiling/`).
Code: `score_zoom.py`, `train_zoom.py`, `make_cache*.py`, `hr_*.py`, `time_zoom*.py`, `figures_zoom.py`; window libs in `model/zoom_nfd/`.
The active / hard-example data-mining study that started here was split out to `experiments/EXP-0073-active-data-mining/` (2026-10-06); it reads EXP-0072's models, `window_cache.pt` and `artifacts/ceiling/resim_*`.
