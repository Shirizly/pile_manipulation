# RUN-0013 — DS-0021/22 wired into FlexData + switched LinearForesight fit on DS-0021 → MODEL-0013

- what: (1) made DS-0021 / DS-0022 readable by FlexData: `FlexData/build_cache.py` builders parameterised by
  dataset dir (new commands `ds0021|splits_ds0021|paths_ds0021`, `ds0022|splits_ds0022|paths_ds0022`; DS-0020/19
  defaults unchanged) and `FlexData/image_mask.py` entries `ds0021|ds0022`; instance `config.yaml` in each DS dir
  (grid/plate/flags identical to DS-0020 v2 / DS-0019, `occupancy.source: image_mask`). Split = EXP-0064
  GroupedParticleDataset rule (per count group, first 90 % of sorted states train). (2) render check
  (`code/flex_cache_check.py`, see DS-0021/22 DATASET.md). (3) EXP-0062 LF recipe on DS-0021 train, (scheme, lambda)
  picked on DS-0021 val only.
- commands (exact argv + provenance in `artifacts/RUN-0013-lf-fit/*.json` / COMMAND.txt, ledger `runs/COMMANDS.jsonl`):
  - `python scripts/run_probe.py --tag exp0064_run0013_caches ... -- bash experiments/EXP-0064-obj-count-effect-study/code/build_flex_caches_ds0021_ds0022.sh` (~5 min, 8 workers; 2 earlier attempts failed on a missing cache dir, fixed by mkdir in the script)
  - `python scripts/run_probe.py --tag exp0064_run0013_cache_check -- python experiments/EXP-0064-obj-count-effect-study/code/flex_cache_check.py`
  - `python scripts/run_probe.py --tag exp0064_run0013_lf_fit --threads 8 --artifact-dir weights/MODEL-0013-... -- python experiments/EXP-0064-obj-count-effect-study/code/fit_lf_ds0021.py --device cpu --out weights/MODEL-0013-linear-foresight-flex-mask-countgroups/checkpoint.pt`
- commit 3373e65b, dirty (this run's FlexData/DS-config edits + other sessions' files); CPU only (GPU held by the
  overnight GNN queue); 2026-10-04 00:00-00:10 CEST.
- data: DS-0021 train 15,193 kept rows / val 1,741 (flags: escaped, out_of_grid, null dropped).
- result: equal-width bins, lambda 300 switched (val accuracy 0.3596; collection edges 0.3557), single lambda 300 (0.2660);
  fit_bins vs Gram max |diff| 1.9e-6. Render check: frame peak (0,0); 99.7 % / 99.9 % of removed mask pixels inside
  the swept region (DS-0021 / DS-0022) vs 47 % / 55 % with z negated.
- outputs: `weights/MODEL-0013-linear-foresight-flex-mask-countgroups/` (checkpoint.pt, fit.json); dataset caches under
  `datasets/DS-0021-*/cache/`, `datasets/DS-0022-*/cache/`; `results/flex_cache_check.json`, `results/figures/data_check/`.
