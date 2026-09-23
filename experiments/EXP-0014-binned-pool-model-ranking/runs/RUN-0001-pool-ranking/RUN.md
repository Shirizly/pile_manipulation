# RUN-0001 — build the dV cache and survey all five rankers

- **experiment:** EXP-0014
- **dataset:** DS-0001 (`Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`), step 0, all 20000 rows
- **models:** MODEL-0001, MODEL-0002, MODEL-0003 (all reused unmodified; nothing fitted here) + `persistence`, `random`
- **commit:** 6ea03278, dirty (see EXPERIMENT.md)
- **seed:** 0
- **status:** completed

## Commands (recorded, not reconstructed)

```
PYTHONPATH=. python -u scripts/probes/binned_pool_cache.py \
    Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm \
    --out experiments/temp/binned-pools/dv_cache_corner.pt

PYTHONPATH=. python -u scripts/probes/pool_survey.py \
    experiments/temp/binned-pools/dv_cache_corner.pt --goal corner \
    --models random,persistence,descriptor,nfd,visual-switched \
    --k-rank 32 --out experiments/temp/binned-pools/survey_corner.json

# six figures, one per (model, {typical,worst}) pool named by the survey:
PYTHONPATH=. python -u scripts/probes/pool_inspect.py \
    experiments/temp/binned-pools/dv_cache_corner.pt --goal corner \
    --slate <s> --models random,<model> --out <...>.png
```

## Outputs

- dV cache (artifact, large, kept in `experiments/temp/binned-pools/dv_cache_corner.pt`; regenerable by the first command above)
- `results/survey_corner.json` — the population statistics, including `slateN` per model
- `results/figures/*.png` + `*.log` — six per-pool figures with their numeric blocks

## Notes

The cache embeds `occ0` per slate and the workspace bounds, because DS-0001's
`step{k}.pt` layout cannot be read by the `PileSweepData` reload path the two
older cache builders use. `pool_common.load_occ0_for_slate` prefers the
embedded copy when present.
