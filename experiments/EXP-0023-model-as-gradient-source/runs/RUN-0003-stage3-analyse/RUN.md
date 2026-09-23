# RUN-0003 — stage 3, metrics

## Exact argv
```
PYTHONPATH=. python -u \
  experiments/EXP-0023-model-as-gradient-source/code/stage3_analyse.py \
  --stage1 .../artifacts/stage1_actions.pt \
  --stage2 .../artifacts/stage2_genesis.pt \
  --out    .../results/metrics.json
```

## Sign convention and the deviation from `DESIGN.md`
`dv` is a COST. `DESIGN.md`'s literal `gradient_gain = dv_grad - dv_rank`
contradicts its own gloss ("negative = worse") under that convention, so the
three difference metrics are negated to preserve the design's MEANING:
`gradient_gain = dv_rank - dv_grad`, `pool_escape = pool_ceiling - dv_grad`,
`regret_vs_oracle = dv_grad - dv_oracle`, all "positive = better".
`capture_vs_oracle = dv_grad / dv_oracle` is unchanged.

## Floors
`random` is the mean true `dv` over the shared 100-action seed pool;
`pool_ceiling` is that pool's minimum (best) true `dv`. `persistence` is NOT
used — it predicts `dv = 0` for every candidate, so its `argmin` returns row
order rather than a prediction (`scripts/probes/binned_pool_cache.py`,
`docs/CODEMAP.md`).

## Known-number reproduction
19 seed-pool rows per state (190 total) are re-executed in Genesis from the
restored snapshot and compared against `experiments/temp/binned-pools/
dv_cache_corner.pt`'s `dv_true`, which was produced by the corpus collection
itself through a different code path. Reported in `results/metrics.json`
under `cache_reproduction`.

## Power check
`between_arm_sd_of_means` (sd of the 6 arm means of `gradient_gain`) against
`between_state_sd_of_means` (sd of the 10 state means). If the former is not
larger, the design cannot separate arms at this n and the record says so.
