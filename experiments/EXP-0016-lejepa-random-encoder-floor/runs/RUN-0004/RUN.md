# RUN-0004 — slateN on DS-0001 through the decoder readout

- experiment: EXP-0016 · run: RUN-0004 · status: completed (3 invocations,
  one per dynamics seed)
- commit `6ea03278`, dirty: true
- data: DS-0001 — `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`,
  20 slates × 1000 candidates, scatter spawn, step 0. No split; a fixed
  evaluation pool, as EXP-0012/0013/0014 use it. Nothing here was fitted on it.
- device: cuda:0 (asserted on the occupancy tensor)
- outputs: `results/slaten_corner.json` (dyn seed 0),
  `results/slaten_corner_dynseed{1,2}.json`

## Commands (recorded)

```
OMP_NUM_THREADS=4 PYTHONPATH=/home/alon/Code/pile_manipulation \
  /home/alon/anaconda3/envs/pme/bin/python -u code/score_slaten.py \
  --enc artifacts/RUN-0001/encoder_seed0.pt --dec artifacts/RUN-0003/decoder.pt \
  --dyn-dir artifacts/RUN-0002 --out results/slaten_corner.json
# and --dyn-seed 1 / --dyn-seed 2 with --out results/slaten_corner_dynseed{1,2}.json
```

## Code path

Reused from `scripts/probes/binned_pool_cache.py` (EXP-0014's builder):
`BinnedSlateCorpus`, `particles_to_occupancy` with the same BOUNDS/GRID/RADIUS,
`control_utility_test.lyapunov`/`lyapunov_weights`, the
`dv = value(after) − value(before)` convention, and the `random` baseline built
from `torch.Generator().manual_seed(0)`. `slateN` itself from the canonical
`Baselines/common/goals.py::slate_n_capture`, `higher_is_better=False`.

**Provenance check:** `persistence` (−0.0042, sem 0.0892) and `random`
(−0.1409, sem 0.1007) reproduce EXP-0014's rows to 4 decimal places.

## Result

slateN (mean over 3 dynamics seeds): K=1 +0.4191, K=4 +0.4117, K=8 +0.3765;
decode-z0 (no dynamics) −0.0042; persistence −0.0042; random −0.1409.
frac(dv_true == 0) = 0.031. ~95 s per invocation.
