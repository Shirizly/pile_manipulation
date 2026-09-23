# RUN-0001 — encode the transition corpus with a frozen random encoder

- experiment: EXP-0016 · run: RUN-0001 · status: completed
- commit `6ea03278`, dirty: true (see EXPERIMENT.md "What was uncommitted")
- data: `Genesis/data/overnight_randlen_train` (192 files, 98304 rows),
  `Genesis/data/overnight_randlen_test` (21 files, 10752 rows) — the PRE-EXISTING
  split, reused. Invariant `randlen-train-test-file-disjoint` holds.
- seed: 0 (`torch.manual_seed` + `torch.cuda.manual_seed_all` before encoder construction)
- device: cuda:0 (asserted on the occupancy tensor inside the loop)
- outputs: `artifacts/RUN-0001/{encoder_seed0.pt, latents_{train,test}.pt,
  occ_sample_{train,test}.pt, encode_cost.json}`

## Command (recorded)

```
cd experiments/EXP-0016-lejepa-random-encoder-floor
OMP_NUM_THREADS=4 PYTHONPATH=/home/alon/Code/pile_manipulation \
  /home/alon/anaconda3/envs/pme/bin/python -u code/encode_corpus.py \
  --seed 0 --out artifacts/RUN-0001
```

## Resolved config

encoder `ResCNNEncoder(input_resolution=64, latent_dim=256, n_res_blocks=2,
channels=[32,64,128,256])`, 3,593,792 params, frozen (`requires_grad_(False)`
on every parameter) and never optimised. Rasteriser
`transforms.functional.particles_to_occupancy`, BOUNDS ±0.064 m, GRID 64,
footprint_radius = 0.5·0.005/pitch — the same constants
`scripts/probes/binned_pool_cache.py` uses. batch 512.

## Result

train 87.5 s, test 8.8 s, peak 1064 MiB. z per-dim std mean 0.0531;
‖dz‖/‖z‖ = 0.0941.
