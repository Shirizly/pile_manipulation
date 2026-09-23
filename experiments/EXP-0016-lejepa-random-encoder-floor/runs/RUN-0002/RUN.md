# RUN-0002 — switched-linear latent dynamics sweep (K × seed)

- experiment: EXP-0016 · run: RUN-0002 · status: completed
- commit `6ea03278`, dirty: true
- input: RUN-0001's latents (no refit of the encoder; the encoder object is not
  even constructed here)
- device: cuda:0 (asserted)
- outputs: `artifacts/RUN-0002/{dyn_K{1,4,8}_seed{0,1,2}.pt, dynamics_metrics.json}`
  — **every fitted operator persisted** with its config, the z-normalisation
  (`mu`, `sd`) it was fitted under, and its own metrics, so a later cell reloads
  rather than refits.

## Command (recorded)

```
OMP_NUM_THREADS=4 PYTHONPATH=/home/alon/Code/pile_manipulation \
  /home/alon/anaconda3/envs/pme/bin/python -u code/fit_dynamics.py \
  --latents artifacts/RUN-0001 --ks 1,4,8 --seeds 0,1,2 --epochs 60 \
  --out artifacts/RUN-0002
```

## Resolved config

AdamW lr 3e-3 wd 1e-4, OneCycleLR, batch 1024, 60 epochs, MSE on Δz.
z fed to the model as (z − mean_train)/std_train(scalar); Δz predicted in raw
latent units, so the reported R² is in the untransformed latent space.
Baseline `Δz = 0` evaluated in the same space (R² = 0 by construction).

## Result

test R² vs Δz=0, mean ± sd over 3 seeds: K=1 +0.2483 ± 0.0004,
K=4 +0.3683 ± 0.0024, K=8 +0.3726 ± 0.0012. 19.7–22.4 s and 248–278 MiB per fit.
