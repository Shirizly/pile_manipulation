# RUN-0003 — post-hoc occupancy decoder on the frozen random encoder

- experiment: EXP-0016 · run: RUN-0003 · status: completed
- commit `6ea03278`, dirty: true
- input: RUN-0001's cached z (detached, loaded from disk) and occupancy subset
  (24576 train / 6144 test pairs); RUN-0002's K=1 and K=8 seed-0 operators
- device: cuda:0 (asserted on z and occ)
- outputs: `artifacts/RUN-0003/{decoder.pt, decoder_metrics.json}`

**No gradient can reach the encoder**: this script never constructs the encoder,
and asserts `not z.requires_grad` on its inputs.

## Command (recorded)

```
OMP_NUM_THREADS=4 PYTHONPATH=/home/alon/Code/pile_manipulation \
  /home/alon/anaconda3/envs/pme/bin/python -u code/train_decoder.py \
  --latents artifacts/RUN-0001 --dyn-dir artifacts/RUN-0002 --epochs 40 \
  --out artifacts/RUN-0003
```

## Resolved config

`OccDecoder(latent_dim=256, out_resolution=64)`, AdamW lr 2e-3 wd 1e-4,
OneCycle, batch 256, 40 epochs, MSE against the soft (splat) occupancy.
`accuracy` uses `fit_linear_foresight.swept_region_mask(half_width_px=6,
pad_px=3)`, which covers 8.1 % of the frame, with persistence (occ0) as the
baseline, per `experiments/METRICS.md`.

## Result

train mse 0.00378, test recon mse 0.0331 (~9× gap). Autoencoding accuracy vs an
empty frame −0.111. Swept-region one-step accuracy +0.137 (K=1) / +0.191 (K=8),
decode-the-true-next-latent ceiling +0.174. Full-frame accuracy −1.00 for all
cells including the ceiling. 256 s, 1616 MiB at batch 256.
