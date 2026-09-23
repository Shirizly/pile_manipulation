# RUN-0002 — encode overnight_randlen_{train,test} with the trained encoder
- experiment: EXP-0019 · commit 6ea03278 (dirty)
- command: `python -u code/encode_states.py --ckpt artifacts/RUN-0001/encoder_lejepa.pt --out artifacts/RUN-0002`
- 98304 train / 10752 test transitions; 75.0 s + 8.2 s on CUDA; status: completed
- outputs: `artifacts/RUN-0002/latents_{train,test}.pt` (z0, z1, a, length_m, encoder config + ckpt path)
