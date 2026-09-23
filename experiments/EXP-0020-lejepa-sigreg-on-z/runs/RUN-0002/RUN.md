# RUN-0002 — encode the transition corpus with the RUN-0001 encoder
- experiment: EXP-0020 · commit 6ea03278 (dirty)
- command: `OMP_NUM_THREADS=4 python -u ../EXP-0019-lejepa-encoder-pushlen-switched/code/encode_states.py --ckpt artifacts/RUN-0001/encoder_lejepa.pt --out artifacts/RUN-0002` (script UNCHANGED)
- 98304 train / 10752 test transitions, deterministic rasterisation, 75.5 s + 8.2 s on CUDA; status: completed.
- outputs: `artifacts/RUN-0002/latents_{train,test}.pt` (z0, z1, a, length_m, encoder_config, ckpt).
