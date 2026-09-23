# RUN-0001 — LeJEPA Stage 1 with SIGReg on BOTH p and z
- experiment: EXP-0020 · commit 6ea03278 (dirty)
- command (recorded, see COMMAND.txt): `OMP_NUM_THREADS=4 python -u ../EXP-0019-lejepa-encoder-pushlen-switched/code/train_encoder.py --epochs 12 --steps-per-epoch 250 --bs 192 --lambda-sigreg 0.02 --sigreg-on-z 0.02 --sean-files 60 --seed 0 --latent-dim 256 --proj-dim 256 --lr 1e-3 --out artifacts/RUN-0001`
- code: EXP-0019's `train_encoder.py` UNCHANGED; only the pre-existing, never-before-run `--sigreg-on-z` flag differs from EXP-0019 RUN-0001.
- data: 252 files = 192 `overnight_randlen_train` + 60 strided `Genesis/data/Sean/**/*_data.pt` -> 214688 states.
- 1406 s on CUDA (RTX 4070 Laptop 8 GB), peak 5811 MiB; status: completed.
- outputs: `artifacts/RUN-0001/{encoder_lejepa.pt,train_curves.json}` (checkpoint carries state_dict + config + projector + train_args + curves).
