# RUN-0001 — LeJEPA Stage 1 encoder training
- experiment: EXP-0019 · commit 6ea03278 (tree dirty — see EXPERIMENT.md)
- command: see `COMMAND.txt`; `python -u code/train_encoder.py --epochs 12 --steps-per-epoch 250 --bs 192 --lambda-sigreg 0.02 --sean-files 60 --out artifacts/RUN-0001`
- data: `Genesis/data/overnight_randlen_train` (192 files) + 60 strided `Genesis/data/Sean/**/*_data.pt` → 214688 states
- device: CUDA, RTX 4070 Laptop 8 GB; peak 5785 MiB; wall 1393 s; status: completed
- outputs: `artifacts/RUN-0001/{encoder_lejepa.pt,train_curves.json}`, log `runs/RUN-0001/stdout.log`
