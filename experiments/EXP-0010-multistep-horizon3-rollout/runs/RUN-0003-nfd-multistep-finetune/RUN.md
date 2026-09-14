# RUN-0003-nfd-multistep-finetune

Fine-tunes NFD (a ~30k-parameter 3-channel UNet, init from
`Baselines/NFD/runs/nfd_3ch/unet_best.pth`) through the SAME closed-loop
3-step rollout objective as RUN-0002, `L = lam*L1 + lam^2*L2 + lam^3*L3`,
but with **full-image unmasked** MSE per step (deliberately the opposite
masking choice from RUN-0002, motivated by RUN-0002's own diagnosis that
the masked objective under-constrains the operator outside the swept
region). Lambda in {0.3, 0.5, 0.7, 0.9}, Adam lr=3e-4, batch=256, 100
epochs (3500 iters/lambda), ~5min/lambda. **Positive result on accuracy,
null on control**: closed-loop step-3 accuracy rises from +0.092
(untrained) to +0.209-0.216 (~2.2x) with no cost to step-1/teacher-forced
accuracy; terminal slateN was already near-ceiling untrained (lyapunov
0.967, 15/0/0 wins vs init at every lambda) and stays there — only
`mass_in_region` moves (0.821 -> ~0.91).

- **Command**: see `COMMAND.txt` (reconstructed — predates
  `run_probe.py`-style logging; original invocation
  `python -u train_nfd_multistep.py` from `experiments/temp/multistep-nfd/`).
- **Commit**: `0ddab20f` (dirty tree — same pre-existing, unrelated
  docs/skills reorganisation as RUN-0001/RUN-0002).
- **Device**: single GPU run (RTX 4070 Laptop, 8GB), ~26min wall clock per
  `train.log` timestamps (23:33:43 start -> 23:59:24 done), ~3.1GB observed
  GPU memory at batch=256 — no gradient checkpointing needed.
- **Data / split**: same raw-file loader and split convention as
  RUN-0002 (35/15 slates, seed 0, `n20_L20mm` + `n20_L40mm`,
  `n20_L10mm` excluded); disjointness asserted in-script.
- **Status**: completed, no errors, no NaN/inf triggered the abort check,
  no lambda needed early stopping.
- **Outputs**: `artifacts/RUN-0003-nfd-multistep-finetune/
  {results_nfd_multistep.json,train.log}` (raw), 4 fitted checkpoints
  (`nfd_lam{0.3,0.5,0.7,0.9}.pth`, kept in
  `experiments/temp/multistep-nfd/`; `nfd_lam0.9.pth` additionally promoted
  to `weights/MODEL-0003-nfd-multistep-finetuned/` — see that MODEL.md for
  the explicit caveats on lambda-unresolvedness and unmeasured control
  benefit), `results/RUN-0003-nfd-multistep-finetune-RESULTS.md` (curated,
  copied verbatim from the source `experiments/temp/multistep-nfd/
  RESULTS.md`).
