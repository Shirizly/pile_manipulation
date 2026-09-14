# RUN-0001-rollout-eval

Closed-loop 3-step rollout evaluation of 4 EXISTING models (no training
here): `model0001_switched`, `model0001_global`, `hybrid94`, `nfd`, plus
MODEL-0002 (descriptor-only) teacher-forced only. Also verifies the
`slates_multistep` row-index-is-trajectory-identity property and the
bin-sequence distribution for the switched operator.

- **Command**: see `COMMAND.txt` (reconstructed — this run predates
  `run_probe.py`-style logging; the exact invocation was
  `python -u rollout.py` from `experiments/temp/multistep-rollout/`, per
  that directory's own `RESULTS.md` header).
- **Commit**: `0ddab20f` (dirty tree — the same pre-existing docs/skills
  reorganisation visible in this session's `git status`, predating this
  run and not touching any file this run reads or writes).
- **Device**: single GPU run, ~5s total wall clock.
- **Data**: loaded directly from the raw `Genesis/data/slates_multistep/
  {n20_L20mm,n20_L40mm}/*_{batch}_data.pt` files (NOT the `*_eval` dataset
  configs, which apply `min_push_length_m` filtering that drops rows and
  would break row==env-identity alignment across steps). 50 slates x 128
  envs x 3 steps per dataset, unfiltered. `n20_L10mm` excluded.
- **Models loaded from**: `weights/MODEL-0001-stage2-visual-switched/
  checkpoint.pt` (switched + global readings of the same file),
  `experiments/temp/stage2-slaten/operators/hybrid94_lam1.0.pt`,
  `Baselines/NFD/runs/nfd_3ch/unet_best.pth`,
  `weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt`.
- **Status**: completed, no errors.
- **Outputs**: `artifacts/RUN-0001-rollout-eval/results_multistep.json`
  (raw), `results/RUN-0001-rollout-eval-RESULTS.md` (curated, copied
  verbatim from the source `experiments/temp/multistep-rollout/RESULTS.md`).
