# Baselines/common — LOG

Agent: A4-common. Branch `baselines/overnight`. Builds the shared plumbing
every baseline model plugs into: pooled dataset configs, one data loader
(particle view + occupancy view), and a generalised eval harness that
replaces `scripts/probes/expB_multistep_eval.py`'s hardcoded linear/mean-delta/
UNet with a pluggable predictor. See `Baselines/ORCHESTRATION_LOG.md` for the
scope this serves.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: IN PROGRESS — filled in as work completes.**

**Files delivered:**
- `configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml` — pooled
  train config (both cells' train splits).
- `Baselines/common/data.py` — `load_cell(cfg_path, split, manifest_path=None,
  tag=None) -> CellData`, `rasterize_particles(raw, run_idx, particles_world)
  -> occ (H,W)`.
- `Baselines/common/eval_baseline.py` — generic scorer. Predictor contract:

  ```python
  class BaselinePredictor(Protocol):
      name: str
      def predict_occ(self, batch: PredictorBatch) -> torch.Tensor:  # (B,64,64)
          ...
  ```

  `PredictorBatch` (defined in `eval_baseline.py`) carries BOTH views for the
  eval split: `occ0 (B,H,W)`, `actions (B,4)` world `[sx,sy,ex,ey]` metres,
  `states (B,20,7)` pre-push particle xyz+quat (world metres), `p_start
  (B,3)`, `p_stop (B,3)`, `angle (B,)`, `run_idx (B,)`, `raw` (the
  `PileSweepData` instance, so a particle-space model can rasterise its own
  prediction via `data.rasterize_particles(batch.raw, run_idx_i,
  predicted_particles_i)`), `workspace_min/max`, `H`, `W`. It never carries
  `states_`/`occ1` (the ground truth) — a predictor cannot see the answer.

**Command another agent runs to score their model:**
```
PYTHONPATH=. python Baselines/common/eval_baseline.py \
    configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml \
    configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \
    Genesis/data/slates_multistep/n20_L20mm/manifest.json \
    --tag <yourmodel>_L20mm --out-prefix Baselines/<YOU>/runs/<yourmodel>_L20mm \
    --predictor yourpkg.module:build_predictor
```
(swap the eval cfg/manifest for the L40mm cell to score that one; `--predictor`
is `module.path:factory_callable`, called with no args, must return an object
with `.name` and `.predict_occ`.)

**Self-test:** see "Self-test result" section below — filled in once run.

**Things that will trip up model agents:** see "Gotchas" section below.

---

## Running notes

- 2026-09-08: started. Read ORCHESTRATION_LOG.md, expB_multistep_eval.py,
  exp0026_kcurve_exact.py, Genesis/training/dataset.py (PileSweepData),
  registry/dataset_registry.py, fit_linear_foresight.py, dmdc_baseline.py.
