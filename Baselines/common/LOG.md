# Baselines/common — LOG

Agent: A4-common. Branch `baselines/overnight`. Builds the shared plumbing
every baseline model plugs into: pooled dataset configs, one data loader
(particle view + occupancy view), and a generalised eval harness that
replaces `scripts/probes/expB_multistep_eval.py`'s hardcoded linear/mean-delta/
UNet with a pluggable predictor. See `Baselines/ORCHESTRATION_LOG.md` for the
scope this serves.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: DONE. Self-test passes.**

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

**Self-test (`PYTHONPATH=. python Baselines/common/eval_baseline.py --self-test`,
uses only the built-in reference predictors, no external model needed):
PASSED.**
- `accuracy[mean-delta]` = 0.08569, `accuracy[linear]` = 0.30053 — exact match
  to `runs_expB/n20_L20mm_accuracy.json` to 5 decimal places (both overall
  and all three `accuracy_by_step` rows).
- `slateK_exact` for `linear`/`mean-delta` at every K in
  `runs_expB/n20_L20mm_kcurve_exact.json` (2,4,8,16,32,64,128) matches to
  <2e-3 by re-running `scripts/probes/exp0026_kcurve_exact.py` UNCHANGED on
  the dv cache this harness wrote.
- Conclusion: the harness (data.py's two views + eval_baseline.py's fit/score
  pipeline) reproduces the register's numbers exactly. Nothing here needed
  correcting against the old numbers.

**Things that will trip up model agents:**
1. `data.py`'s particle view reaches into `PileSweepData` PRIVATE attributes
   (`_resolve_idx`, `_run_lookup`, `_offsets`, `runs`, `configs`,
   `_output_grid`, `to_pxl`, `ctr_in_PXL`, `_draw_particle_grid`) because
   there is no public accessor for per-sample `states`/`states_`/`angle` or
   for the rasteriser. This was a deliberate reuse-over-reimplementation
   choice (verified byte-identical against `occ0`/`occ1`, see below) — flagged
   here in case a future refactor of `Genesis/training/dataset.py` renames
   these and silently breaks `Baselines/common/data.py`.
2. `--predictor` is `module.path:factory_callable` — the module must be
   importable (put your baseline's package on `PYTHONPATH`), the factory
   takes NO arguments and returns an object with `.name` and `.predict_occ`.
   `predict_occ` must return a tensor of EXACTLY `batch.occ0.shape`
   (float, will be cast to float32) — no sigmoid/clamping is applied for you
   (do it yourself if your model outputs logits, as `exp0021_eval.unet_forward`
   does for the UNet: `torch.sigmoid(...).squeeze(1)`).
3. Fitting the reference operators (`linear`/`mean-delta`) on the POOLED
   `n20_L20L40_train` config is noticeably slower than per-cell (180 runs /
   23040 transitions vs 90/11520) — a few minutes, not seconds. This is
   NOT GPU work (all CPU tensor ops), so it does not need `gpu_lock.sh`, but
   budget wall-clock time for it in a script that scores against both eval
   cells.
4. `push_length` is NOT constant in the pooled train pool (20mm vs 40mm) —
   see the pooled config's own header comment. A model that does not
   condition on the action/push-length will fit a mixture and underperform.
5. `rasterize_particles` is per-transition (loops internally over 20
   particles via OpenCV) and has no batched form — a particle-space
   predictor (the GNN) must loop over its batch to rasterise, one call per
   row, passing that row's own `run_idx` (configs can in principle differ
   per source run, though in practice all L20/L40 runs share one box/plate
   geometry).
6. The task brief guessed the particle rasteriser lived under `transforms/`.
   It does not: `transforms.functional.particles_to_occupancy` EXISTS but is
   a different, point-only splat (no box/quaternion handling) used elsewhere
   in the repo (e.g. the MPC oracle) — using it here would silently
   reintroduce the axis/shape mismatch that `PileSweepData._draw_particle_grid`
   was written to fix (C-018). `data.rasterize_particles` calls the dataset's
   own private method instead; see `data.py`'s module docstring.
7. `eval_baseline.py`'s `--goals` accepts `lyapunov_weights`' goal keys
   unchanged (default `corner,center`, per this task's brief) — a `*-pile`
   goal additionally computes `pile_centroid_and_support` first, exactly as
   `expB_multistep_eval.py` does; not needed for the default two goals.

---

## Running notes

- 2026-09-08: started. Read ORCHESTRATION_LOG.md, expB_multistep_eval.py,
  exp0026_kcurve_exact.py, Genesis/training/dataset.py (PileSweepData),
  registry/dataset_registry.py, fit_linear_foresight.py, dmdc_baseline.py.
- Wrote `data.py`; verified `rasterize_particles` reproduces `occ0`/`occ1`
  exactly (mean abs err 0.0 over sampled rows) on `n20_L20mm_eval`, and that
  its manifest-based slate/step tagging matches EXP-B's own counts (20
  slates x 3 steps).
- Wrote `eval_baseline.py`; `--self-test` passes (see summary above).
- Verified `--predictor module:factory` plumbing end-to-end with a throwaway
  dummy (persistence-clone) predictor against the pooled train config and
  the `n20_L20mm_eval` cell — confirms an external model's `predict_occ`
  result flows through into `accuracy.json`/`dv_cache.pt` under its own name
  alongside the reference rows.
- Committed `data.py` + pooled config first (agents were blocked on data.py
  in particular); `eval_baseline.py` committed once its self-test passed.
