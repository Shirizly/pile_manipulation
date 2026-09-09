# Experiment commands — how each recent result was produced

**Purpose.** One place to find the exact invocation behind a recorded number.
Records under `docs/experiments/` name their `provenance.script`, but not the
arguments, and several results turn on an argument (which slate config, which
`--goals`, which `--split`). This file closes that gap.

**Provenance of this file itself.** Commands marked **[recorded]** are copied
verbatim from `runs/<tag>.json`, which `scripts/run_probe.py` writes beside
each log at run time, or from a log/doc that captured the invocation.
Commands marked **[reconstructed]** were not captured anywhere; they are
rebuilt from the documented usage plus the observed output filenames
(`--tag`/`--out-prefix` are recoverable from what landed on disk). Treat a
reconstructed line as accurate in form but unverified in detail.

**Conventions that apply to nearly everything below**

- `PYTHONPATH=.` and the project env: `/home/alon/anaconda3/envs/pme/bin/python`
  (`conda activate pme`). `OMP_NUM_THREADS=4` on a shared machine.
- Long jobs go through `scripts/run_probe.py --tag <tag> --threads 4 -- <cmd>`,
  which forces `python -u`, redirects to `runs/<tag>.log`, records the PID, and
  stamps `utils.git_provenance()` into `runs/<tag>.json`.
- **`run_probe.py` does not detach its child from the launching shell's process
  group**, so a long job dies when that shell exits. Wrap it:
  `setsid nohup python scripts/run_probe.py ... > /dev/null 2>&1 < /dev/null & disown`.
  This cost ~50 min of GPU on a restarted collection; see
  `docs/plan_selection_pressure_validation.md`.
- Baselines CUDA **training** goes through `bash Baselines/common/gpu_lock.sh`
  (a counting semaphore, `GPU_LOCK_SLOTS` default 3). Evaluation does not need
  the wrapper.

---

## 1. Data collection

### Multi-step same-state slates — the EXP-B testbed
50 states x 128 candidate action sequences x 3 steps per cell, placement-aware
starts, perpendicular pushes, constant commanded length. Feeds EXP-0024_v1,
EXP-0026_v1, EXP-0024_v2, EXP-0026_v2, EXP-0027, EXP-0028, EXP-0029 and all of
the Baselines work. **[recorded]**

```
python -u -m Genesis.same_state_slate_collection --n-cubes 20 --n-envs 128 \
  --n-states 50 --n-steps 3 --placement-aware --no-pile-aware \
  --push-length 0.010 --output-root data/slates_multistep --tag n20_L10mm
python -u -m Genesis.same_state_slate_collection --n-cubes 20 --n-envs 128 \
  --n-states 50 --n-steps 3 --placement-aware --no-pile-aware \
  --push-length 0.020 --output-root data/slates_multistep --tag n20_L20mm
python -u -m Genesis.same_state_slate_collection --n-cubes 20 --n-envs 128 \
  --n-states 50 --n-steps 3 --placement-aware --no-pile-aware \
  --push-length 0.040 --output-root data/slates_multistep --tag n20_L40mm
```

`--placement-aware --no-pile-aware` is load-bearing: the two are mutually
exclusive in code (`generate_action_samples` returns on the `pile_aware`
branch before `placement_aware` runs), so passing both silently gives
pile-aware sampling.

### n=50 cell, reduced scope, interrupted at 6 of 20 states **[recorded]**

```
setsid nohup python scripts/run_probe.py --tag n50_L20mm --threads 4 -- \
  python -m Genesis.same_state_slate_collection --n-cubes 50 --n-envs 64 \
  --n-states 20 --n-steps 3 --placement-aware --no-pile-aware \
  --push-length 0.020 --output-root data/slates_multistep --tag n50_L20mm \
  > /dev/null 2>&1 < /dev/null & disown
```

To resume, use a **new `--tag`** and a **different `--seed`**: the batch counter
is `files_in_dir / 3` so appending breaks the
`batch_idx == slate_idx * n_steps + step_idx` grouping invariant, and the state
library is seeded, so re-running with `--seed 0` regenerates the same states
rather than extending them.

---

## 2. Dataset and cell utilities

| what | command | source |
|---|---|---|
| Verify a collected cell (same-state spread, action uniqueness, realized vs commanded length) | `python scripts/probes/verify_slate_cell.py Genesis/data/slates_multistep/n20_L20mm 0.020` | [recorded] |
| Rebuild `manifest.json` after an interrupted collection | `python scripts/probes/rebuild_slate_manifest.py Genesis/data/slates_multistep/n50_L20mm --n-steps 3 --apply` | [recorded] |
| Materialise the 30/20 train/eval slate split | `python scripts/probes/prepare_slate_multistep_split.py` | [reconstructed] |
| Validate the register (recomputes grades; must exit 0) | `python scripts/check_register.py` | [recorded] |
| Regenerate the generated inventory | `python scripts/summarise_register.py > docs/experiments/STATE_OF_PLAY.md` | [recorded] |

---

## 3. EXP-0021 / EXP-0022 / EXP-0023 — granularity, blur-fairness, channels

**EXP-0021** — UNet vs the linear operator across object count and action
sampling (7 cells x 2 strata). **[recorded]**

```
python -u -m training.train /tmp/exp0021_pilot.yaml --no-resume   # pilot
bash scripts/probes/exp0021_train_all.sh                          # all cells
bash scripts/probes/exp0021_eval_all.sh
```

**EXP-0022** — the UNet's advantage is high-frequency: score both model classes
on a common blurred target instead of each in its own configuration.
**[recorded]**

```
python -u scripts/probes/exp0022_blur_fair.py configs/dataset/genesis_granularity_blind_n50.yaml  runs_granularity/unetfilm_blind_n50  --tag blind_n50
python -u scripts/probes/exp0022_blur_fair.py configs/dataset/genesis_granularity_blind_n20.yaml  runs_granularity/unetfilm_blind_n20  --tag blind_n20
python -u scripts/probes/exp0022_blur_fair.py configs/dataset/genesis_granularity_contact_n20.yaml runs_granularity/unetfilm_contact_n20 --tag contact_n20
python -u scripts/probes/exp0022_blur_fair.py configs/dataset/genesis_granularity_contact_n5.yaml  runs_granularity/unetfilm_contact_n5  --tag contact_n5
```

**EXP-0023** — mask-vs-depth channel confound at full episode count, plus the
non-negativity re-run. **[recorded]**

```
python -u scripts/probes/channels.py --glob 'Genesis/data/cube_spectrum/n20/*_data.pt'    --label 'cube n20 PILOT3ep' --cube-size 0.005 --min-push-mm 19.9 --max-episodes 3
python -u scripts/probes/channels.py --glob 'Genesis/data/cube_spectrum/n20/*_data.pt'    --label 'cube n20 FULL'     --cube-size 0.005 --min-push-mm 19.9
python -u scripts/probes/channels.py --glob 'Genesis/data/granularity/n20/**/*_data.pt'   --label 'granularity n20 scattered blind 40mm FULL' --cube-size 0.005 --min-push-mm 39.0
python -u scripts/probes/nonneg_vs_ridge.py
python -u scripts/probes/deposit_profile.py     # EXP-0010 deposit profile, full data
python -u scripts/probes/granularity_characterise.py
```

---

## 4. EXP-0024 / EXP-0025 — control utility and the degradation spectrum

**EXP-0024** — does the UNet's image-accuracy edge buy control utility?
Train in-distribution, then score image accuracy and control side by side.
**[recorded]**

```
python -u -m training.train configs/training/exp0024_unetfilm_cube_spectrum_n20.yaml --no-resume
python -u scripts/probes/exp0024_control_eval.py \
  configs/dataset/genesis_cube_spectrum_n20.yaml \
  configs/dataset/genesis_cube_spectrum_n20_slates.yaml \
  runs_exp0024/unetfilm_cube_spectrum_n20 --tag paired
```

**EXP-0025** — the full degradation spectrum (amplitude, blur, hf-noise,
displacement, wrong-physics) on same-state slates. **[recorded]**

```
python -u scripts/probes/full_spectrum_same_state.py
```

---

## 5. EXP-0026 — the selection-pressure curve (K sweep)

Two stages: cache per-candidate dV once, then sweep the pool size K.
**[recorded]**

```
# stage 1 -- all 50 slates, plus EXP-0025's degradation arms rebuilt in this code path
python -u scripts/probes/exp0026_selection_pressure.py \
  configs/dataset/genesis_cube_spectrum_n20.yaml \
  configs/dataset/genesis_cube_spectrum_n20_slates_all.yaml \
  runs_exp0024/unetfilm_cube_spectrum_n20 \
  --split train --degradations --out runs_exp0026/dv_cache_all50_deg.pt

# stage 1 -- EXP-0024's 49-slate subset, as a reproduction check
python -u scripts/probes/exp0026_selection_pressure.py \
  configs/dataset/genesis_cube_spectrum_n20.yaml \
  configs/dataset/genesis_cube_spectrum_n20_slates.yaml \
  runs_exp0024/unetfilm_cube_spectrum_n20 \
  --split test --out runs_exp0026/dv_cache_exp0024_49.pt

# stage 2 -- the K curve, all arms, paired comparisons
python -u scripts/probes/exp0026_kcurve.py runs_exp0026/dv_cache_all50_deg.pt \
  --goal corner --models all \
  --pairs "UNet-linear,oracle-linear,linear-mean-delta" \
  --out runs_exp0026/kcurve_deg_all50.json
python -u scripts/probes/exp0026_kcurve.py runs_exp0026/dv_cache_exp0024_49.pt \
  --goal corner --out runs_exp0026/kcurve_corner_49.json
```

`--split train` with `genesis_cube_spectrum_n20_slates_all.yaml` is the only
setting that loads all 50 slates: `PileSweepData._assign_group_splits` computes
`round(50 * test_pct/100)` and clamps the total to `n-1`, so the registry path
can never place every group in `test`. That, not the push-length filter, is why
earlier records scored 49.

---

## 6. EXP-0024_v1 / EXP-0026_v1 — the three push-length cells

Train per cell on 30 slates, score on the 20 held-out slates' step-0 pools.
**[recorded]**

```
python -u -m training.train configs/training/expB_unetfilm_slates_multistep_n20_L10mm.yaml --no-resume
python -u -m training.train configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml --no-resume
python -u -m training.train configs/training/expB_unetfilm_slates_multistep_n20_L40mm.yaml --no-resume

python -u scripts/probes/expB_multistep_eval.py \
  configs/dataset/genesis_slates_multistep_n20_L10mm_train.yaml \
  configs/dataset/genesis_slates_multistep_n20_L10mm_eval.yaml \
  Genesis/data/slates_multistep/n20_L10mm/manifest.json \
  runs_expB/unetfilm_slates_multistep_n20_L10mm \
  --tag n20_L10mm --degradations --out-prefix runs_expB/n20_L10mm
# ... identically for L20mm and L40mm, substituting the three paths and the tag
```

Control ranking uses **step-0 candidates only**: at step 0 all 128 envs share an
identical start state, so they form a genuine same-state pool. By steps 1-2
each env has executed its own push and diverged, so using them would reintroduce
the cross-state confound EXP-0012 exists to remove.

---

## 7. EXP-0024_v2 / EXP-0026_v2 — the cost-functional sweep

Same models and pools, only the cost functional changes. `--goals` is the whole
experiment. **[recorded]**

```
# dataset A -- old slates, EXP-0024's checkpoint
python -u scripts/probes/exp0026_selection_pressure.py \
  configs/dataset/genesis_cube_spectrum_n20.yaml \
  configs/dataset/genesis_cube_spectrum_n20_slates_all.yaml \
  runs_exp0024/unetfilm_cube_spectrum_n20 --split train --degradations \
  --goals "ind-square8-pile,ind-square16-pile,ind-stripe-thin-pile,corner" \
  --out runs_exp0026/dv_cache_sharp.pt

# dataset B -- the L20mm cell, in-distribution checkpoint
python -u scripts/probes/expB_multistep_eval.py \
  configs/dataset/genesis_slates_multistep_n20_L20mm_train.yaml \
  configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \
  Genesis/data/slates_multistep/n20_L20mm/manifest.json \
  runs_expB/unetfilm_slates_multistep_n20_L20mm \
  --tag n20_L20mm_sharp --degradations \
  --out-prefix runs_expB/n20_L20mm_sharp \
  --goals "ind-square8-pile,ind-square16-pile,ind-stripe-thin-pile,corner"

# then the K curve per goal
python -u scripts/probes/exp0026_kcurve.py runs_exp0026/dv_cache_sharp.pt \
  --goal ind-stripe-thin-pile --ks 2,4,8,16,31 \
  --out runs_exp0026/kc_sharp_A_ind-stripe-thin-pile.json
python -u scripts/probes/exp0026_kcurve.py runs_expB/n20_L20mm_sharp_dv_cache.pt \
  --goal ind-stripe-thin-pile --ks 2,4,8,16,32,64,128 \
  --out runs_expB/kc_sharp_B_ind-stripe-thin-pile.json

# supporting analyses
python scripts/probes/spectral_concentration.py        # weight-field frequency content
python scripts/probes/functional_degeneracy_screen.py  # dv_true spread / %helpful per goal
```

The `*-pile` goals recompute their target centroid from whatever data the run
loads, so an 8x8 px target can shift a pixel between runs and move the numbers
by ~0.05. Tracked as `goal-placement-pinned` (**broken**) in `INVARIANTS.md`.

---

## 8. EXP-0027 / EXP-0028 / EXP-0029 — verification and mechanism

| record | what it settles | command | source |
|---|---|---|---|
| EXP-0027 | Is `warp-only`'s positive accuracy a bug? (No — rms rewards blur) | `python scripts/probes/warp_blur_diagnostic.py` | [reconstructed] |
| EXP-0028 | Is the L10mm `-0.44` / `0.91` dissociation a computation error? (No) | `python scripts/probes/l10_verification.py` | [reconstructed] |
| EXP-0029 | Mechanism: error projection, geometry-only baseline, shuffle null, paired UNet-linear | `python scripts/probes/exp0029_l10mm_mechanism.py` | [reconstructed] |
| — | earlier warp diagnostic | `python scripts/probes/expB_warp_diagnostic.py` | [reconstructed] |

---

## 9. Action-pool diagnostics — `reports/action_pool_diagnostics.md`

| what | command | source |
|---|---|---|
| Population statistics over all pools (agreement / near-tie / large-loss, pool spread) | `python scripts/probes/pool_survey.py <dv_cache.pt> --goal <goal>` | [reconstructed] |
| One pool: dV histogram, chosen actions drawn in the workspace, `Rk` curve, ordering scatter | `python scripts/probes/pool_inspect.py <dv_cache.pt> --goal <goal> --slate <id>` | [reconstructed] |

Caches used: `runs_exp0026/dv_cache_sharp.pt` (dataset A, 50 pools of ~32) and
`runs_expB/n20_L20mm_sharp_dv_cache.pt` (dataset B, 20 pools of 128). Figures
land in `reports/figs/` and are gitignored; the `.log` beside each figure is
committed.

---

## 10. Baselines — GNN, NFD, SchenckCNN

Filed under `Baselines/ORCHESTRATION_LOG.md` and the per-model `LOG.md` files.
L10mm is excluded from this arm as suspect (EXP-0028 / EXP-0029).

**Training** — through the GPU lock. **[recorded]**

```
bash Baselines/common/gpu_lock.sh python Baselines/GNN/train/train_genesis_gnn_dyn.py \
  --epochs 500 --batch-size 128 --out-dir Baselines/GNN/runs
bash Baselines/common/gpu_lock.sh python Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_3ch.yaml
bash Baselines/common/gpu_lock.sh python Baselines/NFD/train_nfd.py Baselines/NFD/configs/nfd_train_2ch_ablation.yaml
bash Baselines/common/gpu_lock.sh python Baselines/SchenckCNN/train_schenck.py \
  --epochs 100 --ckpt-every 10 --out Baselines/SchenckCNN/runs/schenck.pth
```

**Scoring** — the documented form, with the `--predictor` factory and the
`--tag`/`--out-prefix` pair. The template is recorded in
`Baselines/common/LOG.md`; the eight cells below are **[reconstructed]** from
that template plus the output files present (`{gnn,nfd,nfd_2ch,schenck}` x
`{L20mm,L40mm}`), because no log captured the invocations verbatim.

```
PYTHONPATH=. python Baselines/common/eval_baseline.py \
  configs/dataset/genesis_slates_multistep_n20_L20L40_train.yaml \
  configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml \
  Genesis/data/slates_multistep/n20_L20mm/manifest.json \
  --tag gnn_L20mm --out-prefix Baselines/GNN/runs/gnn_L20mm \
  --predictor Baselines.GNN.predictor:build_predictor
```

Substitute per cell: eval config and manifest (`L20mm` / `L40mm`), `--tag` and
`--out-prefix` (`gnn_*`, `nfd_*`, `nfd_2ch_*`, `schenck_*` under the matching
`Baselines/<model>/runs/`), and `--predictor`
(`Baselines.GNN.predictor:build_predictor`,
`Baselines.NFD.predictor:build_predictor`,
`Baselines.SchenckCNN.predictor:build_predictor` — all three factories exist).

**Self-tests and convention checks** — cheap, run before scoring. **[recorded]**

```
PYTHONPATH=. python Baselines/common/eval_baseline.py --self-test
PYTHONPATH=. python Baselines/GNN/scripts/verify_rasterizer.py
PYTHONPATH=. python Baselines/GNN/scripts/check_geometry.py
PYTHONPATH=. python Baselines/SchenckCNN/check_axes.py
bash Baselines/common/gpu_lock.sh python -c "import Baselines.NFD.nfd_lib"
```

**Timing / inference cost** — the compute axis of the model map. **[recorded]**

```
python Baselines/common/benchmark_time.py
python Baselines/common/benchmark_time.py --force
```

There is **no single "run everything" driver** for the Baselines arm; it is
orchestrated per model through `Baselines/ORCHESTRATION_LOG.md`.

---

## Gaps, stated rather than papered over

- **Five logs have no captured command**: `runs/exp0024_eval.log`,
  `runs/n50_L20mm.log`, `runs/n50_L20mm_b.log`, `runs/sharp_A.log`,
  `runs/sharp_L20.log`. The last two carry only a provenance dict
  (`49c285b1`, `eb390fd2`). The `sharp_*` commands above are the ones that
  produced those outputs, taken from the session that ran them, not from the
  logs.
- **Everything run outside `run_probe.py` is unrecorded by construction** —
  every direct invocation and every `--goals`/`--ks` variation. That is the
  main reason this file exists, and the durable fix is to route analysis runs
  through `run_probe.py` too, not just long ones.
- **`scripts/probes/exp0026_kcurve_exact.py`** is referenced by the Baselines
  logs with `--models`, `--ks` and `--goal corner` but no full invocation was
  captured anywhere.
- **Most probe scripts carry no `Usage` docstring** (34 of 36 under
  `scripts/probes/`); only `rebuild_slate_manifest.py` and
  `verify_slate_cell.py` do. Adding one to each is the cheapest way to keep
  this file from going stale.
