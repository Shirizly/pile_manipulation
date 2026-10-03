---
# ---- identity -------------------------------------------------------------
id: EXP-0003
title: Switching the pixel-space linear-foresight operator by push-length bin beats the single global operator on overnight_randlen -- on image accuracy AND on step-0 control-utility capture, at both 64x64 and 32x32
tier: T1
mode: exploratory
date: 2026-09-10
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On overnight_randlen's held-out test split, fitting Suh & Tedrake 2020's
  switched-linear pixel operator with one operator per push-length bin (6
  equal-width bins over [0, observed max push length], ridge=1.0
  toward-identity) raises BOTH swept-region `accuracy` AND `slateN`/capture
  (step-0 candidate-ranking control utility, averaged over 3 goals x 3
  value functions) over the single global operator fit with the identical
  recipe on the identical train data -- at both 64x64 (this repo's own
  convention) and 32x32 (the paper's own resolution) canonical-frame
  resolutions.

# prediction: omitted -- mode is exploratory (first run of a newly-built
# fitting pipeline plus its first pass through the shared control-utility
# harness; see "What would change the verdict" for the confirmatory
# follow-up this motivates).

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 36b38878
  dirty: true                      # pre-existing dirty tree from concurrent
                                   # Baselines work on this branch (GNN/NFD/
                                   # common files) -- see dirty_files in
                                   # Baselines/LinearForesight/runs/
                                   # operators_res64_accuracy.json for the
                                   # exact list. This record's own new files
                                   # (Baselines/LinearForesight/*) are
                                   # untracked, and the 2 edited shared files
                                   # (Baselines/common/randlen_data.py,
                                   # Baselines/common/eval_report.py) are
                                   # additive (a new optional kwarg, 4 new
                                   # MODELS entries) -- see "What was
                                   # actually run".
  data_commit: "unrecorded (overnight_randlen -- no provenance block in its _0_config.yaml, same gap noted in EXP-0001/EXP-0002)"
  script: "Baselines/LinearForesight/fit_switched.py (fit + image accuracy), Baselines/common/eval_report.py (control-utility capture)"
  data:
    - "configs/dataset/genesis_overnight_randlen_train_all.yaml"
    - "configs/dataset/genesis_overnight_randlen_test_all.yaml"
  code_path: "PileSweepData raster (registry.dataset_registry.build_dataset), via Baselines.common.randlen_data.load_randlen_cell -- same occupancy rasteriser EXP-0001/EXP-0002 use. fit_switched.py calls it with need_step_idx=False (fitting never needs step tagging); eval_report.py calls it with its default need_step_idx=True (capture needs step-0/slate grouping)"
  seed: 0
  split: "overnight_randlen file-level train/test split (scripts/probes/prepare_randlen_split.py, seed 0): 192 train files (98,296 transitions) / 21 test files (10,751 transitions), all 5 spawn/particle-count groups pooled in each"
  runtime: "~7 min per fit_switched.py run (res64 and res32 separately, CPU only, data load dominates at 344-364s each), ~45s for the eval_report.py control-utility pass over all 4 models"

budget:
  declared: "not declared in advance -- direct continuation of building Baselines/LinearForesight this session, extended per user request to add the resolution ablation and control-utility pass"
  spent: "~1h total across both sessions (initial fit+accuracy pipeline, one shared-loader bug fix, this record's extension: bundling single_operator+mean_delta into the same fit run, 2 new predictor classes, 4 new eval_report.py MODELS entries, the res32 rerun, the control-utility pass, this record)"
  outcome: within

design:
  varied:
    model: [linear-switched (6 bins), linear-single]
    resolution: ["32x32", "64x64"]
  held_fixed:
    fit_recipe: "ridge=1.0 toward identity (fit_linear_foresight.fit_operator), IDENTICAL for linear-switched (per-bin) and linear-single (whole train set) at a given resolution -- isolates the effect of switching itself, not a different constraint choice"
    crop: 1.0
    train_data: "same 98,296-transition train_all load for every model at a given resolution (single global operator and all 6 per-bin operators fit from it, IN THE SAME PROCESS -- see 'retrained, not stale' note below)"
    test_data: "same 10,751-transition test_all load, same swept-region mask, for every model"
    goals_and_value_fns: "3 goals (random_quadrant, ring_O, T) x 3 value functions (lyapunov, mass_in_region, signed_mass) -- Baselines/common/goals.py, identical to EXP-0001/EXP-0002"
  baselines: [persistence, mean-delta]
  metric: "accuracy (image, swept region) AND slateN/capture (control-utility, step-0 same-state pools, averaged over goals) -- both keys in experiments/METRICS.md, computed by fit_linear_foresight.metrics and Baselines/common/goals.py::slate_n_capture respectively, the SAME functions EXP-0001/EXP-0002 use"

noise_floor: "not measured -- single file-level train/test split (seed 0), no fold-to-fold sd or bootstrap run for either metric. Aggregate accuracy gaps (e.g. 0.288 vs 0.189 at res64) and capture gaps (e.g. lyapunov 0.780 vs 0.624 at res64) are both large relative to the per-cell spread seen across goals/value-functions (~0.03-0.3), but that is not a measured floor -- flagged as imprecision, not treated as precise."

depends_on: [randlen-train-test-file-disjoint, push-frame-warp-roundtrip, randlen-step0-pool-size-128, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Switched beats single at BOTH resolutions, on BOTH metrics, in every
  cell tested. Image accuracy: res64 0.2883 vs 0.1893 (+0.099); res32 0.2478
  vs 0.1473 (+0.100). Control-utility capture (lyapunov, averaged over 3
  goals): res64 0.780 vs 0.624 (+0.155); res32 0.878 vs 0.582 (+0.296) --
  every one of the 3 goals x 3 value-functions x 2 resolutions = 18 capture
  cells favours switched over single. One resolution-dependent asymmetry
  worth flagging: res32's switched model has LOWER image accuracy than
  res64's (0.248 vs 0.288) but HIGHER control-utility capture (lyapunov
  0.878 vs 0.780) -- accuracy and capture do not rank the two resolutions
  the same way; see "What would change the verdict".
verdict: supported
downgrades: [imprecision, indirectness, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

If switching by push length did nothing (the paper's premise not holding on
this corpus's 3-D physics, or the per-bin data being too thin to fit well),
switched would be at or below single on accuracy, especially in the
thinnest bins, AND the two operators would rank step-0 push candidates
about equally well (capture ~equal). Instead switched wins on both metrics,
at both resolutions, in every cell — including the two thinnest bins for
accuracy (6.7k/7.4k rows) and all 9 goal/value-function cells per
resolution for capture. That consistent, cross-metric, cross-resolution
win is the pattern a real length-dependent transport structure would
produce; noise in an underdetermined per-bin fit would instead show up as
switched losing somewhere (a thin bin, an unfavourable goal), which does
not happen here.

## What was actually run

`Baselines/LinearForesight/fit_switched.py` run twice with all other
defaults (`--n-bins 6 --crop 1.0 --constraint ridge --ridge 1.0`): once
`--res 64` (writes `runs/operators_res64.pt`, this repo's established
resolution) and once `--res 32` (writes `runs/operators_res32.pt`, the
paper's own resolution — the ablation `docs/linear_visual_foresight_
baseline.md` §8 calls for). Each run independently loads the full
train/test split and fits BOTH the 6 per-bin operators and the single
global reference operator from the identical loaded tensors in the same
process (`fit_operator` on `Y0_all`/`Y1_all`, same ridge=1.0 recipe) —
this is what makes "linear-single" in the results below a **freshly
retrained** operator on the current `overnight_randlen` train corpus, not a
previously-saved checkpoint: there is no separate "fit the single operator"
script or file anywhere in this baseline; `Baselines/LinearForesight/
predictor.py::SingleLinearForesightPredictor` reads `single_operator`
straight out of the SAME `operators_res<R>.pt` bundle the switched
predictor reads `operators` from.

Control-utility capture was then scored via the SHARED harness EXP-0001/
EXP-0002 already use, extended with 4 new `MODELS` entries
(`linear_switched_res64`, `linear_single_res64`, `linear_switched_res32`,
`linear_single_res32`, `Baselines/common/eval_report.py`) rather than a
new script:

```bash
PYTHONPATH=. python Baselines/common/eval_report.py \
    --models linear_switched_res64,linear_single_res64,linear_switched_res32,linear_single_res32 \
    --corpora randlen_test \
    --out-prefix Baselines/LinearForesight/runs/control_utility_report
```

This scores the SAME `randlen_test` held-out pool `fit_switched.py`'s own
table above uses, on the standard 3-goal x 3-value-function `slateN`/
capture grid. **This is the step-0 candidate-ranking control-utility proxy
EXP-0001/EXP-0002 use, not a full closed-loop MPC rollout through
Genesis** — no `simple_mpc/` controller was run; see "indirectness" in
Threats and the scope note in `SPEC.md`.

Two small code changes were needed to reach this point (both additive, no
existing behaviour changed):
1. `Baselines/common/randlen_data.py::load_randlen_cell` gained a
   `need_step_idx: bool = True` kwarg (default preserves existing callers'
   behaviour) — `fit_switched.py` passes `False` since fitting never uses
   step tagging, and the pooled `_all.yaml` train config has a handful of
   step-0 rows dropped by `min_push_length_m`, which breaks that tagging's
   `sample_idx // 128` derivation for the rest of the affected file (see
   `LOG.md`). `eval_report.py`'s own `_load_cell` still calls it with the
   default (`True`), since capture needs step-0/slate grouping.
2. `Baselines/LinearForesight/predictor.py` gained
   `SingleLinearForesightPredictor`/`build_predictor_single` alongside the
   existing switched predictor, and `fit_switched.py`'s saved bundle grew
   two keys (`single_operator`, `mean_delta`) so both predictors load from
   one file per resolution.

## Numbers

**Image accuracy** (swept-region, held-out test, 10,751 transitions):

| model | res64 | res32 |
|---|---|---|
| persistence | 0.0000 | 0.0000 |
| mean-delta | 0.0729 | 0.0557 |
| linear-single | 0.1893 | 0.1473 |
| **linear-switched** | **0.2883** | **0.2478** |

**Control-utility capture** (`slateN`, step-0 pools, averaged over the 3
goals — random_quadrant/ring_O/T):

| model | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| linear-single (res64) | 0.6236 | 0.6061 | 0.3912 |
| **linear-switched (res64)** | **0.7795** | **0.6841** | **0.7768** |
| linear-single (res32) | 0.5824 | 0.6290 | 0.4772 |
| **linear-switched (res32)** | **0.8784** | **0.7849** | **0.7764** |

Per-goal breakdown for all 4 models (9 cells each): `Baselines/
LinearForesight/runs/control_utility_report.json`. Per-bin accuracy
breakdown and bin edges (identical bin boundaries at both resolutions,
since binning is by world-metre push length, not by pixel resolution):
`Baselines/LinearForesight/runs/operators_res{32,64}_accuracy.json`. Fitted
operators: `Baselines/LinearForesight/runs/operators_res{32,64}.pt`.

Train bin counts (both resolutions, identical — binning is
resolution-independent): 7420 / 18330 / 24925 / 22275 / 18680 / 6666, all
well above the `MIN_ROWS_PER_BIN=50` identity-fallback floor.

## What would change the verdict

- **The accuracy/capture resolution disagreement is worth its own check.**
  res32's switched operator has lower image accuracy than res64's but
  higher capture. Two live explanations, not yet distinguished: (a) capture
  only needs the CORRECT DIRECTION of mass movement relative to a goal
  region, which a coarser 32x32 grid may preserve even while losing
  pixel-level fidelity a 64x64 grid captures; or (b) this is a single-split
  fluke (no noise floor, see above). A fold-level rerun (different seed
  splits) would distinguish real-vs-noise; a per-bin capture breakdown
  (not just per-goal) would test explanation (a) directly.
- **A noise floor** (fold-to-fold or seed-to-seed sd) for both metrics —
  not measured; the "imprecision" downgrade.
- **A real closed-loop MPC rollout** through `simple_mpc/` (the plan doc's
  Steps 3-5: the Lyapunov cost registered as `image_lyapunov`, an
  enumerating action sampler, a runnable MPC config) would retire
  "indirectness" for the control axis — `slateN`/capture is a one-step
  candidate-ranking proxy, not a multi-step closed-loop regret measurement,
  same caveat EXP-0001/EXP-0002 already carry.
- **`--constraint nonneg`** (the paper's own best-performing fit variant):
  still not run (ridge was deliberately chosen to isolate the switching
  question from the constraint question, `SPEC.md` §3).

## Threats

- **`imprecision`**: no noise floor measured for either metric (single
  seed-0 file split); see above.
- **`indirectness`**: `slateN`/capture is a step-0 candidate-ranking proxy
  for actual closed-loop control performance, not a measurement of it — see
  "What would change the verdict".
- **`untested-dependency`**: `randlen-train-test-file-disjoint` and
  `randlen-step0-pool-size-128` hold (tested, `INVARIANTS.md`);
  `push-frame-warp-roundtrip` was newly registered this session (re-verified
  15/15 passing, but a first formal registration, not a long track record);
  `goal-mask-axis-convention-row-y-col-x` (used by the capture computation)
  is still `unchecked` in `INVARIANTS.md` — inherited from EXP-0001/
  EXP-0002, which carry the same downgrade for the same reason.
- Considered and dismissed: **provenance mismatch** between linear-switched
  and linear-single at a given resolution — both are fit by the identical
  `fit_operator` call on the identical loaded train tensors within the same
  `fit_switched.py` run, so this is not a threat within a resolution.
  Across resolutions (res32 vs res64) the two ARE separate runs/loads —
  flagged, not a threat to the switched-vs-single claim (which is always
  within-resolution), but relevant to the accuracy/capture disagreement
  noted above.

## Unrelated findings

- `Baselines/common/randlen_data.py::load_randlen_cell`'s step-0-completeness
  assertion was, before this session, unconditionally run on every load —
  meaning nothing had ever actually loaded the pooled `_all.yaml` configs
  through it end-to-end before (GNN training bypasses this loader entirely
  via its own `dataset_genesis_gnn.py::_load_rows`). Worth knowing if another
  future caller reaches for `load_randlen_cell` on a pooled config and hits
  the same assertion.
- A dense 64x64 canonical-frame operator bundle (6 per-bin + 1 single) is
  ~470MB on disk (`runs/operators_res64.pt`) vs ~29MB at 32x32
  (`runs/operators_res32.pt`) — a 16x gap matching the D² parameter scaling
  the plan doc's ORCHESTRATION_LOG already flagged for the single-operator
  case. Nothing else in `Baselines/*/runs/` is remotely this size; worth a
  decision (gitignore, LFS, or float16) before it is committed.

## Later evidence (2026-10-03 audit)

Added by the cross-experiment audit (summary: `experiments/SUMMARY.md`). Numbers above are unchanged.

- Asymmetric goal masks (T / random_quadrant / ring_O / stripe / letters) in this record were scored BEFORE the goal-axis fix `28271c09` (2026-09-17) and were never rescored; transpose-invariant goals (corner, center, ...) are unaffected (ISS-003). ISS-003 names this record explicitly.
- EXP-0013 re-fits res32 vs res64 with a fair same-grid scoring (C-019) and reverses this record's resolution observation.
