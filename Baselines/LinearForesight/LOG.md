# Linear Foresight baseline — LOG

## 2026-09-10 — subdir created: switched (per-length-bin) fit on overnight_randlen

**Task:** encapsulate the existing switched-linear visual-foresight code
(`fit_linear_foresight.py`, `transforms/functional.py`'s push-frame warp)
into its own `Baselines/` subdir, comparable to `Baselines/GNN`/`NFD`/
`SchenckCNN`, and use it to fit per-push-length-bin operators on
`Genesis/data/overnight_randlen` — that corpus randomises push length, so
the single global operator `Baselines/common/eval_baseline.py`/
`eval_randlen_indist.py` already fit and use as a "linear" reference row is
the paper's degenerate 1-bin case, not its actual switched design.

Decisions (asked of the user up front, both confirmed):
- **Full baseline integration**, not just a fitting script: `predictor.py`
  implementing the `BaselinePredictor` contract, so this model can be
  scored by the same harness as GNN/NFD/SchenckCNN, not only reported
  standalone.
- **All 5 overnight_randlen groups** (n20 + n50), via
  `Baselines/common/randlen_data.py::load_randlen_cell` (N-agnostic),
  not just the n20 subset `Baselines/common/data.py::CellData` can read.

Built: `model.py` (bin-by-length dispatch over the existing `predict_world`
apply path — no new warp/fit math), `fit_switched.py` (6 equal-width bins
over `[0, observed max push length]` on the TRAIN split, ridge-toward-
identity fit per bin by default, held-out accuracy report against
persistence/mean-delta/the single global operator), `predictor.py`. See
`SPEC.md` for the binning-rule rationale (why low=0 not observed-min, why
high is measured not read from `DATASET.yaml`'s sampler config) and the
resolution/constraint defaults (64×64, ridge — matching the existing
single-operator reference for a fair switched-vs-single comparison, not
the paper's own 32×32/nonneg choices).

**Bug found and fixed while smoke-testing:** `Baselines/common/randlen_data.
load_randlen_cell`'s step-tagging assertion (added for the step-0-candidate-
ranking use case) fires on the pooled `_all.yaml` train config as-is — 5 of
24576 expected step-0 rows are missing (a `min_push_length_m` filter drop
inside one file), which breaks the `sample_idx // 128` step heuristic for
the rest of that file. This is a real data fact (a handful of pushes really
are shorter than the 0.1 mm floor), but it has nothing to do with fitting,
which never looks at `step_idx`. Added `need_step_idx: bool = True` to
`load_randlen_cell` (default preserves existing behaviour for
`eval_report.py`/`eval_randlen_indist.py`) so a caller that only needs
`occ0`/`occ1`/`actions` can skip the per-row `p_start`/`p_stop`/`angle`/
`step_idx` loop and its assertion entirely; `fit_switched.py` now calls it
with `need_step_idx=False`.

**Smoke-tested end-to-end first** (`--max-samples 400 --n-bins 3 --device
cpu`): fit + `runs/operators.pt` + held-out `runs/operators_accuracy.json`
all produced correctly, and `predictor.py::build_predictor()` loads that
bundle and predicts a sane `(20,64,64)` batch in `[0,1]` when driven with a
real `PredictorBatch`.

**Then ran for real** (default settings: 6 bins, ridge=1.0 toward identity,
res=64, CPU — no GPU available in this session's environment; the run took
~7 minutes total, dominated by the 344s data load, each per-bin fit itself
3-10s). Bin counts came out well-balanced (6.7k-24.9k rows/bin, none near
the `MIN_ROWS_PER_BIN=50` identity-fallback floor) — the earlier LOG worry
about thin edge bins under equal-width binning did not materialize on the
full corpus.

**Result** (held-out `overnight_randlen` test, 10,751 transitions, swept-
region `accuracy` — see `docs/experiments/METRICS.md` for the metric,
`Baselines/LinearForesight/runs/operators_accuracy.json` for full numbers):
switching by push-length bin clearly beats the single global operator
already used as the "linear" reference row everywhere else in this repo:

| model | aggregate accuracy |
|---|---|
| linear-switched (this baseline, 6 bins) | **0.2883** |
| linear-single (existing repo-wide reference) | 0.1893 |
| mean-delta | 0.0729 |
| persistence | 0.0000 |

and switching wins in **every individual length bin**, not just on
aggregate (e.g. [40.0,53.3) mm: 0.334 vs 0.295; [53.3,66.7) mm: 0.361 vs
0.298) — the shortest bin ([0,13.3) mm) is negative for every model
including switched (-0.31), i.e. worse than doing nothing, which is
expected: a push that barely moves the pile has almost no signal to fit
against, and even a per-bin-restricted operator cannot avoid modelling
noise as if it were displacement there.

This is a genuine, citable one-step accuracy result specific to
`overnight_randlen`, not just a pipeline correctness check — recorded as
**EXP-0003** / claim **C-003** (`docs/experiments/REGISTER.md`,
`docs/experiments/EXP-0003-switched-linear-foresight-randlen.md`), grade
`very-low` (exploratory, single split, one-step accuracy only — see that
record's Threats section for exactly what would raise it).

## Open questions for whoever runs the first real fit

- ~~Whether 6 equal-width bins actually balance sample counts reasonably~~
  — **resolved below**: bin counts on the full corpus were 6.7k-24.9k rows,
  nowhere near the identity-fallback floor.
- Whether `--constraint nonneg` (the paper's own best-performing variant)
  changes the switched-vs-single comparison; not run here, `ridge` was
  chosen as the default specifically to isolate the switching question
  from the constraint question (see SPEC.md §3).

## 2026-09-10 (same day, follow-up) — resolution ablation + control-utility pass

User asked for three extensions to the above: (1) repeat the fit at 32×32
(the paper's own resolution) alongside the existing 64×64; (2) score
control-utility ("MPC performance"), the same way EXP-0002 did for GNN/NFD;
(3) confirm the single-operator reference used for comparison is retrained
on the current data, not a stale checkpoint — it already was (`fit_switched.
py` always fits `linear-single` fresh, in the same process, on the same
loaded train tensors as the switched operators — see the "retrained, not
stale" note added to `predictor.py`/`SPEC.md`), but this was made more
visible: `single_operator` and `mean_delta` are now bundled INTO the same
`operators_res<R>.pt` file the switched operators are saved to (previously
only used internally by `fit_switched.py`'s own eval table, not persisted
or loadable by a predictor), and `predictor.py` gained
`SingleLinearForesightPredictor`/`build_predictor_single` reading that same
bundle — so both models are always demonstrably fit on identical data.

Also renamed the output convention: `runs/operators.pt` →
`runs/operators_res<R>.pt` (R=64 or 32), so the two resolutions' bundles
never collide. The old `runs/operators.pt`/`runs/operators_accuracy.json`
(pre-extension, missing `single_operator`/`mean_delta`) were deleted, not
kept alongside — superseded, not a separate result.

Control-utility scoring reused the EXISTING shared harness
(`Baselines/common/eval_report.py`, the same script EXP-0001/EXP-0002 ran)
rather than writing a new one: added 4 `MODELS` entries
(`linear_switched_res64`, `linear_single_res64`, `linear_switched_res32`,
`linear_single_res32`), no changes to the harness's own logic. Ran on the
pooled `randlen_test` corpus only (not the piled/scattered/mixed
stratification EXP-0002 added) — "like EXP-0002" was read as "same
methodology", not "also stratify by spawn mode"; that stratification is a
cheap follow-up (`--corpora randlen_piled,randlen_scattered,randlen_mixed`)
if wanted later.

**Result**: switched beats single on BOTH metrics, at BOTH resolutions, in
every cell tested (image accuracy: 2/2 resolutions; capture: 18/18
goal×value-fn×resolution cells). One finding not resolved yet: res32's
switched operator has LOWER image accuracy than res64's (0.248 vs 0.288)
but HIGHER control-utility capture (lyapunov 0.878 vs 0.780) — accuracy and
capture disagree on which resolution is "better". Full numbers, and what
would resolve that disagreement, are in **EXP-0003**
(`docs/experiments/EXP-0003-switched-linear-foresight-randlen.md`, updated
in place rather than superseded — same claim, extended evidence), which
also now depends on `randlen-step0-pool-size-128` and
`goal-mask-axis-convention-row-y-col-x` (the latter still `unchecked`,
inherited downgrade from EXP-0001/EXP-0002).

Bug fixed en route (see `INVARIANTS.md`'s `push-frame-warp-roundtrip` row,
newly registered this session): none new this round — the
`need_step_idx` fix from the same day's earlier entry was reused as-is by
`fit_switched.py`'s two reruns (res64, res32) and by `eval_report.py`'s
unmodified `_load_cell` (which still needs `need_step_idx=True` for
capture's step-0 grouping, and that path was already exercised
successfully by EXP-0001/EXP-0002 before this session touched anything).
