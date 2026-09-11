# Linear Foresight baseline (switched, per-push-length) — design spec

**Source paper:** H.J.T. Suh & R. Tedrake, *The Surprising Effectiveness of
Linear Models for Visual Foresight in Object Pile Manipulation*,
arXiv:2002.09093v3. Full paper summary, fidelity analysis, and integration
plan: [`docs/linear_visual_foresight_baseline.md`](../../docs/linear_visual_foresight_baseline.md)
(referred to below as "the plan doc"). This SPEC is the narrower, Baselines-
subdir-local counterpart of that doc, following the shape `Baselines/GNN/
SPEC.md`/`Baselines/SchenckCNN/SPEC.md` use.

**Status:** fitting + eval-harness integration built and run, at both 64x64
(this repo's own convention) and 32x32 (the paper's own resolution), on
both image accuracy and step-0 control-utility (`slateN`/capture). No
closed-loop MPC integration (the plan doc's Steps 4/5/7 — an enumerating
action sampler, the Lyapunov MPC cost, a runnable config) is built here;
"control-utility" below means the same step-0 candidate-ranking proxy
`Baselines/common/eval_report.py` already uses for GNN/NFD (EXP-0001/
EXP-0002), not an actual closed-loop rollout. This subdir answers one
question: *fit the paper's switched-linear operator on
`Genesis/data/overnight_randlen` and check whether the switching (choosing
`A` by push length) helps over the single global operator fit on the same
data* — see `docs/experiments/EXP-0003-switched-linear-foresight-randlen.md`
for the answer (yes, at both resolutions, on both metrics, in every cell
tested).

## 1. What already existed, and what this subdir adds

The pixel-space switched-linear machinery — the SE(2) push-frame warp
(`transforms/functional.py`), the OLS/non-negative-least-squares fit
(`fit_linear_foresight.py::fit_operator`/`fit_operator_nonneg`), and the
warp→matvec→unwarp→blend apply path (`fit_linear_foresight.py::
predict_world`) — already existed at repo root, built for datasets
collected at ONE fixed push length
(`Genesis/configs/collection_foresight_single_operator.yaml`,
`--perpendicular-pushes --push-length`). `Baselines/common/eval_baseline.py`
and `Baselines/common/eval_randlen_indist.py` additionally already fit and
score a SINGLE global operator (ridge=1.0 toward identity) on
`overnight_randlen` as a "linear" reference row — but that is the paper's
degenerate 1-bin case, not its switched design (their §3.1: **one operator
per push-length bin**, 5 bins in their own setup).

`overnight_randlen` is exactly the corpus the paper's switching was
designed for: push length is randomised per transition rather than fixed
(`Genesis/data/overnight_randlen/DATASET.yaml`; realised length ranges from
sub-millimetre to ~80 mm), so a single operator averages over pushes that
barely touch the pile and pushes that sweep 80 mm through it. This subdir
is the missing piece: **fit one operator per push-length bin on this
corpus, and report whether that beats the single-operator reference.**

Nothing warp/fit/apply-related is duplicated — `model.py` and
`fit_switched.py` both import the existing primitives directly (`from
fit_linear_foresight import ...`) rather than re-implementing them, the same
relationship `Baselines/GNN/predictor.py` has to root-level `model/
gnn_dyn.py`.

## 2. Binning rule

Bin edges are `n_bins` (default 6) **equal-width bins over `[0,
max(push_length_m)]`**, where `push_length_m = ||[ex,ey] - [sx,sy]||` is
computed once per transition from the world-metre action
(`model.py::push_length_m`) — resolution-independent, so bin membership
does not depend on `--res` or a dataset's `resolution_scale`.

Two choices worth being explicit about, since both are judgment calls, not
derivable facts:

- **Low edge is 0, not the observed minimum.** Push length is a physical
  quantity that cannot be negative, and a fixed floor keeps bin edges
  reproducible across different train subsamples (the observed minimum
  would otherwise shift every edge by a data-dependent amount). The
  corpus already filters near-zero pushes (`min_push_length_m: 0.0001`),
  so in practice this differs from using the observed minimum by well
  under a millimetre.
- **High edge is the observed max on the TRAIN split actually loaded**, not
  a value read off `DATASET.yaml`'s `sweep_length_dist` config (which is
  itself marked `unverified` there, and is a sampling *distribution*
  parameter, not a guaranteed realised bound — actuator/wall clamping can
  shift what is actually realised, see the plan doc §7.2's discussion of
  start-nudging). Fitting reads the true range off the data it loaded
  rather than trusting a config value.

6 bins (not the paper's 5) was the number requested for this corpus; the
choice is a CLI flag (`--n-bins`) either way, not a constant.

**Underdetermined bins.** With `n_bins=6` over 192 pooled train files
(~512 transitions/file), each bin gets on the order of 10-20k rows — but
the distribution over length is not uniform (the sampler's `main_range`
concentrates mass in the middle of the range, `DATASET.yaml`), so the
lowest/highest bins can be thin. `fit_switched.py` falls back to the
identity operator (pure persistence) for any bin with fewer than
`MIN_ROWS_PER_BIN=50` rows rather than fitting on noise — the same
"degrade toward persistence, not toward erasing the pile" principle
`fit_linear_foresight.fit_operator`'s `toward_identity` ridge target
already documents.

## 3. Fit recipe

Per bin: `Y_{k+1} = A Y_k` fit by ridge-regularised OLS toward the identity
(`fit_operator`, their eq. 8, `ridge=1.0` by default) or by the paper's own
non-negativity constraint (`fit_operator_nonneg`, their eq. 9, FISTA — see
that function's docstring for why FISTA over per-row QPs). **Default is
`ridge`**, not `nonneg` — deliberately, because the existing single-operator
"linear" reference row this subdir is meant to be compared against also
uses `ridge=1.0`; defaulting the switched fit to `nonneg` would confound
"does switching help" with "does the constraint help", two different
questions. `--constraint nonneg` is available for the second question.

Resolution is a CLI flag (`--res`), run at **both 64×64** (matching
`Baselines/common/eval_baseline.py`'s `R=64`, for direct comparability with
the existing reference) **and 32×32** (the source paper's own resolution —
the ablation plan doc §8 calls for). Each resolution gets its own bundle
file (`runs/operators_res64.pt` / `runs/operators_res32.pt`) so one never
silently overwrites the other. See
`docs/experiments/EXP-0003-switched-linear-foresight-randlen.md` for the
resolution comparison (switching wins at both; a `linear-switched`
accuracy-vs-capture disagreement BETWEEN the two resolutions is flagged
there as still unexplained).

Each `fit_switched.py` run fits and bundles TWO operators together —
`operators` (the `n_bins` per-bin set) AND `single_operator` (the plain
global operator, identical recipe, same loaded train tensors, same
process) — specifically so a switched-vs-single comparison is never
confounded by comparing a freshly-fit operator against a stale, previously
saved one. There is no separate "fit the single operator" script in this
subdir.

## 4. File map

| File | Role |
|---|---|
| `fit_switched.py` | CLI: bins `overnight_randlen`'s TRAIN split by push length, fits one operator per bin PLUS the single global reference operator (same data, same process), evaluates both on the held-out TEST split against persistence/mean-delta, writes `runs/operators_res<R>.pt` + `runs/operators_res<R>_accuracy.json` |
| `model.py` | `push_length_m`, `bin_index`, `predict_switched` — the length-based dispatch to per-bin operators via the existing `predict_world` |
| `predictor.py` | `SwitchedLinearForesightPredictor` + `SingleLinearForesightPredictor` — the `BaselinePredictor` contract (`.name`, `.predict_occ`) so both models plug into `Baselines/common/eval_baseline.py`/`eval_randlen_indist.py`/`eval_report.py` exactly like GNN/NFD/SchenckCNN |
| `runs/operators_res<R>.pt` | fitted bundle at resolution `R`: `{operators: [A_0..A_{n_bins-1}], single_operator, mean_delta, bin_edges, counts, res, crop, constraint, ridge, n_bins, train_cfg, provenance}` |
| `runs/operators_res<R>_accuracy.json` | held-out image-accuracy report from that fit run (aggregate + per-bin) |
| `runs/control_utility_report.json` | step-0 `slateN`/capture report (3 goals x 3 value functions) for all 4 model/resolution combinations, via `eval_report.py` |

## 5. Usage

```bash
# fit both resolutions (all 5 overnight_randlen groups, 6 bins, ridge)
PYTHONPATH=. python Baselines/LinearForesight/fit_switched.py --res 64
PYTHONPATH=. python Baselines/LinearForesight/fit_switched.py --res 32

# score image accuracy + control-utility capture via the shared cross-
# corpus harness, alongside GNN/NFD (Baselines/common/eval_report.py's
# MODELS dict already carries linear_switched_res64/32 and
# linear_single_res64/32)
PYTHONPATH=. python Baselines/common/eval_report.py \
    --models linear_switched_res64,linear_single_res64,linear_switched_res32,linear_single_res32 \
    --corpora randlen_test \
    --out-prefix Baselines/LinearForesight/runs/control_utility_report

# or the pairwise in-distribution harness (single --predictor at a time)
PYTHONPATH=. python Baselines/common/eval_randlen_indist.py \
    --predictor Baselines.LinearForesight.predictor:build_predictor \
    --tag linear_switched_randlen_indist \
    --out-prefix Baselines/LinearForesight/runs/linear_switched_randlen_indist
```

`LINEARFORESIGHT_CKPT` overrides the operator bundle path (default
`runs/operators_res64.pt`), mirroring `GNN_CKPT`/`NFD_CKPT` — point it at
`runs/operators_res32.pt` to score the 32×32 ablation through
`eval_randlen_indist.py`/`eval_baseline.py` directly.

## 6. Known limitations

- **Length is the only switching variable**, matching the paper's own
  design. `fit_linear_foresight.py`'s separate contact-score-bin experiment
  (`--bins`) switches on a different variable (how much pile mass the
  blade will contact) and is not combined with length binning here — that
  would be a 2D switching rule (length × contact), which the plan doc's
  §7.4 discusses in the analogous angle-binning case as a real interface/
  data-budget cost, not attempted in this baseline.
- **No angle/orientation switching** (plan doc §3, §7): pushes are treated
  as perpendicular-push-equivalent via the same warp every other operator
  in this repo uses; `overnight_randlen`'s actual plate yaw is sampled
  semi-independently of push direction (`perpendicular_yaw: true` per
  `DATASET.yaml`, but see the plan doc §7.1 for the general case), so this
  inherits whatever fidelity gap that entails, not a new one.
- **No CLOSED-LOOP MPC hookup, still.** `runs/control_utility_report.json`
  gives step-0 candidate-ranking control-utility (`slateN`/capture) —
  the same proxy EXP-0001/EXP-0002 use for GNN/NFD, and the reason those
  records still carry an `indirectness` downgrade despite testing
  "control" — not an actual multi-step closed-loop rollout. A real
  Lyapunov-cost closed loop through `simple_mpc/` still needs the plan
  doc's Steps 3-5, not built here.
- **A ~470MB operator bundle at res64** (six dense 4096×4096 float32
  matrices plus the single operator; ~29MB at res32). Nothing else in
  `Baselines/*/runs/` is close to this size and there is no `.gitignore`/
  git-LFS handling for it yet — a decision to make before committing
  `runs/operators_res64.pt`.
