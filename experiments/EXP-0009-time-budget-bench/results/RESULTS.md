# Time-budget bench: timing GATE for the equalised-wall-clock control comparison

Produced by `experiments/EXP-0009-time-budget-bench/code/bench.py`. Timing
only -- no control/slateN evaluation was run (that is the larger experiment
this gates). Raw numbers: `artifacts/RUN-0001-timing-sweep/results_timing.json`.
Derived per-model candidate budgets: `budgets.json` (this dir, via
`../code/make_budgets.py`). Full stdout: `artifacts/RUN-0001-timing-sweep/run.log`.

Provenance: commit `0ddab20f` (dirty -- unrelated doc/skill reorg files),
branch `baselines/overnight`. GPU: RTX 4070 Laptop 8GB, idle at run start
(15 MiB used, 0% util) -- contention gate passed, not `--force`d.
`torch.cuda.is_available()=True`, all timings below on `cuda:0` unless noted.

## Device-bug fix

`Baselines/common/eval_baseline.py::_predictor_batch` builds `PredictorBatch`
straight from `load_cell`'s CPU tensors and never calls `.to(device)`; since
`NFDPredictor`/`SchenckPredictor` derive their compute device from
`batch.occ0.device`, both silently run on CPU AS ACTUALLY INVOKED BY THE
SCORER TODAY, while `GNNPredictor` forces its own `cuda-if-available` device
internally regardless of the batch (documented in
`Baselines/common/benchmark_time.py`'s own module docstring). Timing
"NFD/Schenck-on-CPU" against "GNN-on-GPU" would be meaningless.

**Fix applied**: this bench's own `build_batch()` moves every tensor
(`occ0`, `actions`, `p_start`, `p_stop`, `angle`) to `DEVICE` (`cuda`)
*before* handing it to any predictor, for every model -- NFD, Schenck,
GNN, MODEL-0001 (switched+global), MODEL-0002, hybrid14, hybrid94 alike.
Every model wrapper additionally calls `assert_all_on(DEVICE, ...)` on the
tensors actually fed to its forward call, and asserts
`next(model.parameters()).device.type == DEVICE` where the model is an
`nn.Module` -- introspected per call, not assumed. All 8 rows below
verified `device_actual == cuda` (or `cuda:0`); see `results_timing.json`
`results.*.device_actual`.

## Timing table (end-to-end = preprocessing + forward; median ms, K candidates)

| model | device | params | fwd-only @K=128 (ms) | K=1 | K=8 | K=32 | K=64 | K=128 | cand/s @K=128 |
|---|---|---|---|---|---|---|---|---|---|
| nfd (UNet, 3ch) | cuda:0 | 30,541 | 3.62 | 1.12 | 1.12 | 1.32 | 1.57 | 4.03 | 31,783 |
| schenck (16-layer CNN) | cuda:0 | 139,937 | 48.67 | 0.77 | 1.89 | 6.36 | 20.71 | 46.27 | 2,767 |
| gnn (PropNetDiffDen) | cuda:0 | 38,403 | 3.92 | 1.88 | 4.39 | 12.06 | 22.92 | 44.11 | 2,902 |
| model0001_switched (visual, 6-bin) | cuda | n/a (linear op) | 2.36 | 1.49 | 1.49 | 1.49 | 1.99 | 2.83 | 45,246 |
| model0001_global (visual, unswitched) | cuda | n/a | 1.24 | 1.25 | 1.22 | 1.21 | 1.31 | 1.71 | 74,661 |
| model0002_descriptor_only (94-dim) | cuda | n/a | 0.18 | 1.67 | 1.66 | 1.63 | 1.65 | 2.06 | 62,104 |
| hybrid14 (visual+14desc, switched) | cuda | n/a | 1.57 | 2.77 | 2.67 | 2.72 | 2.89 | 3.64 | 35,193 |
| hybrid94 (visual+94desc, switched) | cuda | n/a | 1.58 | 2.77 | 2.72 | 2.70 | 2.83 | 3.64 | 35,129 |

Full curve (all 11 batch sizes: 128,96,64,48,32,24,16,8,4,2,1), IQR, and
forward-only-at-every-K are in `results_timing.json`
(`results.<name>.by_k.<K>.{end2end_ms,fwd_ms,pre_ms}` each carry
`{median,q1,q3,n}`). >=5 warm-up (discarded), 15 timed repeats,
`torch.cuda.synchronize()` before/after every timed region, per
`Baselines/common/benchmark_time.py`'s methodology (reused, not
reinvented).

## GNN Python-loop caveat

`Baselines/GNN/predictor.py`'s `sample_nodes_xy` call loops per-sample in
Python (`for i in range(B)`). Measured: at K=128 this loop alone takes
~40ms of the ~44ms end-to-end cost (fwd-only is 3.9ms) -- i.e. **preprocessing,
not the graph-net forward pass, is >90% of GNN's wall-clock cost at large
K**, and it scales ~linearly in K exactly as a per-row Python loop would.
The single-sample cost (`loop_s / B`, `gnn_one_sample_s_estimate` in
`results_timing.json`) is ~0.31ms/sample at K=128 -- if this were batched
(vectorised FPS/foreground-extraction), GNN's preprocessing cost would be
closer to O(1) than O(B), which would put its *architectural* end-to-end
cost close to its 3.9ms forward-only number even at K=128 -- i.e. much
closer to NFD's curve than to its own measured curve. **This is an
implementation slowdown, not an architectural one**, and is reported as
such rather than silently absorbed into "the GNN is slow."

## GATE QUESTION: does the scaling have any bite?

**No -- the scaling is NOT uniformly flat; it has real bite, and it is
model-dependent**, so the equalised-wall-clock-budget design is NOT
vacuous:

- Three models (`model0001_global`, `model0002_descriptor_only`, and to a
  lesser extent `nfd`) are close to latency-bound: their end-to-end median
  barely moves from K=1 to K=64 (e.g. `model0001_global`: 1.25ms -> 1.31ms,
  a ~5% increase for 64x the candidates) and only start climbing near
  K=128.
- Three others (`schenck`, `gnn`, and `model0001_switched` less severely)
  are clearly throughput-bound: `schenck` goes from 0.77ms (K=1) to 46.3ms
  (K=128), a ~60x increase for 128x the work -- almost perfectly linear,
  i.e. genuinely paying per-candidate.
- `hybrid14`/`hybrid94` sit at an almost flat ~2.7-2.8ms across the ENTIRE
  range K=1..64 (their per-candidate visual+descriptor pipeline is cheap
  relative to fixed launch/warp overhead at these batch sizes), only
  climbing to 3.64ms at K=128 -- flat, but flat at a much higher absolute
  floor than `model0001_global`/`model0002`.

## Derived candidate budgets N_i (reference: fastest model @ K=128)

Fastest model at K=128 is **`model0001_global`** at **1.714 ms**
(unswitched linear operator on 32x32 canonicalised occupancy -- no
descriptor computation, no neural net). N_i = interpolated K at which each
model's own end-to-end median curve crosses 1.714 ms (linear interpolation
between adjacent measured K; see `make_budgets.py`).

| model | N_i (candidates in 1.714 ms) |
|---|---|
| model0001_global | 128.0 (reference) |
| model0002_descriptor_only | 74.5 |
| nfd | 69.6 |
| model0001_switched | 54.2 |
| schenck | 7.0 |
| gnn | **< 1** (K=1 already costs 1.88 ms > 1.714 ms budget) |
| hybrid14 | **< 1** (K=1 already costs 2.77 ms > 1.714 ms budget) |
| hybrid94 | **< 1** (K=1 already costs 2.77 ms > 1.714 ms budget) |

**The budget equalisation bites hard for 3 of 8 candidate models**: GNN,
hybrid14 and hybrid94 cannot even clear ONE candidate evaluation inside the
time the fastest model spends on 128 -- an MPC step built on the
equalised-budget design would have to let them evaluate a single action
(or, if the budget is a hard floor of N=1, borrow time from elsewhere) while
`model0001_global` ranks all 128. Schenck gets a real but survivable
budget (7 of 128 candidates, ~5%). NFD and the two `model0001` variants
land in the 54-128 range, i.e. they are not exempt from the equalisation
but are not crippled by it either.

**Recommendation: run the larger experiment.** The gate is non-vacuous --
if it had come back flat for all 8 (N_i >= 128 everywhere), the equalised
budget would rank every model on the full candidate pool and the control
comparison would degenerate into a pure accuracy comparison; that is not
what was measured. The 3 image+descriptor "hybrid"/GNN models in
particular will need their slateN/control-utility scored at N_i in the
single digits, which is exactly the interesting regime the larger
experiment is meant to probe (does a model that is very accurate per-call
but can only afford 1-7 candidates still beat a cheap model that gets to
rank all 128?).

## RUN-0002 (timing-corrected) -- amendment, not a replacement

See `EXPERIMENT.md`'s "Amendment" section for the full defect writeup
(sync stalls, untimed value function, uncharacterised precision). Numbers
below supersede RUN-0001 IN SCOPE (sync-free fwd, K in {1,32,128} only,
value function timed both ways); RUN-0001's own numbers above are not
invalidated -- they are correct for the narrower thing they measured.

End-to-end median ms, WITHOUT vs WITH the goal-value function
(lyapunov + mass_in_region) timed in the same region, at K=1/32/128:

| model | K=1 (no val / +val) | K=32 (no val / +val) | K=128 (no val / +val) | fwd@128 |
|---|---|---|---|---|
| nfd | 1.122 / 1.189 | 1.751 / 1.797 | 3.116 / 3.225 | 2.542 |
| schenck | 0.774 / 0.819 | 6.444 / 6.334 | 47.242 / 47.336 | 48.821 |
| gnn | 1.903 / 2.149 | 11.849 / 14.685 | 44.544 / 55.306 | 4.881 |
| model0001_switched | 1.755 / 1.828 | 1.788 / 1.850 | 2.166 / 2.165 | 2.534 |
| model0001_global | 1.269 / 1.327 | 1.239 / 1.270 | 1.746 / 1.810 | 1.254 |
| model0002_descriptor_only | 1.870 / 2.504 | 1.884 / 2.502 | 2.262 / 2.814 | 0.301 |
| hybrid14 | 2.938 / 2.961 | 2.834 / 2.937 | 3.862 / 3.947 | 1.801 |
| hybrid94 | 2.960 / 3.029 | 2.866 / 2.925 | 3.900 / 3.935 | 1.819 |

Full table (pre/fwd/end2end, with/without value, IQR):
`artifacts/RUN-0002-timing-corrected/results_timing_v2.json`.

**Sync-fix numerical equivalence** (defect 1): max abs diff between the old
`.any()`-guarded loop and the sync-free fix, same input, K=128 -- exactly
`0.0` for all 4 affected models (model0001_switched, hybrid14, hybrid94,
model0002_descriptor_only). Effect on timings: model0001_switched K=128
end2end 2.83ms (RUN-0001) -> 2.166ms (RUN-0002), a real ~23% drop; the
other 3 models barely moved (within the measured resolution floor below).

**Resolution floor** (defect 3): 3 separate-process runs of schenck@K=128
gave end2end medians 46.09/47.84/48.11ms (~2.0ms / ~4.3% spread) --
RUN-0001's "impossible" fwd_ms@128 (48.67) > end2end_ms@128 (46.27) is a
~2.4ms/5% inversion, fully inside this floor. **Differences under ~5%
relative (or ~2ms at the ~45ms scale) are not resolvable** and are not
used to rank models below.

**Does descriptor-only become fastest once the value function is timed?
No.** model0002_descriptor_only's point-mass value readout costs ~0.6ms
against its own ~0.03ms fwd (its N_i(T=2.5ms) collapses from 128, no
value, to unresolvable/<1 with value) -- a much larger relative hit than
model0001_global's image-space value cost (~0.06ms against ~0.9-1.2ms
fwd). model0001_global stays the most budget-robust model once scoring is
included.

**Budget sweep** (`results/budgets.json`, N_i(T), capped at 128, WITHOUT
value):

| model | T=1.714ms (orig) | T=2.5ms | T=5ms | T=10ms | T=25ms | T=50ms |
|---|---|---|---|---|---|---|
| nfd | 30.2 | 84.6 | 128.0 | 128.0 | 128.0 | 128.0 |
| schenck | 6.1 | 10.4 | 24.1 | 40.4 | 75.7 | 128.0 |
| gnn | None | 2.9 | 10.7 | 26.2 | 70.6 | 128.0 |
| model0001_switched | None | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 |
| model0001_global | 122.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 |
| model0002_descriptor_only | None | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 |
| hybrid14 | None | None | 128.0 | 128.0 | 128.0 | 128.0 |
| hybrid94 | None | None | 128.0 | 128.0 | 128.0 | 128.0 |

`T=1.714ms` (the original single reference) is **degenerate**: 5 of 8
models can't clear even K=1. `T>=50ms` is **vacuous**: everyone saturates
at 128. **`T~=2.5ms` is the most informative single budget** -- it spreads
N_i from ~3 (hybrid14/94, unresolved/near-floor) to 128 without collapsing
the whole field to one value. (The `with_value` sweep is in
`budgets.json` directly; it shifts descriptor-only's numbers sharply
downward per the finding above, everyone else only modestly.)

## Caveats / scope not covered here

- Only ONE real state (`configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`,
  row 0) was used, repeated K times with K real candidate actions cycled
  from the same eval cell -- matching `Baselines/common/benchmark_time.py`'s
  own "same state, many candidate actions" MPC-inner-loop shape. State
  rasterisation (particle -> occupancy) was done once, outside every timed
  region, per the task's explicit scope rule.
- `model0001_switched`/`model0001_global` share one checkpoint
  (`weights/MODEL-0001-stage2-visual-switched/checkpoint.pt`) -- same
  32x32 visual state, switched (6 length-bin operators) vs. a single
  global operator; this is exactly the "operator variant" ablation the
  task asked to time separately.
- No control/slateN scoring was run. That is explicitly out of scope for
  this timing gate.

## RUN-0003 (budgeted-control) -- the control half

See `code/eval_budgeted_control.py`, `artifacts/RUN-0003-budgeted-control/results_control.json`,
`runs/RUN-0003-budgeted-control/{RUN.md,COMMAND.txt}`. Datasets: `n20_L20mm`
+ `n20_L40mm` (n20_L10mm excluded), 20 slates x 128 candidates each. Goal
shape: **corner** only (scope reduction, documented). Value functions:
**lyapunov** (lead) + **mass_in_region**. `N_i(T)` from `budgets.json`'s
`with_value` sweep. slateN = `budgeted_slate_capture`: same
`(chosen-mean)/(best-mean)` formula as `Baselines/common/goals.py::
slate_n_capture`, but the model only RANKS within its drawn `N_i`-sized
subset while `best`/`mean` are always computed over the FULL 128-pool
(oracle/floor never gets easier for a budget-limited model).

**T=2.5ms is flagged as inside RUN-0002's measurement noise floor**
(~5% relative / ~2ms absolute resolution; cross-process drift for fast
models exceeds 30%) -- reported below, not led with. T=5/10ms are the
trustworthy cells.

### Pooled slateN (n20_L20mm + n20_L40mm), lyapunov, mean-of-means across the 2 datasets

5-draw headline (top/bottom/mean over draws 0-4; top==bottom==mean when
N_i=128, no sampling variance) at each budget T. N_i shown per model; a
model with N_i=128 at that T is NOT budget-limited there. `persistence`
and `random` are the mandatory floor baselines, always scored on the full
128-pool (never budget-limited).

| model | K=32 fixed ref | T=2.5ms (NOISE FLOOR) N_i / top-bot-mean | T=5ms N_i / top-bot-mean | T=10ms N_i / top-bot-mean |
|---|---|---|---|---|
| nfd | 0.882 | 79 / 0.978-0.940-0.957 | 128 / 0.979 (no variance) | 128 / 0.979 |
| model0001_switched | 0.895 | 128 / 0.972 | 128 / 0.972 | 128 / 0.972 |
| hybrid14 | 0.881 | **INFEASIBLE** (N_i<1 with value) -- see note | 128 / 0.972 | 128 / 0.972 |
| hybrid94 | 0.885 | **INFEASIBLE** | 128 / 0.972 | 128 / 0.972 |
| model0001_global | 0.835 | 128 / 0.952 | 128 / 0.952 | 128 / 0.952 |
| schenck | 0.889 | 10 / 0.807-0.683-0.734 | 24 / 0.884-0.832-0.863 | 40 / 0.944-0.876-0.909 |
| gnn | 0.890 | 1 / 0.152-(-0.177)-(-0.010) | 8 / 0.783-0.638-0.713 | 20 / 0.870-0.789-0.832 |
| model0002_descriptor_only | 0.667 | **INFEASIBLE** | 128 / 0.663 | 128 / 0.663 |
| persistence (floor) | 0.003 | 128 / -0.167 | 128 / -0.167 | 128 / -0.167 |
| random (floor) | 0.067 | 128 / 0.117 | 128 / 0.117 | 128 / 0.117 |

`mass_in_region` shows the same qualitative ranking at a lower absolute
scale (e.g. K=32 fixed ref: nfd=0.773, model0001_switched=0.791,
model0001_global=0.714, schenck=0.773, gnn=0.767, model0002_descriptor_only
=0.348, persistence=0.022, random=0.030) -- full table in the JSON.

"INFEASIBLE" = `budgets.json`'s `with_value` N_i is `null` at that T (even
K=1 exceeds the time budget once the value function's own cost is
included) -- these models cannot be scored under that budget at all, not
scored at N=0.

### 50-draw distribution vs the 5-draw headline (lyapunov, per-dataset, not pooled)

| dataset | model | T | N_i | 5-draw [top, bottom] | 50-draw [p5, p95] |
|---|---|---|---|---|---|
| n20_L20mm | gnn | 5ms | 8 | [0.689, 0.497] | [0.396, 0.624] |
| n20_L20mm | gnn | 2.5ms | 1 | [0.149, -0.057] | [-0.166, 0.186] |
| n20_L40mm | gnn | 2.5ms | 1 | [0.154, -0.297] | [-0.289, 0.242] |
| n20_L20mm | schenck | 5ms | 24 | [0.792, 0.716] | [0.714, 0.844] |
| n20_L40mm | schenck | 10ms | 40 | [0.975, 0.945] | [0.949, 0.982] |

In 2 of the ~12 budget-limited (dataset,model,T) cells checked by hand
(gnn@n20_L20mm/T=5ms, gnn@n20_L40mm/T=2.5ms), the 5-draw TOP exceeded the
50-draw's own 95th percentile (and one 5-draw BOTTOM fell below the 50-draw
5th percentile) -- with only 20 slates per dataset and N_i as low as 1-8,
5 draws is not always a reliable stand-in for the true sampling
distribution for the LOWEST-N_i, HIGHEST-variance cells (gnn specifically).
For schenck (N_i>=10) the 5-draw range sits inside or very close to the
50-draw [p5,p95] band -- representative there. **Conclusion: the 5-draw
headline is trustworthy for N_i>=~10; for gnn's N_i in {1,8} it can
overstate the achievable spread in either direction and the 50-draw
band should be treated as the more reliable number.**

### Wins/losses/ties (draw 0, lyapunov, pooled qualitatively -- see JSON for
exact per-dataset counts, n=20 slates/dataset)

`model0001_global` vs `persistence`: model0001_global wins essentially
every slate (persistence's raw slateN is near 0/negative by construction --
it never moves the pile). `gnn` vs `schenck` at T=5ms: schenck wins more
slates than gnn loses to it at this budget (N_i=24 vs 8), but the paired
sem (from `head_to_head` in the JSON) is wide given n=20 slates/dataset --
treat any single-dataset head-to-head with n=20 as a POWER LIMITATION, not
a resolved ranking, unless it holds in both datasets AND the paired sem is
small relative to the gap (see JSON `paired_sem_of_diff`).

### Verdict

- **Under a real time budget, NFD is the best control model overall,
  closely followed by the switched-linear MODEL-0001/hybrid family**,
  across BOTH trustworthy budgets (T=5ms, T=10ms) and both value functions.
  model0001_global (the cheapest, most budget-robust model per RUN-0002)
  is close behind but consistently ~0.02-0.05 slateN below nfd/switched at
  T=5/10ms -- i.e. paying for the switched/higher-capacity family is
  affordable once T>=5ms.
- **The ranking does NOT flip with T** in the trustworthy range (T=5,10ms):
  nfd and the switched-linear/hybrid family are at or near ceiling by
  T=5ms already (their N_i=128 there); schenck and gnn are the only models
  still climbing between T=5 and T=10ms, and NEITHER overtakes NFD or the
  linear family at T=10ms -- gnn's T=10ms mean (0.832 pooled) is still
  below model0001_global's T=2.5ms number (0.952), let alone nfd's. **No
  CNN/GNN model overtakes the linear family at any T tested here.**
- At **T=2.5ms** (flagged noise floor), model0002_descriptor_only and both
  hybrids are INFEASIBLE (their own value-inclusive per-candidate cost
  already exceeds 2.5ms for even 1 candidate) -- this is a real, if
  noise-floor-adjacent, finding: the value-function overhead specifically
  penalises the descriptor-only/hybrid family at the tightest budget,
  consistent with RUN-0002's own finding.
- **The 50-draw distribution shows the 5-draw top/bottom is representative
  for schenck (N_i>=10) but can be misleading for gnn (N_i in {1,8})** --
  see table above. This matters specifically for the gnn-vs-schenck
  comparison at T=5ms, which should be read as noisy, not as a settled
  ranking.
- **Every difference at T=2.5ms is inside RUN-0002's measurement noise
  floor** and is not used to rank models there beyond the INFEASIBLE flag
  (which is a hard `null` in budgets.json, not a close call). At T=5/10ms,
  the nfd/model0001-family-vs-schenck/gnn gap is far larger than the noise
  floor (e.g. nfd pooled 0.979 vs gnn pooled 0.713 at T=5ms, lyapunov) and
  is a genuine ranking, not a power limitation.
