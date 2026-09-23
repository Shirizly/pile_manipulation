---
# ---- identity -------------------------------------------------------------
id: EXP-0025
title: >
  A flow/advection output head (predict a per-pixel displacement field and warp
  the occupancy through it) does NOT beat direct next-occupancy prediction at
  pilot scale; coarse flow resolution is the one design choice that helps, and
  it alone beats the direct control on slateN/lyapunov
tier: T1
mode: exploratory
date: 2026-09-23
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On a subset of `slates_multistep/n20_L20mm_train` at ~40 epochs, an NFD UNet
  whose head emits a per-pixel displacement field, through which the current
  occupancy is backward-warped (`grid_sample`), achieves higher `slateN` and
  higher swept-region `accuracy` on the L20mm eval cell than an
  identically-trained UNet emitting the next occupancy directly.

# prediction: omitted -- mode is exploratory. This is a design sweep to find
# which parameters matter, not a pre-registered threshold test.

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: a175b981
  dirty: true                     # concurrent multi-agent session; see EXP-0022
  data_commit: "unrecorded (slates_multistep carries no provenance block)"
  script: >
    Baselines/NFD/train_nfd.py with Baselines/NFD/configs/nfd_train_flow_*.yaml
    and nfd_train_direct_control_L20mm_pilot.yaml (training);
    Baselines/common/eval_report.py (scoring, the same harness every other
    model in this repo is scored through)
  data: ["Genesis/data/slates_multistep/n20_L20mm_train (subset, training)",
         "configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml (scoring)"]
  code_path: >
    Baselines/NFD/flow_nfd_lib.py (training-side flow head) /
    Baselines/NFD/flow_predictor.py::FlowWarpPredictor (eval side);
    backward warping via torch.nn.functional.grid_sample.
    NOTE (post-run refactor, pure code move, no behaviour change, no
    retraining): this code moved to model/flow_nfd/lib.py /
    model/flow_nfd/predictor.py::FlowWarpPredictor after this experiment's
    runs completed -- the sha-plus-path above no longer resolves as written.
    Registered type name (`nfd-flow-warp`) is unchanged.
  seed: 0
  split: "PileSweepData file-granularity hash split, val/test 5/5; eval is the separate held-out L20mm eval cell"
  runtime: "6 cells, ~3-6 min each on a contended GPU; scoring ~20s per cell on CPU"
  runs: [RUN-0001, RUN-0002, RUN-0003, RUN-0004, RUN-0005, RUN-0006, RUN-0007]
  env: "python 3.10, torch 2.11.0+cu130 (anaconda3/envs/pme), RTX 4070 Laptop 8 GB"

budget:
  declared: "70 min wall-clock, 140k tokens"
  spent: "training completed within budget; the delegated agent was terminated by an org spend limit before scoring, and the scoring was completed directly afterwards"
  outcome: exceeded

design:
  varied:
    flow_resolution: ["full 64x64", "coarse 16x16 upsampled"]
    max_displacement_px: [4, 12, 24]
    source_sink_term: [false, true]
    output_head: ["flow/advection", "direct next-occupancy (control)"]
  held_fixed:
    architecture: "UNetModels_modular.UNet, features [4,8,16], final_kernel_size 1"
    action_encoding: "3-channel [occ0, r(p_start), r(p_stop)]"
    loss: "eulerian_combined, mse 1.0, all other terms 0, against the absolute occ1 target"
    optimiser: "Adam lr 1e-4, StepLR, batch 32, augmentation on"
    corpus_subset_and_epochs: "identical across all six cells"
    grid_resolution: "64x64 (resolution_scale 0.5)"
  baselines: [persistence, random, "direct-prediction NFD trained on the same subset for the same epochs (flow_direct_control)"]
  metric: "slateN (leads; 3 goals x 3 value functions), swept-region accuracy (beside it, flagged suspect per experiments/METRICS.md)"

noise_floor: >
  Not measured -- one training run per cell, no seed repetition. The ordering
  among the middle cells must not be over-read. The two conclusions drawn are
  chosen to survive this: the direct control's margin over the best flow cell
  on accuracy (0.294 vs 0.213) and largedisp24's NEGATIVE accuracy are both
  far larger than any plausible single-run spread, whereas the gaps between
  flow_baseline, flow_smalldisp4 and flow_srcsink are not, and no claim rests
  on them.

depends_on: [occ-rasteriser-consistency, goal-mask-axis-convention-row-y-col-x]
establishes: []

# ---- outcome --------------------------------------------------------------
result: >
  Direct-prediction control: accuracy 0.294, slateN 0.608/0.738/0.487
  (lyapunov/mass_in_region/signed_mass). Best flow cell (coarse 16x16):
  accuracy 0.213, slateN 0.754/0.517/0.447. The control wins on accuracy by a
  wide margin and on 2 of 3 value functions; `flow_coarse16` beats it on
  slateN/lyapunov (0.754 vs 0.608), the metric that decides, but loses the
  other two. Design parameters: coarse flow resolution clearly beats full
  per-pixel (0.213 vs 0.141 accuracy); a 24 px displacement bound is
  catastrophic (accuracy -0.027, i.e. WORSE than predicting no change at all);
  a source/sink term hurts (0.089 vs 0.141 for the otherwise-identical cell).
  EXTENSION (RUN-0008..0014, directly-supervised flow): a MODEL-FREE ceiling --
  warping occ0 by the GROUND-TRUTH particle displacement field -- scores
  accuracy 0.167, BELOW the direct control's ACHIEVED 0.294, so no
  translation-only backward-warp flow model can win on accuracy however well it
  is learned. The same ceiling on slateN is 0.813/0.770/0.512, ABOVE every cell
  tried, so the refutation does NOT extend to the deciding metric. Directly
  supervising the flow from particle correspondence reached 0.707/0.559/0.411,
  accuracy 0.180 -- better than the unsupervised full-resolution cell, still
  short of coarse-pooled flow and of the direct control.
verdict: refuted
downgrades: [imprecision, incomplete-design, provenance, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

Only ~1.75% of pixels change per push (`changed_pixel_frac`, EXP-0022's
training logs), so a direct model spends almost all of its output capacity and
loss gradient re-emitting its input. A displacement field is mass-conserving by
construction, residual by construction, and far lower-dimensional. If that
structural advantage is real, a flow head should beat a direct head trained on
the same subset for the same number of epochs with the same action encoding,
loss and optimiser — which is exactly the control included in the sweep. It did
not, so the structural argument does not by itself buy accuracy here.

## The exception worth not burying

`flow_coarse16` **beats the direct control on `slateN`/lyapunov**, 0.754 vs
0.608. `slateN` is the metric this project treats as deciding, and `accuracy`
is explicitly the suspect one. It loses on the other two value functions, so
this is not a win — but it is the kind of single-value-function disagreement
`experiments/METRICS.md` warns about, and it is the one result here that would
justify looking again, particularly on a corpus with more than one push length.

## Caveats that limit how far this refutation reaches

- Short-epoch cells on a data SUBSET: this ranks design choices against each
  other, it does not establish any cell's absolute quality.
- **`n20_L20mm` is a single-push-length corpus.** A displacement field's
  natural advantage is expressing how far material moves as a function of push
  length — the one thing this corpus cannot show. The negative is therefore
  weaker evidence than it looks, and `flow_coarse16`'s lyapunov win
  correspondingly more interesting.
- The prior negative result this approach already had in this project was
  reproduced here with a like-for-like control and a parameter sweep, rather
  than a single configuration — which is the substantive addition.

## What would change the verdict

- **Re-run `flow_coarse16` and the control on `overnight_randlen`** (multi-length).
  This is the single cell worth the GPU time, for the reason above.
- **Seeds.** One run per cell; no noise floor.
- A coarse-resolution sweep (8x8, 16x16, 32x32) — resolution was the only
  parameter that clearly helped, and only two values were tried.

## Threats

- **`imprecision`**: no noise floor (above).
- **`incomplete-design`**: the decisive multi-length cell was not run, and
  resolution — the parameter that mattered — was sampled at only two values.
- **`provenance`**: dirty tree; the delegated agent was terminated mid-task by
  an org spend limit and the scoring was completed separately afterwards, so
  training and scoring did not happen in one uninterrupted pass.
- **`untested-dependency`**: `occ-rasteriser-consistency` is `unchecked`.

## Extension: directly-supervised flow (RUN-0008..0014)

`states`/`states_` carry **consistent particle indexing**, so the flow need not
be a latent variable at all: a ground-truth displacement field can be built from
particle correspondence and the flow supervised directly, removing both failure
modes diagnosed above (drift in unidentified regions, and the capture-radius
limit on finding large displacements photometrically).

**The model-free ceiling is the most valuable number this experiment produced,
and it splits by metric:**

| | accuracy | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| ground-truth flow field (ceiling) | **0.167** | **0.813** | **0.770** | 0.512 |
| `flow_direct_control` (achieved) | 0.294 | 0.608 | 0.738 | 0.487 |
| `flow_coarse16` (achieved) | 0.213 | 0.754 | 0.517 | 0.447 |
| `flow_supervised_masked` (augmented) | 0.180 | 0.707 | 0.559 | 0.411 |
| `flow_supervised` (augmented) | 0.150 | 0.576 | 0.454 | 0.256 |

- **On `accuracy` the parameterisation is structurally dead.** A PERFECT flow
  field scores 0.167, below what the direct control actually achieves (0.294).
  The cause was stratified rather than asserted: accuracy is worst in the
  NEAR-STATIC bin and improves with displacement, i.e. bilinear `grid_sample`
  blurs a near-binary occupancy field even at ground truth, and that blur costs
  most exactly where persistence error is already near zero. **This is the
  opposite of the capture-radius mechanism** that explains the displacement-bound
  results, and it is a distinct, previously undocumented reason flow-style heads
  lose on this metric.
- **On `slateN` the ceiling is ABOVE every cell tried** (0.813 lyapunov vs
  `flow_coarse16`'s 0.754), so there is real headroom this attempt did not
  reach. The refutation is therefore specific to `accuracy` and to the cells
  actually trained — it does not bound the idea on the metric that decides.

**A dropped-augmentation confound was caught and corrected.** The first
supervised cells trained without the x8 augmentation, seeing 768 effective
samples against the other cells' 6,144, which is a sufficient alternative
explanation for underperformance on its own. Re-run with augmentation matched,
every metric improved substantially and **the internal ranking flipped**: the
masked-magnitude-penalty cell went from worst (apparently collapsed, slateN
below the random floor) to best of the two supervised cells (lyapunov
-0.018 -> 0.707). Its apparent collapse was an artifact of data starvation, not
a property of the penalty. The unaugmented runs are kept on disk as an
accidental data ablation.

**The x8 augmentation is not valid for a vector field without extra work**, and
this was verified rather than assumed. `training/trainer.py::_augment_eulerian_batch`
transforms `input`/`target`/`physics`/`push_px` and silently drops anything
else, and even for `push_px` it permutes point COORDINATES — it has no notion of
rotating a vector's COMPONENTS. A separate augmenter rotates the flow target's
components with the transform's linear part, checked by equivariance
(`warp(R(occ0), R_vec(R(flow))) == R(warp(occ0, flow))`) over 40 sample x
rotation x flip combinations: max abs diff 3.8e-6. A negative control that skips
the component rotation — the exact bug — gives max abs diff 1.0, so the check
has power.

## Unrelated findings

- `docs/ARCHITECTURE.md` lists `NCAModels.py` and `SpatTransNet.py` under
  `model/futureintegration/`; they actually live at `model/` top level and are
  imported from there by `registry/model_registry.py`. Stale doc path, noticed
  while looking for the prior flow-style attempt.
- The six checkpoints from this sweep all survived the delegated agent's
  termination and were scored afterwards without retraining — the value of
  persisting every fitted object beside its resolved config, exactly as
  `experiment-log` requires. Had only the metrics been kept, the whole sweep
  would have had to be re-run.
