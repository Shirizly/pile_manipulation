---
# ---- identity -------------------------------------------------------------
id: EXP-0007
title: A FiLM latent predictor beats dz=0 by ~10x on a noise-free geometric target, but naively training the design doc's Stage-2 loss collapses the encoder -- the design has no anti-collapse mechanism
tier: T1
mode: exploratory
date: 2026-09-13
hypothesis: null

# ---- the claim ------------------------------------------------------------
claim: >
  On 6 files (3072 transitions, `Genesis/data/overnight_randlen/piled/cube/
  n20/size0.005/layers4`), a reduced-width FiLM-conditioned residual latent
  predictor (`docs/experimental_design/hybrid_linear_latent.md` Stage 2:
  `L_latent = ||P(z,a) + z - E(T_a(X))||^2` for the analytic, purely
  kinematic whole-pile-shift target `T_a`) beats the trivial `dz=0` (predict
  no change) baseline by roughly 10x on both train and holdout loss --
  PROVIDED an anti-collapse term (a per-channel variance hinge, not part of
  the design doc's stated Stage-2 loss) is added; trained on the doc's loss
  exactly as written, the encoder collapses (z_std ~2e-6, both losses and
  their ratio become meaningless) and the go/no-go question cannot be
  answered at all.

prediction: null  # exploratory go/no-go probe

# ---- how the numbers were made -------------------------------------------
provenance:
  commit: 0ddab20f
  dirty: true                      # same pre-existing unrelated dirty tree
                                    # as EXP-0004..0006; this record's own
                                    # inputs are untracked temp scratch.
  data_commit: "unrecorded (overnight_randlen has no provenance block, same gap as EXP-0001..0006)"
  script: "experiments/temp/latent-killprobe/probe.py"
  data: ["Genesis/data/overnight_randlen/piled/cube/n20/size0.005/layers4/*_data.pt (6 files, raw states/states_/p_starts/p_stops/angles schema, not the registry PileSweepData path)"]
  code_path: "transforms.functional.particles_to_occupancy (rasteriser); transforms.functional.warp_affine_occ (analytic T_a); model/NFDUNetFilm.py FiLMConvBlock/FiLMGenerator (predictor); a reduced-width conv encoder defined in probe.py itself (not a project module)"
  seed: 0
  split: "random disjoint split, 2458 train / 614 holdout (614/3072 ~ 20%), not file-level -- flagged, see Threats"
  runtime: "not recorded precisely; single short probe run per the task's 'GO/NO-GO probe' framing"
  runs: []

budget:
  declared: "not separately declared for this promotion; the original probe ran under an unstated short go/no-go budget"
  spent: "promotion pass: ~10 min / ~15k tokens"
  outcome: within

design:
  varied:
    training_objective: ["L_latent as specified (no regulariser) -- COLLAPSES", "L_latent + VICReg-style per-channel variance hinge (weight 1.0, target std 0.1) -- the fix applied"]
  held_fixed:
    encoder_width: "8->8->16->16->32->32 (REDUCED from the design doc's specified 32->64->128->256 -- a probe, not the real build; see Threats/incomplete-design"
    target: "T_a(X), a pure rigid translation of the whole occupancy field by the world push vector -- analytically identical to a canonical +x shift via to_push_frame/from_push_frame, computed directly instead for less code"
    action_encoding: "[x_p/0.064, y_p/0.064, sin(theta_p), cos(theta_p)], start-of-push reference point, ground-truth plate angle from the dataset's own recorded angles field"
    predictor: "P(z,a), 2 FiLM blocks, cond_dim=4, zero-initialised output conv so P(z,a)=0 at step 0"
    training: "400 steps, batch 64, Adam lr 1e-3"
  baselines: ["dz=0 (predict no latent change) -- the mandatory do-nothing baseline for this claim"]
  metric: "loss_model / loss_baseline ratio in latent space (NOT accuracy or slateN -- this probe never reconstructs or scores an image; also not yet a key in experiments/METRICS.md, see Threats). accuracy/slateN reference rows are not reported here since neither was computed by this probe -- flagged, not silently omitted"

noise_floor: "not measured -- single 400-step run, single train/holdout split (not file-level), no seed sweep. The train/holdout ratio agreement (0.093 vs 0.095) is the only internal consistency check available; not a substitute for a real floor."

depends_on: [push-frame-warp-roundtrip, occ-rasteriser-consistency]
establishes: [hybrid-latent-stage2-anticollapse]

# ---- outcome --------------------------------------------------------------
result: >
  With the variance-hinge fix: loss_model/loss_baseline ratio = 0.093
  (train) / 0.095 (holdout) -- the predictor beats dz=0 by roughly 10x,
  latent non-degenerate (z_std=0.148, mean-abs=0.542). Without any
  anti-collapse term (the design doc's Stage-2 loss exactly as written):
  the encoder collapses (z_std ~2e-6), both losses fall to ~1e-9/1e-11
  together, and their ratio (~39x, i.e. numerically "worse" than baseline)
  is meaningless -- a degenerate solution that trivially satisfies the
  objective, not a measurement of anything.
verdict: supported
downgrades: [imprecision, indirectness, incomplete-design, untested-dependency]
grade: very-low
supersedes: []
invalidated_by: null
---

## Why this test discriminates

The GO/NO-GO question ("can a FiLM latent predictor beat dz=0 on a clean
geometric target") is answered cleanly by the regularised run: a 10x margin
on both train and holdout, with a non-degenerate latent, is not a marginal
result that could plausibly be noise. The collapse finding discriminates a
different, more important question: whether the design doc's Stage-2 loss
AS WRITTEN is sufficient on its own. It is not -- the collapsed run's near-
identical model/baseline losses (both ~1e-9 to ~1e-11) is the specific
signature of `E` mapping every input to (near) the same vector, which
trivially minimizes `||\hat z' - z'||^2` for both the model and the `dz=0`
baseline at once. That signature is diagnostic, not ambiguous: a real
predictive win and a collapsed degenerate solution do not look alike once
`z_std` is checked.

## What was actually run

A single go/no-go probe, explicitly scoped smaller than a real Stage-2
build: encoder width reduced from the design doc's 256-channel bottleneck to
32, 400 training steps (not a full schedule), 6 files / 3072 transitions
(not the full corpus), and a directly-computed analytic `T_a` (mathematically
equivalent to routing through `to_push_frame`/`from_push_frame`, chosen for
less code since there is no rotation to handle in this purely-translational
target). The probe was run twice: once exactly as the design doc specifies
(no regulariser) -- collapsed -- and once with an ad hoc VICReg-style
variance hinge added specifically to get past that collapse. Both runs are
reported; the un-regularised run's numbers are reported as diagnostic of the
collapse, not as a real ratio.

## Numbers

| run | loss_model | loss_baseline (dz=0) | ratio | z_std |
|---|---:|---:|---:|---:|
| no regulariser (as specified) | ~1e-9 | ~1e-11 | ~39 (meaningless -- collapsed) | ~2e-6 |
| + variance hinge, train | 0.00272 | 0.02934 | **0.093** | 0.148 |
| + variance hinge, holdout | 0.00282 | 0.02980 | **0.095** | (same run) |

## What would change the verdict

- **The full-width encoder** (32->64->128->256, per the design doc) -- not
  tested; this probe deliberately used a reduced width to keep the go/no-go
  check cheap. Whether collapse is more or less likely at full width is
  unknown.
- **The real Stage-3 dynamics target** (only material in front of the plate
  moves, not the whole pile) is harder than this probe's purely kinematic
  whole-pile-shift `T_a` -- the 10x margin here is an upper bound on what a
  Stage-3 probe would likely show, not a preview of it (the source RESULTS.md
  says this explicitly).
- **A principled anti-collapse mechanism** added to the design doc itself
  (variance/covariance regulariser, contrastive term, or a frozen/pretrained
  encoder) rather than an ad hoc probe-only patch -- this is the actual
  action item the collapse finding implies; see `establishes` above.

## Threats

- **`imprecision`**: single 400-step run, single non-file-level split, no
  seed sweep; the 0.093/0.095 train/holdout agreement is suggestive but not
  a real noise floor.
- **`indirectness`**: `T_a` is a purely kinematic, noise-free, whole-pile
  rigid shift -- an easier target than real Stage-3 dynamics (where only
  material in front of the plate should move). The 10x margin should not be
  read as a preview of a real dynamics-prediction margin. Also: this
  probe's metric (a latent-space loss ratio) is not `accuracy` or `slateN`
  and is not yet a key in `experiments/METRICS.md` -- flagged here rather
  than added, per this task's instruction to leave `METRICS.md` edits to
  the concurrent slaten-broad work.
- **`incomplete-design`**: reduced encoder width (32 vs the doc's 256-dim
  bottleneck), only 6 files, only 400 steps, no seed/width sweep.
- **`untested-dependency`**: `occ-rasteriser-consistency` is `unchecked` in
  `INVARIANTS.md`, inherited from EXP-0001..0006 for the same reason.
- Considered and dismissed: **provenance mismatch** between the model and
  baseline losses -- both are computed from the same encoder/predictor
  state at every step (`||\hat z' - z'||^2` vs `||z - z'||^2`), not a
  separately-trained baseline, so the ratio is not confounded by that.

## Unrelated findings

- **`docs/experimental_design/hybrid_linear_latent.md` section 9 (Training
  stages, Stage 2) specifies `L_latent = ||\hat z' - z'||^2` with no
  variance, covariance, or contrastive term anywhere in the loss** -- this
  is the design gap this probe's collapse exposes. Tracked as an
  `INVARIANTS.md` tag (`hybrid-latent-stage2-anticollapse`, status
  `broken`, established by this record) rather than left as prose, since
  future work building Stage 2 for real should not have to re-discover this
  by re-hitting the collapse. No pytest exists for it yet -- there is no
  reference implementation of Stage 2 in the codebase to test against
  (only this probe's throwaway `probe.py`); the xfail pattern should be
  applied once a real Stage-2 module exists.
