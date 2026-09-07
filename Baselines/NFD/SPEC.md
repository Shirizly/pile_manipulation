# NFD baseline — design spec (non-FiLM, built on `UNetModels_modular.UNet`)

Status: design only, nothing trained. Written by agent A2-grid-specs for
implementation by a later agent. Source paper: Xue, Cheng, Kachana, Xu,
*"Neural Field Dynamics Model for Granular Object Piles Manipulation"*,
CoRL 2023 (`docs/reference_papers/Neural Field Dynamics Model for
Granular Object Piles Manipulation.pdf`, 17 pages incl. appendix).

**Scope correction (applied):** the orchestrator initially asked for a
non-FiLM NFD "based on the UNet files"; the user then pinned this down
explicitly to `model/UNetModels_modular.py`'s `UNet` class, registered as
`unet-modular` (`registry/model_registry.py:305`), **not**
`model/NFDUNetFilm.py`. Everything below targets `UNet` /`unet-modular`.
`NFDUNetFiLM` appears only as the comparison row (current best measured
model, `accuracy` 0.419 L20mm / 0.504 L40mm, see `Baselines/ORCHESTRATION_LOG.md`).

Every numbered fact below is tagged with where it came from:
**[paper text]**, **[paper Fig.7 / appendix]**, **[repo code]**, or
**[inference]** (my own reasoning — never treat as something the paper says).

---

## 1. What NFD actually is

### 1.1 State representation — **[paper text, §3.1]**

State `s ∈ R^{H×W}`, a single-channel "density field" obtained by segmenting
a top-down RGB image into a one-channel occupancy grid after orthographic
projection. In their sim, 50 cubic 1cm blocks on a table, so a cell's value
is essentially binary occupancy (possibly soft at the pile boundary).

Our state is the same kind of object: `occ0`/`occ1`, 64×64, box 0.128 m
across → 2 mm/px **[task brief / repo convention]**.

### 1.2 Action encoding — **[paper text, §3.1, Fig.7]** — the crux question

The paper does **not** encode the action as a single delta/flow channel and
does **not** hand the network raw pose numbers. It renders the pusher pose
into the *same* density-field representation as the state, via a
differentiable rasterizer (a simplified SoftRasterizer, Liu et al. 2019),
and treats state and action as one unified multi-channel image:

> "we propose mapping the pushing action into a similar spatial field-based
> representation... we assume linear pusher motion during a short interval
> `[t, t+1)` and represent an action as `aₜ := [r(xₜ), r(xₜ₊₁)]`, where
> `xₜ ∈ SE(2)` is the pusher pose at time t and `r: SE(2) → R^{H×W}` is a
> differentiable rendering function that rasterizes a pusher pose into a
> one-channel image."

So the action is **two separate one-channel renders** — the pusher's
oriented footprint rasterized once at its start pose and once at its end
pose — concatenated to the state channel-wise. Fig.7 (the architecture
diagram, reproduced in §3 below) shows exactly **3 input channels** labelled
"3" at the network's input, matching `[state, r(xₜ), r(xₜ₊₁)]`. There is no
merging of the two renders into one channel anywhere in the paper.

**Ablation evidence that this matters** — Appendix C.4, Table 5: replacing
DVF's vector-based action with this field-based action representation
("DVF‑Improved") cuts sequential-prediction error by ~3–4× and rollout cost
by ~2–10× at every dataset size tested (2k/6k/20k examples). The paper
treats the *separateness* of the two channels, not just "the action is
rendered," as the source of the gain — quote: "our proposed field-based
action representation is not limited to pile manipulation and has the
potential to be used in many model-based learning methods."

### 1.3 What the existing repo actually feeds in — **[repo code]**

Two independent code paths build the "action" channel, and they agree with
each other but **disagree with the paper**:

- `Genesis/training/dataset.py::PileSweepData._draw_plate` (training-time
  ground truth construction):
  ```python
  occ1 = draw_plate_soft(start_center, angle, grid_size, plate_dim_x, plate_dim_y, intensity=0.5, sigma=sigma)[0]
  occ2 = draw_plate_soft(end_center,   angle, grid_size, plate_dim_x, plate_dim_y, intensity=1.0, sigma=sigma)[0]
  self._input_grid[1] = 1 - (1 - occ1) * (1 - occ2)          # soft OR, ONE channel
  ```
- `model/eulerian_wrapper.py` (deployment-time, ~line 1361): same pattern,
  `act_start`/`act_end` at intensities 0.5/1.0, then
  `act_ch = torch.maximum(act_start, act_end)` — again one channel.

Both call `transforms/functional.py::draw_plate_soft` — which **is** a
differentiable rasterizer, essentially the same primitive the paper calls
`r(x)` — but then **collapse the two renders into a single channel** via a
soft union, distinguishing start from end only by an asymmetric intensity
(0.5 vs 1.0) rather than by channel identity. `model_registry.py`'s
`in_channels: 2` default for `unetfilm`/`unetfilm-shallow` (`[occupancy,
action]`, `configs/model/unetfilm.yaml`) is a direct consequence of this
one-channel action. (There is also a now-dead `sweep-removed-*` `input_mode`
in the registry giving 3 channels — confirmed **not** the paper's 2-render
scheme: it adds a *swept-region mask*, a different concept, and the
docstring/`oldscripts/train_unet_genesis.py` mark it removed/unused.)

`transforms/functional.py::build_action_delta` (also referenced in the
brief) is a *different, unrelated* function — it builds a per-particle
Gaussian-influence displacement vector field for the **Lagrangian**
(PropNet/GNN) training path, not for any Eulerian/UNet model. It is not
part of the NFD action-encoding story at all.

**Bottom line: the single most important delta to make a faithful NFD is
in_channels 2→3, splitting the current one-channel union action into two
separate full-strength renders — one per pose — exactly as the paper's
`aₜ := [r(xₜ), r(xₜ₊₁)]`.** This does not require writing a new rasterizer:
`draw_plate_soft` already exists and already gets called twice per sample —
the fix is to *stop* combining its two outputs into one channel.

### 1.4 What the network predicts — **[paper text, §3.2, Eq. before it]**

`ŝₜ₊₁ = f_θ(sₜ, aₜ)` — the network predicts the **full next density field
directly**, not a delta and not a flow/warp field. This also distinguishes
it from `model/SpatTransNet.py`'s `EulerianSTN` (registered
`spatial-transformer`), which predicts a *warp field* — that is a different
paper's idea (Gaussian-Splatting-style or STN-style), not NFD's.

Whether the field is predicted with a residual add to the current state is
**not shown in Fig.7**: the figure's legend defines exactly six symbols
(Conv3×3, Conv1×1, MaxPool2×2, skip connection, Up-sampling2×2, "block
copied") and none of them is a residual/add operator, unlike Schenck's
Fig.3 which explicitly draws a "+" after the final 1×1 conv. **[paper Fig.7,
inference from its legend]**: the literal paper architecture predicts the
absolute next field, no residual skip to the input.

By contrast, **[repo code]** `NFDUNetFiLM.forward` ends with
`return self.head(d_features) + current_state` — an explicit residual add —
and `unet-modular`'s own factory defaults to `residual: true` ("predict a
delta added to input channel 0"). So residual-to-current-occupancy is a
repo/`unet-modular` convention, not something Fig.7 shows.
**[inference]**: for a quasistatic single push where only a small local
region of the 64×64 grid changes, residual framing is standard practice and
plausibly still helps even though the paper doesn't draw it — but a
strictly faithful NFD reproduction sets `residual: false`. Recommend trying
both; default to `residual: true` since it is cheap, matches what the
current best in-house model already does, and there is no paper evidence it
would hurt (see Sec. 4, "residual" row).

### 1.5 Loss — **[paper text, §3.2]**

```
L_train = ‖f_θ(sₜ, aₜ) − sₜ₊₁‖²_F
```
`‖·‖_F` is the Frobenius norm, so `L_train` is a **plain sum-of-squared-
errors regression loss** over the H×W grid — equivalent to MSE up to the
`1/(H·W)` normalisation constant. No BCE, no dice, no sharpness/TV
regularisation, no mass term is mentioned anywhere in the paper (main text
or appendix).

**[repo code]** contrast: the actual best-model run
(`configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml`) trains
with `loss.type: eulerian_combined`, weights `mse: 1.0, mass: 0.2` (dice/bce/
sharpness/tv all 0). The `mass` term (`training/losses.py`,
`EulerianCombinedLoss`, "mean absolute mass-conservation error") is a
**repo-only addition**, not in the paper. `EulerianTrainingWrapper`'s own
docstring says the model "returns raw logit... BCEWithLogitsLoss receives
the raw logit" for *some* configurations elsewhere in the repo — that
logit/BCE framing does not apply to NFD; the paper's loss is a plain
regression on a continuous field, so the output head should stay linear
(no sigmoid), which is already how `UNet.final_conv` is defined (plain
`nn.Conv2d`, no activation) — no delta needed there.

**Recommendation:** train with plain MSE (`mse: 1.0`, everything else 0)
for the faithful variant. A small `mass` weight (repo convention, e.g. 0.1–0.2)
is a reasonable second variant to compare, but label it explicitly as
"NFD + repo mass term," not "NFD."

### 1.6 Multi-step / recurrent rollout — **[paper text, §3.2, medium-confidence inference]**

The paper says, in the same paragraph as the single-step loss:

> "To perform sequential prediction for long-horizon, the predicted state
> ŝₜ₊₁ is fed again into the dynamics model along with the proposed action
> aₜ₊₁."

This sentence is the *only* place multi-step behaviour is mentioned, and it
is not accompanied by a summed/BPTT-style training loss equation — the only
loss equation in the paper is the single-step one above. Table 5's "Error
(Sequential Prediction)" column is reported as an **evaluation** metric,
comparing against DVF's own "Seq. Pred." evaluation setting from Suh &
Tedrake — evidence this describes chaining the model's own single-step
predictions at *test/rollout time* (open-loop autoregression for planning),
not a multi-step training objective. **[inference, medium confidence
— the paper is genuinely ambiguous here and I could not resolve it further
from the text]**: training is single-step only; "sequential prediction" is
autoregressive chaining applied after training, purely at rollout/eval
time.

**For our data:** each slate has 3 steps, ~23k transitions pooled
(L20mm+L40mm train). **[repo code]**: the current best in-house model
(`accuracy` 0.419/0.504) is *also* trained purely single-step — the
`expB_unetfilm_*` config feeds 90 independent `(occ0, action) → occ1`-style
rows (30 slates × 3 steps), no rollout/BPTT. Given that (a) this matches the
paper's own recipe as best as it can be determined, (b) it matches the
existing best-model's recipe, and (c) recurrent/BPTT training is real
implementation and tuning effort for an "exploratory scaffolding" baseline —
**recommend single-step training as the core deliverable**. If time
remains, a stretch goal is a 2-step scheduled-sampling fine-tune (feed the
model's own step-0 prediction back in as input for step-1 with some
probability) — but this is optional, not required for the baseline to be
"working, plugged-in, measured" per the orchestration priority order.

### 1.7 Where does FiLM come from? — **[paper text + repo code, high confidence]**

**FiLM is entirely a repo addition, not in the paper.** Evidence:
1. Nothing in the paper's text, Fig.7, or Appendix A/B mentions a
   conditioning vector, material properties, or per-sample modulation of
   any kind. The model's only inputs are the state+action image stack; the
   only output is the predicted field. Section 4 ("Our method can
   generalize to diverse environments with different dynamics", Table 4)
   demonstrates transfer to unseen pusher/object shapes **zero-shot, with
   no retraining and no conditioning input** — the opposite of needing a
   FiLM vector to specialise the network per material.
2. `model/NFDUNetFilm.py`'s own module docstring: *"NFD Shallow U-Net +
   FiLM conditioning... extended with Feature-wise Linear Modulation...
   Reference: Perez et al., 'FiLM: Visual Reasoning with a General
   Conditioning Layer', AAAI 2018."* — a different paper, cited explicitly
   in the file itself, confirming this is a deliberate in-house extension
   layered on top of Xue et al. 2023, not part of it.
3. **[repo data, checked directly]**: the physics vector FiLM conditions on
   (`friction, density, box_friction`, `cond_dim=3`) is **constant across
   the whole training set**, not a per-sample signal — the raw
   `Genesis/data/slates_multistep/n20_L20mm_train/*_data.pt` files carry
   only `states, states_, p_starts, p_stops, angles`, no per-sample physics
   fields, and the deployed config's physics values
   (`configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml`,
   `inference:` block) are fixed scalars (`particle_friction: 0.25,
   particle_density: 750.0, box_friction: 0.3`). A FiLM conditioned on a
   constant vector can only learn a constant per-channel affine shift —
   something a plain `Conv2d`'s bias term already provides. **Given this,
   dropping FiLM for `unet-modular` costs essentially nothing on this
   dataset, and is a fidelity *gain* relative to the paper (which never had
   it), not a loss.** This would change if a later dataset pools cells with
   genuinely varying per-sample physics — flag as a risk if that ever
   becomes the case (see §5, R3).

---

## 2. Concrete deltas — target: `UNetModels_modular.UNet` / `unet-modular`

Since `unet-modular`'s registered factory (`registry/model_registry.py:305`)
calls `EulerianTrainingWrapper(model, uses_physics=False)` and `forward(x)`
consumes **only** the input tensor (no physics argument at all — the
factory's own `uses_physics` default is `false`), **the action must enter
entirely through input channels.** §1.3 already established that this is
exactly what the paper does too — so there is no fidelity loss from
`unet-modular`'s physics-free signature; if anything it forces the more
faithful design (channel-based action, no side-channel conditioning at all).

Do not edit `model/UNetModels_modular.py` or `registry/model_registry.py`.
Everything below is either (a) a `structure_parameters` / factory-config
dict passed to the *existing* `UNet` class unmodified, or (b) new code
under `Baselines/NFD/` (a data-prep step that builds the 3-channel input,
and, if BatchNorm-removal is wanted for max fidelity, a *local copy* of
`UNet`/`DoubleConv` with BN stripped — never edit the shared file).

| `unet-modular` config key | recommended value | source |
|---|---|---|
| `in_channels` | **3** | `[occ0, r(p_start,angle), r(p_stop,angle)]` — paper §3.1, see §1.3 above. **This is the single most important delta.** |
| `out_channels` | 1 | paper Fig.7 (final "1" box) |
| `features` | **`[4, 8, 16]`** | paper Fig.7 encoder widths (top-left labels "3→4→4", then "8,8", then "16,16"). With `unet-modular`'s own bottleneck rule (`bottleneck_ch = features[-1]*2`), this *automatically* gives a 32-channel bottleneck — matching Fig.7's "32,32" boxes exactly, with zero extra config. Verified: `UNet({"features":[4,8,16], "in_channels":3, ...})` builds and runs at 64×64 (checked directly, forward pass returns `(B,1,64,64)`). |
| `kernel_size` | 3 | paper Fig.7: every "Convolution 3×3" symbol in encoder/decoder |
| `final_kernel_size` | **1** | paper Fig.7's very last arrow before the "1" output box is explicitly "Convolution 1×1" (green), distinct from the 3×3 convs elsewhere. `unet-modular` defaults `final_kernel_size = kernel_size` (i.e. 3) unless overridden — **must be set explicitly to 1** to match. |
| `activation` | `relu` | **[inference]**: paper doesn't name an activation; Fig.7 doesn't show one either (not part of its legend); Ronneberger et al. 2015 (the U-Net paper NFD explicitly cites, ref [33]) uses ReLU, and it's the repo default. No stronger paper evidence available. |
| `bottleneck_type` | `None` | Gives exactly Fig.7's bottleneck: `bottleneck_pre` (a `DoubleConv`, i.e. the two "32" conv boxes) with `self.bottleneck = nn.Identity()` after — no SE/FC/Transformer extra processing, which Fig.7 doesn't show either. |
| `bottleneck_kwargs` | `{}` | n/a for `bottleneck_type: None` |
| `residual` | **true (deviation, flagged)** — try `false` too | Fig.7 shows no residual/add symbol at all (§1.4) → faithful value is `false`. `true` is `unet-modular`'s own default and matches what both existing repo models already do; **[inference]** likely still helps for a quasistatic small-per-step field, no paper evidence either way. Recommend training both and reporting which the metrics prefer; do not silently pick one and call it "the paper's." |
| `mixed_blocks` / `activation_list` / `mixed_type` | leave at defaults (`[]` / unused) | These are `unet-modular`'s own experimental multi-activation knobs (mixing relu/silu/gelu/mish per block) — **no counterpart anywhere in the paper.** Using them would not be "NFD," faithfully or otherwise; leave disabled. |
| BatchNorm | kept (repo default) — **flag as a known deviation** | `DoubleConv` (used by every encoder/decoder/bottleneck block) hardcodes `nn.BatchNorm2d` after every conv, with no config flag to disable it. Fig.7 shows no normalisation step at all, and the paper text never mentions BatchNorm. `unet-modular` gives no way to turn this off without a local code copy. **Recommendation**: accept BN as a pragmatic, undocumented deviation for the primary run (it is a well-understood stabiliser and the network is tiny either way); if a stricter-fidelity ablation is wanted later, copy `DoubleConv` into `Baselines/NFD/` with the two `nn.BatchNorm2d(out_ch)` lines removed and compare. |
| Output activation | none (linear) | Matches §1.5 — paper's loss is a plain regression on a continuous field, and `UNet.final_conv` is already a bare `nn.Conv2d` with no activation. No change needed. |

**Measured parameter count** (instantiated directly, not estimated):
`UNet({"features":[4,8,16], "in_channels":3, "out_channels":1, "kernel_size":3,
"final_kernel_size":1, "activation":"relu", "residual":true,
"bottleneck_type":"None"})` → **30,541 parameters**, forward pass on
`(B,3,64,64)` → `(B,1,64,64)` confirmed to run. This is ~2× the paper's own
reported count (Table 2: `1.53e4` params for NFD) — the difference is
almost entirely BatchNorm's extra affine parameters (`2×channels` per BN
layer) partially offset by `DoubleConv`'s `bias=False` convs. Both numbers
are in the same "tens of thousands" regime the paper is making its
efficiency point about (vs. DVF's `6.95e7` and DPI-Net's `3.91e5`,
Table 2) — order-of-magnitude fidelity holds.

### Building the 3rd (and true 2nd) action channel

No new rasterizer is needed. `transforms/functional.py::draw_plate_soft`
already exists and is already called twice per sample in
`Genesis/training/dataset.py::PileSweepData._draw_plate` — the only change
(in new `Baselines/NFD/` code, not in the shared file) is to keep its two
outputs as **two separate channels** instead of unioning them:

```python
r_start = draw_plate_soft(p_start_xy, angle, grid_size, plate_len_px, plate_wid_px, intensity=1.0, sigma=sigma)
r_stop  = draw_plate_soft(p_stop_xy,  angle, grid_size, plate_len_px, plate_wid_px, intensity=1.0, sigma=sigma)
x = torch.stack([occ0, r_start, r_stop], dim=1)   # (B, 3, H, W)
```
Note: with two channels there is no longer any need for the existing code's
asymmetric 0.5/1.0 intensity trick (it existed only to let one union'd
channel disambiguate start vs. end) — both renders can use full intensity
1.0, since the network can trivially tell them apart by channel identity,
which is closer to the paper's actual `r(xₜ), r(xₜ₊₁)` (their rasterizer
also has no reason to weight the two poses differently).

---

## 3. Architecture as read off Fig.7 (paper appendix)

Rendered directly from the PDF (`docs/reference_papers/Neural Field
Dynamics Model for\nGranular Object Piles Manipulation.pdf`, page 12,
"Figure 7: Architecture of proposed neural network" — note: this repo's
filename literally contains an embedded newline between "for" and
"Granular", confirmed via `os.listdir`; pass the literal `\n` when opening
it). Legend: 3×3 conv (dark red), 1×1 conv (green), maxpool 2×2, upsample
2×2, skip connection, "block copied" (concat).

```
input (3ch: state, r(xt), r(xt+1))
  L1:  conv3x3→4, conv3x3→4  ──────────────────────────skip──────┐
        │ maxpool2x2                                              │
  L2:   conv3x3→8, conv3x3→8 ────────────skip───────┐             │
        │ maxpool2x2                                 │             │
  L3:   conv3x3→16, conv3x3→16 ──skip──┐             │             │
        │ maxpool2x2                   │             │             │
  BN:   conv3x3→32, conv3x3→32          │             │             │
        │ upsample2x2                   │             │             │
  L3':  concat(16+32)→conv3x3→16→conv3x3→16 ──────────┘             │
        │ upsample2x2                                 │             │
  L2':  concat(8+16)→conv3x3→8→conv3x3→8 ──────────────┘             │
        │ upsample2x2                                               │
  L1':  concat(4+8)→conv3x3→4→conv3x3→4 ───────────────────────────┘
        │ conv1x1→1
output (1ch: predicted next field)
```

This is exactly reproduced by `unet-modular`'s `features=[4,8,16]` +
default bottleneck-doubling + `final_kernel_size=1`, per §2.

---

## 4. Training recipe

**The paper gives no optimizer/lr/batch-size/epoch numbers for the dynamics
model anywhere in the main text or appendix** — searched the full extracted
text; the only numeric training details in the whole paper are for the
*trajectory optimizer's* cost weights (`α₁=1, α₂=2, α₃=1, α₄=2`, §3.3),
which are irrelevant to training the network. Everything below is
**[inference]**, anchored to (a) this repo's own trainer
(`training/trainer.py::_build_optimizer`, hardcoded `torch.optim.Adam` +
`StepLR`) and (b) the existing best-model's recipe
(`configs/training/expB_unetfilm_slates_multistep_n20_L20mm.yaml`), scaled
for a ~5× smaller model (30.5K vs. the FiLM model's larger width-doubling
[8,16,32] architecture) and a pooled ~23k-transition dataset instead of one
cell's ~11.5k.

| hyperparameter | recommendation | rationale |
|---|---|---|
| optimizer | Adam | repo's only supported optimizer (`trainer.py`); also what Schenck et al. use (5e-4) for a similarly-sized conv net, so at least one paper in this project's set validates Adam for this problem class |
| learning rate | 1e-4 to 3e-4, start with **2e-4** | existing best model uses 1e-4 for a larger net; this net is ~5× smaller with fewer BN-normalised layers, can likely tolerate a somewhat higher LR — no strong evidence either way, treat as a first-guess to sanity-check against a training-loss curve, not a committed value |
| LR schedule | `StepLR`, `step_size=40`, `gamma=0.7` | same family as existing recipe, adjusted step count for more available epochs (below) |
| batch size | 64 | 8 GB GPU has large headroom for a 30.5K-param net on 64×64×3 inputs (existing recipe used 32 for a bigger model); larger batch reduces epoch count needed and BN batch statistics are less noisy |
| epochs | 150, early-stop patience 25 | unlike the existing recipe (`patience: 100` = effectively disabled, because that run had `val_pct: 0`), a genuinely pooled ~23k-transition dataset can afford a real held-out validation slice; recommend `val_pct: 10` (slate-level split, not row-level — copy the existing slate-level split discipline noted in the L20mm config's comments) so early stopping is meaningful |
| grad clip | 1.0 | repo convention (`grad_clip_norm: 1.0` in existing config) |
| mixed precision | on | repo convention, cheap on an 8 GB card |
| loss | `mse: 1.0` only for the faithful run; a `mass: 0.1–0.2` variant as a labelled second run | §1.5 |
| data | pooled L20mm+L40mm train splits per `ORCHESTRATION_LOG.md`'s scope decision; eval per-cell on each cell's own `_eval` split | matches how the number-to-beat (0.419/0.504) was produced structurally, modulo per-cell vs. pooled training (already flagged as an expected small difference in the orchestration log) |

Total expected wall-clock: a 30.5K-parameter, 3-level UNet at 64×64 with
~23k transitions and batch 64 is on the order of a few seconds/epoch on an
RTX 4070 Laptop (the existing FiLM model, several times larger, already
trains a comparable dataset size in well under an hour for 100 epochs per
other logs in this repo) — expect single-digit minutes for the full 150-epoch
run, i.e. this is cheap relative to the GPU-lock contention risk, not a
bottleneck.

---

## 5. Risks / open questions (ranked)

1. **R1 — multi-step training ambiguity (§1.6).** I could not fully resolve
   whether "sequential prediction" in the paper is trained-for (BPTT/
   scheduled sampling) or purely a rollout-time evaluation of a single-step
   -trained model. My reading (single-step training) matches both the only
   loss equation given and the existing best-model's own recipe, but if a
   later agent finds contrary evidence (e.g. in the CoRL supplementary
   video/code, not available here), the training loop would need a rollout
   loss added. Medium confidence, worth a second look if time allows.
2. **R2 — BatchNorm cannot be disabled without a local code copy.** Fig.7
   shows no normalisation; `DoubleConv` hardcodes it. Decided to accept BN
   for the primary run (§2) rather than fork the file, but this means the
   "faithful" run is not literally BN-free. Low-severity, easy to fix later
   (a copy of `UNet`/`DoubleConv` under `Baselines/NFD/model.py` with BN
   stripped, if an ablation is wanted).
3. **R3 — physics conditioning is currently constant, but pooling could
   change that.** §1.7's "dropping FiLM costs nothing" argument depends on
   friction/density/box_friction being dataset-wide constants, verified for
   the L20mm cell's raw `.pt` files and the deployed inference config. If a
   later pooling adds cells with genuinely different physics per sample,
   the physics-free `unet-modular` path would lose real information FiLM
   would have captured — re-verify this assumption if the training data mix
   changes.
4. **R4 — residual framing (`residual: true` vs. `false`) is an
   open choice, not resolved by the paper (§1.4).** Both are cheap to run;
   recommend reporting both rather than picking one silently.
5. **R5 — output/state value range.** The paper calls `s` a "density
   field" (not strictly binary — could exceed the [0,1] range if the
   segmentation/downsampling accumulates overlapping objects per cell), but
   never states the exact range or whether it's clipped/normalised. Our
   `occ0`/`occ1` representation and range should be whatever
   `Baselines/common/data.py` already produces (documented as `(B,64,64)`
   occupancy in that baseline's own `LOG.md`) — treat that as ground truth
   for range/units rather than re-deriving it from the paper, since the
   paper's own description is underspecified here.
6. **R6 — exact figure provenance is a rendered image, not machine text.**
   The channel counts in §3 were read directly off the Fig.7 diagram
   (rendered to PNG and visually inspected, since `pdftoppm`/poppler was
   unavailable and text extraction cannot recover numbers embedded in a
   figure) rather than pulled from `pypdf` text extraction. High confidence
   in the reading (legend is explicit, labels are unambiguous, numbers were
   cross-checked against the params/GFLOP table), but noting the method for
   auditability.
