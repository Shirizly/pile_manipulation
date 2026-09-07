# Schenck CNN baseline — design spec

Status: design only, nothing trained. Source: Schenck, Tompson, Fox, Levine,
*"Learning Robotic Manipulation of Granular Media"*, CoRL 2017
(`docs/reference_papers/Learning Robotic Manipulation of Granular Media.pdf`,
10 pages). Roughly half the length of `Baselines/NFD/SPEC.md` per brief.

Tags as in the NFD spec: **[paper text]**, **[inference]**.

---

## 1. What the model is

The paper evaluates four predictive models for scoop-and-dump manipulation
of pinto beans; the strongest is the **scoop & dump–net**, a fully
convolutional network with no pooling/upsampling at all — every layer stays
at the input's full spatial resolution. Architecture **[paper Fig.3, read
from extracted PDF text — the layer list is textual in this PDF, not
embedded in an image, so no visual rendering was needed]**:

- Two stacked towers, each **16× `Conv 32@3×3` (+ ReLU, implied by "fully
  convolutional... comprised of only convolution and ReLU layers")**,
  followed by one `Conv 1@1×1`, followed by a residual **"+"** (predicted
  delta added back onto the current height-map — the only one of the two
  reference papers whose diagram explicitly draws this residual symbol).
- **Top tower**: input = current height-map `hₜ` + action map (see §2) →
  predicts the height-map *change due to the scoop only*.
- An explicit mass-conservation channel: the top tower's predicted mass
  removed is summed and passed as an extra scalar-broadcast channel into the
  bottom tower.
- **Bottom tower**: input = `hₜ` + top tower's predicted scoop-delta + the
  mass-conservation channel + action map → predicts the *dump* delta.
  Final output = `hₜ` + scoop-delta + dump-delta.
- The simpler **single–net** ablation is just the top tower alone (no
  scoop/dump split, no mass-passthrough), predicting the whole transition in
  one shot — **[paper text, §4.1]**: "the layout of the single–net is
  identical to the top half of Figure 3."

State representation: a **height-map**, a 2D grid where each cell holds the
surface height of the granular media at that location, **[paper text,
§3]** "each cell is approximately 1cm × 1cm." The paper never states the
grid's overall pixel dimensions (tray size in cells) — this is a genuine
gap in the text, not resolved by anything else in the PDF.

## 2. Action encoding — **[paper text, §3.1]**

The real action is a 9D `scoop & dump` parameter vector: start location
(2D), start angle (1D), end location (2D), end angle (1D), roll angle (1D),
dump location (2D) — a scoop-drag-lift-carry-pour primitive, not a push.

This 9D vector is reparameterized into an **"action map"**, an image the
same size as the height-map:
- draw a straight line from the scoop start to end location, its colour
  linearly interpolated red→green along its length (encodes progress /
  direction, not just position);
- draw a blue dot at the dump location;
- tile each of the 3 angles (start angle, end angle, roll angle) across 3
  extra constant-valued channels, concatenated to the position map.

**[inference]**: read literally, the "line drawn in colour from red to
green, plus a blue dot" is most naturally an RGB image (3 channels) encoding
position/progress, and the text explicitly adds "3 extra channels" for the
angles — so the total action-map channel count is most plausibly **6**
(3 position/colour + 3 tiled angle), stacked with the 1-channel height-map
for **7 input channels** total. The paper's own Figure 2 caption only shows
"an action map (without the angles)" as a single illustrative image and
never states a channel count in the text — this exact number (6 vs. some
other split of the colour encoding) is not fully certain from the text
alone; flagged as an open question (§4, R1).

## 3. Loss and training — **[paper text, §6.2]**

- **single–net**: L2 loss on the predicted next height-map.
- **scoop & dump–net**: L2 loss on the final predicted height-map **plus**
  an equally-weighted L2 loss on the top tower's intermediate scoop-only
  prediction (supervised against a recorded real intermediate height-map,
  captured after the scoop but before the dump) — with **gradient stopped**
  from the final loss into the top tower, so the top tower is trained
  *only* by its own intermediate loss and cannot use the dump-visible
  target to shortcut the split.
- Optimizer: **Adam, lr = 5e-4**, mini-batch gradient descent (batch size
  not stated numerically anywhere in the text).
- Schedule: pre-train 30,000 iterations on the hand-coded baseline model's
  outputs (§4.3 in the paper — a geometric heuristic, not a learned model),
  then fine-tune 100,000 iterations on real robot-collected transitions.
  Paper reports this pretraining empirically helped ("we empirically
  determined that the networks performed better with pre-training than
  without").
- Data: ~15,000 real scoop&dump examples total (10,000 collected using the
  hard-coded baseline's policy, 5,000 more collected using a scoop&dump–net
  trained on the first 10,000, i.e. one round of DAgger-like self-improving
  data collection).

## 4. What does not map onto our setup

This is the important part of a "half-length" spec — most of the paper's
machinery is built around an action type and sensing modality we don't have:

- **Scoop-and-dump vs. plate push.** The entire two-tower / mass-passthrough
  architecture exists specifically to separate a *lift-and-carry* action
  (scoop, which removes mass from one place) from a *pour* action (dump,
  which adds mass somewhere else) — with an explicit inter-tower mass
  conservation channel enforcing "what left the scoop must land at the
  dump." Our task is a single continuous plate push: nothing is picked up,
  carried, or poured; there is no scoop phase and no dump phase to split.
  **Reproducing the two-tower split faithfully would mean building an
  architecture for a physical process we don't have** — recommend not doing
  this. The **single–net** ablation (one FCN tower, no split, no mass
  channel) is the right level of fidelity for a plate-push task, and the
  paper's own results (Fig.5a) show single–net and scoop&dump–net had
  *similar* held-out test-set error — the split's benefit shows up in
  closed-loop task performance (Fig.5b/c), which is specifically about
  correctly attributing where scooped mass ends up, a non-issue for a push.
- **9D scoop parameterization vs. 2-endpoint push.** Our action is fully
  described by `p_start`, `p_stop` (world metres) and a single blade
  `angle` — 5 real numbers, no roll angle, no separate start/end scoop
  angle, no dump location. The action-map idea (rasterize the action into
  an image aligned with the state grid) transfers directly and cheaply; the
  9-parameter vector itself does not apply and shouldn't be forced to.
- **Real depth-camera height-maps vs. simulated cube poses.** Their height-
  map comes from a calibrated RealSense-style depth camera, height-
  thresholded and arm-occluded-region-masked, over a bed of beans that can
  pile up in the vertical direction (a genuinely 2.5D signal). Our state is
  simulated: exact cube poses rasterized into a 2D occupancy grid, no depth
  sensor noise, and (per the repo's Genesis pile setup) cubes are pushed
  around on a flat surface without necessarily stacking. If cubes in our
  sim never meaningfully stack, a binary/soft **occupancy** grid (what
  `Baselines/common/data.py` already produces) is the right analogue of
  their height-map and no height channel is needed; if stacking does occur
  non-trivially, an occupancy grid *silently discards* exactly the
  information their height-map was designed to capture (how much is piled,
  not just where) — worth a quick empirical check (see §4 risks) before
  assuming occupancy is an adequate substitute.
- **Mass conservation as an architectural constraint vs. ours.** Their
  explicit "sum of scooped mass, passed to the dump tower" mechanism has no
  meaning without a discrete scoop/dump split. If a mass-conservation prior
  is wanted for a push model, `training/losses.py::EulerianCombinedLoss`'s
  existing `mass` term (already used for the current best UNet-FiLM model,
  weight 0.2) is the repo-native way to add one — not this paper's
  mechanism.

## 5. Recommended minimal faithful adaptation

Given the mismatches above, the honest reproduction target is **not** the
scoop&dump–net but the **single–net**, adapted to our representation:

- Input: `occ0` (64×64, our resolution — 0.128 m box vs. their ~1cm cells,
  scale is a free choice, 64×64 matches the rest of this project's
  convention) concatenated with an action map built from `p_start, p_stop,
  angle`: a rasterized line/plate footprint between the two endpoints
  (reuse `transforms/functional.py::draw_plate_soft`, exactly as recommended
  for the NFD spec — same primitive serves both baselines) plus the single
  yaw `angle` tiled as one constant channel (we have one angle, not three,
  since there's no separate start/end/roll angle for a rigid plate push).
- Architecture: a single tower, ~12-16 layers of `Conv 32@3×3 + ReLU` at
  constant 64×64 resolution (no pooling, matching the paper exactly — this
  is a deliberate difference from NFD's UNet-style downsampling; the
  paper's stated rationale is that mass conservation across the scoop/dump
  boundary requires exact registration, but the flat, no-pooling depth also
  simply gives a large receptive field — 16 layers of 3×3 convs ≈ 33×33 px
  effective receptive field, a large fraction of a 64×64 grid, without
  needing multi-scale pooling), ending in `Conv 1@1×1` then a residual add
  of `occ0`.
- Loss: plain L2 (MSE) on the predicted next occupancy, matching the
  single–net's stated loss exactly — no scoop/dump split loss applies.
- Training: Adam, lr 5e-4 (paper's stated value, one of only two papers in
  this project's set that actually gives one — reuse it rather than
  guessing), batch size not paper-specified, **[inference]** use 32-64 to
  match the other baselines' convention; paper's 30k-iteration
  baseline-pretrain step doesn't apply here (there's no cheap geometric
  baseline predictor for a plate push comparable to their scoop heuristic;
  skip pretraining and just train on real transitions from iteration 0).

## 6. Feasibility (one sentence, as requested)

Feasible and cheap as the single–net (a plain deep 3×3-conv stack, no
pooling, residual output, L2 loss, action rasterized onto the state grid)
— but the paper's headline architecture (the two-tower scoop&dump–net with
inter-tower mass conservation) is not worth reproducing here because it is
built entirely around a lift-carry-pour action our plate-push task doesn't
have.

## Risks / open questions (ranked)

1. **R1 — action-map channel count (§2) is my best reading, not a stated
   number.** The paper never gives an explicit channel count for the
   RGB-line-plus-dot encoding; 6 total action channels (3 colour + 3 tiled
   angle) is inferred from "we draw... a blue dot... tile the 3 angles
   across 3 extra channels", not read off a figure or table.
2. **R2 — height-map vs. occupancy grid may not be equivalent** if cubes
   in the sim ever stack; worth a quick check of the raw slate data (max
   z-overlap across the 20 cubes) before assuming a single occupancy
   channel captures everything a height-map would.
3. **R3 — no stated grid resolution.** "~1cm cells" is given, but total
   tray extent (hence total grid pixel count) is not in the text; our
   64×64/0.128 m convention is carried over from the rest of this project,
   not derived from this paper.
4. **R4 — no stated batch size**, anywhere in the paper; the recipe above
   fills that gap from repo/task convention, not paper evidence.
