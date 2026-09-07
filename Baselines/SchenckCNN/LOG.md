# Baselines/SchenckCNN — LOG

Agent: A2-grid-specs. Branch `baselines/overnight`. Task: design-only spec
for the Schenck et al. (CoRL 2017) granular-media CNN, lower priority than
NFD, roughly half the length. No training performed, no existing repo file
modified.

---

## SUMMARY FOR HANDOFF (read this first)

**Status: DONE.** Deliverable is `Baselines/SchenckCNN/SPEC.md`.

**One-line takeaway:** the paper's headline architecture (the two-tower
"scoop & dump–net" with an inter-tower mass-conservation channel) is built
entirely around a lift-carry-pour scoop action and doesn't apply to a
plate-push task; recommend reproducing the simpler **single–net** ablation
instead (one FCN tower, ~16 layers of `Conv 32@3×3`, no pooling, residual
output add, L2 loss) with the action re-encoded from `p_start/p_stop/angle`
via the same rasterizer recommended for the NFD baseline
(`transforms/functional.py::draw_plate_soft`).

**Key findings (see SPEC.md for full evidence/citations):**
- State: height-map, ~1cm cells, exact grid extent not given in the paper.
- Action: 9D scoop&dump parameter vector, reparameterized into a rasterized
  "action map" (RGB line + dot for position, 3 tiled channels for the 3
  angles) — inferred 6 action channels + 1 height-map = 7 input channels;
  the exact channel split is not stated as a number anywhere in the text,
  flagged as R1.
- Loss: L2 on next state; the scoop&dump–net adds an equally-weighted,
  gradient-stopped L2 on an intermediate scoop-only prediction. Paper gives
  actual optimizer hyperparameters (Adam, lr 5e-4) — one of only two
  reference papers in this project's set that does.
- Feasibility: single–net variant is straightforward and cheap; the
  two-tower split is explicitly not worth reproducing for a push-only task
  (no scoop/dump phases to separate, no mass-conservation boundary to
  enforce).

**Files delivered:** `Baselines/SchenckCNN/SPEC.md`, this log.
**Not delivered (out of scope):** any code, any training run.

---

## Running notes

- 2026-09-08. Extracted full text via `pypdf` (10 pages) — unlike the NFD
  paper, Schenck's Fig.3/Fig.4 layer lists (`Conv:32@3×3` stacks) came
  through as plain extractable text, no need to rasterize the page as an
  image to read them.
- Cross-checked the action-map description (§3.1 in the paper) closely
  since it's the part most directly analogous to NFD's action-rendering —
  confirmed it's conceptually the same idea (render the action into an
  image aligned with the state grid) but a materially different
  parameterization (9D scoop pose vs. our 2-endpoint plate push), so most
  of the value here is the "what doesn't map" analysis rather than a
  literal channel-for-channel port.
- Deliberately did not re-read `model/UNetModels_modular.py` in as much
  depth as for the NFD spec — the recommended single–net architecture
  (constant-resolution stack of 3×3 convs, no pooling) does not map onto
  `UNet`'s down/up-sampling structure at all; noted in the spec that this
  would be a from-scratch small module, not a config choice against the
  existing `UNet` class.
