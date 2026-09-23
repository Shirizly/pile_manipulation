# Warped-NFD derivations

Working notes for the "warped NFD" model (NFD UNet operating on occupancy
warped into the canonical push frame). Records the two derivations behind
`transforms.functional.canonical_plate_channels` and the `push_px`
augmentation in `training/trainer.py::_augment_eulerian_batch`, so the next
session does not have to re-derive them. Both were verified empirically, not
just derived on paper — see the scripts referenced below.

## Conventions this depends on

- `transforms.functional`'s push-frame module: occupancy `(B, H, W)`, `H`
  rows, `W` cols; push endpoints `start_px`/`end_px` are `(B, 2)` pixels in
  **(col, row)** order; canonical +x is the push direction, origin at the
  push midpoint (`push_frame_transform`'s docstring).
- `draw_plate_soft`'s `center` argument is `(B, 2)` pixels in **(row, col)**
  order — i.e. `center[:, 0]` indexes grid dim0, `center[:, 1]` dim1 (pinned
  by `tests/test_grid_convention.py::test_draw_plate_soft_puts_world_x_on_dim0`,
  which notes this contradicts the function's own docstring wording — that
  test is the ground truth, not the docstring).
- These two conventions are **transposed relative to each other**: a
  `push_frame`-style `(col, row)` point must be flipped to `(row, col)`
  before it can be used as a `draw_plate_soft` center.

## 1. Canonical-frame plate scaling (`canonical_plate_channels`)

**Length scaling.** `push_frame_transform` maps a canonical NORMALIZED
offset `x_c` from the midpoint to a world normalized offset `scale * x_c`
along the push direction. A world pixel difference of `L` pixels is a world
normalized difference of exactly `2*L/world_res` (the `_to_normalized`
`+1`/`-1` terms cancel in a difference). Equating `scale * x_c =
2*L/world_res` and converting `x_c` (canonical normalized) to canonical
PIXELS (`2 * length_canon_px / canon_res`) gives

```
length_canon_px = L * canon_res / (world_res * scale)
```

i.e. `L / scale` canonical pixels when `canon_res == world_res` (the literal
case named in this feature's brief). This is a uniform similarity (a
rotation, which is factored out by construction, plus one scalar), so plate
dimensions (`plate_dim_x_px`, `plate_dim_y_px`) and the render `sigma` all
scale by the same factor `k = canon_res / (world_res * scale)`.

`canonical_plate_channels` takes an optional `world_res` (defaulting to
`canon_res`, giving exactly the simplified `L/scale` formula) — this wasn't
in the four-argument literal signature in the brief, but is needed for the
fully general `canon_res != world_res` case and defaults away when unused.

**Angle.** In the canonical frame the push direction is always along the col
axis (dim1). `draw_plate_soft` at `angle=0` puts its length axis (the
`plate_dim_x_px`-sized one) along dim0 (row) — exactly perpendicular to a
push along dim1 (col). So the canonical render always uses `angle=0` for
both plates; only the row-fixed, col-offset centers move with push length.
This is the whole point of the canonical frame: direction and position are
already factored out, only length remains a free parameter.

**Verified** in
`/tmp/.../scratchpad/verify_canonical_plate.py` and `verify_canonical_plate2.py`
(not checked into the repo — throwaway scratch scripts) by building
world-frame `draw_plate_soft` renders for random push poses, warping them
through `to_push_frame` with the same `scale`, and comparing pixel-for-pixel
against `canonical_plate_channels`'s direct render. Three configs tested
(`canon_res == world_res` with `scale != 1`, `scale == 1`, and
`canon_res != world_res`): correlation `0.992`-`0.9998`, small nonzero
max-abs-diff (`0.06`-`0.51` on a `[0,1]` field) consistent with the
world-frame path resampling through `grid_sample` while the canonical path
is drawn analytically — not a convention mismatch. The world-frame plate
angle used in that check is
`plate_angle = pi - phi`, `phi = atan2(delta_row, delta_col)` — see next
section for why.

### Angle relation used in verification: `plate_angle = pi - phi`

`draw_plate_soft`'s own "natural" travel angle (the angle whose `(cos,sin)`
points along an endpoint delta, in ITS `(dim0, dim1)` = `(row, col)` frame)
is `atan2(dim1_diff, dim0_diff) = atan2(col_diff, row_diff)`. `push_frame`'s
`phi = atan2(row_diff, col_diff)` swaps the argument order (a `(col,row)`
vs `(row,col)` point convention swap, as above), and swapping `atan2`'s
arguments reflects the angle: `atan2(col_diff, row_diff) = pi/2 - phi`. The
plate is drawn perpendicular to travel (`+pi/2` in `draw_plate_soft`'s own
frame), so:

```
plate_angle_for_draw_plate_soft = (pi/2 - phi) + pi/2 = pi - phi
```

This only matters for the verification script (comparing against
`draw_plate_soft` world renders); `canonical_plate_channels` itself never
computes this angle, because in the canonical frame the angle is always 0.

## 2. `push_px` augmentation pixel map (`_augment_eulerian_batch`)

`_augment_eulerian_batch` builds 8 views per k in `{0,1,2,3}`:
`xr = rot90(x, k, dims=(-2,-1))`, `xm = flip(xr, dims=[-1])`. Both `xr` and
`xm` are kept (`xs.extend([xr, xm])`), not just the flipped one.

Derived (then verified against `torch.rot90`/`torch.flip` directly, by
scanning every pixel of a small odd-sized grid and checking exact match —
not just a plausible-looking formula) the forward index map "input pixel
`(row=r, col=c)` moves to output pixel `(row_o, col_o)`":

**Rotation only** (`xr`, `k` applied, no flip):
```
k=0: (col_o, row_o) = (c, r)
k=1: (col_o, row_o) = (r, n-1-c)
k=2: (col_o, row_o) = (n-1-c, n-1-r)
k=3: (col_o, row_o) = (n-1-r, c)
```

**Rotation + horizontal flip** (`xm`): apply the rotation-only map above,
then negate the resulting col: `col_o <- n-1-col_o` (flip is along the last
dim = col), row unchanged.

These are **exact integer/affine reflections** (verified over every pixel of
a 7x7 grid, all `k`, both flipped/unflipped — see git history of this file's
companion script, not checked in), so they hold for sub-pixel (float) push
endpoints too, not just integer pixel centers — no `align_corners`-specific
correction is needed beyond this, because the map is a pure index
permutation/reflection, not a resample.

`_augment_push_endpoints(push_px, n, k, flipped)` applies this map to both
`(start_col, start_row)` and `(end_col, end_row)` in `push_px`'s `(B, 4)` =
`[start_col, start_row, end_col, end_row]` layout, producing the same
`[xr, xm]` per-`k` ordering `xs`/`ts` use, so `torch.cat` over all three
lists stays aligned.

**Verified end-to-end**: render a plate from `(start_px, end_px)` via
`draw_plate_soft` (with the `(row,col)` center swap and the `pi - phi` angle
relation above), augment the image through `_augment_eulerian_batch`, then
independently re-render a plate from the AUGMENTED `push_px` and compare to
the augmented image directly. All 8 views matched to `max_abs_diff < 1e-4`,
correlation `> 0.9999997` — see `tests/test_push_px_augmentation.py`
(`test_augmented_push_px_matches_augmented_render`), which is the permanent
regression test for this derivation.

## 3. `WarpedNFDWrapper` / `WarpedNFDPredictor` (this task)

`model/warped_nfd/lib.py` (training) and
`model/warped_nfd/predictor.py::WarpedNFDPredictor`/`build_canonical_stack`
(eval) implement the warped NFD model. Two more traps found while building
and verifying it, plus the plate_mode timing measurement:

**Trap 3: `PileSweepData`'s own plate-render pixel centers are NOT in
`push_frame`'s (col,row) order.** `_extract_sample_in_pxl`'s `plate_pos =
p_start * to_pxl + ctr_in_PXL` is fed straight into `draw_plate_soft` as
`center` throughout the existing NFD code (`nfd_lib.py::_draw_plate`,
`predictor.py::NFDPredictor.predict_occ`) — i.e. it is already in
`draw_plate_soft`'s own (row, col) = (dim0, dim1) = (world_x_pixel,
world_y_pixel) convention (consistent with `_draw_particle_grid`'s
dim0=world_x fix). `transforms.functional`'s push-frame module wants the
TRANSPOSE, (col, row) = (dim1, dim0) = (world_y_pixel, world_x_pixel).
`PileSweepData3ChWarped.push_px_pixels` computes the pixel pair with the
exact same formula as `predict_occ`, then swaps it into push_px's
`[start_col, start_row, end_col, end_row]` layout — do not pass the native
(row,col) pair straight to `push_frame_transform`/`to_push_frame`/
`push_frame_roundtrip`, it silently transposes the whole canonical frame
(this was caught by the train/eval parity check below going from ~1e-6
diff to a completely wrong image when the swap was accidentally omitted
during development).

**Trap 4 (informational, not a bug): the ×8 spatial augmentation does NOT
make the canonical-frame stack identical across all 8 views — only across
the 4 pure rotations.** Measured on a real L20mm_train sample (B=1,
augmented to B=8, `plate_mode`-independent since this is about the warped
occ0/wall channels, not the plates): the 4 un-flipped rotations (`k=0..3`,
no `hflip`) produce an EXACTLY identical (`max_abs_diff` at float32
precision, effectively 0) canonical occ0 and wall channel; the 4 flipped
views are exactly identical to EACH OTHER but differ from the rotation-only
views by a fixed, nonzero amount (mean abs diff 0.049 for occ0, 0.22 for
the wall channel; max 1.0 for both). This is expected, not a bug:
`push_frame_transform` builds a pure ROTATION `R(phi)` (no reflection), so
it is exactly invariant under the 4 proper rotations (which `push_px`'s
augmentation map tracks exactly, per section 2 above) but a horizontally-
flipped world view is a REFLECTION of the original, which the canonical
frame does not undo — the flipped views see a genuinely mirror-image
canonical scene, not a repeat of the unflipped one. Consequence: the ×8
augmentation still produces 8 geometrically meaningful (if only 2-fold-
distinct up to exact symmetry) canonical views for this model, not 8
copies of the same view — training on all 8 is still valid data
augmentation, just not "invariance-testable" in the naive sense of
expecting all 8 to match.

**plate_mode timing (decides the pilot configs' default).** Measured a full
training step (forward + backward + optimizer.step, the wrapped UNet +
push_frame_roundtrip + plate-channel construction) on an idle GPU (`nvidia-
smi` 0% util / 15 MiB used, `uptime` load average ~1.4-1.9 beforehand),
batch_size=32 pre-augmentation / 256 post-augmentation, 64x64 grid
(resolution_scale=0.5), `nfd-unet-warped` with `wall_channel=false`, 8
warm-up steps discarded, `torch.cuda.synchronize()` around each timed
region, median over repeats (see `Baselines/common/benchmark_time.py`'s
methodology; a standalone script was used here rather than that harness
itself, since it times predictor.predict_occ calls, not a training step):

| run | canonical | warp | canonical vs warp |
|---|---|---|---|
| separate processes, 15 reps | 43.32 ms | 40.42 ms | +7.2% |
| separate processes, 30 reps | 34.05 ms | 38.33 ms | -11.2% |
| interleaved (paired), 40 reps each | 39.05 ms | 40.53 ms | -3.6% |

The sign flips between the two separate-process runs (consistent with this
project's documented noise floor for fast, few-tens-of-ms measurements —
`subagent-experimenter` skill: "~5% relative, and >30% cross-process drift
for fast models"), so `plate_mode` cannot be ranked precisely at this
measurement's resolution. Every measurement is nonetheless well under the
user's 10% threshold (the interleaved, paired comparison — the most
methodologically sound of the three, since both arms share the same
process/thermal/scheduler conditions — puts `canonical` 3.6% FASTER than
`warp`), so **`plate_mode: canonical` is the default** in all three pilot
configs (`Baselines/NFD/configs/nfd_train_warped*_L20mm_pilot.yaml`,
`predictor.py::WARPED_DEFAULT_PLATE_MODE`). This is not a strong claim that
canonical IS faster, only that using it costs nothing measurable against
the 10% budget, and it is architecturally simpler (no extra `grid_sample`
warp of the plate channels).

**Epoch-count derivation.** A real (non-smoke) 1-epoch run of
`nfd_train_warped_L20mm_pilot.yaml` (`plate_mode: canonical`, the full
10,496-sample L20mm_train split, batch 32, augmentation on) measured 37.9 s
on the same idle GPU. 20 epochs ~= 12.6 minutes, under the ~15-minute
smoke-config budget with margin — used for all three pilot configs (kept
identical per the task brief).
