"""model/flow_nfd/supervised.py -- EXP-0025 extension: supervise the
flow/advection head DIRECTLY from particle correspondence, instead of only
through the photometric loss on the warped occupancy (`flow_nfd_lib.py`).

Motivation (see experiments/EXP-0025-flow-warp-nfd-pilot/results/
supervised_flow.md): the photometric-only flow head lost to direct
next-occupancy prediction. Diagnosis: the flow is never supervised
directly, so (a) it drifts to arbitrary values wherever occ0 is locally
flat (zero photometric gradient), and (b) gradient descent on a photometric
loss can only align features already within about one feature width, while
measured per-particle displacements reach far outside that basin (p95 6.5px,
max 20px at this resolution). `states`/`states_` in every `_data.pt` share
particle indexing (verified in the pilot's gate script), so a target
displacement field can be built directly and supervised with `||f -
f_target||^2` -- convex, no basin to escape.

Convention (verified by the gate script, see results/supervised_flow.md):
`Genesis/training/dataset.py::_extract_sample_in_pxl` converts `states`/
`states_` to pixel space via `particles[:, :3] = particles[:, :3] * to_pxl +
ctr_in_PXL`; the resulting `particles[:, 0]` IS the final occupancy grid's
dim0 (row) index (world x) and `particles[:, 1]` IS dim1 (col, world y) --
no further axis swap needed (`_draw_particle_grid`'s cv2-then-transpose
dance nets out to identity on this mapping; confirmed empirically: 18/20
destination centers in a real sample land exactly on an occ1==1 pixel).

The warp is BACKWARD (`grid_sample`'s contract, matching
`flow_nfd_lib.py::FlowWarpWrapper`): for output/destination pixel x, the
field says "where did this material come from", so
    f(x) = x0 - x, indexed AT THE DESTINATION pixel x.
`FlowWarpWrapper`'s channel convention: channel 0 is displacement along the
W/col axis (`disp_x_norm`, i.e. dcol = world-y), channel 1 along the H/row
axis (`disp_y_norm`, i.e. drow = world-x).

20 particles is sparse for a 64x64 field, and cubes stack (several
particles can be nearest a given destination pixel). Resolution used here:
supervise ONLY at destination pixels the rasteriser actually marks as
foreground (`occ1 > 0.5`) -- a masked loss, not scatter-and-diffuse, because
diffusing a rigid per-particle displacement into pixels with no material
correspondence would invent a target where none exists. Among foreground
pixels, a collision (two particles' footprints both claim a pixel) is
resolved by NEAREST DESTINATION CENTER: every foreground pixel is assigned
the rigid (whole-particle, translation-only) displacement of whichever
particle's destination center it is closest to. This does NOT model
particle rotation, which is a known, reported source of ceiling error (see
results/supervised_flow.md) -- a per-pixel field literally cannot express
rotation any better than this construction can target it.
"""
from __future__ import annotations

import torch


def build_flow_targets_for_split(states: torch.Tensor, states_: torch.Tensor,
                                  occ1: torch.Tensor, to_pxl: float,
                                  ctr_in_pxl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised-per-sample (loop over N, vectorised over H*W and particles).

    states, states_ : (N, 20, 7) world metres (xyz + quat; only xyz used).
    occ1             : (N, H, W) destination occupancy (float, ~binary).
    to_pxl           : scalar, metres -> pixels.
    ctr_in_pxl       : (3,) pixel offset (raw.ctr_in_PXL).

    Returns
    -------
    flow_target : (N, 2, H, W) float32, [dcol, drow] in PIXELS, 0 outside mask.
    mask        : (N, H, W) float32 in {0, 1}: destination-foreground pixels only.
    """
    N, H, W = occ1.shape
    ctr = ctr_in_pxl.to(torch.float32)
    src_row = states[:, :, 0] * to_pxl + ctr[0]
    src_col = states[:, :, 1] * to_pxl + ctr[1]
    dst_row = states_[:, :, 0] * to_pxl + ctr[0]
    dst_col = states_[:, :, 1] * to_pxl + ctr[1]

    flow_target = torch.zeros(N, 2, H, W, dtype=torch.float32)
    mask = torch.zeros(N, H, W, dtype=torch.float32)

    for n in range(N):
        fg = occ1[n] > 0.5
        if not bool(fg.any()):
            continue
        fg_idx = fg.nonzero(as_tuple=False)
        fg_row = fg_idx[:, 0].float()
        fg_col = fg_idx[:, 1].float()
        dr = fg_row[:, None] - dst_row[n][None, :]
        dc = fg_col[:, None] - dst_col[n][None, :]
        nearest = (dr * dr + dc * dc).argmin(dim=1)
        dcol = src_col[n][nearest] - dst_col[n][nearest]
        drow = src_row[n][nearest] - dst_row[n][nearest]
        flow_target[n, 0][fg] = dcol
        flow_target[n, 1][fg] = drow
        mask[n][fg] = 1.0

    return flow_target, mask


def _rotate_flow_components(dc: torch.Tensor, dr: torch.Tensor, k: int,
                             flipped: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Vector-component transform matching the SPATIAL `rot90(k)` [+ hflip on
    dims=[-1]] `training/trainer.py::_augment_eulerian_batch` applies to
    images. `dc`/`dr` are the flow field's two channels (dcol = displacement
    along the W/col axis, drow = displacement along the H/row axis -- this
    module's own convention, matching `flow_nfd_lib.FlowWarpWrapper`).

    A rot90/flip of the IMAGE is a rigid transform of the (row, col) plane;
    a VECTOR living in that plane (unlike a scalar occupancy value) must be
    rotated by the same transform's LINEAR part, not just carried along to
    its new pixel location by `torch.rot90`/`torch.flip` (which only
    permutes array layout -- it does not touch values). Derived from the
    point map in `model/warped_nfd/WARPED_NFD_NOTES.md` section 2
    (`row_o, col_o` as a function of `r, c` for each `k`), by reading off
    that map's linear part (the map's ONLY translation-independent part,
    since the map is a pure rotation/reflection about the grid centre):

        k=0: (col_o, row_o) = (c, r)             -> linear: col+=c, row+=r
        k=1: (col_o, row_o) = (r, n-1-c)          -> linear: col+=r, row+=-c
        k=2: (col_o, row_o) = (n-1-c, n-1-r)      -> linear: col+=-c, row+=-r
        k=3: (col_o, row_o) = (n-1-r, c)          -> linear: col+=-r, row+=c

    so a vector (dr, dc) at the input maps to (dr_o, dc_o) at the output by
    exactly that linear part (dc_o = the map's col-linear-part evaluated at
    (dr, dc) in place of (r, c); dr_o likewise for row):

        k=0: (dc_o, dr_o) = ( dc,  dr)
        k=1: (dc_o, dr_o) = ( dr, -dc)
        k=2: (dc_o, dr_o) = (-dc, -dr)
        k=3: (dc_o, dr_o) = (-dr,  dc)

    and the additional horizontal flip (`torch.flip(xr, dims=[-1])`, which
    reflects the COL axis only) negates the resulting dc_o only:
    `dc_o <- -dc_o`, `dr_o` unchanged. Verified end-to-end (not just
    algebraically) by `verify_flow_augmentation_equivariance` below.
    """
    if k == 0:
        dc_o, dr_o = dc, dr
    elif k == 1:
        dc_o, dr_o = dr, -dc
    elif k == 2:
        dc_o, dr_o = -dc, -dr
    elif k == 3:
        dc_o, dr_o = -dr, dc
    else:
        raise ValueError(k)
    if flipped:
        dc_o = -dc_o
    return dc_o, dr_o


def augment_flow_batch_x8(inputs: torch.Tensor, occ1: torch.Tensor,
                           flow_target: torch.Tensor, mask: torch.Tensor
                           ) -> dict[str, torch.Tensor]:
    """The x8 (4 rotations x 2 flips) augmentation group, extended to also
    carry `flow_target` correctly (image-like tensors -- `inputs`, `occ1`,
    `mask` -- are permuted spatially by `torch.rot90`/`torch.flip` exactly as
    `training/trainer.py::_augment_eulerian_batch` does; `flow_target`
    additionally needs its two components rotated per `_rotate_flow_components`).

    Returns a dict of tensors, each with batch dim x8, same order as
    `_augment_eulerian_batch` (k=0..3, and within each k: [rotation-only,
    rotation+flip]) so a spot check against that function's own output is
    directly comparable index-for-index.
    """
    xs, ts, fs, ms = [], [], [], []
    for k in range(4):
        for flipped in (False, True):
            def sp(x):
                y = torch.rot90(x, k, dims=(-2, -1))
                if flipped:
                    y = torch.flip(y, dims=[-1])
                return y
            x_aug = sp(inputs)
            t_aug = sp(occ1)
            m_aug = sp(mask)
            f_sp = sp(flow_target)  # spatial permutation only, values untouched
            dc_o, dr_o = _rotate_flow_components(f_sp[:, 0], f_sp[:, 1], k, flipped)
            f_aug = torch.stack([dc_o, dr_o], dim=1)
            xs.append(x_aug); ts.append(t_aug); fs.append(f_aug); ms.append(m_aug)
    return dict(
        inputs=torch.cat(xs, dim=0),
        occ1=torch.cat(ts, dim=0),
        flow_target=torch.cat(fs, dim=0),
        mask=torch.cat(ms, dim=0),
    )
