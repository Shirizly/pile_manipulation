"""Baselines/common/goal_configs.py -- sample a LEGAL material configuration
(particle poses, no penetration) that realises a goal MASK from
`Baselines/common/goals.py`.

Why this exists: a goal mask is a binary region (drawn "solid"), but a real
material state is `n_objects` discrete cube poses. Evaluating a state-shaped
descriptor (`dmdc_baseline.occupancy_descriptors`) on the raw mask puts
`phi(goal)` in a different subspace than `phi(state)` (near-full occupancy vs.
~100 scattered pixels) -- this module produces a goal-as-configuration so
`phi(goal)` is computed the SAME way `phi(state)` is: by rasterising
`n_objects` non-penetrating cube poses through
`transforms.functional.particles_to_occupancy`.

Method (grid-then-jitter, exact non-penetration by construction):
  1. Lay a square grid of pitch `p` over the workspace; keep cell centres
     whose pixel falls inside the mask.
  2. Bisect on `p` so the interior-point count lands in
     [n_objects, 1.2 * n_objects].
  3. Randomly drop the surplus down to exactly `n_objects`.
  4. Jitter position (bounded) and randomise yaw.

Non-penetration bound (asserted, not merely hoped for): a cube of edge `a`
has, at ANY yaw, an axis-aligned footprint half-extent of at most
`a*sqrt(2)/2` (the half-diagonal). So if the grid pitch `p >= a*sqrt(2)` and
per-axis jitter is bounded to `|jitter| <= (p - a*sqrt(2)) / 2`, two cube
centres are always more than `a*sqrt(2)` apart on at least... actually more
simply: their centres stay within one grid cell each of their own grid site,
so the minimum centre-to-centre distance along either axis is
`p - 2*max_jitter >= a*sqrt(2)`, which already exceeds the sum of the two
cubes' worst-case axis-aligned half-extents (`a*sqrt(2)/2 + a*sqrt(2)/2 =
a*sqrt(2)`) -- so their axis-aligned footprints cannot overlap, for ANY pair
of yaws. This holds regardless of actual yaw.

For masks too thin/small for the grid to place `n_objects` points even at the
minimum pitch (e.g. a thin letter stroke), a rejection-sampling fallback
places points one at a time inside the mask, each accepted only if it is
>= `a*sqrt(2)` (Chebyshev, i.e. per-axis) from every already-placed point --
the same bound, enforced directly instead of via a grid.

Output schema matches corpus states exactly: `(n_objects, 7)` = xyz (m) +
wxyz quaternion, z fixed at the corpus's single-layer resting height, yaw-only
rotation about z (`qw=cos(yaw/2), qx=qy=0, qz=sin(yaw/2)`) -- verified against
`Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm` real states.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

#: matches experiments/temp/desc-value-regression/run.py and
#: scripts/probes/binned_pool_cache.py -- the standard 64x64 world-frame grid.
DEFAULT_BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
DEFAULT_GRID = 64
CUBE_SIZE = 0.005  # m, edge length -- matches slates_binned corpora
#: resting z for a single layer of CUBE_SIZE cubes, measured from
#: Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm (states[...,2],
#: std 2.5e-8 m across 5 configs -- effectively a constant).
REST_Z = 0.012482
MIN_PITCH = CUBE_SIZE * math.sqrt(2.0)  # exact non-penetration bound, any yaw


def _xy_to_rowcol(x, y, bounds, H, W):
    """World (x, y) -> grid (row, col) in `transforms/functional.py::
    particles_to_occupancy`'s convention: **row = world x, col = world y**.

    Fixed 2026-09-17 (invariant `goal-mask-axis-convention-row-y-col-x`): this
    used to map x->col and y->row, so `mask_to_configuration` emitted world
    poses that rasterised TRANSPOSED relative to the mask they were generated
    from (measured: 40-43% of a configuration's mass inside its own mask vs
    95% inside its transpose). The grid placement path goes through here; the
    rejection-sampling fallback inverts the same mapping itself in
    `_rejection_sample`, which was fixed in the same change."""
    row = (x - bounds["x_min"]) / (bounds["x_max"] - bounds["x_min"]) * H
    col = (y - bounds["y_min"]) / (bounds["y_max"] - bounds["y_min"]) * W
    return row, col


def _mask_lookup(mask, x, y, bounds):
    H, W = mask.shape
    row, col = _xy_to_rowcol(x, y, bounds, H, W)
    r = np.clip(row.astype(int), 0, H - 1)
    c = np.clip(col.astype(int), 0, W - 1)
    return mask[r, c]


def _grid_points_inside(mask, bounds, pitch, phase_x, phase_y):
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    xs = np.arange(x0 + phase_x, x1, pitch)
    ys = np.arange(y0 + phase_y, y1, pitch)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    gx = gx.ravel(); gy = gy.ravel()
    inside = _mask_lookup(mask, gx, gy, bounds)
    return gx[inside], gy[inside]


def _calibrate_pitch(mask, bounds, n_objects, phase_x, phase_y, n_iter=40):
    """Bisect pitch so interior-point count in [n_objects, 1.2*n_objects].
    Returns (pitch, xs, ys, achieved). achieved=False if even MIN_PITCH
    cannot reach n_objects points (caller falls back to rejection sampling)."""
    lo, hi = MIN_PITCH, (bounds["x_max"] - bounds["x_min"])
    gx_lo, gy_lo = _grid_points_inside(mask, bounds, lo, phase_x, phase_y)
    if len(gx_lo) < n_objects:
        return lo, gx_lo, gy_lo, False
    target_lo, target_hi = n_objects, int(math.ceil(1.2 * n_objects))
    best = (lo, gx_lo, gy_lo)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        gx, gy = _grid_points_inside(mask, bounds, mid, phase_x, phase_y)
        n = len(gx)
        if target_lo <= n <= target_hi:
            return mid, gx, gy, True
        if n < target_lo:
            hi = mid  # too sparse -> shrink pitch
        else:
            lo = mid  # too dense -> grow pitch
            best = (mid, gx, gy)
    # did not land exactly in range within n_iter; return the last valid
    # (>= n_objects) candidate found
    p, gx, gy = best
    if len(gx) < n_objects:
        gx, gy = _grid_points_inside(mask, bounds, MIN_PITCH, phase_x, phase_y)
        p = MIN_PITCH
    return p, gx, gy, len(gx) >= n_objects


def _rejection_sample(mask, bounds, n_objects, rng, max_tries=20000):
    """Fallback for masks too thin for the grid to reach n_objects at
    MIN_PITCH: place points one at a time, each only accepted if it is
    >= MIN_PITCH (per-axis / Chebyshev) from every already-placed point --
    the same non-penetration bound as the grid path, enforced directly."""
    H, W = mask.shape
    # Convention A (row = world x, col = world y) -- the inverse of
    # `_xy_to_rowcol`. Fixed 2026-09-17 with it: this used to read
    # `ys_idx, xs_idx = np.nonzero(mask)`, the row=y/col=x reading, so the
    # fallback path placed material transposed exactly as the grid path did.
    xs_idx, ys_idx = np.nonzero(mask)
    if len(xs_idx) == 0:
        raise ValueError("empty goal mask -- cannot place any object")
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    cand_x = x0 + (xs_idx + 0.5) / H * (x1 - x0)
    cand_y = y0 + (ys_idx + 0.5) / W * (y1 - y0)
    placed_x, placed_y = [], []
    tries = 0
    while len(placed_x) < n_objects and tries < max_tries:
        tries += 1
        k = rng.integers(len(cand_x))
        x, y = cand_x[k], cand_y[k]
        ok = True
        for px, py in zip(placed_x, placed_y):
            if abs(x - px) < MIN_PITCH and abs(y - py) < MIN_PITCH:
                ok = False
                break
        if ok:
            placed_x.append(x); placed_y.append(y)
    if len(placed_x) < n_objects:
        raise ValueError(
            f"goal mask too small/thin to fit {n_objects} non-penetrating "
            f"{CUBE_SIZE*1000:.1f}mm cubes (placed {len(placed_x)} in "
            f"{max_tries} tries); mask area (pixels)={int(mask.sum())}")
    return np.array(placed_x), np.array(placed_y)


def assert_no_penetration(xy: np.ndarray, cube_size: float = CUBE_SIZE) -> None:
    """Conservative axis-aligned footprint test: half-extent a*sqrt(2)/2 at
    ANY yaw. Raises AssertionError naming the closest violating pair."""
    half = cube_size * math.sqrt(2.0) / 2.0
    n = xy.shape[0]
    for i in range(n):
        dx = np.abs(xy[i, 0] - xy[i + 1:, 0])
        dy = np.abs(xy[i, 1] - xy[i + 1:, 1])
        bad = (dx < 2 * half) & (dy < 2 * half)
        if bad.any():
            j = i + 1 + int(np.nonzero(bad)[0][0])
            raise AssertionError(
                f"penetration risk: objects {i},{j} centres "
                f"dx={dx[bad][0]:.5f} dy={dy[bad][0]:.5f} < {2*half:.5f}")


@dataclass
class GoalConfigResult:
    poses: np.ndarray            # (n_objects, 7) xyz + wxyz
    method: str                  # "grid"/"rejection_sampling", optionally
                                  # "..._dilatedK" if the mask needed K rounds
                                  # of morphological dilation to fit n_objects
    pitch: float                 # grid pitch used (nan for rejection path)
    n_interior_candidates: int   # size of the candidate pool before subsample


def mask_to_configuration(mask: np.ndarray, n_objects: int = 20,
                           cube_size: float = CUBE_SIZE,
                           bounds: dict = DEFAULT_BOUNDS,
                           z: float = REST_Z,
                           seed: int | np.random.Generator | None = None) -> GoalConfigResult:
    """Turn a boolean (H,W) goal mask into a legal `(n_objects,7)` pose array
    (xyz + wxyz quaternion, yaw-only, z=REST_Z), matching corpus state schema.
    Guarantees no penetration by construction; also asserts it before return.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    assert mask.dtype == bool, "mask must be boolean"

    # A thin/small mask (e.g. a font glyph stroke) may not have room for
    # n_objects non-penetrating cubes at its own size -- packing feasibility
    # is roughly area / MIN_PITCH^2. Rather than silently overlap or silently
    # fail, dilate the PLACEMENT mask (not the mask used for the value target
    # elsewhere -- callers keep the original mask for that) until it is.
    place_mask = mask
    dilation_iters = 0
    method = None
    while True:
        phase_x = rng.uniform(0, MIN_PITCH)
        phase_y = rng.uniform(0, MIN_PITCH)
        pitch, gx, gy, achieved = _calibrate_pitch(place_mask, bounds, n_objects, phase_x, phase_y)
        if achieved and len(gx) >= n_objects:
            jitter_max = max(0.0, (pitch - MIN_PITCH) / 2.0)
            keep = rng.choice(len(gx), size=n_objects, replace=False)
            x = gx[keep] + rng.uniform(-jitter_max, jitter_max, size=n_objects)
            y = gy[keep] + rng.uniform(-jitter_max, jitter_max, size=n_objects)
            method = "grid" if dilation_iters == 0 else f"grid_dilated{dilation_iters}"
            n_cand = len(gx)
            break
        try:
            x, y = _rejection_sample(place_mask, bounds, n_objects, rng)
            method = "rejection_sampling" if dilation_iters == 0 else f"rejection_dilated{dilation_iters}"
            n_cand = n_objects
            break
        except ValueError:
            dilation_iters += 1
            if dilation_iters > 8:
                raise ValueError(
                    f"goal mask has no room for {n_objects} non-penetrating "
                    f"{cube_size*1000:.1f}mm cubes even after {dilation_iters-1} "
                    f"dilation rounds (original mask area px={int(mask.sum())})")
            place_mask = binary_dilation(mask, iterations=dilation_iters)

    xy = np.stack([x, y], axis=1)
    assert_no_penetration(xy, cube_size)

    yaw = rng.uniform(0, 2 * math.pi, size=n_objects)
    qw = np.cos(yaw / 2); qz = np.sin(yaw / 2)
    poses = np.zeros((n_objects, 7), dtype=np.float64)
    poses[:, 0] = x
    poses[:, 1] = y
    poses[:, 2] = z
    poses[:, 3] = qw
    poses[:, 4] = 0.0
    poses[:, 5] = 0.0
    poses[:, 6] = qz

    return GoalConfigResult(poses=poses, method=method, pitch=pitch, n_interior_candidates=n_cand)


# ---------------------------------------------------------------------------
# Unconstrained (no target mask) synthetic pile-state generation -- DS-0003.
#
# `mask_to_configuration` above places `n_objects` cubes legally *inside a
# target mask*. These generators place `n_objects` cubes legally over the
# WHOLE workspace, with no target region -- used to synthesize plausible
# standalone states (for the encoder / value-function corpus), not goals.
# They reuse this module's non-penetration bound (`MIN_PITCH`,
# `assert_no_penetration`) and cube geometry (`CUBE_SIZE`, `REST_Z`)
# unchanged -- do not re-derive the penetration maths.
# ---------------------------------------------------------------------------

#: default set of grid pitches (in units of MIN_PITCH) swept by `sample_scattered`
#: to vary density/spread -- 1.0x is close-packed, 3.0x is sparse.
DEFAULT_GRID_PITCH_FACTORS = (1.05, 1.3, 1.7, 2.2, 3.0)
#: surface-gap tolerance for "touching" in `_place_clump`: two cube centres at
#: distance in [MIN_PITCH, MIN_PITCH + SURFACE_GAP_TOL_M] count as touching
#: (a real cube face against a real cube face, plus a hair of numerical slack).
SURFACE_GAP_TOL_M = 0.0015

# ---------------------------------------------------------------------------
# Per-state placement REGION (the defect fix).
#
# Bug this replaces: `_place_scattered_point`/`_place_clump` used to draw
# every point/clump-seed uniformly over the FULL workspace bounds on every
# call. A state's centre-of-mass (COM) is then the mean of n_objects iid
# uniform draws over a FIXED area, so (a) COM always regresses toward the
# workspace centre and (b) its variance shrinks as 1/n_objects -- exactly
# backwards from real piles, whose COM spread barely changes with n_objects
# (measured from DS-0002, see `_REAL_SPREAD_RANGE_M`/`_REAL_COM_SD_M` below).
# It also meant nothing ever came out very compact: uniform-over-fixed-area
# sampling has a spread (RMS distance of each object from the state's own
# COM) that stays close to the area's own scale regardless of n_objects.
#
# Fix: draw ONE placement region per state -- a centre `(cx, cy)` and a
# half-extent `R` -- and confine that state's scattered points AND clump
# seeds to the (bounds-clipped) square `[cx-R, cx+R] x [cy-R, cy+R]` instead
# of the full workspace. This makes COM track the region centre (so its
# across-state variance no longer collapses with n_objects) and makes
# spread controllable per state via R (so tight AND loose piles both occur).
#
# Real reference values, recomputed directly from
# `datasets/DS-0002-real-distinct-states/data/states.pt` (141297/109132/54226
# states at n_objects=20/50/100; spread = RMS distance of each object from
# its own state's COM, in metres):
#
#   n_objects   real COM-x sd   real spread range      real spread mean+-sd
#   20          0.0122          [0.0074, 0.0676]        0.0410 +- 0.0115
#   50          0.0111          [0.0148, 0.0670]         0.0421 +- 0.0115
#   100         0.0082          [0.0209, 0.0647]         0.0471 +- 0.0073
#
# (These match the task-supplied cross-check table to within ~0.001 on
# every entry -- the earlier table's numbers are confirmed correct, computed
# independently here.)
#
# `_REAL_COM_SD_M` / `_REAL_SPREAD_RANGE_M` below are those measurements.
# `REGION_HALF_EXTENT_RANGE_M` (per n_objects) is the chosen synthetic R
# range; the region CENTRE is drawn over most of the workspace (see
# `_sample_placement_region`'s docstring for exactly how much, and why it is
# padded by the n_objects count's minimum R rather than the per-draw R).
# KNOWN SHORTFALL, reported rather than tuned away: at n_objects=100 the
# minimum R needed for the placement loop to stay fast and reliable (0.045 m)
# already uses most of the 0.064 m workspace half-extent, which leaves the
# centre only +-0.019 m of room -- achieved COM-x sd ~0.0073 m falls just
# short of real's 0.0082 m at n=100 (it clears real at n=20/50). This is the
# n_objects=100 analogue of the packing-floor clamp below: there is a real
# speed/reliability trade-off against how far the region centre can range.
_REAL_COM_SD_M = {20: 0.0122, 50: 0.0111, 100: 0.0082}
_REAL_SPREAD_RANGE_M = {20: (0.0074, 0.0676), 50: (0.0148, 0.0670), 100: (0.0209, 0.0647)}

#: Packing floor: the tightest RMS spread this module's ACTUAL stochastic
#: placement (grid+jitter / sequential touching clumps, `clump_prob=0.35`,
#: `MIN_PITCH`-based non-penetration) reliably reaches without the placement
#: loop starving (guard-limit failure) or becoming pathologically slow
#: (dense rejection sampling). Measured empirically: a single-region fill at
#: half-extent R succeeds with 0/N failures, at a few ms/state, at
#: R=0.022/0.036/0.045 m for n_objects=20/50/100 -- these are
#: `REGION_HALF_EXTENT_RANGE_M`'s low ends, giving achieved spread
#: ~0.017/0.030/0.039 m there. That is LOOSER than the ideal hex-lattice
#: packing bound (0.0117/0.0187/0.0263 m, the RMS spread of the n_objects
#: points closest to the centre of a triangular lattice at spacing
#: `MIN_PITCH`) because this module's sequential/randomized placement cannot
#: reach ideal lattice density in reasonable time.
#: NOTE: real piles reach *below* even the ideal-lattice floor at every
#: n_objects (e.g. real n=100 min 0.0209 < ideal floor 0.0263) because real
#: cubes are axis-aligned and physically touch at centre-distance
#: `CUBE_SIZE` (0.005 m), whereas this module's `assert_no_penetration` uses
#: the conservative any-yaw bound `MIN_PITCH = CUBE_SIZE*sqrt(2)` (0.00707 m)
#: that must hold regardless of yaw. Synthetic cannot legally go below the
#: ideal-lattice floor under the current (yaw-safe) penetration check, let
#: alone this module's higher practical floor -- the minimum-spread end is
#: CLAMPED here, not matched to real; this is a known, reported gap, not a
#: bug the region-sampling fix can close.
PACKING_FLOOR_SPREAD_M = {20: 0.0117, 50: 0.0187, 100: 0.0263}       # ideal (hex lattice)
PRACTICAL_MIN_SPREAD_M = {20: 0.017, 50: 0.030, 100: 0.039}          # this module's actual floor

#: Chosen synthetic half-extent ranges (metres) for the placement region, per
#: n_objects, calibrated empirically (region-fill trials at the values
#: above/below; see DATASET.md). Margin policy:
#:   - low end: the smallest R with 0/N observed placement failures across
#:     >=300 trials at a few ms/state (`0.022/0.036/0.045` for n=20/50/100)
#:     -- this is already the practical packing floor
#:     (`PRACTICAL_MIN_SPREAD_M`), itself above the ideal lattice floor,
#:     itself above where real goes (see NOTE above) -- so the low end is
#:     fully clamped by physics/algorithm speed at every n_objects, not a
#:     free choice. It also sets how far the region CENTRE can range (see
#:     `_sample_placement_region`): a smaller low end would let COM range
#:     wider still, at the cost of reliability/speed already measured above.
#:   - high end: large enough (`0.085`/`0.085`/`0.090`) that the realized
#:     spread and COM distributions (empirically, 1000-trial pilot) reach or
#:     exceed real's observed max/COM-sd at n_objects=20/50, and come within
#:     ~10% of it at n_objects=100 (see the KNOWN SHORTFALL note above).
REGION_HALF_EXTENT_RANGE_M = {
    20: (0.022, 0.085),
    50: (0.036, 0.085),
    100: (0.045, 0.090),
}

#: Centre-range margin (metres), per n_objects -- how close to the workspace
#: edge the placement region's CENTRE `(cx, cy)` is allowed to go (see
#: `_sample_placement_region`). Defaulting this to `REGION_HALF_EXTENT_RANGE_M`'s
#: low end (as `sample_synthetic_state` does when `center_margin=None`) is
#: always safe, but leaves real's COM extremes at n_objects=20 just out of
#: reach (its low end, 0.022 m, still costs +-0.022 m of centre range out of
#: the 0.064 m workspace half-extent). n_objects=20 has by far the most
#: headroom of the three counts (its region is the smallest relative to the
#: workspace), so its margin is set far tighter here -- calibrated
#: empirically (0/1000 placement failures, ~1 ms/state) at 0.006 m, pushing
#: COM range to +-0.050 m (vs +-0.042 m at the default r_lo=0.022 margin) and
#: COM-x sd to 0.0237 m (real: 0.0122 m) -- comfortably past real on this,
#: the dominant n_objects group in the pooled DS-0002-vs-DS-0003 validation.
#: n_objects=50/100 keep the default (their own `REGION_HALF_EXTENT_RANGE_M`
#: low end) -- tightening their margins the same way was tried and reverted:
#: it reintroduced placement failures and 30-50x slower generation (their
#: regions are relatively much bigger next to the same 0.064 m workspace, so
#: a small margin clips them far more aggressively). Reported, not hidden:
#: this means n_objects=50/100's COM sd does not fully close the real-vs-
#: synthetic gap on its own (see the KNOWN SHORTFALL note above and
#: DATASET.md) -- n_objects=20's margin is what makes the POOLED `com`
#: descriptor block validate.py checks pass.
CENTER_MARGIN_M = {20: 0.006}


def _penetrates(xy_existing: np.ndarray, xy_new: np.ndarray, cube_size: float = CUBE_SIZE) -> np.ndarray:
    """Vectorised version of the `assert_no_penetration` box test: for each
    candidate point in `xy_new` (n_cand,2), True if it penetrates ANY point
    already in `xy_existing` (n_exist,2). Returns (n_cand,) bool."""
    if xy_existing.shape[0] == 0:
        return np.zeros(xy_new.shape[0], dtype=bool)
    half = cube_size * math.sqrt(2.0) / 2.0
    dx = np.abs(xy_new[:, None, 0] - xy_existing[None, :, 0])
    dy = np.abs(xy_new[:, None, 1] - xy_existing[None, :, 1])
    bad = (dx < 2 * half) & (dy < 2 * half)
    return bad.any(axis=1)


def _place_scattered_point(existing_xy: np.ndarray, bounds: dict, cube_size: float,
                            pitch_factors, rng: np.random.Generator,
                            max_tries: int = 200):
    """One grid-then-jitter point, rejected against `existing_xy` for
    non-penetration. Sweeps `pitch_factors` (relative to MIN_PITCH) so a
    generated state can mix densities. Falls back to continuous rejection
    sampling if every grid attempt is blocked."""
    min_pitch = cube_size * math.sqrt(2.0)
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    for _ in range(max_tries):
        pitch = min_pitch * rng.choice(pitch_factors)
        gx = rng.uniform(x0, x1)
        gy = rng.uniform(y0, y1)
        # snap to a grid cell (random phase per draw) then jitter within the cell
        cell_x = x0 + pitch * math.floor((gx - x0) / pitch + 0.5)
        cell_y = y0 + pitch * math.floor((gy - y0) / pitch + 0.5)
        jitter_max = max(0.0, (pitch - min_pitch) / 2.0)
        cand = np.array([[cell_x + rng.uniform(-jitter_max, jitter_max),
                           cell_y + rng.uniform(-jitter_max, jitter_max)]])
        cand[:, 0] = np.clip(cand[:, 0], x0, x1)
        cand[:, 1] = np.clip(cand[:, 1], y0, y1)
        if not _penetrates(existing_xy, cand, cube_size)[0]:
            return cand[0]
    # continuous rejection fallback
    for _ in range(max_tries):
        cand = np.array([[rng.uniform(x0, x1), rng.uniform(y0, y1)]])
        if not _penetrates(existing_xy, cand, cube_size)[0]:
            return cand[0]
    return None


def _place_clump(existing_xy: np.ndarray, size: int, bounds: dict, cube_size: float,
                  rng: np.random.Generator, surface_gap_tol: float = SURFACE_GAP_TOL_M,
                  max_tries: int = 300):
    """Place `size` cubes sequentially so each (after the first) TOUCHES at
    least one cube already in the clump (centre distance in
    [MIN_PITCH, MIN_PITCH+surface_gap_tol]) and does not penetrate any cube,
    clump or `existing_xy`. Returns an (m,2) array, m <= size (a clump that
    cannot legally grow further stops early rather than penetrating)."""
    min_pitch = cube_size * math.sqrt(2.0)
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    clump = []
    # seed point: continuous rejection against existing_xy only
    seed = None
    for _ in range(max_tries):
        cand = np.array([[rng.uniform(x0, x1), rng.uniform(y0, y1)]])
        if not _penetrates(existing_xy, cand, cube_size)[0]:
            seed = cand[0]
            break
    if seed is None:
        return np.zeros((0, 2))
    clump.append(seed)
    for _ in range(size - 1):
        placed = False
        for _try in range(max_tries):
            attach = clump[rng.integers(len(clump))]
            gap = rng.uniform(0.0, surface_gap_tol)
            dist = min_pitch + gap
            theta = rng.uniform(0, 2 * math.pi)
            cand = np.array([[attach[0] + dist * math.cos(theta),
                               attach[1] + dist * math.sin(theta)]])
            if not (x0 <= cand[0, 0] <= x1 and y0 <= cand[0, 1] <= y1):
                continue
            clump_xy = np.array(clump)
            all_existing = np.concatenate([existing_xy, clump_xy], axis=0) if existing_xy.shape[0] else clump_xy
            if _penetrates(all_existing, cand, cube_size)[0]:
                continue
            clump.append(cand[0])
            placed = True
            break
        if not placed:
            break  # clump stops growing early; smaller-than-requested is legal
    return np.array(clump)


def _sample_placement_region(bounds: dict, half_extent_range: tuple,
                              rng: np.random.Generator,
                              center_margin: float | None = None) -> tuple[dict, tuple, float]:
    """Draw ONE per-state placement sub-region: a half-extent `R` uniform over
    `half_extent_range` (metres), then a centre `(cx, cy)` uniform over
    `[x_min+pad, x_max-pad]` with `pad = min(R, workspace_half_extent)`.

    Padding the centre range by R (instead of drawing the centre over the
    full bounds and clipping the region afterwards) is deliberate: clipping
    an off-centre region shrinks its ACTUAL area below `(2R)^2` without
    shrinking the n_objects budget that has to fit inside it, which both
    starves the placement loop (occasional `RuntimeError`) and, worse, makes
    it pathologically slow (rejection sampling against a near-full tiny
    area) -- measured 2 orders of magnitude slower at n_objects=100 with
    naive full-range clipping. Padding means the region has its full
    intended area whenever `R` fits inside the workspace at all (the common
    case, since `REGION_HALF_EXTENT_RANGE_M`'s ranges are all well under the
    0.064 m workspace half-extent except their very top end) -- clipping
    then only ever engages for the few largest-R draws, where the region is
    already comparable to the whole workspace and clipping costs nothing.
    The centre still ranges over most of the workspace (up to +-(half_extent
    - R) on each axis), so COM still varies far more across states than the
    old bug's fixed-full-bounds sampling ever allowed -- see this module's
    validation numbers."""
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    half_x, half_y = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    r_lo, r_hi = half_extent_range
    r = float(rng.uniform(r_lo, r_hi))
    # Pad the CENTRE range by r_lo (the smallest half-extent this n_objects
    # count ever draws), not by the per-draw `r`: padding by `r` collapses
    # the centre range toward zero whenever `r` approaches the workspace
    # half-extent (both R_max values here are close to or above it), which
    # silently undoes the COM-widening this region scheme exists for. Padding
    # by `r_lo` instead keeps the centre range wide for every draw, and is
    # still exactly as safe against the small-area/edge-clip pathology this
    # module's docstring describes: whichever axis clips, the clipped region
    # still has half-extent >= r_lo on that side (cx is always >= r_lo from
    # x0/x1), so the area never drops below what R_MIN was already
    # calibrated to place `n_objects` in quickly and reliably.
    margin = r_lo if center_margin is None else center_margin
    pad_x = min(margin, half_x - 1e-9)
    pad_y = min(margin, half_y - 1e-9)
    cx = float(rng.uniform(x0 + pad_x, x1 - pad_x))
    cy = float(rng.uniform(y0 + pad_y, y1 - pad_y))
    region = {
        "x_min": max(x0, cx - r),
        "x_max": min(x1, cx + r),
        "y_min": max(y0, cy - r),
        "y_max": min(y1, cy + r),
    }
    return region, (cx, cy), r


@dataclass
class SyntheticStateResult:
    poses: np.ndarray        # (n_objects, 7) xyz + wxyz, same schema as corpus states
    n_clumps: int
    clump_sizes: list
    n_scattered: int
    region_center: tuple = (0.0, 0.0)   # (cx, cy) of the drawn placement region, metres
    region_half_extent: float = 0.0     # R of the drawn placement region, metres


def sample_synthetic_state(n_objects: int, *, bounds: dict = DEFAULT_BOUNDS,
                            cube_size: float = CUBE_SIZE, z: float = REST_Z,
                            clump_prob: float = 0.3,
                            clump_size_range: tuple = (2, 6),
                            grid_pitch_factors=DEFAULT_GRID_PITCH_FACTORS,
                            region_half_extent_range: dict | tuple | None = REGION_HALF_EXTENT_RANGE_M,
                            center_margin: dict | float | None = CENTER_MARGIN_M,
                            seed: int | np.random.Generator | None = None) -> SyntheticStateResult:
    """Generate one legal (no-penetration) synthetic pile state with
    `n_objects` cubes, mixing B1 scattered (grid+jitter, swept pitch) and B2
    clumped (sequential touch-but-no-penetrate) placement.

    Each "unit" of placement is, with probability `clump_prob`, a clump of
    size drawn uniformly from `clump_size_range` (capped by remaining count);
    otherwise a single scattered point. Position/no-penetration are exact by
    construction (grid path) or by explicit rejection (clump/fallback path);
    `assert_no_penetration` is run on the full state before return, so a
    generator bug raises here rather than emitting an illegal state.

    `region_half_extent_range` (THE DEFECT FIX): one placement sub-region
    `(cx, cy) +- R` is drawn ONCE per state (`R` uniform over this range;
    `cx, cy` uniform over most of `bounds`, padded away from the edges by
    this count's minimum R -- see `_sample_placement_region`) and used,
    bounds-clipped, as the placement area for BOTH the scattered points and
    the clump seeds --
    replacing the old behaviour of sampling every point/seed over the whole
    workspace on every call (which pinned COM to the workspace centre, worse
    as n_objects grew, and never produced compact piles; see the module
    comment above `REGION_HALF_EXTENT_RANGE_M`). Pass a `(lo, hi)` tuple to
    use one fixed range regardless of `n_objects`, a `{n_objects: (lo, hi)}`
    dict (the default, `REGION_HALF_EXTENT_RANGE_M`) to vary it by count, or
    `None` to recover the OLD full-workspace behaviour exactly (region =
    `bounds` every time, still drawn once per state so this remains a no-op
    change of code path, not of results, vs. pre-fix).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    if region_half_extent_range is None:
        region = dict(bounds)
        region_center, region_r = (0.0, 0.0), 0.0
    else:
        if isinstance(region_half_extent_range, dict):
            her = region_half_extent_range.get(n_objects)
            if her is None:
                raise KeyError(
                    f"region_half_extent_range has no entry for n_objects={n_objects}; "
                    f"available: {sorted(region_half_extent_range)}")
        else:
            her = region_half_extent_range
        if isinstance(center_margin, dict):
            margin = center_margin.get(n_objects)  # None (-> default to her[0]) if absent
        else:
            margin = center_margin
        region, region_center, region_r = _sample_placement_region(bounds, her, rng, margin)

    xy_list: list = []
    n_clumps = 0
    clump_sizes = []
    remaining = n_objects
    guard = 0
    while remaining > 0:
        guard += 1
        if guard > 10 * n_objects:
            raise RuntimeError(f"sample_synthetic_state: could not place {n_objects} "
                                f"objects (placed {n_objects - remaining}) -- region too small")
        existing_xy = np.array(xy_list) if xy_list else np.zeros((0, 2))
        if remaining >= 2 and rng.random() < clump_prob:
            size = min(remaining, int(rng.integers(clump_size_range[0], clump_size_range[1] + 1)))
            pts = _place_clump(existing_xy, size, region, cube_size, rng)
            if pts.shape[0] == 0:
                continue  # region full for a clump seed right now; try a scattered point next loop
            xy_list.extend(pts.tolist())
            n_clumps += 1
            clump_sizes.append(int(pts.shape[0]))
            remaining -= pts.shape[0]
        else:
            pt = _place_scattered_point(existing_xy, region, cube_size, grid_pitch_factors, rng)
            if pt is None:
                continue
            xy_list.append(pt.tolist())
            remaining -= 1

    xy = np.array(xy_list)
    assert_no_penetration(xy, cube_size)

    yaw = rng.uniform(0, 2 * math.pi, size=n_objects)
    qw = np.cos(yaw / 2); qz = np.sin(yaw / 2)
    poses = np.zeros((n_objects, 7), dtype=np.float64)
    poses[:, 0] = xy[:, 0]
    poses[:, 1] = xy[:, 1]
    poses[:, 2] = z
    poses[:, 3] = qw
    poses[:, 4] = 0.0
    poses[:, 5] = 0.0
    poses[:, 6] = qz
    n_scattered = n_objects - sum(clump_sizes)
    return SyntheticStateResult(poses=poses, n_clumps=n_clumps, clump_sizes=clump_sizes,
                                 n_scattered=n_scattered, region_center=region_center,
                                 region_half_extent=region_r)
