"""Baselines/GNN/perception.py -- deployment-realistic node construction:
read the occupied pixels off the SAME top-down occupancy raster every
other baseline in this repo already consumes
(`Baselines/common/data.py::rasterize_particles`) as a point cloud, then
Farthest Point Sample (FPS) it down to a fixed node count. This plays the
role of the paper's own perception module (RGB-D -> foreground point cloud
-> FPS, SPEC.md section 1), per an explicit 2026-09-10 direction from the
task owner: every model in this project must be able to run from the
camera raster alone -- a real deployment has no privileged per-cube 3D
pose, only a top-down image. See SPEC.md's "CORRECTION" section for the
full history (an earlier version of this baseline fed ground-truth
simulator cube centroids directly to the graph nodes; that was wrong, not
a deliberate simplification).

Terminology note: our raster is already PURE foreground by construction --
it is a rendered top-down silhouette of the pile alone (no table, no
clutter, no sensor noise), unlike the paper's real RGB-D frame, which
genuinely needs depth-thresholding to separate the pile from the
background it was captured against (`dataset_gnn_dyn.py:98`,
`depth < 0.599/0.8`). There is no segmentation/background-subtraction step
here to mimic that -- `foreground_points_xy` below just reads out which
pixels are already occupied (value 1) as a point cloud. The function name
is kept for continuity with the paper's step this replaces, not because
anything is being extracted FROM a background.

What this module does NOT do, on purpose:
  * it never reads per-cube 3D height/orientation -- our raster (like a
    real top-down camera with no depth channel) carries none, so every
    node gets a single FIXED z (`Z_CONST`) and, when a predicted particle
    needs rendering back to occupancy, a FIXED, axis-aligned footprint at
    the real cube's own size (`rasterize_nodes_as_cubes`, `CUBE_SIZE_M`) --
    never a per-node orientation (2026-09-10 direction: no circles either,
    a cube footprint at the correct physical size, unrotated). This is a
    real capability gap versus the pile's true geometry, not an oversight.
  * it never uses the simulator's exact cube centroid as a node's INPUT
    position. A node's input (x, y) is exactly the raster-derived,
    pixel-quantized point FPS selected -- nothing privileged.

What it DOES use privileged simulator state for, and why that's fine:
  * TRAINING LABELS ONLY (`track_displacement`, called from the dataset,
    never from `predictor.py`). To know how far a given raster-observed
    point should move (the regression target), this tracks the nearest
    real cube (KDTree match, exactly `dataset_gnn_dyn.py:108-109`'s own
    `first_particles_tree.query` step) and reads THAT cube's true
    displacement from the simulator. This is the standard sim-training
    pattern (privileged label, realistic input) -- a real deployment would
    need actual frame-to-frame tracking (optical flow/ICP) to reconstruct
    this same displacement signal from a live camera feed. At inference
    (`predictor.py`) none of this tracking machinery runs at all; only
    foreground-extraction + FPS on the given occupancy is used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

N_PARTICLES = 20          # fixed node count -- SPEC.md's own scope (no resolution regressor)

FOREGROUND_THRESH = 0.5   # occupancy is a hard {0,1} grid; any positive value is foreground

# metres -- local-mean radius after FPS, mirrors utils.py::recenter (and
# dataset_gnn_dyn.py:101's use of it). Chosen just under half the 5mm cube
# edge so it averages a single cube's own occupied pixels without blending
# into a touching neighbour's.
RECENTER_R = 0.003

# Every node gets this z. Height is not observable from a top-down
# occupancy raster, so it is never estimated, measured, or looked up --
# see SPEC.md hazard on orientation/height, now widened to position too.
Z_CONST = 0.0

# metres -- real cube edge length, `data_collection.sampled.particle_sizes`
# / `data_collection.particle_size` in every cell's own `_0_config.yaml`
# (confirmed identical -- 0.005 m -- in both the slates_multistep cells and
# every overnight_randlen group, n20 and n50 alike). Used as the fixed
# footprint size for `rasterize_nodes_as_cubes` -- a node's rasterised
# footprint is always THIS size, never a per-cell/per-particle size lookup
# (which would require a real per-particle identity we no longer have,
# and would break the moment a model's chosen node count differs from a
# cell's true particle count, e.g. n_particles=30 scored on an n50 cell).
CUBE_SIZE_M = 0.005


def _fps_indices(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy farthest-point sampling to a fixed count `n` over 2-D
    `points`. Returns the SELECTED INDICES into `points`. Pads by tiling if
    the candidate set is smaller than `n` (degenerate/heavily-occluded
    frame; not expected given ~120-180 occupied pixels per real transition
    at this cell's resolution, but keeps this from ever crashing -- same
    under-population handling as `model/eulerian_wrapper.py::_fps_np`)."""
    m = points.shape[0]
    if m < n:
        reps = int(np.ceil(n / m))
        return np.tile(np.arange(m), reps)[:n]

    start = int(rng.integers(m))
    selected = [start]
    dist = np.linalg.norm(points - points[start], axis=-1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=-1))
    return np.asarray(selected, dtype=np.int64)


def foreground_points_xy(occ: np.ndarray, to_pxl: float, ctr_in_pxl,
                          thresh: float = FOREGROUND_THRESH) -> np.ndarray:
    """occ: (H,W) top-down occupancy grid, dim0=world_x axis, dim1=world_y
    axis (the post-transpose convention `_draw_particle_grid` returns --
    see `Genesis/training/dataset.py`'s own docstring on why: OpenCV draws
    in (col,row) order, the opposite of this repo's (x,y) convention
    elsewhere, and skipping the transpose silently swaps the two world
    axes). `occ` is already pure foreground (no
    background/clutter rendered into it at all, module docstring) -- this
    just reads out the occupied pixels as a point cloud, it does not
    segment anything out of a noisier scene the way a real depth-threshold
    step would. Returns (M,2) world-frame xy points, one per occupied
    pixel, at pixel centres -- the exact inverse of `rasterize_particles`'s
    forward transform (`Baselines/common/data.py:175-176`:
    `world*to_pxl + ctr_in_PXL`), never re-derived from scratch."""
    idx = np.argwhere(occ >= thresh)
    if idx.shape[0] == 0:
        raise ValueError("occupancy grid has no foreground pixels above threshold")
    ctr = np.asarray(ctr_in_pxl, dtype=np.float64)[:2]
    return (idx.astype(np.float64) - ctr[None, :]) / to_pxl


def _recenter(candidates_xy: np.ndarray, sampled_xy: np.ndarray, r: float = RECENTER_R) -> np.ndarray:
    """Replace each FPS-selected point with the mean of all candidate
    points within radius `r` of it -- purely raster-derived (only uses the
    SAME candidate cloud FPS ran on), mirrors `utils.py::recenter`."""
    out = sampled_xy.copy()
    for j, p in enumerate(sampled_xy):
        near = candidates_xy[np.linalg.norm(candidates_xy - p, axis=-1) <= r]
        if near.shape[0] > 0:
            out[j] = near.mean(axis=0)
    return out


def sample_nodes_xy(occ: np.ndarray, to_pxl: float, ctr_in_pxl, n_particles: int = 20,
                     seed: int | None = None, thresh: float = FOREGROUND_THRESH) -> np.ndarray:
    """The full deployment-time recipe: read `occ`'s occupied pixels as a
    point cloud (already pure foreground, see module docstring -- no
    background/clutter to segment out), FPS to `n_particles`, recenter.
    Returns (n_particles,2) world xy -- these ARE
    the node input positions, no privileged lookup involved anywhere in
    this function. `seed` fixes the FPS starting point for reproducibility
    (pass e.g. the dataset row index); omit for an independently
    randomised start each call."""
    rng = np.random.default_rng(seed)
    candidates_xy = foreground_points_xy(occ, to_pxl, ctr_in_pxl, thresh)
    sel = _fps_indices(candidates_xy, n_particles, rng)
    sampled_xy = candidates_xy[sel]
    return _recenter(candidates_xy, sampled_xy)


def track_displacement(sampled_xy: np.ndarray, gt_cur_xy: np.ndarray,
                        gt_next_xy: np.ndarray) -> np.ndarray:
    """TRAINING-LABEL-ONLY helper (see module docstring) -- never called
    from `predictor.py`. For each raster-derived node in `sampled_xy`, find
    its nearest real cube in `gt_cur_xy` ((20,2), privileged simulator
    state) and return that cube's true (dx,dy) displacement to
    `gt_next_xy`. Mirrors `dataset_gnn_dyn.py:108-109`'s own KDTree-
    tracking step; duplicates are possible (two raster points snapping to
    the same real cube) and are NOT deduplicated, exactly as the reference
    recipe does not deduplicate -- an honest reflection of real tracking
    ambiguity between adjacent/touching cubes, not a bug to hide."""
    tree = cKDTree(gt_cur_xy)
    _, nearest_idx = tree.query(sampled_xy, k=1)
    return gt_next_xy[nearest_idx] - gt_cur_xy[nearest_idx]


def rasterize_nodes_as_cubes(node_xy: np.ndarray, to_pxl: float, ctr_in_pxl,
                              H: int, W: int, cube_size_m: float = CUBE_SIZE_M) -> np.ndarray:
    """Axis-aligned, FIXED-size box rasterisation of `node_xy` ((n,2) world
    xy) -- the node-count-agnostic replacement for
    `PileSweepData._draw_particle_grid`/the old
    `rasterize_particles_batch` (removed 2026-09-10). Those read a per-cell
    config's own `n_particles`/`particle_sizes` list, which assumes one
    node per REAL cube -- no longer true once node count is a free
    hyperparameter decoupled from true particle count (SPEC.md's
    "CORRECTION" section): scoring a 30-node model on an n50 cell would
    silently truncate to 30 boxes or index past the end of `node_xy`,
    depending on which is larger. This function instead draws exactly
    `node_xy.shape[0]` boxes, every one the same real cube size, at yaw=0
    (2026-09-10 direction: cubes, not circles, no orientation -- this is a
    low-resolution raster regardless of shape choice).

    Every box is axis-aligned (no `cv2.boxPoints`/rotation needed at
    yaw=0), so this uses plain numpy slice assignment, which correctly ORs
    overlapping boxes together -- unlike naively batching multiple
    `cv2.fillPoly` contours in one call (the documented trap in this
    project's rasteriser history), plain slice assignment has no
    even-odd-winding-rule surprise.

    Same world->pixel transform as every other rasteriser here
    (`world*to_pxl + ctr_in_PXL`); `to_pxl`/`ctr_in_pxl` come from the
    cell's own `PileSweepData` instance (`raw.to_pxl`/`raw.ctr_in_PXL`),
    never re-derived. Returns a NEW (H,W) float32 occupancy grid.
    """
    grid = np.zeros((H, W), dtype=np.float32)
    half_px = 0.5 * cube_size_m * to_pxl
    ctr = np.asarray(ctr_in_pxl, dtype=np.float64)[:2]
    px = np.asarray(node_xy, dtype=np.float64) * to_pxl + ctr[None, :]
    for x, y in px:
        r0 = max(0, int(round(x - half_px)))
        r1 = min(H, int(round(x + half_px)))
        c0 = max(0, int(round(y - half_px)))
        c1 = min(W, int(round(y + half_px)))
        if r1 > r0 and c1 > c0:
            grid[r0:r1, c0:c1] = 1.0
    return grid


def rasterize_nodes_as_cubes_batch(node_xy: np.ndarray, to_pxl: float, ctr_in_pxl,
                                    H: int, W: int, cube_size_m: float = CUBE_SIZE_M) -> np.ndarray:
    """Batched form of `rasterize_nodes_as_cubes`: `node_xy` is (B,n,2).
    Returns (B,H,W) float32. A plain per-row loop -- this is pure numpy
    slice assignment (no cv2/Python-object overhead per particle), so it
    is already fast; no thread pool needed (contrast the old cv2-based
    `rasterize_particles_batch`, which needed one for exactly that
    per-particle Python/cv2 cost)."""
    B = node_xy.shape[0]
    out = np.empty((B, H, W), dtype=np.float32)
    for i in range(B):
        out[i] = rasterize_nodes_as_cubes(node_xy[i], to_pxl, ctr_in_pxl, H, W, cube_size_m)
    return out


def resample_occupancy_through_nodes(occ: np.ndarray, to_pxl: float, ctr_in_pxl,
                                      n_particles: int, seed: int | None = None,
                                      cube_size_m: float = CUBE_SIZE_M) -> np.ndarray:
    """Ground-truth-side helper for the accuracy report: take an existing
    occupancy grid (e.g. ground-truth `occ0`/`occ1`), FPS-sample it down to
    `n_particles` nodes exactly as `sample_nodes_xy` does for a model's own
    INPUT, then rasterise those nodes back to occupancy
    (`rasterize_nodes_as_cubes`). This does NOT touch a model at all --
    it exists so a node-count-limited model's prediction can be compared
    against a ground truth that has gone through the SAME node-count
    bottleneck, isolating dynamics-prediction error from the
    representational loss of using only `n_particles` nodes to describe an
    arbitrary pile (see `docs/experiments/METRICS.md`, "GNN accuracy" for
    why this comparison is the fair one)."""
    nodes_xy = sample_nodes_xy(occ, to_pxl, ctr_in_pxl, n_particles=n_particles, seed=seed)
    return rasterize_nodes_as_cubes(nodes_xy, to_pxl, ctr_in_pxl, occ.shape[0], occ.shape[1], cube_size_m)
