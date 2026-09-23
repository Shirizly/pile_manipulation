"""Grid -> per-object relaxation toward the centroid ("push them together").

Rigid row/column translation does not work: with random yaws one pair per row
is always near contact, so the row cannot move at all (measured: spread
0.0289 -> 0.0283 at n=100, i.e. nothing). Relaxing each object individually
lets cubes slip into the gaps their neighbours leave, which is what actually
compacts a pile.

Contact uses the EXACT separating-axis test; the axis-aligned box test in
goal_configs.py cannot express contact (it rejects legal diagonal contact by
sqrt(2)).
"""
import numpy as np
from Baselines.common.cube_overlap import overlaps_pairs


def _legal_move(xy, yaw, size, k, new_pt, nbrs, tol):
    if len(nbrs) == 0:
        return True
    m = len(nbrs)
    return not overlaps_pairs(np.repeat(new_pt[None], m, 0), np.full(m, yaw[k]),
                              xy[nbrs], yaw[nbrs], size, tol).any()


def sample_compacted_state(n_objects, rng, size=0.005, compaction=1.0,
                           n_sweeps=26, tol=-1e-6, seed_jitter=0.008,
                           yaw_kappa=0.0, pitch_factor=1.0):
    """`tol` NEGATIVE inflates the squares during placement, so every accepted
    move keeps a real margin and the finished state verifies at tol=0; a
    positive tol lets slack accumulate into genuine overlaps (measured: 0/8
    legal at full compaction).  `yaw_kappa` concentrates yaws around a random
    common heading (von Mises): real cubes settle into partly-aligned
    orientations and so pack tighter than uniform-random yaws allow."""
    # Seed pitch from the ACTUAL yaw spread, not the any-yaw worst case: a
    # near-aligned pile may legally start at ~size, and starting at the
    # worst-case 1.41*size leaves relaxation too far to travel.
    _probe = (rng.vonmises(0.0, yaw_kappa, 256) if yaw_kappa > 0
              else rng.uniform(0, 2 * np.pi, 256))
    _w = np.abs(np.cos(_probe)) + np.abs(np.sin(_probe))
    # `pitch_factor` > 1 seeds a LOOSE lattice. Without it the seed is always
    # cube-scale, so every state comes out tight however little it is
    # compacted -- measured: n=100 spanned 0.0237-0.0288 against real's
    # 0.0230-0.0637, i.e. the dispersed end was missing entirely.
    pitch = size * float(np.quantile(_w, 0.97)) * 1.02 * pitch_factor
    side = int(np.ceil(np.sqrt(n_objects))) + 1
    gx, gy = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float64)
    d = np.linalg.norm(cells - cells.mean(0), axis=1)
    cells = cells[np.argsort(d + rng.uniform(0, 0.7, len(d)))[:n_objects]]
    xy = cells * pitch
    xy += rng.uniform(-seed_jitter, seed_jitter, xy.shape) * pitch
    # NOTE: do NOT wrap with % (pi/2) here. A square is symmetric mod pi/2, so
    # the wrap looks harmless, but it maps a concentrated von Mises spread back
    # across the whole quarter-turn and destroys the concentration outright
    # (measured: yaw_kappa had exactly zero effect on packing until this went).
    if yaw_kappa > 0:
        yaw = rng.vonmises(rng.uniform(0, 2 * np.pi), yaw_kappa, n_objects)
    else:
        yaw = rng.uniform(0, 2 * np.pi, n_objects)

    n_eff = max(1, int(round(compaction * n_sweeps)))
    step = 0.45 * size
    for s in range(n_eff):
        centroid = xy.mean(0)
        # neighbour list: only nearby cubes can ever block a step this size
        D = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        np.fill_diagonal(D, 1e9)
        nbr = [np.nonzero(D[k] < 3.2 * size)[0] for k in range(n_objects)]
        for k in rng.permutation(n_objects):
            v = centroid - xy[k]
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            for trial in (v / nv * step,                      # toward centroid
                          np.array([-v[1], v[0]]) / nv * step * rng.choice([-1., 1.])):
                cand = xy[k] + trial
                if _legal_move(xy, yaw, size, k, cand, nbr[k], tol):
                    xy[k] = cand
                    break
        step *= 0.88
    xy -= xy.mean(0)
    return xy, yaw


def sample_multicluster_state(n_objects, rng, size=0.005, bounds=None,
                              compaction=None, yaw_kappa=None, n_clusters=None,
                              max_place_tries=60, **kw):
    """A state as 1..k independently-compacted clusters placed in the workspace.

    A SINGLE compacted blob cannot reach the dispersed end of the real
    distribution: material filling the entire workspace uniformly only reaches
    spread ~0.052, while real states reach 0.064 -- so those states are
    multi-cluster or boundary-hugging, not one loose pile. Measured with one
    blob: n=100 spanned 0.0236-0.0288 against real's 0.0219-0.0643.

    `n_clusters` = 1 gives the compact extreme; more, pushed apart, give the
    dispersed extreme with the right structure.
    """
    if bounds is None:
        bounds = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}
    if n_clusters is None:
        n_clusters = int(rng.integers(1, 5))
    n_clusters = max(1, min(n_clusters, max(1, n_objects // 4)))
    # split objects across clusters, each at least 3
    cuts = rng.dirichlet(np.ones(n_clusters)) * n_objects
    counts = np.maximum(3, np.round(cuts).astype(int))
    while counts.sum() > n_objects:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n_objects:
        counts[np.argmin(counts)] += 1

    theta0 = rng.uniform(0, 2 * np.pi)
    disp = rng.uniform(0.0, 1.0) ** 0.7      # per-state dispersion: 0 = stacked
                                             # at centre, 1 = pushed to corners
    xy_all, yaw_all = [], []
    for ci, c in enumerate(counts):
        comp = rng.uniform(0.05, 1.0) if compaction is None else compaction
        kap = rng.uniform(0.0, 6.0) if yaw_kappa is None else yaw_kappa
        # loose seed lattice, scaled so a single cluster can still fit: this is
        # the dispersed end of the family
        pf = 1.0 + rng.uniform(0.0, 1.0) ** 1.5 * (14.0 / np.sqrt(max(4, int(c))))
        xy_c, yaw_c = sample_compacted_state(int(c), rng, size=size, pitch_factor=pf,
                                             compaction=comp, yaw_kappa=kap, **kw)
        placed = False
        for _ in range(max_place_tries):
            r = np.sqrt(((xy_c) ** 2).sum(1)).max() + size
            # Cluster centres on a ring of radius `disp` * max, angles spread
            # apart. Uniform placement keeps every cluster near the middle and
            # caps spread at ~0.037 (n=100); real reaches 0.064, which needs
            # clusters pushed out to opposite corners.
            ax = max(0.0, (bounds["x_max"] - bounds["x_min"]) / 2 - r)
            ay = max(0.0, (bounds["y_max"] - bounds["y_min"]) / 2 - r)
            cx = rng.uniform(-ax, ax)
            cy = rng.uniform(-ay, ay)
            cand = xy_c + np.array([cx, cy])
            if not xy_all:
                xy_all.append(cand); yaw_all.append(yaw_c); placed = True; break
            prev = np.concatenate(xy_all)
            d = np.linalg.norm(cand[:, None] - prev[None], axis=-1).min()
            if d > size * 1.45:           # clusters must not interpenetrate
                xy_all.append(cand); yaw_all.append(yaw_c); placed = True; break
        if not placed:
            return None, None
    return np.concatenate(xy_all), np.concatenate(yaw_all)
