"""State synthesizer: scattered singletons + compact clumps, mixed.

A state is built from UNITS -- a unit is either one cube (scatter) or a small
compacted clump. The clump fraction sets local density (contacts per cube);
the placement-region extent sets overall spread. Those two knobs map onto the
two statistics that characterise a real pile, so they can be calibrated
against real post-sweep states directly.

Clumps come from `pile_compaction.sample_compacted_state`, and every placement
is checked with the EXACT separating-axis test in `cube_overlap` -- the
axis-aligned box test in goal_configs.py rejects legal diagonal contact by
sqrt(2) and so cannot represent a genuinely compact clump at all.
"""
import numpy as np

from Baselines.common.cube_overlap import overlaps_pairs
from Baselines.common.pile_compaction import sample_compacted_state

DEFAULT_BOUNDS = {"x_min": -0.064, "x_max": 0.064, "y_min": -0.064, "y_max": 0.064}


def _blocked(xy_new, yaw_new, xy_ex, yaw_ex, size, tol):
    if len(xy_ex) == 0:
        return False
    m, e = len(xy_new), len(xy_ex)
    i, j = np.meshgrid(np.arange(m), np.arange(e), indexing="ij")
    i, j = i.ravel(), j.ravel()
    return bool(overlaps_pairs(xy_new[i], yaw_new[i], xy_ex[j], yaw_ex[j], size, tol).any())


def synthesize_state(n_objects, rng, size=0.005, bounds=None, region_frac=None,
                     clump_frac=None, clump_size_max=12, tol=-1e-6, tries=420):
    """One legal state. `region_frac` in (0,1] scales the placement region and
    therefore the spread; `clump_frac` in [0,1] is the share of cubes that go
    into compact clumps and therefore the contact count."""
    b = DEFAULT_BOUNDS if bounds is None else bounds
    # Defaults calibrated against REAL POST-SWEEP states (DS-0002, role=
    # post_sweep), which are mostly dispersed with a compact tail: spread
    # p5/p50/p95 = .0136/.0453/.0554 (n=20), .0197/.0484/.0558 (n=50),
    # .0372/.0510/.0592 (n=100); contacts p50 = 0.90/1.56/2.66.
    # Region extent sets SPREAD, clump share sets CONTACTS, so they calibrate
    # against those two statistics almost independently.
    if region_frac is None:
        # MIXTURE, not a power curve: real spread has a long compact tail
        # (p5 .0136) under a wide median (p50 .0453) at n=20, and no single
        # power transform gives both -- biasing wide enough for the median
        # erases the tail (measured p5 .0307 vs .0136) and vice versa.
        if rng.uniform() < 0.20:
            region_frac = rng.uniform(0.13, 0.46)              # compact tail
        else:
            region_frac = 0.78 + 0.22 * rng.uniform()          # dispersed bulk
    if clump_frac is None:
        # denser piles at higher object counts: real contacts/cube rise with n
        # while spread also rises, i.e. many clumps spread out, not one blob
        expo = float(np.interp(n_objects, [20, 50, 100], [1.6, 1.0, 0.35]))
        clump_frac = rng.uniform() ** expo

    # --- partition into units -------------------------------------------------
    n_in_clumps = int(round(clump_frac * n_objects))
    sizes, rem = [], n_in_clumps
    while rem >= 2:
        s = int(rng.integers(2, max(3, min(clump_size_max, rem) + 1)))
        s = min(s, rem)
        sizes.append(s); rem -= s
    n_single = n_objects - sum(sizes)

    units = []
    for s in sizes:
        xy_c, yaw_c = sample_compacted_state(
            s, rng, size=size, compaction=rng.uniform(0.6, 1.0),
            yaw_kappa=rng.uniform(0.0, 4.0), n_sweeps=14)
        units.append((xy_c, yaw_c))
    for _ in range(n_single):
        units.append((np.zeros((1, 2)), rng.uniform(0, 2 * np.pi, 1)))
    rng.shuffle(units)

    # --- place units in a region, largest first (hardest to fit) -------------
    hx = (b["x_max"] - b["x_min"]) / 2 * region_frac
    hy = (b["y_max"] - b["y_min"]) / 2 * region_frac
    # Region centre ranges over the FULL workspace, not inset by the region
    # half-extent: inset pins a wide region to the middle, so material can
    # never pile against an edge and |COM| stays far below real (measured p95
    # 0.0117 vs real 0.0281 at n=100). Units falling outside are rejected by
    # the per-unit bounds check below, which is what produces edge-hugging.
    # MIXTURE. Inset-only pins a wide region to the middle and |COM| stays far
    # below real (p95 .0117 vs .0281 at n=100); full-range-only lets the
    # boundary clip the region, which fixes |COM| but compresses spread
    # (p50 .0301 vs .0453 at n=20). Real needs both populations.
    if rng.uniform() < 0.35:                       # edge-hugging
        cx0 = rng.uniform(b["x_min"], b["x_max"])
        cy0 = rng.uniform(b["y_min"], b["y_max"])
    else:                                          # centred, free to be wide
        cx0 = rng.uniform(b["x_min"] + hx, b["x_max"] - hx)
        cy0 = rng.uniform(b["y_min"] + hy, b["y_max"] - hy)
    units.sort(key=lambda u: -len(u[0]))
    # Uniform fill of a region caps spread at 0.816*half-width; real post-sweep
    # p95 sits ABOVE that, so some states must concentrate toward the region
    # boundary (material swept outward). `edge` pushes sampled positions out
    # along the square's max-norm.
    edge = rng.uniform(0.0, 1.3) if rng.uniform() < 0.5 else 0.0

    xy_all = np.zeros((0, 2)); yaw_all = np.zeros((0,))
    for xy_u, yaw_u in units:
        r = np.linalg.norm(xy_u, axis=1).max() if len(xy_u) > 1 else 0.0
        ok = False
        for _ in range(tries):
            ux, uy = rng.uniform(-1, 1), rng.uniform(-1, 1)
            if edge > 0:
                m = max(abs(ux), abs(uy), 1e-9)
                ux, uy = ux * (m ** (1.0 / (1.0 + edge))) / m, uy * (m ** (1.0 / (1.0 + edge))) / m
            px = cx0 + ux * hx; py = cy0 + uy * hy
            cand = xy_u + np.array([px, py])
            if (cand[:, 0].min() < b["x_min"] + size or cand[:, 0].max() > b["x_max"] - size
                    or cand[:, 1].min() < b["y_min"] + size or cand[:, 1].max() > b["y_max"] - size):
                continue
            if not _blocked(cand, yaw_u, xy_all, yaw_all, size, tol):
                xy_all = np.concatenate([xy_all, cand]); yaw_all = np.concatenate([yaw_all, yaw_u])
                ok = True; break
        if not ok:
            return None, None
    return xy_all, yaw_all
