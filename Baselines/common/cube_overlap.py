"""Exact separating-axis overlap test for yaw-rotated squares (2D footprints).

Replaces the axis-aligned circumscribed-box test in goal_configs.py, which
rejects 91-99% of REAL simulated states (measured on DS-0002): it demands
|dx| >= a*sqrt(2) OR |dy| >= a*sqrt(2), an axis-aligned keep-out square using
the circumscribed diameter per axis, so two cubes sitting diagonally at
dx=dy=a (centre distance a*sqrt(2), genuinely legal at ANY yaw) are refused.
"""
import numpy as np


def _axes(yaw):
    """(m,2,2): for each yaw, its 2 unique edge normals (unit)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], axis=-2)


def overlaps_rect_pairs(xy_a, yaw_a, half_a, xy_b, yaw_b, half_b, tol=0.0):
    """Exact SAT overlap for arbitrary yaw-rotated RECTANGLES, pairwise a[i]
    vs b[i] -- generalizes `overlaps_pairs` (equal squares) to two boxes of
    independently-chosen (half_length, half_width), e.g. a thin blade vs a
    cube footprint (2026-09-28, EXP-0059 tool-placement audit).

    `half_a`/`half_b`: (2,) or (m,2) half-extents along EACH box's own local
    axes (`_axes(yaw)`'s axis 0 = the box's own "length" direction
    (cos, sin), axis 1 = its "width" direction (-sin, cos)) -- e.g. for the
    Genesis plate, `half=(tool_length/2, tool_width/2)` with `yaw` = blade
    yaw (`Genesis/configs/basic.yaml plate.size`, this repo's blade-yaw
    convention). `tol` > 0 shrinks BOTH boxes (a contact gap < tol counts as
    separated); `tol` < 0 INFLATES both (a gap smaller than |tol| counts as
    overlapping) -- the margin variant of a legality check.
    Returns bool (m,).
    """
    xy_a, xy_b = np.asarray(xy_a, dtype=float), np.asarray(xy_b, dtype=float)
    ha = np.broadcast_to(np.asarray(half_a, dtype=float), xy_a.shape) - 0.5 * tol
    hb = np.broadcast_to(np.asarray(half_b, dtype=float), xy_b.shape) - 0.5 * tol
    d = xy_b - xy_a                                     # (m,2)
    A, B = _axes(yaw_a), _axes(yaw_b)                   # (m,2,2)
    sep = np.zeros(d.shape[0], dtype=bool)
    for axes in (A, B):
        for k in range(2):
            n = axes[:, k, :]                           # (m,2)
            # projection radius of a rectangle = h_l * |n.axis0| + h_w * |n.axis1|
            ra = ha[:, 0] * np.abs((A[:, 0] * n).sum(-1)) + ha[:, 1] * np.abs((A[:, 1] * n).sum(-1))
            rb = hb[:, 0] * np.abs((B[:, 0] * n).sum(-1)) + hb[:, 1] * np.abs((B[:, 1] * n).sum(-1))
            sep |= np.abs((d * n).sum(-1)) > (ra + rb)
    return ~sep


def overlaps_pairs(xy_a, yaw_a, xy_b, yaw_b, size, tol=0.0):
    """Exact SAT overlap for squares of edge `size`, pairwise a[i] vs b[i].

    `tol` > 0 shrinks both squares (treats a contact gap < tol as separated),
    which is how 'touching' is expressed without float equality. A thin
    wrapper around `overlaps_rect_pairs` with equal half-extents on both
    sides (numerically identical to the pre-2026-09-28 direct implementation
    -- see this module's docstring history).
    Returns bool (m,).
    """
    half = np.array([0.5 * size, 0.5 * size])
    return overlaps_rect_pairs(xy_a, yaw_a, half, xy_b, yaw_b, half, tol=tol)


def any_overlap_matrix(xy, yaw, size, tol=0.0, ignore_self=True):
    """(n,n) bool overlap matrix for one configuration."""
    n = xy.shape[0]
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    ov = overlaps_pairs(xy[i], yaw[i], xy[j], yaw[j], size, tol).reshape(n, n)
    if ignore_self:
        np.fill_diagonal(ov, False)
    return ov


def state_is_legal(xy, yaw, size, tol=0.0):
    return not any_overlap_matrix(xy, yaw, size, tol).any()
