"""Shared matplotlib primitives for drawing cubes/pushes in metre- or mm-space
top-down plots: a rotated square patch (from a yaw angle, for real cube
footprints), a push arrow, and a blade/plate line segment perpendicular to a
push direction.

Consolidated here (2026-09-28, EXP-0059 visual-debug task) because the SAME
three bits of drawing code were independently re-derived in
`transition_panel.py::push_arrow` and inline inside
`experiments/EXP-0051-*/code/demo_gifs.py::draw_frame` (blade endpoints +
arrow), and a third near-identical copy was about to be written for
`experiments/EXP-0059-*/code/retrieval_debug.py`. Per project-overview's
Visualization guidance ("plotting lives with the thing being plotted", no
`viz/`): this is colocated with the other shared probe-plotting utility
(`pool_common.py`), not a new top-level package, and holds ONLY the small
geometric primitives that were genuinely duplicated -- not a general
plotting framework. Callers still build their own figures/axes/titles.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle


def cube_patch(center, yaw: float, size: float, **kw) -> Rectangle:
    """A `size` x `size` square centred at `center` (x, y), rotated `yaw`
    radians CCW about its own centre -- the rotated-square footprint of one
    real cube (yaw from `model.retrieval.frame.yaw_from_quat` or equivalent).
    `**kw` forwards to `Rectangle` (facecolor, alpha, edgecolor, ...)."""
    cx, cy = float(center[0]), float(center[1])
    return Rectangle((cx - size / 2, cy - size / 2), size, size,
                      angle=float(np.degrees(yaw)), rotation_point="center", **kw)


def push_arrow(ax, start, stop, color="k", lw=1.2, **kw):
    """One push drawn as an arrow start -> stop (any consistent units)."""
    ax.annotate("", xy=(float(stop[0]), float(stop[1])),
                xytext=(float(start[0]), float(start[1])),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, **kw))


def blade_endpoints(center, direction, half_width):
    """`center`: (x, y) the blade/plate's own centre. `direction`: (dx, dy),
    the push heading (need not be unit length). `half_width`: half the
    blade's extent. -> (p0, p1), the two blade endpoints, perpendicular to
    `direction`, straddling `center`."""
    d = np.asarray(direction, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    perp = np.array([-d[1], d[0]])
    c = np.asarray([center[0], center[1]], dtype=float)
    return c - perp * half_width, c + perp * half_width
