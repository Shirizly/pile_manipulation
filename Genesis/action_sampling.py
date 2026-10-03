"""
Genesis/action_sampling.py — action-distribution shaping that has to be aware of
how a batch is simulated, not just of what a single action should look like.

Why this exists
---------------
Environments in a batch step in lockstep, so a batch costs what its *worst*
member costs — twice over:

  step count      ``plate_velocity_translation`` derives ``sweep_steps`` from
                  the LONGEST travel distance in the batch, so every env runs
                  for as long as the longest one, however short its own push.
  per-step cost   each step costs what the densest contact graph in the batch
                  costs.

Sampling each env's action independently therefore makes batching much worse
than it needs to be. Measured at 100 objects (see
tests/scaling_investigation/probe_action_coupling.py), going from 1 to 8 envs:

  identical action in every env    13.47 s -> 15.21 s   (1.13x for 8x the work)
  independently sampled actions    13.34 s -> 40.17 s   (3.01x)

with the gap decomposing into 1.54x from step count and 1.72x from contact
complexity. The step-count half is removable at almost no cost to the data:
share one travel *distance* across the batch while every env keeps its own start
point, direction and blade yaw. Distance is one of five action dimensions and it
still varies fully from batch to batch — only its within-batch variance is given
up, and that variance was buying nothing except a longer sweep for everybody.

Action-space *restriction* also lives here
--------------------------------------------
The second group of helpers restricts which actions are drawn at all, rather
than reshaping their cost. Both restrictions exist for the switched-linear
visual-foresight baseline (docs/linear_visual_foresight_baseline.md §7), which
needs one operator per push length in a frame where the blade face is normal to
the push:

  perpendicular    the push travels along the blade's face normal, so the
                   5-DOF action (start, direction, yaw) collapses to the 4-DOF
                   action that planar-pushing work assumes (Mason 1986).
  fixed length     every push travels the same distance, so a whole dataset
                   supports a single transition operator.

They are plain torch geometry with no Genesis dependency, so they are unit
testable without a GPU (tests/test_action_sampling.py).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

_EPS = 1e-12


def ray_box_max_travel(starts_xy: torch.Tensor, direction: torch.Tensor,
                       low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """How far can we travel along `direction` before leaving [low, high]?

    Per axis the limiting bound is `high` when moving positively along it and
    `low` when moving negatively; an axis we are not moving along cannot limit
    us, hence the infinities.

    Parameters
    ----------
    starts_xy : (..., 2) ray origins.
    direction : (..., 2) unit (or at least non-zero) directions.
    low, high : (..., 2) per-entry box bounds.

    Returns
    -------
    (..., 1) non-negative maximum travel distance.
    """
    inf = torch.full_like(direction, float("inf"))
    t_axis = torch.where(
        direction > _EPS, (high - starts_xy) / torch.where(direction > _EPS, direction, inf),
        torch.where(
            direction < -_EPS, (low - starts_xy) / torch.where(direction < -_EPS, direction, inf),
            inf,
        ),
    )
    return t_axis.min(dim=-1, keepdim=True).values.clamp(min=0.0)


def equalize_travel_distance(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                             low: torch.Tensor, high: torch.Tensor,
                             target: torch.Tensor):
    """Rescale each env's push to a shared travel distance, staying in bounds.

    Each env keeps its own start point and direction; only the distance along
    that direction changes. Where the shared distance would take a push outside
    its sampling box, it is truncated at the boundary — those envs travel less,
    which is the safe direction to err since the batch's step count follows the
    longest push, not the shortest.

    Parameters
    ----------
    starts_xy, stops_xy : (..., 2) push endpoints.
    low, high           : (..., 2) per-entry sampling bounds (they depend on the
                          blade yaw, so they are not a single global box).
    target              : (..., 1) desired travel distance, broadcastable.

    Returns
    -------
    (new_stops_xy, clipped_mask) — the mask marks entries that could not reach
    the target distance without leaving their box.
    """
    delta = stops_xy - starts_xy
    dist = delta.norm(dim=-1, keepdim=True)
    direction = delta / (dist + _EPS)

    t_max = ray_box_max_travel(starts_xy, direction, low, high)

    t = torch.minimum(target, t_max)
    clipped = (target > t_max + 1e-9).squeeze(-1)
    return starts_xy + direction * t, clipped


def shared_batch_distance(dist: torch.Tensor, env_dim: int = 0) -> torch.Tensor:
    """Pick the travel distance a whole batch will share, per sample.

    Takes one environment's own draw rather than a summary statistic. A median
    or mean over environments would concentrate as the batch grows — with many
    envs it would converge to the population median and the *between-batch*
    variation in push length would quietly disappear. One env's draw is an exact
    sample from the same distribution a single-env run would see, so the
    marginal distribution of push lengths across batches is unchanged.
    """
    return dist.select(env_dim, 0).unsqueeze(env_dim)


# ---------------------------------------------------------------------------
# Action-space restriction: perpendicular pushes and fixed push length
# ---------------------------------------------------------------------------


def blade_normal(angles: torch.Tensor) -> torch.Tensor:
    """Unit normal of the blade face, for blade yaw `angles`.

    The blade's long axis is ``(cos θ, sin θ)`` — read off how
    ``SandboxManipulation.generate_action_samples`` shrinks its sampling box,
    which subtracts ``cos(θ)·tool_length/2`` from the x half-extent — so the
    face normal, the only direction a *perpendicular* push can travel, is
    ``(−sin θ, cos θ)``.

    Returns (..., 2). Note the sign is arbitrary: a blade at yaw θ can push
    along either ``+n̂`` or ``−n̂``, and callers must choose (see
    `constrain_push`, which randomizes it). Because yaw is drawn from
    ``(−π/2, π/2)``, ``cos θ > 0`` always, so always taking ``+n̂`` would send
    every push into the ``+y`` half-plane — a badly skewed dataset that no
    aggregate statistic would reveal.
    """
    return torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)


def sampling_box(angles: torch.Tensor, granular_vol, tool_length: float,
                 tool_width: float, safety_margin: float):
    """Per-yaw bounds for the blade centre, as (low, high) each (..., 2).

    The box shrinks with yaw because the blade's axis-aligned footprint does:
    at θ=0 its length eats into x and its width into y, and they swap at
    θ=±π/2. Extracted so the bounds can be *recomputed* after
    placement-aware sampling replaces the yaw — the box that was correct for
    the drawn yaw is stale for the replacement.
    """
    cos, sin = torch.cos(angles), torch.sin(angles).abs()
    half_x = granular_vol[0] / 2 - (cos * tool_length / 2
                                    + sin * tool_width / 2 + safety_margin)
    half_y = granular_vol[1] / 2 - (sin * tool_length / 2
                                    + cos * tool_width / 2 + safety_margin)
    high = torch.stack([half_x, half_y], dim=-1)
    return -high, high


class ConstrainedPush(NamedTuple):
    """Result of `constrain_push`.

    starts_xy, stops_xy : the push, after restriction.
    starts_moved        : starts nudged to make a fixed-length push fit (see
                          `constrain_push`); the nudge is at most `length`.
    truncated           : pushes that could NOT be made to travel the requested
                          length. These are not in the requested length bin and
                          must not be fitted as though they were.
    """
    starts_xy: torch.Tensor
    stops_xy: torch.Tensor
    starts_moved: torch.Tensor
    truncated: torch.Tensor


def constrain_push(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                   angles: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
                   *, perpendicular: bool = False, length: float | None = None,
                   generator: torch.Generator | None = None) -> ConstrainedPush:
    """Restrict pushes to the blade normal and/or to a fixed travel distance.

    Blade yaw is never changed, so this composes with placement-aware start
    sampling (which chooses start *and* yaw) as long as it runs *after* it —
    running before would let the yaw be replaced underneath and silently break
    perpendicularity.

    Parameters
    ----------
    starts_xy, stops_xy : (..., 2) push endpoints as drawn.
    angles              : (...,)   blade yaw, radians.
    low, high           : (..., 2) sampling bounds *for these angles*.
    perpendicular       : send the push along ±`blade_normal(angles)` instead
                          of along the drawn start→stop direction.
    length              : if given, every push travels exactly this distance
                          (metres) instead of the drawn one, so the whole
                          dataset supports one transition operator.
    generator           : optional torch.Generator for the ± sign draw.

    Returns
    -------
    A `ConstrainedPush`. See the notes below on how the tray boundary is
    handled, which differs between the two restrictions on purpose.
    """
    no_flag = torch.zeros(starts_xy.shape[:-1], dtype=torch.bool,
                          device=starts_xy.device)
    if not perpendicular and length is None:
        return ConstrainedPush(starts_xy, stops_xy, no_flag, no_flag)

    delta = stops_xy - starts_xy
    drawn_dist = delta.norm(dim=-1, keepdim=True)
    target = (torch.full_like(drawn_dist, float(length))
              if length is not None else drawn_dist)

    if perpendicular:
        n_hat = blade_normal(angles)
        sign = torch.randint(0, 2, starts_xy.shape[:-1], generator=generator,
                             device=starts_xy.device,
                             dtype=starts_xy.dtype).mul_(2).sub_(1)
        direction = sign.unsqueeze(-1) * n_hat

        # A push blocked by the tray wall on the drawn side is retried on the
        # other: the ± choice is free, so spend it on reaching the target
        # length rather than fighting the boundary.
        reach = ray_box_max_travel(starts_xy, direction, low, high)
        flipped = ray_box_max_travel(starts_xy, -direction, low, high)
        direction = torch.where((reach < target) & (flipped > reach),
                                -direction, direction)
    else:
        direction = delta / (drawn_dist + _EPS)

    # Where the requested travel does not fit inside the tray from the start it
    # was drawn at, move the START rather than shortening the push. Shortening
    # would be the obvious fix and it is the wrong one twice over: a shortened
    # fixed-length push silently leaves the requested bin and contaminates a
    # single-operator fit, and a shortened free-length push distorts the very
    # length distribution this restriction is supposed to leave alone (measured
    # on 20k draws: shortening truncates 22.5% and drags the mean push from
    # 105 mm to 94 mm). Nudging the start keeps yaw, direction and length all
    # exact and costs a start displacement of at most the push length.
    #
    # Feasible starts satisfy both `low <= start <= high` and
    # `low <= start + d <= high`, i.e. `low - d <= start <= high - d`.
    # Intersecting: [low + relu(-d), high - relu(d)].
    #
    # Note this is not a rare corner case even in a tray many times the push
    # length: along an OBLIQUE normal the box extent through a start near a
    # corner can be well under 2*length, because the binding constraint is
    # whichever axis the diagonal reaches first. Measured in the 270 mm tray
    # over 20k draws — fixed 40 mm: 2.5% nudged, 0% truncated; fixed 100 mm:
    # 15.2% nudged, 0% truncated; free length: 22.2% nudged, 0.4% truncated
    # with the mean push length preserved to 4 decimal places.
    d = direction * target
    lo_feasible = low + (-d).clamp(min=0.0)
    hi_feasible = high - d.clamp(min=0.0)

    # Genuinely infeasible only when the travel exceeds the box extent along
    # `direction` — i.e. no start point in the tray admits this push at all.
    # Unreachable for a sane fixed length; common for a free length drawn as
    # the distance between two points in the box, which can exceed the box's
    # extent along any single axis.
    infeasible = (lo_feasible > hi_feasible).any(dim=-1)
    hi_feasible = torch.maximum(hi_feasible, lo_feasible)

    new_starts = starts_xy.clamp(min=lo_feasible, max=hi_feasible)
    moved = ((new_starts - starts_xy).abs().amax(dim=-1) > 1e-9) & ~infeasible

    if infeasible.any():
        # Nothing else to do for these but travel as far as the tray allows.
        t_max = ray_box_max_travel(new_starts, direction, low, high)
        d = torch.where(infeasible.unsqueeze(-1), direction * t_max, d)

    return ConstrainedPush(new_starts, new_starts + d, moved, infeasible)


def duplicate_action_mask(starts_xy: torch.Tensor, headings: torch.Tensor,
                          pos_tol: float, angle_tol: float) -> torch.Tensor:
    """Flag every action that coincides with another in the same batch.

    Used by multi-step same-state slate collection (see
    ``Genesis/same_state_slate_collection.py``) to enforce that a slate's X
    candidate action sequences never repeat the same (start, heading) at a
    given step -- otherwise two "independent" candidates would in fact be one
    push simulated twice.

    ``headings`` is the actual travel direction (``atan2(dy, dx)`` of
    stop-start), not the blade yaw: a perpendicular push can travel along
    either ``+`` or ``-`` the blade normal for the same yaw, and those are
    different actions. Compared circularly over the full ``2*pi`` range since
    a travel direction has no restricted domain the way blade yaw does.

    Parameters
    ----------
    starts_xy  : (n, 2) push start points, metres.
    headings   : (n,) travel direction, radians, any range.
    pos_tol    : two starts closer than this (metres) count as the same point.
    angle_tol  : two headings closer than this (radians, circular) count as
                 the same direction.

    Returns
    -------
    (n,) bool -- True for every action that has at least one match elsewhere
    in the batch. Both members of a colliding pair are flagged (not just the
    later one), so a caller can resample both instead of arbitrarily keeping
    whichever came first.
    """
    n = starts_xy.shape[0]
    dpos = torch.cdist(starts_xy, starts_xy)
    dh = headings.unsqueeze(0) - headings.unsqueeze(1)
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    close = (dpos < pos_tol) & (dh.abs() < angle_tol)
    close.fill_diagonal_(False)
    return close.any(dim=1)


def relative_blade_angle(starts_xy: torch.Tensor, stops_xy: torch.Tensor,
                         angles: torch.Tensor) -> torch.Tensor:
    """Angle between the push direction and the blade face normal, in [0, π/2].

    Zero means a perpendicular push (the blade plows), π/2 means the push runs
    along the blade's own axis (it shears). The diagnostic that says whether
    restricting to perpendicular pushes gives anything up — see
    docs/linear_visual_foresight_baseline.md §7.5.

    Computed as atan2(|cross|, |dot|) rather than acos(|dot|): acos is
    ill-conditioned exactly where this is used most, near zero, where float32
    rounding in the dot product turns an exactly-perpendicular push into a
    5e-4 rad "error". atan2 is stable across the whole range.
    """
    delta = stops_xy - starts_xy
    direction = delta / (delta.norm(dim=-1, keepdim=True) + _EPS)
    n_hat = blade_normal(angles)
    dot = (direction * n_hat).sum(dim=-1)
    cross = direction[..., 0] * n_hat[..., 1] - direction[..., 1] * n_hat[..., 0]
    return torch.atan2(cross.abs(), dot.abs())


# ---------------------------------------------------------------------------
# Pile-aware action sampling: start at the pile, sweep through it
# ---------------------------------------------------------------------------
#
# Blind sampling draws the blade's start point uniformly over the tray. With a
# compact pile that wastes most of the budget: the blade spends its sweep
# crossing empty tray, and a large share of pushes barely touch the material.
# Measured on the 40 mm dataset (docs/linear_visual_foresight_baseline.md, and
# reports/linear_foresight_report.md 2.3): a typical push had only ~14% of the
# pile in its path, and the bottom half of pushes by contact produced so little
# change that no model beat "predict nothing moved" on them. Those transitions
# cost full simulation time and carry almost no signal.
#
# `pile_contact_starts` instead places the blade one particle-width from the
# pile's near face along the chosen push direction, laterally aligned so the
# swath actually contains material. Every simulated push then starts in contact
# and sweeps through the pile for its whole length.


def pile_contact_starts(particles_xy: torch.Tensor, headings: torch.Tensor,
                        blade_half_length: float, clearance: float,
                        min_swath: int = 3, jitter: float = 0.5,
                        max_tries: int = 8,
                        generator: torch.Generator | None = None):
    """Blade start points that touch the pile and sweep through it.

    For each entry, works in the frame of its own push direction: project every
    particle onto the push axis (``a``) and onto the lateral axis (``l``). Pick a
    lateral offset centred on a randomly chosen particle so the swath is
    guaranteed to contain at least that one, then set the along-axis start just
    behind the nearest particle *inside the swath* — so the blade begins in
    contact rather than driving through empty tray to reach the pile.

    Parameters
    ----------
    particles_xy : (..., N, 2) particle centres, metres. Parked/inactive
        particles must be excluded by the caller; they would drag the pile's
        apparent near face out to the parking area.
    headings : (...,) push direction, radians.
    blade_half_length : half the blade's length, metres — the swath half-width.
    clearance : how far behind the nearest particle to start, metres. One
        particle width is the intended value: close enough that no sweep
        distance is wasted, far enough that the blade is not initialised
        overlapping a particle (which the physics would resolve as a violent
        push-out).
    min_swath : keep re-drawing the lateral offset until at least this many
        particles fall inside the swath. The point of the exercise: a push that
        clips one corner of the pile carries little more information than one
        that misses it.
    jitter : lateral offset is the chosen particle's own lateral coordinate plus
        a uniform draw over +/- ``jitter * blade_half_length``. 0 centres the
        blade exactly on a particle every time (biased); 1 lets the particle sit
        anywhere across the blade face (uniform, but sometimes at the very edge).
    max_tries : lateral re-draws before giving up on ``min_swath``.

    Returns
    -------
    (starts_xy, n_in_swath, ok) — ``starts_xy`` is (..., 2); ``n_in_swath``
    counts particles in the accepted swath; ``ok`` is False where ``min_swath``
    was never reached (a caller should drop or resample those rather than
    simulate a push through nothing).
    """
    u = torch.stack([torch.cos(headings), torch.sin(headings)], dim=-1)
    nvec = torch.stack([-torch.sin(headings), torch.cos(headings)], dim=-1)

    a = (particles_xy * u.unsqueeze(-2)).sum(-1)          # (..., N) along push
    lat = (particles_xy * nvec.unsqueeze(-2)).sum(-1)     # (..., N) lateral

    shape = headings.shape
    N = particles_xy.shape[-2]
    dev = particles_xy.device
    best_c = torch.zeros(shape, device=dev, dtype=particles_xy.dtype)
    best_n = torch.zeros(shape, device=dev, dtype=torch.long)
    ok = torch.zeros(shape, device=dev, dtype=torch.bool)

    for _ in range(max_tries):
        pick = torch.randint(0, N, shape, generator=generator, device=dev)
        c = torch.gather(lat, -1, pick.unsqueeze(-1)).squeeze(-1)
        off = (torch.rand(shape, generator=generator, device=dev,
                          dtype=particles_xy.dtype) * 2.0 - 1.0)
        c = c + off * jitter * blade_half_length

        in_swath = (lat - c.unsqueeze(-1)).abs() <= blade_half_length
        n_in = in_swath.sum(-1)
        # Keep a draw if it is the best seen so far; accept outright once
        # min_swath is met, and stop re-drawing those entries.
        better = (n_in > best_n) & ~ok
        best_c = torch.where(better, c, best_c)
        best_n = torch.where(better, n_in, best_n)
        ok = ok | (best_n >= min_swath)
        if bool(ok.all()):
            break

    in_swath = (lat - best_c.unsqueeze(-1)).abs() <= blade_half_length
    # Nearest particle inside the swath, measured along the push direction.
    big = torch.finfo(a.dtype).max
    a_near = torch.where(in_swath, a, torch.full_like(a, big)).min(dim=-1).values
    # If the swath is empty (min_swath unreachable), fall back to the pile's
    # overall near face so the returned start is still finite and sensible.
    a_all = a.min(dim=-1).values
    a_near = torch.where(best_n > 0, a_near, a_all)

    a_start = a_near - clearance
    starts = a_start.unsqueeze(-1) * u + best_c.unsqueeze(-1) * nvec
    return starts, best_n, ok


# ---------------------------------------------------------------------------
# Touchdown legality (ISS-010 fix, 2026-09-28)
# ---------------------------------------------------------------------------
#
# `_pile_aware_stops` (Genesis/sandbox_manipulation_clean.py) used to clamp a
# collision-free `pile_contact_starts` draw into the sampling box with no cube
# check, and the clamped position routinely lands ON a cube -- measured 44-56%
# of rows in DS-0008/9/11/12/13 (docs: experiments/OPEN_ISSUES.md ISS-010).
# `pile_aware_action_batch` below is the Genesis-free replacement: it composes
# `pile_contact_starts` + the box clamp + a fixed/free push length exactly as
# `_pile_aware_stops` did, but tests the FINAL (post-clamp) touchdown footprint
# against every particle with an exact SAT test and REDRAWS (a fresh heading,
# never a shortened/lengthened push) any illegal slot, up to `max_redraws`
# times.


def _rect_axes(yaw: torch.Tensor) -> torch.Tensor:
    """(...,) radians -> (..., 2, 2): each box's own two unique edge normals."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([torch.stack([c, s], dim=-1),
                        torch.stack([-s, c], dim=-1)], dim=-2)


def overlaps_rect_pairs_torch(xy_a: torch.Tensor, yaw_a: torch.Tensor, half_a,
                              xy_b: torch.Tensor, yaw_b: torch.Tensor, half_b,
                              tol: float = 0.0) -> torch.Tensor:
    """Torch/batched/broadcastable equivalent of
    `Baselines.common.cube_overlap.overlaps_rect_pairs` (exact SAT test for
    yaw-rotated rectangles), kept Genesis-free so it is usable both inside a
    live sim step and in a plain unit test. Numerically identical to the numpy
    version (see `tests/test_action_sampling.py`); the duplication is
    deliberate -- the numpy one is used by offline audit/curation scripts
    (`experiments/EXP-0059-*/code/audit_tool_placement.py`) that have no torch
    device to keep tensors on, this one is used inside the sampler where
    everything is already a (possibly CUDA) torch tensor and converting to
    numpy every redraw iteration would force a device sync per attempt.

    xy_a, xy_b : (..., 2) box centres, any common broadcastable leading shape.
    yaw_a, yaw_b : (...,) radians.
    half_a, half_b : (..., 2) or broadcastable (e.g. (2,)) half-extents along
        each box's OWN local axes (axis 0 = (cos,sin) "length" direction,
        axis 1 = (-sin,cos) "width" direction).
    tol : > 0 shrinks both boxes (a contact gap < tol counts as separated);
        < 0 inflates both (the margin variant).
    Returns bool, broadcast shape of `yaw_a`/`yaw_b`.
    """
    xy_a, xy_b = torch.as_tensor(xy_a), torch.as_tensor(xy_b)
    half_a = torch.as_tensor(half_a, dtype=xy_a.dtype, device=xy_a.device)
    half_b = torch.as_tensor(half_b, dtype=xy_b.dtype, device=xy_b.device)
    ha, hb = half_a - 0.5 * tol, half_b - 0.5 * tol
    d = xy_b - xy_a
    A, B = _rect_axes(yaw_a), _rect_axes(yaw_b)
    sep = torch.zeros(torch.broadcast_shapes(yaw_a.shape, yaw_b.shape),
                      dtype=torch.bool, device=xy_a.device)
    for axes in (A, B):
        for k in range(2):
            n = axes[..., k, :]
            ra = (ha[..., 0] * (A[..., 0, :] * n).sum(-1).abs()
                 + ha[..., 1] * (A[..., 1, :] * n).sum(-1).abs())
            rb = (hb[..., 0] * (B[..., 0, :] * n).sum(-1).abs()
                 + hb[..., 1] * (B[..., 1, :] * n).sum(-1).abs())
            sep = sep | ((d * n).sum(-1).abs() > (ra + rb))
    return ~sep


def quat_yaw(quat: torch.Tensor) -> torch.Tensor:
    """(..., 4) quaternion in (w, x, y, z) order -> (...,) yaw, radians.

    `2*atan2(qz, qw)`, exact for a pure z-rotation. Duplicated (deliberately,
    to keep this module Genesis-free and dependency-free of `model/`) from
    `model/retrieval/frame.py::yaw_from_quat`, which carries the same
    docstring note on why this is exact enough for a real single-layer cube's
    small residual roll/pitch.
    """
    w, z = quat[..., 0], quat[..., 3]
    return 2.0 * torch.atan2(z, w)


def pile_aware_action_batch(particles_xy: torch.Tensor, particles_yaw: torch.Tensor,
                            cube_half_xy, headings: torch.Tensor,
                            blade_half_length: float, blade_half_width: float,
                            granular_vol, safety_margin: float, clearance: float,
                            push_length=None, push_length_lo: float | None = None,
                            start_gap_range: tuple[float, float] | None = None,
                            min_swath: int = 3, max_tries: int = 8,
                            max_redraws: int = 200,
                            generator: torch.Generator | None = None):
    """Legal-by-construction pile-aware touchdown + push, Genesis-free.

    Composes `pile_contact_starts` (collision-free touchdown before the box
    clamp) with the same box-clamp + push-length logic
    `sandbox_manipulation_clean.py::_pile_aware_stops` used, but checks the
    FINAL touchdown footprint against every particle (exact SAT,
    `overlaps_rect_pairs_torch`) and redraws a fresh heading for any illegal
    slot -- never shortens or lengthens the push, per the data-collection
    skill's rule on wall-shortened pushes.

    Parameters
    ----------
    particles_xy : (E, N, 2) this env's ACTIVE particle centres, metres.
    particles_yaw : (E, N) their yaws, radians (see `quat_yaw`).
    cube_half_xy : (2,) or (E, N, 2) each particle's half-extent, metres.
    headings : (E, S) initial push-direction draw (radians); only entries that
        turn out illegal are ever redrawn.
    blade_half_length, blade_half_width : blade footprint half-extents, metres.
    granular_vol, safety_margin : forwarded to `sampling_box`.
    clearance : blade-to-nearest-particle gap at touchdown, metres (one
        particle width is the intended value). Used AS GIVEN (fixed, the old
        behaviour) unless `start_gap_range` is set.
    push_length : None (drawn uniformly per slot) or a scalar/tensor fixed
        target distance, exactly as `_pile_aware_stops` accepted.
    push_length_lo : lower bound for the free-length draw; defaults to
        `clearance` (matching the old behaviour, which used the material's
        own particle size there).
    start_gap_range : None (default -- OLD behaviour: every touchdown sits
        exactly `clearance` metres behind the nearest swath particle's
        CENTRE, fixed) or a `(lo_margin, hi_margin)` pair, metres. When set,
        the start gap -- the coordinator's 2026-09-28 spec, measured ALONG
        THE PUSH AXIS from the blade's FRONT FACE to the NEAR FACE of the
        first cube its swath will contact -- is instead SAMPLED per action,
        uniformly, in `[lo_margin, L - hi_margin]` where `L` is the (scalar)
        `push_length` -- e.g. `(0.005, 0.005)` with `L=0.02` gives a 5-15 mm
        gap and that cube travels 5-15 mm. Applied at the SAME point the old
        fixed distance was (`pile_contact_starts`'s own `clearance`
        argument), not as a post-hoc shift: since that argument is CENTRE-
        based (`a_start = a_near[particle centre] - clearance`), the sampled
        face-to-face gap is converted once, uniformly, into the centre-based
        value `pile_contact_starts` expects (`+ blade_half_width + cube_half`,
        both along the push axis) -- see `draw_gap`. Requires a UNIFORM
        particle size (raises `NotImplementedError` otherwise: converting
        face-gap to centre-clearance needs the contacted cube's half-extent
        before it is known WHICH cube that is) and a SCALAR `push_length` (`L`
        must be known before `pile_contact_starts` runs, which a free-length
        draw -- depends on the post-clamp box -- or a per-env tensor length
        cannot provide; raises `NotImplementedError`). The legality
        check+redraw below still runs after it, and every redraw attempt
        draws a FRESH gap along with the fresh heading; a redraw ALSO fires
        whenever the box clamp moves the touchdown at all (see `full_draw`),
        because the clamp acts in world (x, y) and any movement there
        invalidates the along-push-axis gap the draw was built with, whether
        or not the moved point happens to still be legal -- found via a real
        smoke test after the first version of this fix only redrew on
        illegality (gap p95 ~34mm against a requested 5-15mm window; unit
        test: `test_a_clamped_draw_is_redrawn_not_silently_kept_with_a_corrupted_gap`).
    max_redraws : redraw attempts per illegal-or-gap-out-of-window slot before
        giving up and accepting the best found as a last resort (default 200:
        action sampling is cheap next to the physics step it precedes, so a
        generous budget costs little total collection time -- see the
        docstring's ETA note in `_pile_aware_action_legal`). Illegal touchdowns
        are reported via the returned `n_illegal`, never silently dropped; a
        slot that never finds an in-window candidate is reported via the
        returned `gap_out_of_window` mask instead of silently accepted.

    Returns
    -------
    action_starts_xy, action_stops_xy : (E, S, 2)
    angles : (E, S) blade yaw (perpendicular-push convention)
    ok : (E, S) bool, `pile_contact_starts`' own `min_swath` flag (last draw)
    n_illegal_remaining : int, touchdowns still illegal after `max_redraws`
    n_redraws_used : int, redraw rounds actually run (<= max_redraws)
    gap_out_of_window : (E, S) bool, True where `start_gap_range` was set but
        no legal AND in-window candidate was found within `max_redraws` --
        the accepted touchdown is legal but its along-push-axis gap is not
        guaranteed to be in the requested window. All-False when
        `start_gap_range` is None.
    """
    E, S = headings.shape
    N = particles_xy.shape[1]
    dev, dtype = particles_xy.device, particles_xy.dtype
    half_blade = torch.tensor([blade_half_length, blade_half_width],
                              device=dev, dtype=dtype)
    cube_half = torch.as_tensor(cube_half_xy, device=dev, dtype=dtype)
    if cube_half.ndim == 1:
        cube_half = cube_half.view(1, 1, 2).expand(E, N, 2)
    lo_len = clearance if push_length_lo is None else push_length_lo

    if start_gap_range is not None:
        if push_length is None or torch.is_tensor(push_length):
            raise NotImplementedError(
                "start_gap_range requires a SCALAR push_length: L must be "
                "known before pile_contact_starts runs, which a free-length "
                "draw or a per-env tensor length cannot provide. Not "
                "implemented for that case -- see this function's docstring.")
        gap_lo, gap_hi_margin = start_gap_range
        gap_L = float(push_length)
        if gap_L - gap_hi_margin <= gap_lo:
            raise ValueError(
                f"start_gap_range={start_gap_range} leaves no room in a push "
                f"of length {gap_L} m (need L > lo_margin + hi_margin)")
        # `pile_contact_starts`'s own `clearance` is CENTRE-based (a_start =
        # a_near[particle CENTRE] - clearance), but the requested window is
        # FACE-based (blade FRONT FACE to the first-contact cube's NEAR FACE,
        # the coordinator's 2026-09-28 spec). Converting once here, uniformly,
        # keeps the sampled window's semantics correct without a post-hoc
        # shift elsewhere: centre_clearance = face_gap + blade_half_width (the
        # blade's own half-thickness along the push axis) + cube_half (the
        # particle's own half-extent along the push axis). Requires uniform
        # particle size -- true for every dataset this task touches; asserted
        # rather than silently averaged over a mixed-size pile.
        cube_half_ax = cube_half[..., 0].reshape(-1)
        if not torch.allclose(cube_half_ax, cube_half_ax[:1].expand_as(cube_half_ax),
                              atol=1e-6):
            raise NotImplementedError(
                "start_gap_range assumes a UNIFORM particle size (needed to "
                "convert the requested face-to-face gap into pile_contact_starts' "
                "centre-based clearance without knowing which particle will be "
                "picked); this pile has mixed sizes. Not implemented for that case.")
        gap_cube_half = float(cube_half_ax[0])

    def draw_gap():
        if start_gap_range is None:
            return clearance
        gap_lo, gap_hi_margin = start_gap_range
        gap_L = float(push_length)
        span = gap_L - gap_hi_margin - gap_lo
        face_gap = gap_lo + torch.rand((E, S), generator=generator, device=dev,
                                       dtype=dtype) * span
        return face_gap + blade_half_width + gap_cube_half

    pxy = particles_xy.unsqueeze(1).expand(-1, S, -1, -1)     # (E,S,N,2)
    pyaw = particles_yaw.unsqueeze(1).expand(-1, S, -1)        # (E,S,N)
    chalf = cube_half.unsqueeze(1).expand(-1, S, -1, -1)       # (E,S,N,2)

    def illegal_mask(starts_xy, angles):
        bxy = starts_xy.unsqueeze(2).expand(-1, -1, N, -1)
        byaw = angles.unsqueeze(-1).expand(-1, -1, N)
        ov = overlaps_rect_pairs_torch(bxy, byaw, half_blade, pxy, pyaw, chalf)
        return ov.any(dim=-1)

    def full_draw(hdg):
        starts_xy, n_in, ok = pile_contact_starts(
            pxy, hdg, blade_half_length=blade_half_length, clearance=draw_gap(),
            min_swath=min_swath, max_tries=max_tries, generator=generator)
        angles = hdg + torch.pi / 2
        angles = torch.remainder(angles + torch.pi / 2, torch.pi) - torch.pi / 2
        u_dir = torch.stack([torch.cos(hdg), torch.sin(hdg)], dim=-1)
        low, high = sampling_box(angles, granular_vol, 2 * blade_half_length,
                                 2 * blade_half_width, safety_margin)
        # The clamp moves the touchdown in WORLD (x, y), which is not the same
        # axes `pile_contact_starts` placed it in (push/lateral) -- so it can
        # break the along-push-axis gap the clamped draw was built with,
        # whether or not the moved point happens to land ON a cube (the
        # original ISS-010 mechanism). Rather than trust a coarse "was this
        # clamped" proxy (an earlier version of this fix did, and it let a
        # clamped-but-still-legal draw silently keep an arbitrary gap), the
        # scoring below RECOMPUTES the actual realized gap after the clamp
        # (`_gap_window_ok`) and redraws on THAT, directly.
        starts_xy = starts_xy.clamp(min=low, max=high)
        t_max = ray_box_max_travel(starts_xy, u_dir, low, high)
        if push_length is None:
            span = (t_max - lo_len).clamp_min(0.0)
            L = lo_len + torch.rand_like(span) * span
        else:
            if torch.is_tensor(push_length):
                L = push_length.to(t_max).reshape(t_max.shape).expand_as(t_max).clone()
            else:
                L = torch.full_like(t_max, float(push_length))
            L = torch.minimum(L, t_max)
        stops_xy = starts_xy + u_dir * L
        return starts_xy, stops_xy, angles, ok, u_dir

    def draw_headings():
        return torch.rand((E, S), generator=generator, device=dev,
                          dtype=dtype) * (2 * torch.pi)

    def _gap_bad(starts_xy, u_dir):
        """True where, measured EXACTLY the way the coordinator's spec and the
        offline audit both do (along-push-axis, blade FRONT FACE to the NEAR
        FACE of the first cube ahead in the swept swath), this touchdown has
        either no contact cube ahead at all, or one outside the requested
        `[lo_margin, L - hi_margin]` window -- regardless of whether the box
        clamp actually moved the point. Only meaningful when `start_gap_range`
        is set; returns all-False otherwise (old behaviour, unchanged)."""
        if start_gap_range is None:
            return torch.zeros(starts_xy.shape[:-1], dtype=torch.bool, device=dev)
        gap_lo, gap_hi_margin = start_gap_range
        gap_L = float(push_length)
        nvec = torch.stack([-u_dir[..., 1], u_dir[..., 0]], dim=-1)
        rel = pxy - starts_xy.unsqueeze(2)                     # (E,S,N,2)
        a = (rel * u_dir.unsqueeze(2)).sum(-1)
        lat = (rel * nvec.unsqueeze(2)).sum(-1)
        in_swath = lat.abs() <= blade_half_length
        near_face_a = a - gap_cube_half
        ahead = near_face_a > blade_half_width
        cand = in_swath & ahead
        has_contact = cand.any(dim=-1)
        gap_vals = torch.where(cand, near_face_a - blade_half_width,
                               torch.full_like(near_face_a, float("inf")))
        min_gap = gap_vals.min(dim=-1).values
        in_window = (min_gap >= gap_lo - 1e-9) & (min_gap <= (gap_L - gap_hi_margin) + 1e-9)
        return ~(has_contact & in_window)

    def _score(illegal, gap_bad):
        """0 = legal & gap in window (or start_gap_range unset) -- best.
        1 = legal but gap out of window / no contact ahead -- accept only if
            nothing better is found within `max_redraws` (last resort; flagged
            via the returned `gap_out_of_window`).
        2 = illegal -- must never be preferred over either, even to fix (1);
            this ordering is what keeps the clamp/gap-precision redraw from
            ever regressing the hard legality guarantee (an earlier version
            without this ordering let illegal touchdowns rise to 2-5/32 in a
            real smoke run when both conditions shared one undifferentiated
            redraw budget)."""
        return torch.where(illegal, torch.full_like(illegal, 2, dtype=torch.long),
                           torch.where(gap_bad, torch.ones_like(illegal, dtype=torch.long),
                                      torch.zeros_like(illegal, dtype=torch.long)))

    starts_xy, stops_xy, angles, ok, u_dir = full_draw(headings)
    best_score = _score(illegal_mask(starts_xy, angles), _gap_bad(starts_xy, u_dir))

    n_redraws = 0
    while bool((best_score > 0).any()) and n_redraws < max_redraws:
        n_redraws += 1
        new_hdg = draw_headings()
        new_starts, new_stops, new_angles, new_ok, new_u_dir = full_draw(new_hdg)
        new_score = _score(illegal_mask(new_starts, new_angles), _gap_bad(new_starts, new_u_dir))
        better = new_score < best_score
        m2 = better.unsqueeze(-1)
        starts_xy = torch.where(m2, new_starts, starts_xy)
        stops_xy = torch.where(m2, new_stops, stops_xy)
        angles = torch.where(better, new_angles, angles)
        ok = torch.where(better, new_ok, ok)
        best_score = torch.where(better, new_score, best_score)

    n_illegal = int((best_score == 2).sum())
    gap_out_of_window = best_score == 1
    return starts_xy, stops_xy, angles, ok, n_illegal, n_redraws, gap_out_of_window


def legalize_pushes(acts: torch.Tensor, cube_xy: torch.Tensor, cube_yaw: torch.Tensor, cube_half,
                    blade_half_length: float, blade_half_width: float,
                    box: tuple[float, float] | None = None,
                    step: float = 0.0005, max_back: float = 0.04, max_side: float = 0.015):
    """Make planned pushes touchdown-legal (ISS-010 / ISS-013, user rule 2026-10-03: in Genesis a
    push whose blade lands on a cube is not part of the task).

    acts (K, 4) [sx, sy, ex, ey]; cube_xy (K, n, 2); cube_yaw (K, n); cube_half scalar or (2,).
    The blade is a (2*blade_half_length x 2*blade_half_width) rectangle centred on the start, long
    axis perpendicular to the push. An illegal push is translated (start and end together, length and
    heading kept) by the SMALLEST offset that makes it legal (exact SAT,
    `Baselines/common/cube_overlap.overlaps_rect_pairs`), searching backwards along the push
    (0..max_back, `step` grid) and sideways along the blade (|lateral| <= max_side, 4*step grid);
    if `box` (lo, hi) is given the shifted start AND end must stay inside it. Returns
    (acts', shift_m (K,), ok (K,)): shift 0 = already legal; ok False = no legal offset found (push
    returned unchanged).
    """
    from Baselines.common.cube_overlap import overlaps_rect_pairs
    import numpy as np
    acts = acts.detach().float().cpu().clone()
    K, n = cube_xy.shape[:2]
    cxy = cube_xy.detach().float().cpu().numpy(); cyaw = cube_yaw.detach().float().cpu().numpy()
    half_c = np.broadcast_to(np.asarray(cube_half, dtype=np.float64).reshape(-1), (2,)).copy()
    half_b = np.array([blade_half_length, blade_half_width])
    backs = np.arange(0.0, max_back + 1e-12, step)
    pos = np.arange(4 * step, max_side + 1e-12, 4 * step)
    sides = np.concatenate([[0.0], pos, -pos])                            # includes 0 exactly
    B, S = np.meshgrid(backs, sides, indexing="ij")
    B, S = B.ravel(), S.ravel()
    order = np.argsort(np.hypot(B, S), kind="stable")
    B, S = B[order], S[order]
    shift = torch.zeros(K); ok = torch.ones(K, dtype=torch.bool)
    for k in range(K):
        s, e = acts[k, :2].numpy().astype(np.float64), acts[k, 2:].numpy().astype(np.float64)
        d = e - s
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        u = d / L; v = np.array([-u[1], u[0]])
        yaw = math.atan2(u[1], u[0]) + math.pi / 2
        P = s[None] - B[:, None] * u[None] + S[:, None] * v[None]          # candidate starts (M, 2)
        keep = np.ones(len(P), dtype=bool)
        if box is not None:
            for Q in (P, P + d[None]):
                keep &= (Q >= box[0]).all(1) & (Q <= box[1]).all(1)
        idx = np.flatnonzero(keep)
        found = False
        for c0 in range(0, len(idx), 256):                                # nearest offsets first
            ci = idx[c0:c0 + 256]
            m = len(ci)
            hit = overlaps_rect_pairs(np.repeat(P[ci], n, 0), np.full(m * n, yaw), half_b,
                                      np.tile(cxy[k], (m, 1)), np.tile(cyaw[k], m), half_c).reshape(m, n).any(1)
            free = np.flatnonzero(~hit)
            if len(free):
                j = ci[free[0]]
                if j != 0 or B[j] > 0 or S[j] != 0:
                    acts[k, :2] = torch.as_tensor(P[j], dtype=acts.dtype)
                    acts[k, 2:] = torch.as_tensor(P[j] + d, dtype=acts.dtype)
                shift[k] = float(np.hypot(B[j], S[j]))
                found = True
                break
        ok[k] = found
    return acts, shift, ok
