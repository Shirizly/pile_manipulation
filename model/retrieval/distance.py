"""Configurable push-frame chamfer distance + GPU-batched exhaustive bank
search (EXP-0059, R1 follow-up).

The v0 `NearestTransitionPredictor` (see `predictor.py`'s history) used a
plain (uncapped, unweighted, mean) symmetric chamfer distance and scored
`accuracy_1 = 0.172` against the designer's own crude probe of `0.295`
(capped Chamfer + a mismatch penalty for unmatched cubes, greedy match, sum
not mean). This module makes every one of those knobs explicit and
independently switchable, so the discrepancy can be attributed rather than
guessed at, and adds a GPU-batched exhaustive top-k search so a config sweep
stays fast against the ~12k-transition bank.

Distance (per query row q, bank row j), all in q's/j's OWN push frame:

    term_1 = reduce_{c in Q}  w(c) * cost(min_{c' in B_j} ||c - c'||)
    term_2 = reduce_{c' in B_j} w(c') * cost(min_{c in Q} ||c' - c||)
    d(Q, B_j) = term_1 + term_2

  * `Q`/`B_j` = query/bank-row-j cubes with push-frame (u, v) inside
    `window_u`/`window_v` (the broad "is this cube near the swath at all"
    set -- design doc Section 3's Level 2 "context/buffer").
  * `w(c)` = 1.0, or `corridor_weight` (>1) if `c` additionally lies in the
    tighter swept CORRIDOR the plate itself passes through (Level 0 "direct
    interaction": `u` in `[0, push_len + corridor_u_pad]`,
    `|v| <= corridor_v_halfwidth`) -- upweights cubes the tool actually
    touches over ones merely nearby.
  * `cost(d) = min(d, cap)` if `cap` is set (a CAPPED distance: beyond `cap`,
    further distance stops making two configurations look more different --
    otherwise one wildly-mismatched cube can dominate the whole sum/mean),
    replaced by a flat `mismatch_penalty` instead of the capped value when
    `d > cap` and `mismatch_penalty` is given separately from `cap` (a
    distinct constant cost for "no plausible match", vs. "the closest match
    is merely far").
  * `reduce` is `mean` (weighted by `w`) or `sum`.
  * an optional `wall_weight > 0` appends `wall_weight * wall_distance(c)`
    as a third coordinate before the cdist, so two cubes at the same (u, v)
    but different distance-to-the-nearest-tray-wall (computed by mapping
    back through `frame.push_frame_to_world` -- physically identical to
    comparing against `frame.tray_corners_push_frame`'s geometry, see that
    function's docstring) read as less similar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .bank import TransitionBank, OCC_BOUNDS
from .frame import push_frame_to_world

# Blade half-width (plate.size[0]/2, Genesis/configs/basic.yaml) + one cube
# half-footprint (simple_mpc.adapters.OCC_CUBE_SIZE=0.005) of slack.
BLADE_HALF_WIDTH = 0.02
CUBE_SIZE = 0.005

DEFAULT_WINDOW_U: Tuple[float, float] = (-0.005, 0.045)
DEFAULT_WINDOW_V: Tuple[float, float] = (-0.050, 0.050)
DEFAULT_CORRIDOR_V_HALFWIDTH = BLADE_HALF_WIDTH + CUBE_SIZE   # 0.025
DEFAULT_CORRIDOR_U_PAD = CUBE_SIZE                            # 0.005


@dataclass
class DistanceConfig:
    window_u: Tuple[float, float] = DEFAULT_WINDOW_U
    window_v: Tuple[float, float] = DEFAULT_WINDOW_V
    cap: Optional[float] = None                 # metres; None = uncapped chamfer
    mismatch_penalty: Optional[float] = None    # None -> falls back to `cap` (plain capped chamfer)
    corridor_v_halfwidth: float = DEFAULT_CORRIDOR_V_HALFWIDTH
    corridor_u_pad: float = DEFAULT_CORRIDOR_U_PAD
    corridor_weight: float = 1.0                # 1.0 disables corridor upweighting
    wall_weight: float = 0.0                     # 0.0 disables the wall-distance feature
    reduction: str = "mean"                      # "mean" | "sum"

    def __post_init__(self):
        assert self.reduction in ("mean", "sum")
        if self.mismatch_penalty is not None:
            assert self.cap is not None, "mismatch_penalty needs `cap` set (it is the cost for d > cap)"


def window_mask(uv: torch.Tensor, window_u, window_v) -> torch.Tensor:
    """uv: (..., n, 2) -> (..., n) bool."""
    u, v = uv[..., 0], uv[..., 1]
    return (u >= window_u[0]) & (u <= window_u[1]) & (v >= window_v[0]) & (v <= window_v[1])


def corridor_weight_mask(uv: torch.Tensor, push_len: torch.Tensor, cfg: DistanceConfig) -> torch.Tensor:
    """uv: (..., n, 2), push_len: (...,) broadcastable -> (..., n) float weights,
    1.0 outside the corridor, `cfg.corridor_weight` inside it. The corridor's
    u-extent scales with THIS row's own push length (a longer push sweeps a
    longer corridor), matching how `bank.push_len` already varies per row."""
    u, v = uv[..., 0], uv[..., 1]
    u_hi = (push_len + cfg.corridor_u_pad).unsqueeze(-1) if push_len.dim() == u.dim() - 1 else push_len + cfg.corridor_u_pad
    in_corridor = (u >= 0.0) & (u <= u_hi) & (v.abs() <= cfg.corridor_v_halfwidth)
    return torch.where(in_corridor, torch.full_like(u, cfg.corridor_weight), torch.ones_like(u))


def wall_distance(uv: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor,
                  bounds: dict = OCC_BOUNDS) -> torch.Tensor:
    """uv: (B, n, 2) push-frame coords, p_start/p_stop: (B, >=2) -> (B, n)
    metres to the nearest AXIS-ALIGNED tray wall. Computed by mapping back
    to world (`push_frame_to_world`) and taking the min of the 4 axis-
    aligned slab distances -- physically the same quantity as measuring
    against `frame.tray_corners_push_frame`'s (rotated, in push-frame)
    rectangle, but far simpler since the tray IS axis-aligned in world."""
    xy = push_frame_to_world(uv, p_start, p_stop)
    dx0 = xy[..., 0] - bounds["x_min"]; dx1 = bounds["x_max"] - xy[..., 0]
    dy0 = xy[..., 1] - bounds["y_min"]; dy1 = bounds["y_max"] - xy[..., 1]
    return torch.minimum(torch.minimum(dx0, dx1), torch.minimum(dy0, dy1)).clamp_min(0.0)


def _augment(uv: torch.Tensor, wall_dist: Optional[torch.Tensor], wall_weight: float) -> torch.Tensor:
    if wall_weight and wall_weight > 0.0 and wall_dist is not None:
        return torch.cat([uv, (wall_weight * wall_dist).unsqueeze(-1)], dim=-1)
    return uv


def _cost(d: torch.Tensor, cfg: DistanceConfig) -> torch.Tensor:
    if cfg.cap is None:
        return d
    mismatch = cfg.mismatch_penalty if cfg.mismatch_penalty is not None else cfg.cap
    return torch.where(d > cfg.cap, torch.full_like(d, mismatch), d.clamp(max=cfg.cap))


def chamfer_distance_chunk(query_pts: torch.Tensor, query_w: torch.Tensor,
                           bank_pts: torch.Tensor, bank_w: torch.Tensor,
                           cfg: DistanceConfig) -> torch.Tensor:
    """query_pts: (Bq, n, D), query_w: (Bq, n) combined mask*corridor-weight
    (0 outside the window). bank_pts: (Tc, n, D), bank_w: (Tc, n) likewise.
    -> (Bq, Tc) symmetric chamfer distance."""
    Bq, n, D = query_pts.shape
    Tc = bank_pts.shape[0]
    BIG = 1e6
    d = torch.cdist(query_pts.reshape(Bq, 1, n, D).expand(Bq, Tc, n, D).reshape(Bq * Tc, n, D),
                    bank_pts.reshape(1, Tc, n, D).expand(Bq, Tc, n, D).reshape(Bq * Tc, n, D)
                    ).reshape(Bq, Tc, n, n)
    cost = _cost(d, cfg)

    qmask = query_w > 0
    bmask = bank_w > 0
    # term_1: for each query cube, nearest (in-window) bank cube
    t1_raw = cost.masked_fill(~bmask.view(1, Tc, 1, n), BIG).min(dim=3).values  # (Bq,Tc,n) over query axis
    wq = query_w.view(Bq, 1, n).expand(Bq, Tc, n)
    if cfg.reduction == "mean":
        t1 = (t1_raw * wq).sum(dim=2) / wq.sum(dim=2).clamp_min(1e-9)
    else:
        t1 = (t1_raw * wq).sum(dim=2)
    # term_2: for each bank cube, nearest (in-window) query cube
    t2_raw = cost.masked_fill(~qmask.view(Bq, 1, n, 1), BIG).min(dim=2).values  # (Bq,Tc,n) over bank axis
    wb = bank_w.view(1, Tc, n).expand(Bq, Tc, n)
    if cfg.reduction == "mean":
        t2 = (t2_raw * wb).sum(dim=2) / wb.sum(dim=2).clamp_min(1e-9)
    else:
        t2 = (t2_raw * wb).sum(dim=2)
    return t1 + t2


def query_points_and_weights(states0: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor,
                             cfg: DistanceConfig):
    """states0: (B, n, >=2) world cube xy (or full 7-dim state). -> (pts (B,n,D),
    w (B,n), uv (B,n,2)) for use as the QUERY side of `chamfer_distance_chunk`
    / `topk_search`."""
    from .frame import world_to_push_frame
    uv = world_to_push_frame(states0[..., :2], p_start, p_stop)
    push_len = (p_stop[:, :2] - p_start[:, :2]).norm(dim=-1)
    wmask = window_mask(uv, cfg.window_u, cfg.window_v).float()
    cweight = corridor_weight_mask(uv, push_len, cfg)
    w = wmask * cweight
    wall = wall_distance(uv, p_start, p_stop) if cfg.wall_weight > 0 else None
    pts = _augment(uv, wall, cfg.wall_weight)
    valid = wmask.sum(dim=1) > 0
    return pts, w, uv, valid


def bank_points_and_weights(bank: TransitionBank, cfg: DistanceConfig):
    """Precompute the BANK side once per config (cheap: T ~ 12k rows).

    Also returns `valid` (T,) bool: False for a bank row with NO cube inside
    its own window (nothing to match against) -- `topk_search` must exclude
    these explicitly, because a weighted mean/sum over an empty weight set
    would otherwise silently read as a *perfect* (distance 0) match."""
    wmask = window_mask(bank.uv0, cfg.window_u, cfg.window_v).float()
    cweight = corridor_weight_mask(bank.uv0, bank.push_len, cfg)
    w = wmask * cweight
    wall = wall_distance(bank.uv0, bank.p_starts, bank.p_stops) if cfg.wall_weight > 0 else None
    pts = _augment(bank.uv0, wall, cfg.wall_weight)
    valid = wmask.sum(dim=1) > 0
    return pts, w, valid


def full_distance_matrix(query_pts: torch.Tensor, query_w: torch.Tensor,
                         bank_pts: torch.Tensor, bank_w: torch.Tensor,
                         cfg: DistanceConfig, chunk: int = 1500, device: str = None,
                         bank_valid: torch.Tensor = None) -> torch.Tensor:
    """Like `topk_search` but returns the FULL (Bq, T) distance matrix rather
    than a top-k reduction -- for diagnostics that need to rank or examine
    EVERY bank candidate (e.g. an oracle-donor search over "the whole
    bank"), not just the metric's own top-k. Same bank-chunking, same
    `bank_valid` handling (invalid rows get +inf, never a spurious 0)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    query_pts, query_w = query_pts.to(device), query_w.to(device)
    T = bank_pts.shape[0]
    INF = float(1e9)
    out = torch.empty(query_pts.shape[0], T, device=device)
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        bp = bank_pts[lo:hi].to(device)
        bw = bank_w[lo:hi].to(device)
        d_chunk = chamfer_distance_chunk(query_pts, query_w, bp, bw, cfg)
        if bank_valid is not None:
            bv = bank_valid[lo:hi].to(device)
            d_chunk = d_chunk.masked_fill(~bv.view(1, -1), INF)
        out[:, lo:hi] = d_chunk
    return out.cpu()


def topk_search(query_pts: torch.Tensor, query_w: torch.Tensor,
                bank_pts: torch.Tensor, bank_w: torch.Tensor,
                cfg: DistanceConfig, k: int = 1, chunk: int = 1500,
                device: str = None, bank_valid: torch.Tensor = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exhaustive GPU-batched top-k search over the whole bank.

    query_pts/query_w: (Bq, n, D)/(Bq, n) (from `query_points_and_weights`).
    bank_pts/bank_w: (T, n, D)/(T, n) (from `bank_points_and_weights`).
    bank_valid: (T,) bool, bank rows with nothing in their own window (see
    `bank_points_and_weights`) -- excluded from the search entirely (given
    +inf distance) rather than allowed to win on a spurious 0/0 "distance".
    -> (idx (Bq, k) long, dist (Bq, k) float), sorted ascending by distance.

    Chunks over the BANK dimension (T) so a (Bq, Tc, n, n) intermediate
    tensor stays bounded regardless of the bank's total size; the whole
    query batch is processed together (no per-sample Python loop) so this
    is the "exhaustive but batched/fast" search the sweep needs.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    query_pts, query_w = query_pts.to(device), query_w.to(device)
    T = bank_pts.shape[0]
    INF = float(1e9)
    best_val = None
    best_idx = None
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        bp = bank_pts[lo:hi].to(device)
        bw = bank_w[lo:hi].to(device)
        d_chunk = chamfer_distance_chunk(query_pts, query_w, bp, bw, cfg)  # (Bq, hi-lo)
        if bank_valid is not None:
            bv = bank_valid[lo:hi].to(device)
            d_chunk = d_chunk.masked_fill(~bv.view(1, -1), INF)
        idx_chunk = torch.arange(lo, hi, device=device).view(1, -1).expand(d_chunk.shape[0], -1)
        if best_val is None:
            cat_val, cat_idx = d_chunk, idx_chunk
        else:
            cat_val = torch.cat([best_val, d_chunk], dim=1)
            cat_idx = torch.cat([best_idx, idx_chunk], dim=1)
        kk = min(k, cat_val.shape[1])
        best_val, top_pos = torch.topk(cat_val, kk, dim=1, largest=False, sorted=True)
        best_idx = torch.gather(cat_idx, 1, top_pos)
    return best_idx.cpu(), best_val.cpu()
