"""Geometric, TRUTH-FREE "interaction set" (EXP-0059 coordinator follow-up B,
2026-09-28; design doc `docs/experimental_design/retrieval_based_modeling.md`
section 6.1 "Affected set" -- this module implements ONLY that first step,
not the full section-6 compositional-retrieval machinery: groups, halos,
mirror augmentation and non-overlap assembly are all out of scope here).

Why this exists: the retrieval algorithm must never see uninvolved parts of
the state -- for both nearest-neighbour search AND delta transfer, the only
state of a transition that matters is the few cubes that interact with it.
This module computes that reduced set, ONCE per transition, from geometry
alone (pre-push cube positions, in the transition's OWN push frame, plus the
push length) -- no ground truth, so a QUERY (including a rollout step, which
by construction has no future truth to consult) can always compute it, and a
BANK transition's set is computed identically at build time.

Definition (design doc 6.1, `m_v`/`tau` defaults per that doc):

  direct   : cubes with |v| < blade_half_width + h + m_v  AND
             u in [-h, push_len + h]
             (h = cube_size / 2; "the blade sweeps them, +/- half a cube")
  chain    : repeatedly add any cube c whose centre is within
             d_c = cube_size + tau of a cube a already in the set, AHEAD of
             it ((c - a).u > 0) and within `angle_max_deg` of the push
             direction -- stopping once the CUMULATIVE forward gap (shortest
             path length through the chain, not straight-line distance from
             the direct set) exceeds `push_len`.

`tau` (contact-chain slack) is tuned on TRAINING data only (`tune_tau.py`
below / this module's `__main__`) against the bank's own truth "moved" set
(`TransitionBank.moved`, displacement > 1 mm) -- recall/precision of the
geometric rule against ground truth, never used at query time (queries have
no truth to check against; the frozen `tau` is baked into `DEFAULT_TAU`).
"""
from __future__ import annotations

import math
from typing import Optional

import torch

CUBE_SIZE = 0.005            # metres, matches model.retrieval.distance.CUBE_SIZE
BLADE_HALF_WIDTH = 0.02      # metres, matches model.retrieval.distance.BLADE_HALF_WIDTH
DEFAULT_M_V = 0.001          # metres, "swept" lateral slack (design doc 6.1)
# Contact-chain slack, TUNED (not the design doc's own tau in {1,2,3}mm default range --
# 2026-09-28, coordinator follow-up B): those values only reach 82-90% recall of the bank's
# truth `moved` set on LEGAL (ISS-010-clean) DS-0008+DS-0010 training rows. Swept 1-20mm x
# angle_max_deg in {60,90}: tau=12mm, angle_max_deg=60 is the smallest slack reaching the
# requested ~95% recall (measured 0.960 recall / 0.899 precision on legal-only training rows,
# pooled over cubes) -- `experiments/EXP-0059-*/code/tune_interaction_set.py`,
# `experiments/EXP-0059-*/results/interaction_set_tuning.json`.
DEFAULT_TAU = 0.012          # metres
DEFAULT_ANGLE_MAX_DEG = 60.0


def interaction_set(uv: torch.Tensor, push_len, cube_size: float = CUBE_SIZE,
                    blade_half_width: float = BLADE_HALF_WIDTH,
                    m_v: float = DEFAULT_M_V, tau: float = DEFAULT_TAU,
                    angle_max_deg: float = DEFAULT_ANGLE_MAX_DEG,
                    max_iters: Optional[int] = None) -> torch.Tensor:
    """uv: (B, n, 2) OR (n, 2) push-frame cube centres (`frame.world_to_push_frame`,
    each transition in its OWN frame). push_len: (B,) or scalar, metres.
    -> (B, n) or (n,) bool mask, True for cubes in the geometric interaction
    set. Deterministic, truth-free, batched (no ground truth or Python
    per-row loop -- cheap enough to call on a ~100k-row bank at once).
    """
    squeeze = uv.dim() == 2
    if squeeze:
        uv = uv.unsqueeze(0)
    B, n, _ = uv.shape
    if not torch.is_tensor(push_len):
        push_len = torch.full((B,), float(push_len), dtype=uv.dtype, device=uv.device)
    push_len = push_len.reshape(B).to(uv.dtype)

    h = cube_size / 2.0
    u, v = uv[..., 0], uv[..., 1]
    L1 = push_len.unsqueeze(-1)
    direct = (v.abs() < blade_half_width + h + m_v) & (u >= -h) & (u <= L1 + h)   # (B, n)

    d_c = cube_size + tau
    cos_max = math.cos(math.radians(angle_max_deg))
    # delta[b, a, c] = uv[b, c] - uv[b, a]
    delta = uv.unsqueeze(1) - uv.unsqueeze(2)                 # (B, n, n, 2)
    dist = delta.norm(dim=-1)                                 # (B, n, n)
    cosang = delta[..., 0] / dist.clamp_min(1e-9)
    edge_ok = (dist <= d_c) & (cosang > cos_max)              # a -> c allowed (c ahead of a)

    inf = torch.tensor(float("inf"), dtype=uv.dtype, device=uv.device)
    reach = torch.where(direct, torch.zeros_like(u), inf.expand_as(u)).clone()
    L3 = push_len.reshape(B, 1, 1)                            # (B, 1, 1), broadcasts vs (B,n,n)
    iters = max_iters if max_iters is not None else n
    for _ in range(iters):
        cand = reach.unsqueeze(2) + dist                      # (B, n, n): via a -> c
        cand = torch.where(edge_ok, cand, inf)
        cand = torch.where(cand <= L3, cand, inf)
        best_new = cand.min(dim=1).values                     # (B, n)
        new_reach = torch.minimum(reach, best_new)
        if torch.equal(new_reach, reach):
            reach = new_reach
            break
        reach = new_reach

    in_set = torch.isfinite(reach)
    return in_set[0] if squeeze else in_set


def recall_precision(geometric: torch.Tensor, moved: torch.Tensor):
    """geometric, moved: (B, n) bool -> dict(recall, precision, n_moved, n_geometric).
    recall = P(geometric | moved); precision = P(moved | geometric), pooled
    over every cube in every transition (not averaged per-row), matching how
    `TransitionBank.moved_count_histogram` already pools."""
    moved = moved.bool()
    geometric = geometric.bool()
    tp = (geometric & moved).sum().item()
    n_moved = moved.sum().item()
    n_geometric = geometric.sum().item()
    return dict(recall=tp / n_moved if n_moved else float("nan"),
               precision=tp / n_geometric if n_geometric else float("nan"),
               n_moved=int(n_moved), n_geometric=int(n_geometric), n_tp=int(tp))
