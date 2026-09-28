"""Tests for the configurable push-frame chamfer distance / GPU-batched
top-k bank search (EXP-0059 follow-up: `model/retrieval/distance.py`).

Genesis-free, no data files needed -- everything here is small synthetic
tensors on CPU (`device="cpu"` passed explicitly so these pass on a
GPU-less CI runner too).
"""
import torch

from model.retrieval.bank import TransitionBank
from model.retrieval.distance import (
    DistanceConfig, window_mask, corridor_weight_mask, wall_distance,
    chamfer_distance_chunk, query_points_and_weights, bank_points_and_weights,
    topk_search, OCC_BOUNDS,
)


def _make_state(xy, z=0.0125):
    n = xy.shape[0]
    q = torch.zeros(n, 4); q[:, 0] = 1.0   # identity quaternion
    zcol = torch.full((n, 1), z)
    return torch.cat([xy, zcol, q], dim=1)


def test_window_mask_basic():
    uv = torch.tensor([[[0.0, 0.0], [10.0, 10.0], [0.02, 0.0]]])
    m = window_mask(uv, (-0.005, 0.045), (-0.05, 0.05))
    assert m.tolist() == [[True, False, True]]


def test_corridor_weight_mask_scales_with_push_len():
    cfg = DistanceConfig(corridor_weight=3.0, corridor_v_halfwidth=0.025, corridor_u_pad=0.005)
    uv = torch.tensor([[[0.01, 0.0], [0.05, 0.0], [0.0, 0.04]]])
    push_len = torch.tensor([0.02])
    w = corridor_weight_mask(uv, push_len, cfg)
    # cube 0: u=0.01 in [0, 0.025] and |v|=0 <= 0.025 -> in corridor -> weight 3
    # cube 1: u=0.05 > 0.025 -> outside -> weight 1
    # cube 2: |v|=0.04 > 0.025 -> outside -> weight 1
    assert torch.allclose(w[0], torch.tensor([3.0, 1.0, 1.0]))


def test_wall_distance_matches_axis_aligned_box():
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])   # push along +x
    # push-frame (0,0) IS world (0,0): centre of a +-0.064 box -> wall dist 0.064
    # push-frame (0.064, 0) is world (0.064, 0): exactly on the +x wall -> 0
    uv = torch.tensor([[[0.0, 0.0], [0.064, 0.0]]])
    d = wall_distance(uv, p_start, p_stop, OCC_BOUNDS)
    assert abs(float(d[0, 0]) - 0.064) < 1e-5
    assert abs(float(d[0, 1]) - 0.0) < 1e-5


def test_cap_and_mismatch_penalty_change_the_distance():
    """Single query cube / single bank cube (isolates the `_cost` mechanics
    from the nearest-neighbour selection over multiple cubes): a raw
    distance of 5cm, with cap=2cm and mismatch_penalty=4cm.

    Expected ordering: CAPPED-ONLY (no separate mismatch_penalty -- reads as
    `min(d, cap)`) < CAPPED-WITH-MISMATCH (a flat, larger cost for "no
    plausible match") < UNCAPPED (the full 5cm counts twice, symmetric)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0]])
    window = dict(window_u=(-0.5, 0.5), window_v=(-0.5, 0.5))
    query_xy = torch.tensor([[[0.0, 0.0]]])
    bank_xy = torch.tensor([[[0.05, 0.0]]])   # 5cm away

    def _d(cfg):
        qw = window_mask(query_xy, cfg.window_u, cfg.window_v).float()
        bw = window_mask(bank_xy, cfg.window_u, cfg.window_v).float()
        return float(chamfer_distance_chunk(query_xy, qw, bank_xy, bw, cfg)[0, 0])

    d_uncapped = _d(DistanceConfig(**window))
    d_capped_only = _d(DistanceConfig(**window, cap=0.02))
    d_capped_mismatch = _d(DistanceConfig(**window, cap=0.02, mismatch_penalty=0.04))

    assert abs(d_uncapped - 2 * 0.05) < 1e-6
    assert abs(d_capped_only - 2 * 0.02) < 1e-6
    assert abs(d_capped_mismatch - 2 * 0.04) < 1e-6
    assert d_capped_only < d_capped_mismatch < d_uncapped


def test_topk_search_excludes_invalid_bank_rows():
    """A bank row with nothing inside ITS OWN window must never win, even
    if it happens to sit at the exact query location (which, unmasked,
    would read as a perfect 0-distance match)."""
    p_start = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    p_stop = torch.tensor([[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]])
    xy0 = torch.tensor([[[0.01, 0.0]], [[9.0, 9.0]]])   # row 0 in-window, row 1 far outside any window
    xy1 = xy0.clone(); xy1[:, 0, 0] += 0.005
    states0 = torch.stack([_make_state(xy0[0]), _make_state(xy0[1])])
    states1 = torch.stack([_make_state(xy1[0]), _make_state(xy1[1])])
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t", "t"])

    cfg = DistanceConfig()
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, cfg)
    assert bank_valid.tolist() == [True, False]

    query_xy = torch.tensor([[[0.01, 0.0]]])
    q_start, q_stop = p_start[:1], p_stop[:1]
    q_pts, q_w, _, _ = query_points_and_weights(_make_state(query_xy[0])[None], q_start, q_stop, cfg)
    idx, dist = topk_search(q_pts, q_w, bank_pts, bank_w, cfg, k=2, device="cpu", bank_valid=bank_valid)
    assert int(idx[0, 0]) == 0             # the valid row wins
    assert float(dist[0, 1]) >= 1e9 - 1    # the invalid row is pushed to +inf, not silently 0


def test_topk_search_orders_by_distance():
    p_start = torch.tensor([[0.0, 0.0, 0.0]] * 3)
    p_stop = torch.tensor([[0.02, 0.0, 0.0]] * 3)
    xy0 = torch.tensor([[[0.01, 0.0]], [[0.015, 0.0]], [[0.03, 0.0]]])
    xy1 = xy0.clone(); xy1[:, 0, 0] += 0.005
    states0 = torch.stack([_make_state(xy0[i]) for i in range(3)])
    states1 = torch.stack([_make_state(xy1[i]) for i in range(3)])
    bank = TransitionBank.from_states(states0, states1, p_start, p_stop, source=["t"] * 3)

    cfg = DistanceConfig()
    bank_pts, bank_w, bank_valid = bank_points_and_weights(bank, cfg)
    query_xy = torch.tensor([[[0.0095, 0.0]]])   # closest to bank row 0
    q_pts, q_w, _, _ = query_points_and_weights(_make_state(query_xy[0])[None], p_start[:1], p_stop[:1], cfg)
    idx, dist = topk_search(q_pts, q_w, bank_pts, bank_w, cfg, k=3, device="cpu", bank_valid=bank_valid)
    assert idx[0].tolist() == [0, 1, 2]
    assert dist[0, 0] <= dist[0, 1] <= dist[0, 2]
