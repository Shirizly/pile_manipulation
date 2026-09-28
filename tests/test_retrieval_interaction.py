"""Tests for `model/retrieval/interaction.py`'s truth-free geometric
interaction set (EXP-0059 coordinator follow-up B, 2026-09-28)."""
import torch

from model.retrieval.interaction import interaction_set, recall_precision


def test_direct_set_includes_swept_cube_excludes_far_cube():
    # push from (0,0) to (0.02,0) along +x; one cube right in the blade's
    # path, one cube far away laterally and along u.
    uv = torch.tensor([[[0.01, 0.0], [0.06, 0.06]]])
    mask = interaction_set(uv, push_len=0.02)
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False


def test_squeezes_unbatched_input():
    uv = torch.tensor([[0.01, 0.0], [0.06, 0.06]])
    mask = interaction_set(uv, push_len=0.02)
    assert mask.shape == (2,)
    assert bool(mask[0]) and not bool(mask[1])


def test_chain_closure_propagates_forward_within_push_length():
    # cube 0 (u=20mm) is directly swept (right at the direct set's u boundary);
    # cube 1 (u=25mm) sits just ahead of it, OUTSIDE the direct set's u range
    # ([-2.5mm, 22.5mm]) on its own, but within contact range (7.07mm) and
    # ahead along u -- pulled in by chain closure once tau is large enough
    # (d_c = cube_size + tau must exceed 7.07mm), not at tau=0 (d_c=5mm).
    uv = torch.tensor([[[0.02, 0.0], [0.025, 0.005]]])
    mask_no_chain = interaction_set(uv, push_len=0.02, tau=0.0)   # d_c = cube_size only (5mm) -- too small
    mask_chain = interaction_set(uv, push_len=0.02, tau=0.012)    # default tuned tau -> d_c=17mm
    assert bool(mask_chain[0, 0])           # directly-swept cube always in
    assert not bool(mask_no_chain[0, 1])    # d_c=5mm < 7.07mm gap: not pulled in
    assert bool(mask_chain[0, 1])           # d_c=17mm > 7.07mm gap, ahead, within push_len: pulled in


def test_chain_closure_stops_beyond_push_length():
    # a long straight chain of cubes spaced just within contact range of each
    # other, extending far past the push length -- the tail must NOT be
    # pulled in once the CUMULATIVE forward gap (from the nearest directly-
    # swept cube, not from index 0) exceeds push_len.
    n = 12
    step = 0.006  # within default d_c (cube_size + tau = 17mm) but the CHAIN sums past push_len
    us = step * torch.arange(n)
    uv = torch.stack([us, torch.zeros(n)], dim=-1)[None]
    push_len = 0.02
    mask = interaction_set(uv, push_len=push_len, tau=0.012)
    # direct set (u <= push_len + h = 0.0225): indices 0-3 (u = 0, 6, 12, 18 mm), reach=0.
    # chain closure from index 3 (u=18mm): +6mm -> 24mm (reach 6mm, in), +6 -> 30mm (12mm, in),
    # +6 -> 36mm (18mm, in), +6 -> 42mm (24mm > push_len=20mm, OUT).
    expected_in = torch.tensor([True] * 7 + [False] * (n - 7))
    assert torch.equal(mask[0], expected_in), (mask[0].tolist(), expected_in.tolist())
    assert not bool(mask[0, -1])  # the far tail is excluded
    assert bool(mask[0, 6]) and not bool(mask[0, 7])  # the exact cutoff is where we computed it


def test_recall_precision_perfect_when_geometric_equals_moved():
    moved = torch.tensor([[True, False, True], [False, False, True]])
    rp = recall_precision(moved, moved)
    assert rp["recall"] == 1.0 and rp["precision"] == 1.0
    assert rp["n_moved"] == 3 and rp["n_tp"] == 3


def test_recall_precision_handles_no_moved_cubes():
    moved = torch.zeros(2, 3, dtype=torch.bool)
    geometric = torch.tensor([[True, False, False], [False, False, False]])
    rp = recall_precision(geometric, moved)
    assert rp["n_moved"] == 0
    import math
    assert math.isnan(rp["recall"])
