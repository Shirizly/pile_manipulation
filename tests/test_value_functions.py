"""Capacity-aware value functions and goal-aware candidate selection (EXP-0055 / EXP-0056)."""
import numpy as np
import torch

from Baselines.common.goals import dist_field_from_mask, letter_mask
from simple_mpc import goal_aware_sampling as gas
from simple_mpc import value_functions as vf
from simple_mpc.adapters import occ_for_scoring


def _occ(xy):
    return occ_for_scoring(torch.tensor(np.c_[xy, np.full(len(xy), 0.0025)])[None].float())


def _mask():
    return letter_mask("T", 64, 64)


def test_uniform_layout_beats_packed_on_coverage_metrics():
    m = _mask(); xy = vf.uniform_layout(m)
    packed = np.r_[xy[:10], xy[:10] + 0.003]
    e_u, c_u = vf.coverage_metrics(_occ(xy), m)
    e_p, c_p = vf.coverage_metrics(_occ(packed), m)
    assert float(e_u) < float(e_p) and float(c_u) > float(c_p)
    assert abs(float(vf.uniform_target(m).sum()) - 1) < 1e-5


def test_sliced_emd_orders_and_is_differentiable():
    m = _mask(); t = vf.uniform_target(m)
    d = vf.SlicedEMD(t, res=32, n_dirs=8, device="cpu")
    xy = vf.uniform_layout(m)
    far = xy.copy(); far[:, 0] = -0.05
    occ = torch.cat([_occ(xy), _occ(far)]).requires_grad_(True)
    v = d(occ)
    assert float(v[0]) < float(v[1])
    v.sum().backward()
    assert torch.isfinite(occ.grad).all()


def test_linearised_field_raises_potential_of_full_goal_parts():
    m = _mask(); xy = vf.uniform_layout(m)
    packed = _occ(np.r_[xy[:10], xy[:10] + 0.003])[0]
    phi = vf.linearised_emd_field(packed, vf.uniform_target(m), n_iter=200)
    mm = torch.as_tensor(m); full = packed > 0.05 * packed.max()
    assert float(phi[full & mm].mean()) > float(phi[~full & mm].mean())
    assert float(phi[~mm].mean()) > float(phi[~full & mm].mean())


def test_crowding_penalty_zero_under_capacity_positive_when_crowded():
    m = _mask(); c = vf.CrowdingPenalty(m, device="cpu")
    t = vf.uniform_target(m)[None]
    xy = vf.uniform_layout(m)
    crowded = np.repeat(xy[:2], 10, 0)
    assert float(c(t)) < 1e-6
    assert float(c(_occ(crowded))) > float(c(_occ(xy)))


def test_carry_moves_swath_cubes_to_stop_line_only():
    xy = torch.tensor([[0.0, 0.01], [0.0, 0.03], [0.03, 0.0]])
    act = torch.tensor([[0.0, 0.0, 0.0, 0.04]])        # push +y from the origin, blade along x
    ins, dest = gas.carry(xy, act)
    assert ins[0].tolist() == [True, True, False]
    assert torch.allclose(dest[0, 0], torch.tensor([0.0, 0.0425]), atol=1e-6)
    assert torch.allclose(dest[0, 2], xy[2])


def test_selectors_return_requested_candidates():
    m = _mask(); d = torch.from_numpy(dist_field_from_mask(m)).float()
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(20, 2, generator=g) * 0.08 - 0.04
    bank = torch.cat([xy[torch.randint(0, 20, (512,), generator=g)] - 0.008,
                      xy[torch.randint(0, 20, (512,), generator=g)] + 0.02], 1)
    for f in (gas.select_misplaced, gas.select_deposit):
        out = f(bank, xy, m, d, 64, g)
        assert out.shape == (64, 4)
    p = gas.ot_proposals(xy, m, 16, g)
    assert p is None or p.shape == (16, 4)


def test_crowd_floor_does_not_penalise_isolated_cubes_in_a_large_goal():
    from Baselines.common.goals import quadrant_mask
    m = quadrant_mask(64, 64, 0)
    xy = vf.uniform_layout(m)                      # 20 cubes spread over a 32x32 px quadrant
    occ = _occ(xy)
    assert float(vf.CrowdingPenalty(m, device="cpu")(occ)) > 0            # the flaw
    assert float(vf.CrowdingPenalty(m, device="cpu", floor_single=True)(occ)) < 1e-3
    crowded = np.repeat(xy[:2], 10, 0)
    assert float(vf.CrowdingPenalty(m, device="cpu", floor_single=True)(_occ(crowded))) > 0.01
