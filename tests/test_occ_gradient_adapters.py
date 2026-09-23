"""The occupancy-grid gradient adapters (`simple_mpc/adapters.py`, EXP-0023)
must be the SAME forward as the offline predictors they wrap, and must carry
gradient to the action.

Invariant `occ-gradient-adapter-matches-offline-predictor`.

The failure this guards against is specific and silent: a `Baselines/*`
predictor's `predict_occ` is `@torch.no_grad()`, so an adapter that
reimplemented the forward could drift from it (a different plate sigma, a
different pixel convention) and nothing would notice, while an adapter that
called it directly would return a detached tensor and optimise nothing.
"""
from __future__ import annotations

import math
import pytest
import torch

from simple_mpc.adapters import (OCC_ADAPTERS, OCC_GRID, SlateRawStub,
                                 assert_dv_convention, make_occ_adapter)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTOR_ARMS = [a for a in OCC_ADAPTERS if a != "linear_switched_soft"
                  and not a.startswith("linear")]


def _fixture(B=3):
    torch.manual_seed(0)
    occ = torch.zeros(B, OCC_GRID, OCC_GRID, device=DEV)
    occ[:, 20:34, 20:34] = 1.0
    act = torch.tensor([[-0.03, -0.02, 0.01, 0.01],
                        [0.02, 0.03, -0.02, -0.01],
                        [0.00, -0.04, 0.03, 0.02]], device=DEV)[:B]
    return occ, act


@pytest.mark.parametrize("arm", sorted(OCC_ADAPTERS))
def test_gradient_reaches_every_action_component(arm):
    ad = make_occ_adapter(arm, DEV)
    assert_dv_convention(ad)
    occ, act = _fixture()
    a = act.clone().requires_grad_(True)
    dv = ad.dv(occ, a)
    assert torch.isfinite(dv).all()
    g, = torch.autograd.grad(dv.sum(), a)
    assert torch.isfinite(g).all(), f"{arm}: non-finite gradient"
    assert (g.norm(dim=1) > 0).all(), f"{arm}: a row has an all-zero gradient"
    per_component = g.abs().mean(dim=0)
    assert (per_component > 0).all(), (
        f"{arm}: gradient is exactly zero in component(s) "
        f"{[i for i, v in enumerate(per_component) if v == 0]}")


@pytest.mark.parametrize("arm", sorted(PREDICTOR_ARMS))
def test_adapter_forward_equals_offline_predictor(arm):
    """`PredictorGradientAdapter` must reproduce its predictor's own
    `predict_occ` bit-for-bit (up to float noise) -- it calls the same
    function, so any difference means the batch it builds diverges from the
    one `eval_report.py` builds."""
    ad = make_occ_adapter(arm, DEV)
    occ, act = _fixture()
    with torch.no_grad():
        mine = ad.predict_step(occ, act)
        batch = ad._batch(occ, act)
        theirs = ad.predictor.predict_occ(batch).to(mine.device)
    assert torch.allclose(mine, theirs, atol=1e-6), (
        f"{arm}: adapter forward differs from predict_occ by "
        f"{(mine - theirs).abs().max():.3g}")


def test_soft_gate_agrees_with_hard_gate_at_bin_centres():
    """`soft_bin_weights` must collapse to the hard `bin_index` one-hot at
    every bin centre -- that is what makes it a relaxation of the fitted
    model rather than a different model."""
    from Baselines.LinearForesight.model import (bin_centres, bin_index,
                                                 soft_bin_weights)
    edges = torch.linspace(0.0, 0.08, 7)
    centres = bin_centres(edges)
    w = soft_bin_weights(centres, edges)
    assert torch.allclose(w.sum(dim=1), torch.ones(len(centres)))
    assert torch.equal(w.argmax(dim=1), torch.arange(len(centres)))
    assert torch.allclose(w.max(dim=1).values, torch.ones(len(centres)), atol=1e-6)
    assert torch.equal(bin_index(centres, edges), torch.arange(len(centres)))
    # halfway between two centres the soft gate is a 50/50 blend, where the
    # hard gate jumps -- this is the whole point, so pin it
    mid = 0.5 * (centres[2] + centres[3])
    wm = soft_bin_weights(mid[None], edges)[0]
    assert math.isclose(float(wm[2]), 0.5, abs_tol=1e-5)
    assert math.isclose(float(wm[3]), 0.5, abs_tol=1e-5)


def test_soft_gate_carries_length_gradient_where_hard_gate_cannot():
    from Baselines.LinearForesight.model import soft_bin_weights
    edges = torch.linspace(0.0, 0.08, 7)
    # bin width 0.08/6 = 0.01333, centres at 0.00667, 0.0200, 0.0333, ...
    L = torch.tensor([0.031], requires_grad=True)     # between centres 1 and 2
    w = soft_bin_weights(L, edges)
    assert float(w[0, 1]) > 0 and float(w[0, 2]) > 0
    for b in (1, 2):
        g, = torch.autograd.grad(w[0, b], L, retain_graph=True)
        assert float(g.abs()) > 0.0, f"soft gate carries no length gradient into bin {b}"


def test_dv_sign_convention_is_a_cost():
    ad = make_occ_adapter("nfd_3ch", DEV)
    g = OCC_GRID
    dwf = ad.dw.reshape(-1)
    near = torch.zeros(1, g, g, device=DEV)
    far = torch.zeros(1, g, g, device=DEV)
    near[0, int(dwf.argmin()) // g, int(dwf.argmin()) % g] = 1.0
    far[0, int(dwf.argmax()) // g, int(dwf.argmax()) % g] = 1.0
    assert float(ad.value(near)) < float(ad.value(far)), (
        "lyapunov must be a COST: lower = closer to the goal")


def test_slate_raw_stub_geometry():
    """The stub must reproduce the +/-64 mm / 64 px slate convention every
    DS-0001 number already uses."""
    raw = SlateRawStub()
    assert math.isclose(raw.to_pxl, OCC_GRID / 0.128)
    assert torch.allclose(raw.ctr_in_PXL[:2], torch.tensor([32.0, 32.0]))
