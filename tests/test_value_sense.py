"""Sense-safe difference metrics and the dv-convention guard (TODO M4).

`dv = value(after) - value(before)` is a COST for `lyapunov` and a VALUE for
the mass functions (experiments/METRICS.md, SIGN section). These tests pin:
the difference metrics reproduce EXP-0023's cost-sense numbers exactly, they
are mirror-symmetric under a sense flip, and `assert_dv_convention` fails
loudly when an adapter's declared sense disagrees with its value function.
Genesis-free, CPU-only.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pytest
import torch

from Baselines.common.goals import (gradient_benchmark_metrics, higher_is_better_for,
                                    improvement, mass_in_region)
from simple_mpc.adapters import OCC_GRID, assert_dv_convention


def test_unknown_value_fn_raises():
    with pytest.raises(KeyError):
        higher_is_better_for("lyapunov_typo")
    assert higher_is_better_for("lyapunov") is False
    assert higher_is_better_for("mass_in_region") is True


def test_sense_is_a_required_keyword():
    with pytest.raises(TypeError):
        improvement(1.0, 0.0)                       # no default sense
    with pytest.raises(TypeError):
        gradient_benchmark_metrics(0.0, 0.0, 0.0, np.zeros(3))


def test_mirror_symmetry():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r, g, o = rng.normal(size=3)
        pool = rng.normal(size=20)
        a = gradient_benchmark_metrics(r, g, o, pool, higher_is_better=False)
        b = gradient_benchmark_metrics(-r, -g, -o, -pool, higher_is_better=True)
        for k in ("gradient_gain", "pool_escape", "regret_vs_oracle", "capture_vs_oracle"):
            assert a[k] == pytest.approx(b[k], abs=1e-12), k


def test_reproduces_exp0023_cost_sense_rows():
    """The cost case must equal EXP-0023's hard-coded formulae on its own data."""
    paths = glob.glob("experiments/EXP-0023-*/results/metrics.json")
    if not paths:
        pytest.skip("EXP-0023 results not on disk")
    rows = json.load(open(paths[0]))["rows"]
    for r in rows:
        m = gradient_benchmark_metrics(r["dv_rank"], r["dv_grad"], r["dv_oracle"],
                                       np.array([r["pool_ceiling"]]), higher_is_better=False)
        for k in ("gradient_gain", "pool_escape", "regret_vs_oracle"):
            assert m[k] == pytest.approx(r[k], abs=1e-7), (k, r["slate"], r["arm"])


class _StubAdapter:
    """Geometry of a real adapter (a distance field `dw`), value fn swappable."""
    def __init__(self, value_fn, higher_is_better):
        yy, xx = torch.meshgrid(torch.arange(OCC_GRID), torch.arange(OCC_GRID), indexing="ij")
        self.device = "cpu"
        self.dw = ((xx.float() ** 2 + yy.float() ** 2).sqrt()) / OCC_GRID   # goal at a corner
        self.mask = (self.dw < 0.3).float()
        self.value_fn, self.higher_is_better = value_fn, higher_is_better

    def value(self, occ):
        if self.value_fn == "lyapunov":
            return (occ * self.dw).flatten(1).sum(1)
        return mass_in_region(occ, self.mask)


def test_guard_accepts_correct_declarations():
    assert_dv_convention(_StubAdapter("lyapunov", False))
    assert_dv_convention(_StubAdapter("mass_in_region", True))


@pytest.mark.parametrize("value_fn,hib", [("lyapunov", True), ("mass_in_region", False)])
def test_guard_rejects_wrong_declared_sense(value_fn, hib):
    with pytest.raises(AssertionError):
        assert_dv_convention(_StubAdapter(value_fn, hib))


def test_guard_rejects_value_fn_that_disagrees_with_its_registered_sense():
    """Declared consistently with the registry, but the value function itself
    is inverted: the geometric check must still catch it."""
    ad = _StubAdapter("lyapunov", False)
    ad.value = lambda occ: -(occ * ad.dw).flatten(1).sum(1)
    with pytest.raises(AssertionError):
        assert_dv_convention(ad)
