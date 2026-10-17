"""G4 (RAL plan, 2026-10-15): EXP-0050 oracle.py --legal. Genesis-free: the legaliser on CPU tensors, the inf-masked CEM
refit, and an end-to-end oracle.main() run against a fake simulator (genesis and simple_mpc.genesis_oracle stubbed)
checking that every executed push is touchdown-legal and that the new per-decision fields are recorded."""
import json, math, sys, types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "experiments/EXP-0050-oracle-ceiling/code"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(CODE))

from Genesis.action_sampling import legalize_pushes, legalize_pushes_torch  # noqa: E402
import oracle_legal as OL  # noqa: E402


def _state(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    xy = (torch.rand(n, 2, generator=g) - 0.5) * 0.04
    yaw = (torch.rand(n, generator=g) - 0.5) * 1.5
    q = torch.stack([torch.cos(yaw / 2), torch.zeros(n), torch.zeros(n), torch.sin(yaw / 2)], 1)
    return torch.cat([xy, torch.full((n, 1), 0.0125), q], 1), yaw


def _pushes(n=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    s = (torch.rand(n, 2, generator=g) - 0.5) * 0.06
    th = torch.rand(n, generator=g) * 2 * math.pi
    return torch.cat([s, s + 0.03 * torch.stack([torch.cos(th), torch.sin(th)], 1)], 1)


def test_legaliser_cpu_matches_reference():
    st, yaw = _state()
    acts = _pushes()
    a1, s1, ok1 = OL.make_legaliser(st)(acts)
    assert a1.device.type == "cpu"
    box = OL.legal_box()
    a0, s0, ok0 = legalize_pushes(acts, st[None, :, :2].expand(len(acts), -1, -1), yaw[None].expand(len(acts), -1),
                                  OL.CUBE_HALF, OL.BLADE_HL, OL.BLADE_HW, box=box)
    assert torch.equal(torch.as_tensor(ok0), ok1)
    assert torch.allclose(torch.as_tensor(a0).float(), a1, atol=1e-5)
    assert int((s1 > 0).sum()) > 0, "test state should make some pushes illegal"
    # the legalised pushes are legal as they stand: a second pass moves nothing
    a2, s2, ok2 = OL.make_legaliser(st)(a1[ok1])
    assert ok2.all() and float(s2.max()) == 0.0


def test_legaliser_matches_harness_call():
    """same answer as legalize_pushes_torch called the way wide_planner.make_legal_scorer calls it."""
    st, yaw = _state(seed=3)
    acts = _pushes(seed=4)
    a1, _, ok1 = OL.make_legaliser(st)(acts)
    a2, _, ok2 = legalize_pushes_torch(acts, st[None, :, :2], yaw[None], 0.0025, 0.02, 0.001, box=(-0.064 + 0.004, 0.064 - 0.004))
    assert torch.equal(ok1, ok2) and torch.equal(a1, a2)


def test_masked_values_and_refit():
    ok = torch.tensor([True, False, True, False])
    v = OL.masked_values(torch.tensor([2.0, 1.0]), ok)
    assert v.tolist() == [2.0, float("inf"), 1.0, float("inf")]
    pop = torch.tensor([[0.0, 0.0, 0.03, 0.0], [9.0, 9.0, 9.0, 9.0], [0.01, 0.0, 0.04, 0.0], [9.0, 9.0, 9.0, 9.0]])
    g = torch.Generator().manual_seed(0)
    new = OL.cem_refit(pop, v, 3, 50, g, 0.02, 0.07)       # only the 2 finite rows may be elites
    assert new.shape == (50, 4) and float(new.abs().max()) < 0.1
    assert OL.cem_refit(pop, torch.full((4,), float("inf")), 2, 5, g, 0.02, 0.07) is None
    one = OL.cem_refit(pop, torch.tensor([1.0, float("inf"), float("inf"), float("inf")]), 2, 5, g, 0.02, 0.07)
    assert one.shape == (5, 4) and torch.isfinite(one).all()


# ---------------------------------------------------------------- end-to-end with a fake simulator
class _FakeSim:
    def __init__(self, n_envs, rec):
        self.n, self.rec = n_envs, rec
        self._particle_state = None; self._operation_height = 0.01
        self._settle_steps = self._clearance_ctrl_steps = 0

    def set_material_properties(self, *_):
        pass

    def set_particle_state(self, pos, quat):
        s = torch.cat([pos, quat], -1).float().cpu()
        self._particle_state = s.expand(self.n, -1, -1).clone() if s.shape[0] == 1 else s.clone()

    def update_material_state(self):
        pass

    def generate_action_samples(self, m, pile_aware=True, min_swath_particles=3):
        # pushes aimed through the pile centroid from random sides -- many land ON a cube (illegal touchdowns)
        st = self._particle_state[0]; c = st[:, :2].mean(0)
        th = torch.rand(self.n, m) * 2 * math.pi; u = torch.stack([torch.cos(th), torch.sin(th)], -1)
        s = c - u * (torch.rand(self.n, m, 1) * 0.02); e = s + u * 0.03
        z = torch.zeros(self.n, m, 1)
        return torch.cat([s, z], -1), torch.cat([e, z], -1), None

    def execute_action(self, start, end, ang):
        self.rec.append(torch.cat([start[:, :2], end[:, :2]], 1).cpu().clone())
        s, e = start[:, None, :2].cpu(), end[:, None, :2].cpu(); d = e - s; L = d.norm(dim=-1, keepdim=True); u = d / L
        r = self._particle_state[:, :, :2] - s; t = (r * u).sum(-1, keepdim=True)
        q = (r * torch.stack([-u[..., 1], u[..., 0]], -1)).sum(-1, keepdim=True).abs()
        hit = (t > 0) & (t < L) & (q < 0.02)
        self._particle_state[:, :, :2] += torch.where(hit, (L - t) * u, torch.zeros_like(r))


class _FakeEnv:
    def __init__(self, cfg, n_envs):
        self.rec = []; self._sim = _FakeSim(n_envs, self.rec); self._real_settle_steps = self._real_clearance_steps = 0
        _FakeEnv.last = self

    def destroy(self):
        pass


@pytest.fixture
def fake_genesis(monkeypatch):
    gs = types.ModuleType("genesis"); gs.device = "cpu"
    go = types.ModuleType("simple_mpc.genesis_oracle"); go.GenesisOracleEnv = _FakeEnv
    monkeypatch.setitem(sys.modules, "genesis", gs)
    monkeypatch.setitem(sys.modules, "simple_mpc.genesis_oracle", go)
    sys.modules.pop("oracle", None)
    yield
    sys.modules.pop("oracle", None)


def _corpus_available():
    return (REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys").exists()


@pytest.mark.skipif(not _corpus_available(), reason="start-state corpus not on disk")
def test_oracle_legal_end_to_end(fake_genesis, tmp_path, monkeypatch):
    import oracle
    monkeypatch.setattr(sys, "argv", ["oracle.py", "--mode", "cem", "--legal", "--tag", "t", "--goals", "letter_O_w20",
                                      "--starts", "40", "--steps", "2", "--n-envs", "8", "--cem-pop", "8", "--cem-iters", "2",
                                      "--cem-elite", "2", "--seed-base", "0", "--out-dir", str(tmp_path)])
    oracle.main()
    e = json.loads((tmp_path / "t.json").read_text())["episodes"][0]
    for k in ("quats", "rel", "reach_frac", "exec_reach", "illegal_frac", "legal_shift_m", "entering", "leaving", "legal_fail", "sims_used"):
        assert k in e, k
    assert len(e["quats"]) == len(e["states"]) == 3 and len(e["quats"][0][0]) == 4
    assert len(e["rel"]) == 3 and 0 <= e["rel"][-1] <= 1.5
    assert max(e["illegal_frac"]) > 0, "fake sampler should propose some illegal touchdowns"
    # every action the fake sim executed (all candidate evaluations) was legal on its episode's state at that time:
    # re-legalising the executed decisions moves nothing
    st = [torch.tensor(np.c_[np.asarray(s), np.asarray(q)]).float() for s, q in zip(e["states"], e["quats"])]
    for t, act in enumerate(e["actions"]):
        if act is None:
            continue
        _, sh, ok = OL.make_legaliser(st[t])(torch.tensor([act]).float())
        assert bool(ok[0]) and float(sh[0]) == 0.0


def test_default_flag_off_unchanged_parser():
    """--legal is opt-in: the source still contains the original unlegalised CEM branch, and --legal is rejected outside cem."""
    src = (CODE / "oracle.py").read_text()
    assert 'elif a.mode == "cem" and a.legal:' in src and 'elif a.mode == "cem":' in src
    assert 'ap.add_argument("--legal", action="store_true"' in src
