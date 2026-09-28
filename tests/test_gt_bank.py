"""`simple_mpc.gt_bank.GroundTruthBank`: store / look up simulated outcomes
without ever mixing execution paths or fingerprints. Genesis-free."""
from __future__ import annotations

import pytest
import torch

from simple_mpc.gt_bank import GroundTruthBank, state_key

FP = {"config_sha": "abc", "settle": 60}


def _state(seed=0, n=20):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 7, generator=g)


def _sim(actions):
    """A fake deterministic simulator: final = f(action), shape (B, 20, 7)."""
    return actions.sum(1, keepdim=True)[:, :, None].expand(-1, 20, 7).clone()


def test_roundtrip_and_skip_duplicates(tmp_path):
    bank = GroundTruthBank(tmp_path)
    s, a = _state(), torch.randn(5, 4)
    assert bank.add(s, a, _sim(a), "oracle_rollout_full", FP, "test") == 5
    assert bank.add(s, a[:2], _sim(a[:2]) + 1, "oracle_rollout_full", FP, "test") == 0
    hit, f = GroundTruthBank(tmp_path).lookup(s, a, "oracle_rollout_full")   # fresh, from disk
    assert hit.all() and torch.equal(f, _sim(a))                            # first write kept


def test_never_crosses_sim_paths(tmp_path):
    bank = GroundTruthBank(tmp_path)
    s, a = _state(), torch.randn(3, 4)
    bank.add(s, a, _sim(a), "oracle_rollout_full", FP, "test")
    hit, _ = bank.lookup(s, a, "oracle_rollout_reduced")
    assert not hit.any()
    with pytest.raises(KeyError):
        bank.lookup(s, a, "some_unregistered_path")


def test_fingerprint_mismatch_raises(tmp_path):
    bank = GroundTruthBank(tmp_path)
    s, a = _state(), torch.randn(3, 4)
    bank.add(s, a, _sim(a), "oracle_rollout_full", FP, "test")
    with pytest.raises(ValueError):
        bank.add(s, a + 1, _sim(a + 1), "oracle_rollout_full", {**FP, "settle": 30}, "test")


def test_evaluate_only_simulates_misses(tmp_path):
    bank = GroundTruthBank(tmp_path)
    s, a = _state(), torch.randn(6, 4)
    bank.add(s, a[:4], _sim(a[:4]), "oracle_rollout_full", FP, "first")
    calls = []

    def sim(x):
        calls.append(x.shape[0])
        return _sim(x)
    f = bank.evaluate(s, a, "oracle_rollout_full", FP, "second", sim)
    assert calls == [2] and torch.equal(f, _sim(a))
    assert bank.record("oracle_rollout_full", state_key(s))["sources"] == ["first", "second"]


def test_state_key_distinguishes_states():
    assert state_key(_state(0)) != state_key(_state(1))
    assert state_key(_state(0)) == state_key(_state(0).clone())


def test_does_not_persist_a_views_whole_storage(tmp_path):
    big = torch.rand(5000, 20, 7)
    bank = GroundTruthBank(tmp_path)
    a = torch.randn(2, 4)
    bank.add(big[0], a, big[1:3], "oracle_rollout_full", FP, "test")
    f = next((tmp_path / "oracle_rollout_full").glob("*.pt"))
    assert f.stat().st_size < 100_000, f"{f.stat().st_size} bytes: a view's storage was saved"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA to reproduce")
def test_works_under_cuda_default_device(tmp_path):
    """Genesis calls torch.set_default_device('cuda'); the bank must not care."""
    bank = GroundTruthBank(tmp_path)
    s, a = _state(), torch.randn(4, 4)
    bank.add(s, a[:2], _sim(a[:2]), "oracle_rollout_full", FP, "t")
    prev = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        f = bank.evaluate(s, a, "oracle_rollout_full", FP, "t", lambda x: _sim(x.cpu()))
    finally:
        torch.set_default_device(prev)
    assert torch.equal(f, _sim(a))
