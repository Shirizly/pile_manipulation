"""Invariant `genesis-snapshot-restore-repeat-determinism` -- BROKEN (EXP-0027).

EXP-0024 found repeats of an action bit-identical, but tested only ONE
rollout_candidates call after a fresh reset. EXP-0027's probes showed that
re-running the IDENTICAL 32-action batch from the IDENTICAL snapshot a second
time changes dv (lyapunov, corner) by up to 7e-3 and final positions by up to
17 mm, that outcomes depend on the other actions in the batch, and that an
un-reset env differs systematically from a scene.reset() one (up to 0.048).

strict xfail: the day this passes, the bug is fixed -- drop the marker and flip
the tag to `fixed` in experiments/INVARIANTS.md.

Opt-in (builds a Genesis scene, ~1 min, needs a GPU):
    RUN_GENESIS_TESTS=1 python -m pytest tests/test_genesis_repeat_determinism.py
"""
import glob
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("RUN_GENESIS_TESTS") != "1" or not torch.cuda.is_available(),
                                reason="opt-in Genesis test: set RUN_GENESIS_TESTS=1 (needs GPU)")
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.xfail(strict=True, reason="genesis-snapshot-restore-repeat-determinism is BROKEN (EXP-0027)")
def test_same_batch_twice_from_same_snapshot_is_identical():
    pytest.importorskip("genesis")
    sys.path.insert(0, str(REPO))
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.oracle_mpc import load_oracle_config
    ref = glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage1_actions.pt"))
    if not ref:
        pytest.skip("EXP-0023 artifact (a real settled n20 state + actions) not on disk")
    r = torch.load(ref[0], weights_only=False)
    st = r["states0"][0].float()
    acts = torch.cat([r["pool_actions"][0][:32].float()])
    K = 32
    cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K)
    import genesis as gs
    snap = {"pos": st[None, :, 0:3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
    try:
        p1 = env.rollout_candidates(acts[:, None, :], snap, use_rollout_fidelity=False, record=False)
        p2 = env.rollout_candidates(acts[:, None, :], snap, use_rollout_fidelity=False, record=False)
    finally:
        env.destroy()
    assert float((p1 - p2).abs().max()) < 1e-6
