"""simple_mpc/learned_mpc.py -- planners with a learned-model objective.
Genesis-free: a toy differentiable 'model' stands in for a real adapter."""
import glob
import sys
from pathlib import Path

import pytest
import torch

from simple_mpc.adapters import OCC_GRID
from simple_mpc.learned_mpc import ModelObjective, plan, project_push, run_episode


class ToyAdapter:
    """predict_step: moves occupancy mass toward the push end point, smoothly
    (differentiable in the action); enough to give planners a real signal."""
    device = "cpu"

    def predict_step(self, occ, act):
        B = occ.shape[0]
        g = torch.linspace(-0.064, 0.064, OCC_GRID)
        X, Y = torch.meshgrid(g, g, indexing="ij")
        ex, ey = act[:, 2].view(B, 1, 1), act[:, 3].view(B, 1, 1)
        blob = torch.exp(-((X - ex) ** 2 + (Y - ey) ** 2) / (2 * 0.01 ** 2))
        return 0.5 * occ + 0.5 * blob / blob.sum((1, 2), keepdim=True) * occ.sum((1, 2), keepdim=True)


def _state(seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(20, 7); s[:, :2] = (torch.rand(20, 2, generator=g) - 0.5) * 0.08
    return s


def _goal():
    g = torch.linspace(0, 1, OCC_GRID)
    return (g[:, None] + g[None, :]) / 2          # distance-like field, low at one corner


def test_project_push_matches_exp0023():
    ref = glob.glob("experiments/EXP-0023-*/code/stage1_optimise.py")
    if not ref:
        pytest.skip("EXP-0023 code not present")
    sys.path.insert(0, str(Path(ref[0]).parent))
    from stage1_optimise import project
    a = (torch.rand(500, 4) - 0.5) * 0.2
    x, hx = project(a); y, hy = project_push(a)
    assert torch.equal(x, y) and torch.equal(hx, hy)


@pytest.mark.parametrize("planner", ["rank", "gd", "cem", "mppi"])
def test_planners_never_worse_than_best_candidate(planner):
    torch.manual_seed(0)
    obj = ModelObjective(ToyAdapter(), _state(), _goal())
    cand = (torch.rand(64, 4) - 0.5) * 0.1
    with torch.no_grad():
        best_rank = float(obj(project_push(cand)[0]).min())
    plan(obj, cand, planner, budget_s=0.05)      # warm-up: torch's first optimizer use lazy-imports (~1 s)
    out = plan(obj, cand, planner, budget_s=0.3)
    assert out["pred_dv"] <= best_rank + 1e-9
    # budget respected up to one iteration (the stop rule is predictive)
    assert out["time_s"] <= max(0.3, out["rank_s"]) + out["max_iter_s"] + 0.2
    a, hit = project_push(out["action"][None])
    assert torch.allclose(a[0], out["action"], atol=1e-6)      # returned action is legal


def test_run_episode_calls_checkpoint_hook_each_step():
    seen = []
    rec = run_episode(ToyAdapter(), _state(), _goal(),
                      execute=lambda a: _state(1), sample_candidates=lambda p: (torch.rand(16, 4) - 0.5) * 0.1,
                      planner="rank", budget_s=0.05, n_steps=3, on_step=lambda r: seen.append(len(r["actions"])))
    assert seen == [1, 2, 3] and len(rec["values"]) == 4


def test_rank_uses_its_budget_when_it_can_resample():
    obj = ModelObjective(ToyAdapter(), _state(), _goal())
    draw = lambda: (torch.rand(32, 4) - 0.5) * 0.1
    plan(obj, draw(), "rank", budget_s=0.05, more_candidates=draw)   # warm-up
    out = plan(obj, draw(), "rank", budget_s=0.4, more_candidates=draw)
    assert out["n_evals"] > 32 and out["n_iters"] >= 1


def test_ensemble_objective_is_mean_of_member_dv():
    a, b = ToyAdapter(), ToyAdapter()
    act = (torch.rand(5, 4) - 0.5) * 0.1
    single = ModelObjective(a, _state(), _goal())(act)
    ens = ModelObjective([a, b], _state(), _goal())(act)
    assert torch.allclose(single, ens, atol=1e-6)      # identical members -> identical objective


def test_batched_runner_matches_per_env_bookkeeping():
    """run_episodes_batched: each env gets its OWN action, values follow its own state,
    and the bank passed to the planner is that env's bank."""
    from simple_mpc.learned_mpc import run_episodes_batched
    torch.manual_seed(0)
    K, P = 3, 20
    parts0 = torch.stack([_state(s) for s in range(K)])
    state = {"p": parts0.clone()}
    seen = []

    def sample(m):
        # env k's candidates all end at x = 0.01 * k, so each env's pick is traceable
        c = (torch.rand(K, m, 4) - 0.5) * 0.08
        c[:, :, 2] = torch.arange(K).float()[:, None] * 0.01
        return c

    def execute(a):
        seen.append(a.clone())
        state["p"] = state["p"].clone(); state["p"][:, :, 0] += 0.001
        return state["p"]
    eps = [dict(adapter=ToyAdapter(), dist=_goal(), planner=pl, budget_s=0.05) for pl in ("rank", "gd", "cem")]
    recs = run_episodes_batched(eps, parts0, execute, sample, n_steps=2, n_cand=16, bank_per_env=64)
    assert len(seen) == 2 and seen[0].shape == (K, 4)
    assert torch.allclose(seen[0][0, 2], torch.tensor(0.0), atol=0.02)      # rank picked from env 0's bank
    for r in recs:
        assert len(r["values"]) == 3 and len(r["actions"]) == 2
        assert abs(sum(r["true_dv"]) - (r["values"][-1] - r["values"][0])) < 1e-6
