"""EXP-0075 T0b harness extensions (2026-10-15), Genesis-free and CPU-only: budget_s -> evaluation counts, spill filter (+ zero-candidate
guard), gather-to-disc goal, swept-rectangle stray-reach fraction, and the shared per-push diagnostics."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "experiments/EXP-0075-closed-loop-benchmark/code"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(CODE))


@pytest.fixture(scope="module")
def wp():
    import wide_planner
    return wide_planner


@pytest.fixture()
def cpu(wp, monkeypatch):
    monkeypatch.setattr(wp, "DEV", "cpu")
    return wp


# ---- budget_s ------------------------------------------------------------------------------------------------------
def test_budget_counts_match_clean_table():
    import time_clean as tc
    T = json.loads((REPO / "experiments/EXP-0075-closed-loop-benchmark/results/timing_clean.json").read_text())
    for model, grp in (("vanilla64bal", "vanilla64"), ("ens2bal", "ens128"), ("lf", "lf"), ("zoom64", "zoom64"), ("ens128_s1", "ens128")):
        for B in (0.1, 0.3, 1.0):
            for legal in (False, True):
                row = "H1_harness" + ("_legal" if legal else "")
                for pl, key in (("cem", "cem"), ("cem+gd", "cem_gd")):
                    b = tc.budget_counts(model, 1, B, pl, legal=legal)
                    want = T["n_eval"][grp][row][str(B)][key]
                    assert (b["n_eval"], b["gd_steps"]) == (want["n_eval"], want["gd_steps"])
                    assert b["timing_row"] == row and b["timing_group"] == grp and len(b["timing_sha256"]) == 12
                    kw = b["plan_kw"]
                    assert kw["n_pool"] == kw["pop"] == max(8, b["n_eval"] // 5) and kw["iters"] == 4
                    assert kw["n_pool"] + kw["iters"] * kw["pop"] <= max(b["n_eval"], 40)
                    assert ("gd_steps" in kw) == (key == "cem_gd")
    r = tc.budget_counts("vanilla64bal", 1, 0.3, "rank")
    assert r["plan_kw"] == {"rank_n": r["n_eval"]}


def test_budget_legal_row_costs_more():
    import time_clean as tc
    a, b = tc.budget_counts("vanilla64bal", 1, 1.0, "cem"), tc.budget_counts("vanilla64bal", 1, 1.0, "cem", legal=True)
    assert b["n_eval"] < a["n_eval"] and b["c_ms"] > a["c_ms"]


def test_resolve_cell_defaults_unchanged_and_overrides():
    pytest.importorskip("genesis")      # closed_loop imports the simulator wrapper at module level (no simulator is built here)
    import closed_loop as cl
    base = {"model": "ens2bal", "planner": "cem", "H": 1, "fetch": True}
    c, rec = cl.resolve_cell(base)
    assert c == base and rec is None
    c, rec = cl.resolve_cell({"model": "vanilla64bal", "planner": "cem+gd", "H": 1, "budget_s": 0.3, "n_pool": 100})
    assert c["planner"] == "cem" and c["n_pool"] == 100 and c["pop"] == rec["plan_kw"]["pop"] and c["gd_steps"] == rec["gd_steps"] > 0
    assert rec["overridden"] == ["n_pool"] and "budget_s" not in c


# ---- geometry: swept rectangle, stray reach, spill ------------------------------------------------------------------
def test_swept_hits_geometry(cpu):
    wp = cpu
    act = torch.tensor([[0.0, 0.0, 0.03, 0.0]])            # push along +x, blade spans y in [-0.02, 0.02]
    pts = torch.tensor([[0.015, 0.0], [0.015, 0.019], [0.015, 0.021], [0.035, 0.0], [-0.0005, 0.0]])
    assert wp.swept_hits(act, pts).tolist() == [[True, True, False, False, True]]
    assert wp.swept_hits(act, pts, pad_perp=0.0025, pad_along=0.0025).tolist() == [[True, True, True, False, True]]


def test_bank_reach_frac(cpu):
    wp = cpu
    from Baselines.common.goals import named_goal_mask
    mask = torch.from_numpy(named_goal_mask("disc_n20", 64, 64))
    xy = torch.tensor([[0.0, 0.0], [0.05, 0.05]])           # one cube inside the disc, one stray at (50, 50) mm
    acts = torch.tensor([[0.04, 0.05, 0.055, 0.05],         # sweeps through the stray
                         [-0.05, -0.05, -0.03, -0.05],      # nowhere near
                         [-0.01, 0.0, 0.01, 0.0],           # sweeps only the in-goal cube
                         [0.05, 0.03, 0.05, 0.047]])        # ends 3 mm short of the stray centre (< blade 1 + cube 2.5 mm) -> reaches it
    assert wp.bank_reach_frac(acts, xy, mask) == pytest.approx(0.5)
    assert math.isnan(wp.bank_reach_frac(acts, xy[:1], mask))   # no stray -> nan


def _disc_parts(xy):
    p = torch.zeros(len(xy), 7); p[:, :2] = torch.tensor(xy); p[:, 2] = 0.0125; p[:, 3] = 1.0
    return p


def test_spill_scorer_counts_filled_goal_pixels(cpu):
    wp = cpu
    from Baselines.common.goals import named_goal_mask
    mask = torch.from_numpy(named_goal_mask("disc_n20", 64, 64))
    sp = wp.make_spill_scorer(_disc_parts([[0.0, 0.0], [0.05, 0.05]]), mask)
    v = sp(torch.tensor([[-0.01, 0.0, 0.01, 0.0], [0.04, 0.05, 0.055, 0.05]]))
    assert v[0] > 0.3 and v[1] == 0                         # through the in-goal cube vs only the out-of-goal one
    empty = wp.make_spill_scorer(_disc_parts([[0.05, 0.05]]), mask)
    assert float(empty(torch.zeros(3, 4)).abs().sum()) == 0


class _Obj:                                                 # cost = x of the push end (lower = better)
    def cost(self, s):
        return s[:, -1, 2].float()


def test_spill_filter_rejects_and_guard_falls_back(cpu):
    wp = cpu
    bank = torch.tensor([[-0.03, 0.0, -0.01, 0.0], [-0.02, 0.03, 0.0, 0.03], [0.0, -0.04, 0.02, -0.04]] * 4)
    spill = lambda a: (a[:, 1].abs() < 1e-6).float() * 5.0  # rows on y = 0 "spill" 5 cubes
    out = wp.plan_seq(_Obj(), bank, "rank", 1, rank_n=12, spill=spill, spill_thr=1.0)
    assert out["spill_reject_frac"] == pytest.approx(1 / 3) and out["spill_fallback"] == 0 and abs(float(out["action"][1])) > 1e-6
    out0 = wp.plan_seq(_Obj(), bank, "rank", 1, rank_n=12)
    assert float(out0["action"][1]) == 0.0                  # unfiltered: the y = 0 row is best
    allbad = lambda a: torch.full((len(a),), 9.0)
    out2 = wp.plan_seq(_Obj(), bank, "rank", 1, rank_n=12, spill=allbad, spill_thr=1.0)
    assert out2["spill_fallback"] == 1 and math.isfinite(out2["cost"]) and torch.equal(out2["action"], out0["action"])
    torch.manual_seed(0); out3 = wp.plan_seq(_Obj(), bank, "cem", 1, n_pool=12, pop=12, iters=2, spill=allbad, spill_thr=1.0)
    assert out3["spill_fallback"] == 1 and math.isfinite(out3["cost"])


def test_plan_seq_defaults_unchanged_by_new_kwargs(cpu):
    wp = cpu
    bank = torch.rand(64, 4) * 0.06 - 0.03
    torch.manual_seed(1); a = wp.plan_seq(_Obj(), bank, "cem", 1, n_pool=16, pop=16, iters=2)
    torch.manual_seed(1); b = wp.plan_seq(_Obj(), bank, "cem", 1, n_pool=16, pop=16, iters=2, diag_xy=torch.tensor([[0.05, 0.05]]), diag_mask=torch.zeros(64, 64))
    assert torch.equal(a["action"], b["action"]) and a["cost"] == b["cost"] and "pool_reach_frac" in b and "pool_reach_frac" not in a


# ---- gather-to-disc goal -----------------------------------------------------------------------------------------
def test_disc_goal():
    from Baselines.common.goals import named_goal_mask, disc_radius_for_cubes, parse_disc_goal, letter_goal_mask, DISC_PACKING
    r = disc_radius_for_cubes(20)
    assert math.pi * r ** 2 * DISC_PACKING == pytest.approx(20 * 0.005 ** 2) and r == pytest.approx(0.01508, abs=1e-5)
    m = named_goal_mask("disc_n20", 64, 64)
    pitch = 0.128 / 63
    assert m.sum() * pitch ** 2 == pytest.approx(math.pi * r ** 2, rel=0.1)
    assert np.array_equal(m, m.T) and np.array_equal(m, m[::-1]) and m[31, 31] and m[32, 32] and not m[0, 0]
    assert parse_disc_goal("disc_r15") == pytest.approx(0.015) and parse_disc_goal("letter_O") is None
    assert np.array_equal(named_goal_mask("letter_O_w20", 64, 64), letter_goal_mask("letter_O_w20", 64, 64))
    vs = json.loads((REPO / "experiments/EXP-0075-closed-loop-benchmark/results/vstar_disc.json").read_text())["disc_n20"]
    assert 0.9 < vs["mass_frac_best_placement"] <= 1.0 and vs["mask_px"] == int(m.sum())


# ---- shared per-push diagnostics -----------------------------------------------------------------------------------
def test_transition_diag_entering_leaving_and_value():
    import analyse
    from Baselines.common.goals import named_goal_mask, dist_field_from_mask
    m = named_goal_mask("disc_n20", 64, 64); d = dist_field_from_mask(m)
    S = np.zeros((3, 2, 3)); S[..., 2] = 0.0125
    S[0, 0, :2] = [0.05, 0.0]; S[0, 1, :2] = [0.0, 0.0]          # stray, in
    S[1, 0, :2] = [0.0, 0.005]; S[1, 1, :2] = [0.0, 0.0]         # stray enters
    S[2, 0, :2] = [0.0, 0.005]; S[2, 1, :2] = [-0.05, 0.0]       # other leaves
    dg = analyse.transition_diag(S, torch.from_numpy(d).float(), torch.from_numpy(m))
    assert dg["entering"].tolist() == [1, 0] and dg["leaving"].tolist() == [0, 1]
    assert dg["true_dv"][0] < 0 < dg["true_dv"][1]               # value is a cost: entering lowers it
    np.testing.assert_allclose(dg["true_dv"], np.diff(dg["value"]))
