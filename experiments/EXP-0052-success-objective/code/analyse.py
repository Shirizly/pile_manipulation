"""EXP-0052: success-objective cells (mass_w1, mass_w3) vs EXP-0051 lyapunov-only control,
paired by (planner, goal, start) for nfd_residual_worldframe_noaug_ep43. -> results/analysis.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask
from simple_mpc.adapters import occ_for_scoring
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
M = "nfd_residual_worldframe_noaug_ep43"
def load(p, cell=None):
    out = {}
    for e in json.loads(Path(p).read_text())["episodes"]:
        if e.get("complete") and e["model"] == M and (cell is None or e["cell"] == cell):
            S = torch.tensor(e["states"]).float(); occ = occ_for_scoring(S)
            m = torch.from_numpy(goal_mask(e["goal"])).float().reshape(1, -1)
            f = occ.reshape(len(S), -1); mf = ((f * m).sum(1) / f.sum(1)).numpy()
            v = np.array(e["values"]); vs = VS[e["goal"]]
            out[(e["planner"], e["goal"], e["start"])] = dict(mf=mf, rel=mf / vs["mass_frac_best_placement"],
                ach=(v[0] - v) / (v[0] - vs["vstar_map_value"]), plan=np.array(e["plan_time_s"]))
    return out
ctrl = load(REPO / "experiments/EXP-0051-task-success-completion-time/results/success_states.json")
res = {}
rng = np.random.default_rng(0)
for cell in ("mass_w1", "mass_w3"):
    X = load(REPO / "experiments/EXP-0052-success-objective/results/objective_states.json", cell)
    for pl in ("gd", "cem"):
        for grp, goals in (("letters", ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z"]), ("two_squares", ["two_squares"])):
            ks = [k for k in X if k[0] == pl and k[1] in goals and k in ctrl]
            r = {}
            for kk in (8, 24):
                d = np.array([X[k]["mf"][kk] - ctrl[k]["mf"][kk] for k in ks])
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
                r[f"mass_k{kk}"] = dict(obj=float(np.mean([X[k]["mf"][kk] for k in ks])), ctrl=float(np.mean([ctrl[k]["mf"][kk] for k in ks])),
                                        diff=float(d.mean()), ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
                r[f"ach_k{kk}"] = dict(obj=float(np.mean([X[k]["ach"][kk] for k in ks])), ctrl=float(np.mean([ctrl[k]["ach"][kk] for k in ks])))
            for th in (0.8, 0.9):
                r[f"complete_{th}"] = dict(obj=float(np.mean([(X[k]["rel"][1:] >= th).any() for k in ks])),
                                           ctrl=float(np.mean([(ctrl[k]["rel"][1:] >= th).any() for k in ks])))
            r["n"] = len(ks)
            res[f"{cell}/{pl}/{grp}"] = r
            print(f"{cell:8s} {pl:4s} {grp:11s} n={len(ks):2d}: in-goal mass k8 {r['mass_k8']['obj']:.2f} vs {r['mass_k8']['ctrl']:.2f} "
                  f"({r['mass_k8']['diff']:+.3f}); k24 {r['mass_k24']['obj']:.2f} vs {r['mass_k24']['ctrl']:.2f} ({r['mass_k24']['diff']:+.3f} "
                  f"[{r['mass_k24']['ci'][0]:+.3f},{r['mass_k24']['ci'][1]:+.3f}]); lyap ach k24 {r['ach_k24']['obj']:.2f} vs {r['ach_k24']['ctrl']:.2f}; "
                  f"complete@0.8 {r['complete_0.8']['obj']:.2f} vs {r['complete_0.8']['ctrl']:.2f}, @0.9 {r['complete_0.9']['obj']:.2f} vs {r['complete_0.9']['ctrl']:.2f}")
Path(REPO / "experiments/EXP-0052-success-objective/results/analysis.json").write_text(json.dumps(res, indent=1))
