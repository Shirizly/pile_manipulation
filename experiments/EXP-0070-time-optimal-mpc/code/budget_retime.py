"""EXP-0070: re-time EXP-0045 (planning budget 0.03-1 s per push, 24 pushes, 8 goals x 4 starts, GD/CEM,
2 models; no states recorded, so completion is LYAPUNOV-based: first k with achieved fraction
(V0 - Vk)/(V0 - V*) >= 0.9, V* from EXP-0046) under the physical clock (plan_time_s + action_time),
censored at 24 pushes. Not legal-action data (pre-ISS-013). -> results/budget_retime.json"""
import json, os, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_time import episode_action_times
d = json.loads((REPO / "experiments/EXP-0045-decision-time-vs-actions/results/time_vs_actions.json").read_text())
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
out = {}
for e in d["episodes"]:
    v = np.array(e["values"]); vs = VS.get(e["goal"], {}).get("vstar_map_value", 0.0)
    ach = (v[0] - v) / max(v[0] - vs, 1e-9); hit = np.nonzero(ach[1:] >= 0.9)[0]
    K = len(e["actions"]); k = int(hit[0]) + 1 if len(hit) else K
    t = np.array(episode_action_times(e["actions"])["total"]); p = np.array(e["plan_time_s"])
    fam = "quadrant" if e["goal"].startswith("quadrant") else "letter"
    for key in (f"{e['model'][:12]}|{e['planner']}|{e['budget_s']}|{fam}", f"ALL|{e['planner']}|{e['budget_s']}|ALL"):
        o = out.setdefault(key, dict(done=[], pushes=[], time=[], exec=[], plan=[], flat2=[]))
        o["done"].append(len(hit) > 0); o["pushes"].append(k); o["time"].append(float(p[:k].sum() + t[:k].sum()))
        o["exec"].append(float(t[:k].sum())); o["plan"].append(float(p[:k].sum())); o["flat2"].append(float(p[:k].sum() + 2.0 * k))
res = {k: {m: float(np.mean(v)) for m, v in o.items()} | {"n": len(o["done"])} for k, o in sorted(out.items())}
p = REPO / "experiments/EXP-0070-time-optimal-mpc/results/budget_retime.json"; tmp = Path(str(p) + ".tmp")
tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, p)
for k, v in res.items():
    print(f"{k:40s} n{v['n']:3d} done {v['done']:.2f} pushes {v['pushes']:5.1f} time {v['time']:5.1f} (exec {v['exec']:4.1f} plan {v['plan']:4.1f}) old-clock {v['flat2']:5.1f}")
