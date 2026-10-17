"""EXP-0070: summarise chain_probe_chain.json (Genesis ground truth, no model). Per visited plate-down
state: best free vs best chained push under (i) dV alone, (ii) progress rate -dV / (t_plan + t), (iii)
J = t + etc(V+dV) - etc(V); the share of states where the best action under each criterion is chained;
paired differences. Rotation-speed sensitivity: t_rot recomputed at omega_rot in {pi/2, pi, 2 pi} rad/s.
The free set includes the action the lyap planner actually executed (#0), so 'best free' >= planner.
-> results/chain_analyse.json"""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_time import trap_time

HERE = Path(__file__).resolve().parents[1]
r = json.loads((HERE / "results/chain_probe_chain.json").read_text())
etcm = json.loads((HERE / "results/etc_map_v0.json").read_text())
tp = r["t_plan"]
out = {}
for w in (math.pi / 2, math.pi, 2 * math.pi):
    rows = []
    for s in r["states"]:
        f = etcm[s["goal"]]; f = f["pass2"] if f["n_completed"] < 0.5 * f["n_episodes"] else f["pass1"]
        etc = lambda V: f["power"]["c"] * np.clip(V, 1e-6, None) ** f["power"]["p"]
        F, C = s["free"], s["chain"]
        dVf, dVc = np.array(F["dV"]), np.array(C["dV"])
        tf = np.array(F["t"])
        tc = trap_time(torch.tensor(np.abs(np.array(C["dphi_deg"])) * math.pi / 180, dtype=torch.float64), w, 20.0).numpy() \
            + trap_time(torch.tensor(C["L"], dtype=torch.float64)).numpy()
        V0 = s["V0"]
        Jf, Jc = tf + etc(V0 + dVf) - etc(V0), tc + etc(V0 + dVc) - etc(V0)
        rf, rc = -dVf / (tp + tf), -dVc / (tp + tc)
        rows.append(dict(goal=s["goal"], best_dV_free_random=dVf[1:].min(), best_rate_free_random=rf[1:].max(),
                         best_J_free_random=Jf[1:].min(), best_dV_free=dVf.min(), best_dV_chain=dVc.min(), planner_dV=dVf[0],
                         best_rate_free=rf.max(), best_rate_chain=rc.max(), planner_rate=rf[0],
                         best_J_free=Jf.min(), best_J_chain=Jc.min(), planner_J=Jf[0],
                         t_best_rate_free=tf[rf.argmax()], t_best_rate_chain=tc[rc.argmax()],
                         dphi_best_rate_chain=C["dphi_deg"][int(rc.argmax())], L_best_rate_chain=C["L"][int(rc.argmax())]))
    R = {k: np.array([x[k] for x in rows]) for k in rows[0] if k != "goal"}
    out[f"omega_{w:.2f}"] = dict(
        n_states=len(rows),
        chain_wins_dV=float(np.mean(R["best_dV_chain"] < R["best_dV_free"])),
        chain_wins_rate=float(np.mean(R["best_rate_chain"] > R["best_rate_free"])),
        chain_wins_J=float(np.mean(R["best_J_chain"] < R["best_J_free"])),
        chain_beats_planner_rate=float(np.mean(R["best_rate_chain"] > R["planner_rate"])),
        chain_wins_vs_random_free_dV=float(np.mean(R["best_dV_chain"] < R["best_dV_free_random"])),
        chain_wins_vs_random_free_rate=float(np.mean(R["best_rate_chain"] > R["best_rate_free_random"])),
        chain_wins_vs_random_free_J=float(np.mean(R["best_J_chain"] < R["best_J_free_random"])),
        frac_chain_improving=float(np.mean(R["best_dV_chain"] < 0)),
        mean=dict({k: float(v.mean()) for k, v in R.items()}),
        median_rate_ratio_chain_over_free=float(np.median(R["best_rate_chain"] / np.maximum(R["best_rate_free"], 1e-9))),
        best_rate_chain_dphi_hist={str(k): int(v) for k, v in zip(*np.unique(R["dphi_best_rate_chain"], return_counts=True))},
        best_rate_chain_L_hist={str(k): int(v) for k, v in zip(*np.unique(R["L_best_rate_chain"], return_counts=True))})
p = HERE / "results/chain_analyse.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
print(json.dumps(out, indent=1))
