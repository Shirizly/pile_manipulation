"""EXP-0070: in recorded (lyapunov-planned) legal episodes, is a push's physical time related to its
progress? Per push: t_act (+ transit, sweep parts) vs -dV (true lyapunov drop) and d(mass_rel), Spearman,
per goal family; and progress per second of (plan + t_act). Pushes after completion are excluded.
-> results/time_vs_progress.json"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
HERE = Path(__file__).resolve().parents[1]
rows = json.loads((HERE / "results/retro_episodes.json").read_text())
LEGAL = ("EXP-0051_legal", "EXP-0054_legal", "EXP-0066")
out = {}
for fam in ("quadrant", "two_squares", "letter"):
    T, TR, SW, DV, DM, PL = [], [], [], [], [], []
    for r in rows:
        if r["fam"] != fam or not r["src"].startswith(LEGAL):
            continue
        V = np.array(r["V"]); M = np.array(r["mass_rel"]); last = r["kstar"] or len(r["plan"])
        for j in range(last):
            T.append(r["t_act"]["total"][j]); TR.append(r["t_act"]["transit"][j]); SW.append(r["t_act"]["sweep"][j])
            DV.append(V[j] - V[j + 1]); DM.append(M[j + 1] - M[j]); PL.append(r["plan"][j])
    T, TR, SW, DV, DM, PL = map(np.array, (T, TR, SW, DV, DM, PL))
    if len(T) < 20:
        continue
    q = np.quantile(T, [0, .25, .5, .75, 1])
    out[fam] = dict(n=int(len(T)),
                    rho_t_dV=float(spearmanr(T, DV).correlation), rho_transit_dV=float(spearmanr(TR, DV).correlation),
                    rho_sweep_dV=float(spearmanr(SW, DV).correlation), rho_t_dmass=float(spearmanr(T, DM).correlation),
                    by_time_quartile=[dict(t=float(T[(T >= a) & (T <= b)].mean()), dV=float(DV[(T >= a) & (T <= b)].mean()),
                                           dmass=float(DM[(T >= a) & (T <= b)].mean()),
                                           dV_per_s=float(DV[(T >= a) & (T <= b)].sum() / (T + PL)[(T >= a) & (T <= b)].sum()))
                                      for a, b in zip(q[:-1], q[1:])])
    print(fam, json.dumps(out[fam]))
p = HERE / "results/time_vs_progress.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
