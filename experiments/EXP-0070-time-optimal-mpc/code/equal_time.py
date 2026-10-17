"""EXP-0070: censoring-free comparison -- in-goal mass rel reached by a given WALL time (planning + physical
action time, new clock), per cell, paired with lyap by (goal, start). mass at time T = mass after the last
push completed by T (step function); episodes shorter than T hold their last value.
Usage: equal_time.py TAG [BASE_TAG]  (BASE_TAG: take the lyap rows from another results file, e.g. v1 vs v0b)"""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)
tag = sys.argv[1]; base_tag = sys.argv[2] if len(sys.argv) > 2 else tag
rows = json.loads((HERE / f"results/{tag}_episodes.json").read_text())
base_rows = json.loads((HERE / f"results/{base_tag}_episodes.json").read_text())
TS = (4.0, 8.0, 12.0, 16.0, 20.0)


def m_at(r, T):
    t = np.cumsum(np.array(r["plan"]) + np.array(r["t_act"]["total"]))
    k = int(np.searchsorted(t, T, side="right"))          # pushes finished by T
    return r["mass_rel"][k]


base = {(r["goal"], r["start"]): r for r in base_rows if r["cell"] == "lyap"}
out = {}
for c in sorted({r["cell"] for r in rows} - {"lyap"}):
    for fam in ("ALL", "quadrant", "two_squares", "letter"):
        P = [(r, base[(r["goal"], r["start"])]) for r in rows if r["cell"] == c and (fam == "ALL" or r["fam"] == fam)
             and (r["goal"], r["start"]) in base]
        if not P:
            continue
        res = {}
        for T in TS:
            d = np.array([m_at(a, T) - m_at(b, T) for a, b in P])
            bs = rng.choice(d, (4000, len(d))).mean(1)
            res[str(T)] = dict(mean=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), n=len(d),
                               base=float(np.mean([m_at(b, T) for _, b in P])))
        out[f"{c}|{fam}"] = res
        print(f"{c:12s} {fam:12s} " + "  ".join(f"T{T:g}: {v['base']:.2f}{v['mean']:+.3f}[{v['lo']:+.3f},{v['hi']:+.3f}]" for T, v in zip(TS, res.values())))
p = HERE / f"results/equal_time_{tag}{'' if base_tag == tag else '_vs_' + base_tag}.json"
tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
