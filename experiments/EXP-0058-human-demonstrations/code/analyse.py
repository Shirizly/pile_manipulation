"""EXP-0058: human demonstrations vs the EXP-0057 perfect-model oracle on the same tasks.

Per task and overall: solved?, pushes to solve (first k >= 1 with in-goal mass >= 0.9 x the
EXP-0046 optimum -- the EXP-0055 analyse.py completion rule), final in-goal mass rel. optimum,
mean think time (human only).  Oracle cells: EXP-0057 oracle_A_32x8 and oracle_A_default.

Usage:  python experiments/EXP-0058-human-demonstrations/code/analyse.py [operator ...]
        (default: every operator folder under results/)
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "experiments/EXP-0058-human-demonstrations"
OPT = {g: v.get("mass_frac_best_placement", 1.0) for g, v in
       json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text()).items()}
ORACLE = ["oracle_A_32x8", "oracle_A_default"]
THETA, MAX_PUSHES = 0.9, 20


def summarise(goal, mass_frac, think=None):
    rel = np.array(mass_frac[:MAX_PUSHES + 1]) / OPT.get(goal, 1.0)
    hit = np.nonzero(rel[1:] >= THETA)[0]
    k = int(hit[0]) + 1 if len(hit) else None
    return dict(solved=k is not None, k=k, final_rel=float(rel[-1]),
                think=float(np.mean(think)) if think else None)


def load_human(op):
    out = {}
    for p in sorted((HERE / "results" / op).glob("*_s*.json")):
        r = json.loads(p.read_text())
        s = summarise(r["goal"], r["mass_frac"], r["think_s"])
        s["finished"] = r.get("finished", False); s["n"] = len(r["actions"])
        out[(r["goal"], r["start"])] = s
    return out


def load_oracle(tag):
    d = json.loads((REPO / f"experiments/EXP-0057-oracle-budget-ablation/results/{tag}.json").read_text())
    return {(e["goal"], e["start"]): summarise(e["goal"], e["mass_frac"]) for e in d["episodes"]}


def fmt(s):
    if s is None:
        return f"{'-':>15s}"
    return f"{('k=%2d' % s['k']) if s['solved'] else '  no':>5s} rel {s['final_rel']:.2f}   "


def main():
    res = HERE / "results"
    ops = sys.argv[1:] or (sorted(p.name for p in res.iterdir() if p.is_dir()) if res.exists() else [])
    orc = {t: load_oracle(t) for t in ORACLE}
    for op in ops:
        hum = load_human(op)
        print(f"\n== operator {op}: {len(hum)} task files ==")
        print(f"{'goal':12s} {'s':>3s} | {'human':>15s} think  n  | " + " | ".join(f"{t:>15s}" for t in ORACLE))
        for (g, st), h in sorted(hum.items()):
            flag = "" if h["finished"] else "  (in progress)"
            print(f"{g:12s} {st:3d} | {fmt(h)} {h['think'] or 0:4.0f}s {h['n']:2d} | "
                  + " | ".join(fmt(orc[t].get((g, st))) for t in ORACLE) + flag)
        done = [k for k, h in hum.items() if h["finished"]]
        if not done:
            continue
        print("overall on the finished tasks (n=%d):" % len(done))
        rows = [("human", hum)] + [(t, orc[t]) for t in ORACLE]
        for name, S in rows:
            ss = [S[k] for k in done if k in S]
            ks = [s["k"] for s in ss if s["solved"]]
            th = [s["think"] for s in ss if s["think"] is not None]
            print(f"  {name:18s} solved {len(ks):2d}/{len(ss):2d}  median k {np.median(ks) if ks else float('nan'):4.1f}  "
                  f"mean final rel {np.mean([s['final_rel'] for s in ss]):.2f}"
                  + (f"  mean think {np.mean(th):.1f}s/push" if th else ""))


if __name__ == "__main__":
    main()
