"""EXP-0076 tail-switch analysis: every tail start (recorded state where an episode first reached mass 0.8) is continued by the ens and the v64
planner (2 planner seeds each, <= 8 pushes). Paired over tail starts: solved rate, time to 0.95 on starts both solve, censored time
(unsolved = time of all 8 pushes), split by the model that ran the bulk phase. Seed-to-seed differences of the SAME model = noise floor.
usage: python tail_switch_analyse.py -> results/tail_switch.md"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; EXP = HERE.parent
S = {s["id"]: s for s in json.load(open(EXP / "results/tail_scenarios.json"))["scenarios"]}
RUNS = {(m, sd): EXP / f"results/runs/TS_{m}_s{sd}.json" for m in ("ens", "v64") for sd in (0, 1)}
R = {k: json.load(open(p))["scenarios"] for k, p in RUNS.items() if p.exists()}
ids = sorted(set.intersection(*[set(i for i, r in v.items() if r.get("complete")) for v in R.values()])) if R else []
rng = np.random.default_rng(0)


def stat(r):
    return bool(r["solved"]), float(r["time_s"]), int(r["pushes"])


def boot(d, n=4000):
    d = np.asarray(d, float)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


L = [f"# Tail-switch experiment ({len(ids)} tail starts complete in every run; runs present: {sorted(R)})", ""]
for bulk in ("ens", "v64", "all"):
    sel = [i for i in ids if bulk == "all" or S[i]["bulk_model"] == bulk]
    if not sel:
        continue
    L += [f"## bulk phase by {bulk} ({len(sel)} tail starts; entry mass {np.mean([S[i]['mass0'] for i in sel]):.3f})", "",
          "| tail model, seed | solved | time to 0.95 on solved | censored time (unsolved = all 8 pushes) |", "|---|---|---|---|"]
    for k in sorted(R):
        st = [stat(R[k][i]) for i in sel]
        L.append(f"| {k[0]} s{k[1]} | {sum(s for s, _, _ in st)}/{len(st)} | {np.mean([t for s, t, _ in st if s]):.2f} | {np.mean([t for _, t, _ in st]):.2f} |")
    L += ["", "paired differences (b - a) over tail starts: solved-rate difference, censored time, time on starts both solve [95 % bootstrap CI]"]
    pairs = [(("ens", 0), ("ens", 1), "noise floor ens"), (("v64", 0), ("v64", 1), "noise floor v64"),
             (("ens", 0), ("v64", 0), "v64 - ens seed 0"), (("ens", 1), ("v64", 1), "v64 - ens seed 1")]
    for a, b, nm in pairs:
        if a not in R or b not in R:
            continue
        A = [stat(R[a][i]) for i in sel]; Bs = [stat(R[b][i]) for i in sel]
        ds = boot([int(y[0]) - int(x[0]) for x, y in zip(A, Bs)]); dc = boot([y[1] - x[1] for x, y in zip(A, Bs)])
        both = [(x, y) for x, y in zip(A, Bs) if x[0] and y[0]]; dt = boot([y[1] - x[1] for x, y in both])
        L.append(f"- {nm}: solved {ds[0]:+.3f} [{ds[1]:+.3f}, {ds[2]:+.3f}]; censored time {dc[0]:+.2f} s [{dc[1]:+.2f}, {dc[2]:+.2f}]; "
                 f"time on both-solved (n {len(both)}) {dt[0]:+.2f} s [{dt[1]:+.2f}, {dt[2]:+.2f}]")
    if all(k in R for k in RUNS):          # seed-averaged per start
        e = np.array([[stat(R[("ens", sd)][i])[1] for sd in (0, 1)] for i in sel]).mean(1); v = np.array([[stat(R[("v64", sd)][i])[1] for sd in (0, 1)] for i in sel]).mean(1)
        es = np.array([[stat(R[("ens", sd)][i])[0] for sd in (0, 1)] for i in sel]).mean(1); vs = np.array([[stat(R[("v64", sd)][i])[0] for sd in (0, 1)] for i in sel]).mean(1)
        d1, d2 = boot(vs - es), boot(v - e)
        L.append(f"- seed-averaged v64 - ens: solved {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]; censored time {d2[0]:+.2f} s [{d2[1]:+.2f}, {d2[2]:+.2f}]")
    L.append("")
(EXP / "results/tail_switch.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
