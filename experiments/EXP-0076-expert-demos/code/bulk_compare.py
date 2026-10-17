"""Bulk-phase comparison over all n20 scenarios: ens (f20_all) vs v64 4x budget (F1_v64_all), paired per scenario.
Time / pushes to the first state with mass >= 0.8 (censored at the whole episode if never reached), per stratum; plus whole-episode outcome."""
import json, numpy as np
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
A = json.load(open(EXP / "results/runs/f20_all.json"))["scenarios"]; B = json.load(open(EXP / "results/runs/F1_v64_all.json"))["scenarios"]
ids = sorted(i for i in A if i in B and B[i].get("complete")); rng = np.random.default_rng(0)


def to(r, thr):
    m = np.array(r["masses"]); t = np.concatenate([[0], np.cumsum(r["push_times"])]); h = np.nonzero(m >= thr)[0]
    return (float(t[h[0]]), int(h[0]), True) if len(h) else (float(t[-1]), len(m) - 1, False)


def boot(d):
    d = np.asarray(d, float); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return f"{d.mean():+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"


L = [f"# bulk phase: v64 (F1_v64_all) - ens (f20_all), {len(ids)} paired n20 scenarios", ""]
for thr in (0.8, 0.95):
    X = [(to(A[i], thr), to(B[i], thr)) for i in ids]
    both = [(a, b) for a, b in X if a[2] and b[2]]
    L.append(f"- to mass {thr}: reached ens {sum(a[2] for a, _ in X)} / v64 {sum(b[2] for _, b in X)}; censored time {boot([b[0] - a[0] for a, b in X])} s; "
             f"on both-reached (n {len(both)}) time {boot([b[0] - a[0] for a, b in both])} s, pushes {np.mean([b[1] - a[1] for a, b in both]):+.2f}")
L += ["", "| stratum | n | to 0.8: v64 - ens censored time | to 0.95: solved ens / v64 |", "|---|---|---|---|"]
for st in sorted(set((A[i]["init"], A[i]["kind"]) for i in ids)):
    s = [i for i in ids if (A[i]["init"], A[i]["kind"]) == st]
    L.append(f"| {st[0]} {st[1]} | {len(s)} | {boot([to(B[i], 0.8)[0] - to(A[i], 0.8)[0] for i in s])} | {sum(A[i]['solved'] for i in s)} / {sum(B[i]['solved'] for i in s)} |")
(EXP / "results/bulk_compare.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
