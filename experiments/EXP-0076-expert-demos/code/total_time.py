"""Recompute the bulk (270 n20) and tail-switch (96 starts x 2 seeds) comparisons with TOTAL time = physical push time + planning time
per decision. Two planning-time variants: 'rec' = recorded plan_s of each decision (wall clock of the proposal step on this GPU; some ens
decisions in f20_all were GPU-contended), 'fix' = a constant per-decision cost = that run's median plan_s. Bank sampling (Genesis, equal for
both models) is not in plan_s. -> results/total_time.md"""
import json, numpy as np
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; rng = np.random.default_rng(0)
J = lambda t: json.load(open(EXP / f"results/runs/{t}.json"))["scenarios"]


def curve(r, mode, med):
    p = np.array(r["plan_s"]) if mode == "rec" else np.full(len(r["push_times"]), med)
    return np.concatenate([[0], np.cumsum(np.array(r["push_times"]) + p)]), np.concatenate([[0], np.cumsum(r["push_times"])]), np.array(r["masses"])


def to(r, thr, mode, med):
    tt, tp, m = curve(r, mode, med); h = np.nonzero(m >= thr)[0]
    return (tt[h[0]], tp[h[0]], True) if len(h) else (tt[-1], tp[-1], False)


def boot(d):
    d = np.asarray(d, float); b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return f"{d.mean():+.2f} [{np.percentile(b, 2.5):+.2f}, {np.percentile(b, 97.5):+.2f}]"


def med(R):
    return float(np.median([x for r in R.values() for x in r["plan_s"]]))


L = ["# Total time (physical + planning) comparisons, v64 - ens (positive = v64 slower)", ""]
A, B = J("f20_all"), J("F1_v64_all"); mA, mB = med(A), med(B); ids = sorted(i for i in A if i in B)
L += [f"## Bulk / whole episode, {len(ids)} paired n20 scenarios (ens f20_all median plan {mA:.2f} s / decision, v64 F1_v64_all {mB:.2f} s)", "",
      "| target | time | v64 - ens censored [95 % CI] | mean ens | mean v64 |", "|---|---|---|---|---|"]
for thr in (0.8, 0.95):
    for mode, lab in (("phys", "physical only"), ("rec", "physical + recorded planning"), ("fix", "physical + median planning")):
        k = 1 if mode == "phys" else 0; mm = "rec" if mode == "phys" else mode
        a = [to(A[i], thr, mm, mA)[k] for i in ids]; b = [to(B[i], thr, mm, mB)[k] for i in ids]
        L.append(f"| mass {thr} | {lab} | {boot(np.array(b) - np.array(a))} s | {np.mean(a):.2f} | {np.mean(b):.2f} |")
S = {s["id"]: s for s in json.load(open(EXP / "results/tail_scenarios.json"))["scenarios"]}
T = {(m, sd): J(f"TS_{m}_s{sd}") for m in ("ens", "v64") for sd in (0, 1)}; M = {k: med(v) for k, v in T.items()}
tids = sorted(set.intersection(*[set(v) for v in T.values()]))
L += ["", f"## Tail switch: {len(tids)} recorded 0.8-mass states, each finished by ens and by v64 (2 seeds, <= 8 pushes; censored = all 8 pushes)",
      f"median plan s / decision: " + ", ".join(f"{k[0]} s{k[1]} {v:.2f}" for k, v in M.items()), "",
      "| bulk states from | time | v64 - ens censored, seed-averaged [95 % CI] | mean ens | mean v64 | ens seed floor |", "|---|---|---|---|---|---|"]
for bulk in ("ens", "v64", "all"):
    sel = [i for i in tids if bulk == "all" or S[i]["bulk_model"] == bulk]
    for mode, lab in (("phys", "physical only"), ("rec", "physical + recorded planning"), ("fix", "physical + median planning")):
        k = 1 if mode == "phys" else 0; mm = "rec" if mode == "phys" else mode
        v = {key: np.array([to(T[key][i], 0.95, mm, M[key])[k] for i in sel]) for key in T}
        e = (v[("ens", 0)] + v[("ens", 1)]) / 2; w = (v[("v64", 0)] + v[("v64", 1)]) / 2
        L.append(f"| {bulk} (n {len(sel)}) | {lab} | {boot(w - e)} s | {e.mean():.2f} | {w.mean():.2f} | {boot(v[('ens', 1)] - v[('ens', 0)])} |")
L += ["", "## Composite pipelines (approximate): mean total time = bulk part (270 scenarios, start -> 0.8, physical + median planning)",
      "+ tail part (tail-switch, states from that bulk model, physical + median planning, censored). Tail starts from v64 bulk come from 36 scenarios only.", "",
      "| bulk model | tail model | bulk part s | tail part s | total s |", "|---|---|---|---|---|"]
bulk_part = {"ens": np.mean([to(A[i], 0.8, "fix", mA)[0] for i in ids]), "v64": np.mean([to(B[i], 0.8, "fix", mB)[0] for i in ids])}
for bm in ("ens", "v64"):
    sel = [i for i in tids if S[i]["bulk_model"] == bm]
    for tm in ("ens", "v64"):
        tp = np.mean([(to(T[(tm, 0)][i], 0.95, "fix", M[(tm, 0)])[0] + to(T[(tm, 1)][i], 0.95, "fix", M[(tm, 1)])[0]) / 2 for i in sel])
        L.append(f"| {bm} | {tm} | {bulk_part[bm]:.2f} | {tp:.2f} | {bulk_part[bm] + tp:.2f} |")
(EXP / "results/total_time.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


# ---- censoring check: only episodes / tail starts that BOTH models solve (censoring favours the model that fails more)
L2 = ["", "## Censoring check: pairs where both models reach the target (total = physical + median planning)", "",
      "| comparison | n both | v64 - ens physical | v64 - ens total | solved ens / v64 |", "|---|---|---|---|---|"]
for thr in (0.8, 0.95):
    X = [(to(A[i], thr, "fix", mA), to(B[i], thr, "fix", mB)) for i in ids]; both = [(a, b) for a, b in X if a[2] and b[2]]
    L2.append(f"| bulk/episode to {thr} (270) | {len(both)} | {boot([b[1] - a[1] for a, b in both])} | {boot([b[0] - a[0] for a, b in both])} | "
              f"{sum(a[2] for a, _ in X)} / {sum(b[2] for _, b in X)} |")
for sd in (0, 1):
    X = [(to(T[('ens', sd)][i], 0.95, "fix", M[('ens', sd)]), to(T[('v64', sd)][i], 0.95, "fix", M[('v64', sd)])) for i in tids]
    both = [(a, b) for a, b in X if a[2] and b[2]]
    L2.append(f"| tail seed {sd} (96) | {len(both)} | {boot([b[1] - a[1] for a, b in both])} | {boot([b[0] - a[0] for a, b in both])} | "
              f"{sum(a[2] for a, _ in X)} / {sum(b[2] for _, b in X)} |")
(EXP / "results/total_time.md").write_text((EXP / "results/total_time.md").read_text() + "\n".join(L2) + "\n"); print("\n".join(L2))
