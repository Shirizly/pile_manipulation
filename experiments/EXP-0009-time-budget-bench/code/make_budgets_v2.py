"""make_budgets_v2.py -- RUN-0002 budget SWEEP: N_i(T) for T in a range of
reference budgets (not just the original single 1.714ms reference), using
the corrected (sync-free, value-timed) end2end curve at K in {1,32,128}.
Reports N_i both WITHOUT and WITH the value function timed in, capped at
128, for T in {T_ref (original), 2.5, 5, 10, 25, 50} ms.
"""
import json

with open("experiments/EXP-0009-time-budget-bench/artifacts/RUN-0002-timing-corrected/results_timing_v2.json") as f:
    payload = json.load(f)

results = payload["results"]
T_REF = 1.7144269986602012  # RUN-0001's original reference (model0001_global @ K=128, RUN-0001 numbers)
BUDGETS_MS = [T_REF, 2.5, 5, 10, 25, 50]
CAP = 128


def curve(r, key):
    xs = sorted(int(k) for k in r["by_k"])
    ys = [r["by_k"][str(k)][key]["median"] for k in xs]
    return xs, ys


def interp_n(xs, ys, t):
    if ys[0] > t:
        return None
    for i in range(len(xs) - 1):
        if ys[i] <= t <= ys[i + 1]:
            if ys[i + 1] == ys[i]:
                n = xs[i + 1]
            else:
                frac = (t - ys[i]) / (ys[i + 1] - ys[i])
                n = xs[i] + frac * (xs[i + 1] - xs[i])
            return min(n, CAP)
        if ys[i] <= t and ys[i + 1] <= t:
            continue
    if ys[-1] <= t:
        return float(CAP)  # flat/still under budget at largest measured K -- capped, not extrapolated
    return None


out = {"budgets_ms_swept": BUDGETS_MS, "cap": CAP,
       "note": "N_i(T) interpolated on RUN-0002's corrected (sync-free) "
               "end2end curve at K in {1,32,128} -- coarser interpolation "
               "than RUN-0001's 11-point curve; sufficient to see where "
               "the comparison is/isn't degenerate. 'without_value' uses "
               "end2end_ms; 'with_value' uses end2end_with_value_ms "
               "(lyapunov+mass_in_region both computed).",
       "per_model": {}}

for name, r in results.items():
    xs_v, ys_v = curve(r, "end2end_ms")
    xs_wv, ys_wv = curve(r, "end2end_with_value_ms")
    row = {"without_value": {}, "with_value": {}}
    for t in BUDGETS_MS:
        n = interp_n(xs_v, ys_v, t)
        row["without_value"][f"{t:g}ms"] = round(n, 1) if n is not None else None
        n2 = interp_n(xs_wv, ys_wv, t)
        row["with_value"][f"{t:g}ms"] = round(n2, 1) if n2 is not None else None
    out["per_model"][name] = row

with open("experiments/EXP-0009-time-budget-bench/results/budgets.json") as f:
    old = json.load(f)
out["run0001_reference_budgets"] = old

with open("experiments/EXP-0009-time-budget-bench/results/budgets.json", "w") as f:
    json.dump(out, f, indent=2)

for name, row in out["per_model"].items():
    print(f"{name:28s} " + " ".join(f"{t:g}ms:{row['without_value'][f'{t:g}ms']}"
                                     for t in BUDGETS_MS))
print("\nwrote experiments/EXP-0009-time-budget-bench/results/budgets.json")
