"""make_budgets.py -- derive per-model candidate budgets N_i from
results_timing.json: N_i = number of candidates model i can evaluate
(end-to-end, i.e. pre+fwd) in the time the FASTEST model (by its own
end-to-end median at K=128) takes to do K=128. Interpolated linearly on
the measured (K, end2end_ms median) curve; if the model's cheapest
measured point (K=1) already exceeds the reference time, N_i is reported
as < 1 (clamped to 1 candidate/step floor is NOT assumed -- reported
honestly as "cannot clear the budget for even one candidate").
"""
import json

with open("experiments/EXP-0009-time-budget-bench/artifacts/RUN-0001-timing-sweep/results_timing.json") as f:
    payload = json.load(f)

results = payload["results"]
ks = sorted(payload["ks"])

# reference: fastest model's end2end median at K=128
K_REF = 128
fastest_name, fastest_t = min(
    ((name, r["by_k"][str(K_REF)]["end2end_ms"]["median"]) for name, r in results.items()),
    key=lambda x: x[1])
print(f"fastest model @ K={K_REF}: {fastest_name} = {fastest_t:.4f} ms  <- REFERENCE BUDGET")

def curve(r):
    xs = sorted(int(k) for k in r["by_k"])
    ys = [r["by_k"][str(k)]["end2end_ms"]["median"] for k in xs]
    return xs, ys

def interp_n(xs, ys, t):
    """Largest K (real-valued, linearly interpolated) s.t. end2end(K) <= t.
    ys is expected to be roughly increasing in K (may be locally flat/noisy
    at small K); we walk from the smallest K upward and find the crossing."""
    if ys[0] > t:
        return None  # can't clear budget even at the smallest measured K
    for i in range(len(xs) - 1):
        if ys[i] <= t <= ys[i + 1]:
            if ys[i + 1] == ys[i]:
                return float(xs[i + 1])
            frac = (t - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + frac * (xs[i + 1] - xs[i])
        if ys[i] <= t and ys[i + 1] <= t:
            continue
    if ys[-1] <= t:
        return float(xs[-1])
    return None

budgets = {"reference_model": fastest_name, "reference_time_ms": fastest_t,
           "reference_k": K_REF, "method": "linear interpolation on measured "
           "(K, end2end_ms median) curve; N_i = largest K at/under the "
           "reference model's own K=128 end2end median time",
           "budgets": {}}

for name, r in results.items():
    xs, ys = curve(r)
    n = interp_n(xs, ys, fastest_t)
    if n is None:
        budgets["budgets"][name] = {
            "N_i": None,
            "note": f"even K={xs[0]} costs {ys[0]:.3f}ms > reference {fastest_t:.3f}ms; "
                    f"cannot clear the budget for even one candidate at measured resolution"}
        print(f"  {name:28s} N_i = <{xs[0]}  (K={xs[0]} costs {ys[0]:.3f}ms > budget)")
    else:
        budgets["budgets"][name] = {"N_i": round(n, 1), "N_i_floor": int(n)}
        print(f"  {name:28s} N_i = {n:.1f}")

with open("experiments/EXP-0009-time-budget-bench/results/budgets.json", "w") as f:
    json.dump(budgets, f, indent=2)
print("\nwrote experiments/EXP-0009-time-budget-bench/results/budgets.json")
