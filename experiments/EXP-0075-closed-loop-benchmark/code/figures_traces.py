"""predicted terminal value of the BEST sequence found so far, over the optimisation process, for each intensive method (results/intensive.json 'traces')."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; F = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures"
d = json.load(open(R / "intensive.json")); tr = d["traces"]; ceil = d["gd"]["cost"]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for n, c in (("ref", "tab:blue"), ("random", "tab:orange"), ("cem10k", "tab:green"), ("gd", "tab:brown")):
    t = np.array(tr[n]); x, y = t[:, 0], t[:, 1]
    if n == "gd": x = x + 0  # GD evaluations counted as (sequences in the batch) x (steps); it starts from the best of the 24 initial sequences (themselves taken from the CEM-10k run + the beam)
    ax[0].plot(x, y, "o-" if len(x) < 30 else "-", color=c, ms=3, label=f"{n}: final {y[-1]:+.3f} ({int(x[-1]):,} evals)")
for n, c, m in (("greedy", "tab:red", "s"), ("beam", "tab:purple", "D")):
    st = np.array(tr[n + "_steps"]); ax[0].plot(st[-1, 0], st[-1, 1], m, color=c, ms=9, label=f"{n}: final {st[-1,1]:+.3f} ({int(st[-1,0]):,} evals)")
    ax[1].plot(st[:, 2], st[:, 1], m + "-", color=c, label=f"{n} (best value after j pushes)")
ax[0].axhline(ceil, color="k", ls=":", lw=1); ax[0].text(300, ceil - 0.012, f"best found (GD) {ceil:+.3f}", fontsize=8); ax[0].axhline(0.9 * ceil, color="gray", ls=":", lw=1); ax[0].text(300, 0.9 * ceil + 0.005, "90 % of it", fontsize=8, color="gray")
ax[0].set_xscale("log"); ax[0].set_xlabel("sequence evaluations (log)"); ax[0].set_ylabel("predicted terminal value change of the best sequence so far (lower = better)"); ax[0].legend(fontsize=8); ax[0].set_title("optimisation progress, letter_T_w20 start 40, H = 4")
ax[1].set_xlabel("push j (greedy / beam construction)"); ax[1].set_ylabel("best predicted value after j pushes"); ax[1].set_xticks([1, 2, 3, 4]); ax[1].legend(fontsize=8); ax[1].set_title("greedy / beam: value of the best prefix")
fig.tight_layout(); fig.savefig(F / "intensive_traces.png", dpi=90)
need = {}
for n in ("ref", "random", "cem10k", "gd"):
    t = np.array(tr[n]); hit = np.nonzero(t[:, 1] <= 0.9 * ceil)[0]; need[n] = int(t[hit[0], 0]) if len(hit) else None
print("evaluations to reach 90 % of the best found:", need)
json.dump(need, open(R / "intensive_evals_to_90pct.json", "w"))
