"""EXP-0042 pilot analysis: per-cell mean capture (paired over (state, goal) trials),
differences vs the planner's 1 s default with state-bootstrap CIs, the repeat-cell noise
floor, and evaluations per decision. -> results/analysis_<tag>.json"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
d = json.loads((HERE / f"results/trials_{tag}.json").read_text())
states = sorted(d["states"], key=int); cells = d["cells"]
goals = sorted({t["goal"] for t in d["states"][states[0]]["trials"]})
C = np.full((len(cells), len(states), len(goals)), np.nan); N = np.full_like(C, np.nan); T = np.full_like(C, np.nan)
for si, s in enumerate(states):
    for t in d["states"][s]["trials"]:
        ci, gi = cells.index(t["cell"]), goals.index(t["goal"])
        C[ci, si, gi] = t["capture"] if t["capture"] is not None else np.nan
        N[ci, si, gi] = t["n_evals"]; T[ci, si, gi] = t["time_s"]
rng = np.random.default_rng(0)
def boot(x):  # x (S, G) paired differences -> state bootstrap of the mean
    m = np.nanmean(x, 1); b = [np.nanmean(m[rng.integers(0, len(m), len(m))]) for _ in range(5000)]
    return float(np.nanmean(x)), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
out = {"n_states": len(states), "goals": goals, "cells": {}}
for ci, c in enumerate(cells):
    pl = c.split("_")[0]; ref = cells.index(f"{pl}_budget1.0")
    diff, ci_ = boot(C[ci] - C[ref])
    out["cells"][c] = dict(capture=float(np.nanmean(C[ci])), vs_default=diff, ci=ci_,
                           evals=float(np.nanmean(N[ci])), time_s=float(np.nanmean(T[ci])),
                           trial_sd_diff=float(np.nanstd(C[ci] - C[ref])))
for pl in ("rank", "gd", "cem"):
    r = out["cells"][f"{pl}_default_repeat"]
    out[f"{pl}_repeat_sd_single_trial"] = r["trial_sd_diff"] / np.sqrt(2)
(HERE / f"results/analysis_{tag}.json").write_text(json.dumps(out, indent=1))
print(f"{'cell':22s} capture  vs 1s-default [95% CI]      evals   time")
for c, r in out["cells"].items():
    flag = " *" if (r["ci"][0] > 0 or r["ci"][1] < 0) and not c.endswith("budget1.0") else ""
    print(f"{c:22s} {r['capture']:6.3f}  {r['vs_default']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  {r['evals']:7.0f}  {r['time_s']:.2f}{flag}")
for pl in ("rank", "gd", "cem"):
    print(pl, "single-trial repeat sd", round(out[f"{pl}_repeat_sd_single_trial"], 3))
