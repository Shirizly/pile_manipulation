"""EXP-0070 loop step (pre-declared rule, see EXPERIMENT.md "The loop as the user specified it"):
  loop_step.py PREV_TAG MAP_IN MAP_OUT  -> prints lam2 and writes results/loop_step_<PREV_TAG>.json
Reads results/<PREV_TAG>_episodes.json (analyse_pilot.py output) and equal_time_<PREV_TAG>.json."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from etc_v2 import transitions, evaluate
prev, map_in, map_out = sys.argv[1:4]
eq = json.loads((HERE / f"results/equal_time_{prev}.json").read_text())
cells = sorted({k.split("|")[0] for k in eq})
score = {c: eq[f"{c}|ALL"]["12.0"]["mean"] + eq[f"{c}|ALL"]["16.0"]["mean"] for c in cells}
best = max(cells, key=lambda c: score[c])
cond_a = eq[f"{best}|ALL"]["12.0"]["mean"] >= 0 and eq[f"{best}|ALL"]["16.0"]["mean"] >= 0
M = json.loads(Path(map_in).read_text())["goals"]
rows = [r for r in json.loads((HERE / f"results/{prev}_episodes.json").read_text()) if r["cell"] == best]
res = []
for g in M:
    R = [r for r in rows if r["goal"] == g]
    if R:
        T = transitions(R, "gap"); e = evaluate(T, M[g]["gap|td"]); res.append((e["bellman_mean"], e["n"]))
bm = float(sum(b * n for b, n in res) / sum(n for _, n in res))
cond_b = abs(bm) < 0.5
lam2 = 0.75 if (cond_a and cond_b) else (0.5 if (cond_a or cond_b) else 0.25)
out = dict(best_cell=best, equal_time_12_16=[eq[f"{best}|ALL"]["12.0"]["mean"], eq[f"{best}|ALL"]["16.0"]["mean"]],
           cond_a=cond_a, bellman_mean_on_new=bm, cond_b=cond_b, lam2=lam2)
(HERE / f"results/loop_step_{prev}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
# refit: all data + the best cell's new trajectories x3 (fit reads <tag>_episodes.json, so write the best cell's rows)
(HERE / f"results/{prev}_{best}_episodes.json").write_text(json.dumps(rows))
subprocess.run([sys.executable, str(HERE / "code/etc_v2.py"), "fit", map_out, "--data", "retro", "pilot_v0b", "pilot_v1", "pilot_b01",
                f"{prev}_{best}", "--weight-new", f"{prev}_{best}=3"], check=True)
print("LAM2", lam2)
