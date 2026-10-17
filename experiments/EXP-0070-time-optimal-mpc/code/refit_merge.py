"""EXP-0070 policy-iteration step: refit etc_g on the closed-loop episodes of ONE cell of a pilot
(default the time-aware cell, i.e. data from the new criterion's own policy), keep the v0 fit for goals
with too few completed states. Usage: refit_merge.py PILOT_TAG CELL BASE_MAP OUT_NAME"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
tag, cell, base, out = sys.argv[1:5]
rows = [r for r in json.loads((HERE / f"results/{tag}_episodes.json").read_text()) if r["cell"] == cell]
tmp = HERE / f"results/{tag}_{cell}_rows.json"; tmp.write_text(json.dumps(rows))
subprocess.run([sys.executable, str(HERE / "code/fit_etc.py"), out + "_raw", str(tmp)], check=True)
new = json.loads((HERE / f"results/{out}_raw.json").read_text()); old = json.loads(Path(base).read_text())
merged = {}
for g, v in old.items():
    if g in new and "pass1" in new[g]:
        merged[g] = dict(new[g], refit_from=f"{tag}:{cell}")
    else:
        merged[g] = dict(v, refit_from="v0 (kept: too few completed states in the refit data)")
(HERE / f"results/{out}.json").write_text(json.dumps(merged, indent=1))
for g, v in merged.items():
    print(g, v["refit_from"], v.get("n_episodes"), v.get("n_completed"))
