"""EXP-0033 (T0): across-model rank agreement between `accuracy` and lyapunov
slateN on the EXP-0028 soft-scored harness reports; all 12 models vs the NFD
family only. Seconds; reads JSON only."""
import json, sys
from pathlib import Path
from scipy import stats
REPO = Path(__file__).resolve().parents[3]
for gs in ("default", "many"):
    d = json.load(open(REPO / f"experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-{gs}/report.json"))
    for c in d:
        for label, pick in (("all", lambda m: m not in ("persistence", "random")), ("nfd", lambda m: m.startswith("nfd"))):
            ms = [m for m in d[c] if pick(m)]
            acc = [d[c][m]["accuracy"] for m in ms]
            sl = [d[c][m]["capture"]["averaged_over_goals"]["lyapunov"] for m in ms]
            t = stats.kendalltau(acc, sl)
            print(f"{gs:7s} {c:12s} {label:3s} n={len(ms):2d} Kendall {t.statistic:+.2f} p {t.pvalue:.3f}")
