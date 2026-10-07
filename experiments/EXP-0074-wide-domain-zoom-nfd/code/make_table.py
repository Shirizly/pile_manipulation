"""results/eval_*.json -> results/table.md (macro mean over the 8 test shards; per object-count bins; per-shard acc1 for the dense/scattered extremes)."""
import json, glob, os
R = "experiments/EXP-0074-wide-domain-zoom-nfd/results/"; rows = []
for f in sorted(glob.glob(R + "eval_*.json")):
    n = os.path.basename(f)[5:-5]
    if n.startswith("narrowtest_") or n.startswith("dom_"): continue
    d = json.load(open(f)); m = d["MEAN"]
    rows.append((n, m, d))
L = ["| model | acc1 | slateN | roll1 | roll2 | roll3 | roll4 | acc1 n20 | acc1 n50 | acc1 n100 | roll4 n20 | roll4 n50 | roll4 n100 |", "|---|" + "---|" * 12]
for n, m, d in rows:
    L.append(f"| {n} | " + " | ".join(f"{m[k]:.3f}" for k in ("acc1", "slateN", "roll_1", "roll_2", "roll_3", "roll_4")) + " | " + " | ".join(f"{d[f'MEAN_n{b}']['acc1']:.3f}" for b in (20, 50, 100)) + " | " + " | ".join(f"{d[f'MEAN_n{b}']['roll_4']:.3f}" for b in (20, 50, 100)) + " |")
open(R + "table.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
