"""Collect results/posthoc_fix_{val,test}_<model>.json + posthoc_fix_selected.json -> results/posthoc_fix_val.json and results/posthoc_fix_val.md."""
import json
R = "experiments/EXP-0074-wide-domain-zoom-nfd/results/"; M = ["z128", "w128", "z64", "w64"]; K = ["slateN", "acc1", "roll_1", "roll_2", "roll_3", "roll_4"]
sel = json.load(open(R + "posthoc_fix_selected.json")); T = {"z128": "zoom", "z64": "zoom", "w128": "vanilla", "w64": "vanilla"}
val = {m: json.load(open(R + f"posthoc_fix_val_{m}.json")) for m in M}; test = {m: json.load(open(R + f"posthoc_fix_test_{m}.json")) for m in M}
json.dump(dict(selected=sel, val=val, test=test), open(R + "posthoc_fix_val.json", "w"), indent=1)
f = lambda d: " ".join(f"{d[k]:.3f}" for k in K)
L = ["# Post-hoc mass fix, selected on VAL only (EXP-0074)", "", "Fix: per step, on the pasted-frame delta: zero |delta|<thr, then rescale positive and negative parts to equal mean mass (balance). Internal model state untouched.",
     "Selection rule: per model TYPE, maximise mean over the type's two models (zoom: z128,z64; vanilla: w128,w64) of val MEAN(acc1)+val MEAN(slateN). Columns: " + " / ".join(K) + ".", "",
     "## Selected", ""]
for t, s in sel.items(): L.append(f"- {t}: {s['mode']} thr={s['thr']} (score {s['score']:.3f}; grid {', '.join(f'{k}={v:.3f}' for k, v in s['all_scores'].items())})")
L += ["", "## VAL grid (mean over 8 shards; slateN acc1 roll1-4)", "", "| model | setting | " + " | ".join(K) + " |", "|---|---|" + "---|" * len(K)]
for m in M:
    for s, d in val[m].items(): L.append(f"| {m} | {s} | " + " | ".join(f"{d[k]:.3f}" for k in K) + " |")
L += ["", "## TEST (selected setting vs none vs balance)", "", "| model | setting | " + " | ".join(K) + " |", "|---|---|" + "---|" * len(K)]
for m in M:
    for s, d in test[m].items(): L.append(f"| {m} | {s}{' (selected)' if s == '%s@%s' % (sel[T[m]]['mode'], sel[T[m]]['thr']) else ''} | " + " | ".join(f"{d[k]:.3f}" for k in K) + " |")
sk = lambda m: "%s@%s" % (sel[T[m]]["mode"], sel[T[m]]["thr"])
L += ["", "## Zoom minus vanilla gaps on TEST (same resolution)", "", "| pair | fix | " + " | ".join(K) + " |", "|---|---|" + "---|" * len(K)]
for z, w in (("z128", "w128"), ("z64", "w64")):
    for nm, a, b in (("none", "none@0.0", "none@0.0"), ("selected", sk(z), sk(w)), ("balance", "balance@0.0", "balance@0.0")):
        L.append(f"| {z}-{w} | {nm} | " + " | ".join(f"{test[z][a][k] - test[w][b][k]:+.3f}" for k in K) + " |")
L += ["", "## Test per scattered shard slateN (none -> selected)", ""]
for m in M: L.append(f"- {m}: " + ", ".join(f"{sh} {test[m]['none@0.0']['per_shard'][sh]['slateN']:.3f}->{test[m][sk(m)]['per_shard'][sh]['slateN']:.3f}" for sh in test[m]["none@0.0"]["per_shard"] if sh.startswith("scattered")))
open(R + "posthoc_fix_val.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
