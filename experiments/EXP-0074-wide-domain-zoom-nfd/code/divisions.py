"""Which model is best on which division of the wide test data? From the existing per-shard / per-push-length eval JSONs (no retest).
Divisions: scene type (scattered / inbetween / piled), object count (n20/50/100), the 8 scene x n shards, push-length bins (<35, 35-55, >=55 mm; macro over shards).
Metrics: acc1, slateN, roll_4. Models: LF wide, normal NFD 64 (w64_s0_ms4), zoom64 (z64_f8_ms4), vanilla128, zoom128 (mean of 3 seeds for the 128 pair). Also the same with the val-selected post-hoc mass balance (4 NFD models)."""
import json, numpy as np
E = "experiments/EXP-0074-wide-domain-zoom-nfd/results/"; S = ["scattered", "inbetween", "piled"]; N = [20, 50, 100]
SH = [f"{s}_n{n}" for s in S for n in N if not (s == "piled" and n == 100)]
ld = lambda n: json.load(open(E + f"eval_{n}.json"))
def avg(names):
    ds = [ld(n) for n in names]; return {sh: {k: float(np.mean([d[sh][k] for d in ds])) if ds[0][sh].get(k) is not None else None for k in ds[0][sh] if k != "n_pools"} for sh in SH}
M = {"LF": ld("lf_wide"), "vanilla64": ld("w64_s0_ms4"), "zoom64": ld("z64_f8_ms4"),
     "vanilla128": avg(["w128_f8_ms4", "w128_f8_s1_ms4", "w128_f8_s2_ms4"]), "zoom128": avg(["z128_f8_ms4", "z128_f8_s1_ms4", "z128_f8_s2_ms4"])}
B = {}
for nm, f in (("vanilla64", "w64"), ("zoom64", "z64"), ("vanilla128", "w128"), ("zoom128", "z128")):
    ps = json.load(open(E + f"posthoc_fix_test_{f}.json"))["balance@0.0"]["per_shard"]; B[nm] = {sh: ps[sh] for sh in SH}
def agg(mod, shards, key):
    v = [mod[sh][key] for sh in shards if mod[sh].get(key) is not None]; return float(np.mean(v)) if v else float("nan")
divs = [("scene: scattered", [s for s in SH if s.startswith("scattered")]), ("scene: inbetween", [s for s in SH if s.startswith("inbetween")]), ("scene: piled", [s for s in SH if s.startswith("piled")]),
        ("n = 20", [s for s in SH if s.endswith("_n20")]), ("n = 50", [s for s in SH if s.endswith("_n50")]), ("n = 100", [s for s in SH if s.endswith("_n100")])] + [(sh, [sh]) for sh in SH]
out = ["# Best model per division (existing test evals; no retest). Models: " + ", ".join(M) + ". 128-px entries = mean of 3 seeds. Differences < ~.01 are within what we can resolve (seed spread <= .001 but one fixed test draw, ~60 pools/shard).\n"]
def table(models, metric, keymap, title, divs_):
    out.append(f"\n## {title}\n| division | " + " | ".join(models) + " | best |"); out.append("|---|" + "---|" * (len(models) + 1))
    for name, shards in divs_:
        vals = {m: agg(models_src[m], shards, keymap) for m in models}; best = max(vals, key=lambda m: vals[m]); srt = sorted(vals.values())[::-1]
        out.append(f"| {name} | " + " | ".join(f"{vals[m]:.3f}" for m in models) + f" | **{best}** (+{srt[0]-srt[1]:.3f}) |")
models_src = M
for key, ttl in (("acc1", "one-step accuracy"), ("roll_4", "accuracy after 4 pushes"), ("slateN", "slateN")): table(list(M), key, key, ttl + " (raw predictions)", divs)
Ld = [("L < 35 mm", "acc1_L<35"), ("L 35-55 mm", "acc1_L35-55"), ("L >= 55 mm", "acc1_L>=55")]
out.append("\n## one-step accuracy by push length (macro over shards, raw)\n| bin | " + " | ".join(M) + " | best |"); out.append("|---|" + "---|" * (len(M) + 1))
for nm, k in Ld:
    v = {m: agg(M[m], SH, k) for m in M}; b = max(v, key=lambda m: v[m]); s = sorted(v.values())[::-1]; out.append(f"| {nm} | " + " | ".join(f"{v[m]:.3f}" for m in M) + f" | **{b}** (+{s[0]-s[1]:.3f}) |")
out.append("\n## same by push length x scene type (zoom128 minus vanilla128, raw acc1)\n| scene | L<35 | 35-55 | >=55 |\n|---|---|---|---|")
for s in S: sh = [x for x in SH if x.startswith(s)]; out.append(f"| {s} | " + " | ".join(f"{agg(M['zoom128'], sh, k)-agg(M['vanilla128'], sh, k):+.3f}" for _, k in Ld) + " |")
models_src = B
for key, ttl in (("acc1", "one-step accuracy"), ("roll_4", "accuracy after 4 pushes"), ("slateN", "slateN")): table(list(B), key, key, ttl + " WITH the val-selected mass-balance fix (4 NFD models)", divs)
# oracle gain of per-division model choice (scene x n shards), raw
for key in ("acc1", "roll_4", "slateN"):
    best_single = max(M, key=lambda m: agg(M[m], SH, key)); ora = np.mean([max(M[m][sh][key] for m in M) for sh in SH])
    out.append(f"\n- per-shard oracle choice ({key}): {ora:.3f} vs best single model {best_single} {agg(M[best_single], SH, key):.3f} (+{ora-agg(M[best_single], SH, key):.3f})")
open(E + "divisions.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
