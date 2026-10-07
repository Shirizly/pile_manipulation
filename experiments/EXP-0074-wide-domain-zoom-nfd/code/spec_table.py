import json, numpy as np, os
E = "experiments/EXP-0074-wide-domain-zoom-nfd/results/"
ld = lambda n: json.load(open(E + f"eval_{n}.json")) if os.path.exists(E + f"eval_{n}.json") else None
SH = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
out = ["# Post-training specialisation of the single-step generalists (no new data) -- one-step acc1 / slateN [/ roll4 for n-domains], mean over the domain's test shards; control = generalist continued on all rows for ~the same number of steps\n"]
for kind, gen in (("zoom", "z128_f8"), ("world", "w128_f8")):
    G, C = ld(gen), ld(f"sp_{kind}_control")
    out.append(f"\n## {kind} 128 ([8,16,32])\n| domain | generalist | control (+all-data steps) | specialist | spec - gen | spec - control |\n|---|---|---|---|---|---|")
    for n in (20, 50, 100):
        sh = [s for s in SH if s.endswith(f"_n{n}")]; S = ld(f"sp_{kind}_n{n}")
        if S is None: continue
        f = lambda d, k: float(np.mean([d[s][k] for s in sh])) if d else float("nan")
        cell = lambda d: f"{f(d,'acc1'):.3f} / {f(d,'slateN'):.3f} / {f(d,'roll_4'):.3f}" if d else "n/a"
        out.append(f"| n = {n} | {cell(G)} | {cell(C)} | {cell(S)} | {f(S,'acc1')-f(G,'acc1'):+.3f} / {f(S,'slateN')-f(G,'slateN'):+.3f} / {f(S,'roll_4')-f(G,'roll_4'):+.3f} | " + (f"{f(S,'acc1')-f(C,'acc1'):+.3f} / {f(S,'slateN')-f(C,'slateN'):+.3f} / {f(S,'roll_4')-f(C,'roll_4'):+.3f}" if C else "n/a") + " |")
    R = ld(f"sp_{kind}_Lrouted")
    if R:
        f = lambda d, k: float(np.mean([d[s][k] for s in SH])); lb = {"L<35": "acc1_L<35", "L35-55": "acc1_L35-55", "L>=55": "acc1_L>=55"}
        out.append(f"\n push-length specialists routed by bin (edges 35/55 mm), one-step: acc1 {f(R,'acc1'):.3f} (generalist {f(G,'acc1'):.3f}, control {f(C,'acc1') if C else float('nan'):.3f}); slateN {f(R,'slateN'):.3f} (generalist {f(G,'slateN'):.3f}, control {f(C,'slateN') if C else float('nan'):.3f})")
        for nm, k in lb.items(): out.append(f"  - {nm}: routed {np.mean([R[s][k] for s in SH if R[s].get(k) is not None]):.3f} vs generalist {np.mean([G[s][k] for s in SH if G[s].get(k) is not None]):.3f}" + (f" vs control {np.mean([C[s][k] for s in SH if C[s].get(k) is not None]):.3f}" if C else ""))
open(E + "specialists.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
