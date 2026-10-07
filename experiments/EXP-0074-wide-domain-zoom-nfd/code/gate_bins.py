"""Per push-length bin comparison of the members (balanced predictions, TEST): acc1 within the bin, and slateN of ranking ONLY among candidates inside the bin (pools restricted to the bin; pools with <4 candidates skipped)."""
import sys, json, numpy as np, torch
R0 = "experiments/EXP-0074-wide-domain-zoom-nfd/"
SH = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
D = {sp: {m: torch.load(R0 + f"artifacts/gate_{sp}_{m}.pt", weights_only=False) for m in ("z", "w", "lf")} for sp in ("val", "test")}
BINS = [("L<35", 0, 35), ("L35-55", 35, 55), ("L>=55", 55, 999)]
def cap(vp, vt):
    den = vt.mean(0) - vt.min(0); ok = den > 1e-9
    return np.where(ok, (vt.mean(0) - vt[vp.argmin(0), np.arange(vt.shape[1])]) / np.where(ok, den, 1), np.nan)
out = {}
for sp in ("val", "test"):
    for nm, lo, hi in BINS:
        for m in ("z", "w", "lf"):
            A, S = [], []
            for sh in SH:
                r = D[sp][m][sh]["rows"]; k = (r["L"] >= lo) & (r["L"] < hi)
                if k.sum() > 20: A.append(1 - np.sqrt(r["numb"][k].sum() / r["den"][k].sum()))
                cs = []
                for p in D[sp][m][sh]["pools"]:
                    k = (p["L"] >= lo) & (p["L"] < hi)
                    if k.sum() >= 4: cs.append(np.nanmean(cap(p["vpb"][k], p["vt"][k])))
                if cs: S.append(np.mean(cs))
            out[(sp, nm, m)] = (float(np.mean(A)), float(np.mean(S)), len(S))
for sp in ("val", "test"):
    print(sp, "(balanced) acc1 within bin / slateN ranking only among candidates inside the bin (#shards with pools)")
    for nm, _, _ in BINS: print(" ", nm.ljust(8), "  ".join(f"{m}: {out[(sp,nm,m)][0]:.3f}/{out[(sp,nm,m)][1]:.3f}({out[(sp,nm,m)][2]})" for m in ("z", "w", "lf")))
json.dump({f"{a}|{b}|{c}": v for (a, b, c), v in out.items()}, open(R0 + "results/gating_bins.json", "w"), indent=1)
