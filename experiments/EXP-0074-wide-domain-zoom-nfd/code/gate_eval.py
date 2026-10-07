"""Hard-gating analysis from the cached per-candidate statistics (gate_cache.py): zoom128 (z), vanilla128 (w), linear foresight (lf).
Metrics as in eval_wide: per shard acc1 (pooled ratio of means), slateN (mean over goals of mean capture over pools), roll_k (chains), macro-mean over the 8 shards.
Gating rules pick ONE member per candidate push (L-rules), per chain / scene (scene, n rules) or per (shard, push-length bin); rules are selected on VAL (J = acc1 + slateN)
and reported on TEST. usage: gate_eval.py [raw|bal]"""
import sys, json, itertools, numpy as np, torch
R0 = "experiments/EXP-0074-wide-domain-zoom-nfd/"
variant = sys.argv[1] if len(sys.argv) > 1 else "bal"
SH = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
NAMES = ["zoom", "vanilla", "lf"]
D = {sp: {m: torch.load(R0 + f"artifacts/gate_{sp}_{m}.pt", weights_only=False) for m in ("z", "w", "lf")} for sp in ("val", "test")}
numk = "numb" if variant == "bal" else "num"; vpk = "vpb" if variant == "bal" else "vp"
scene = lambda sh: sh.split("_")[0]; nobj = lambda sh: int(sh.split("_n")[1])
EDGES = (35, 55)


def assign(rule, sh, L):
    """rule(sh, L array) -> model index array (0 zoom, 1 vanilla, 2 lf)."""
    return rule(sh, L)


def metrics(sp, rule, shards=SH, rolls=True):
    A, S, Rk = [], [], {k: [] for k in range(4)}
    for sh in shards:
        rows = [D[sp][m][sh]["rows"] for m in ("z", "w", "lf")]; L = rows[0]["L"]; a = assign(rule, sh, L)
        num = np.choose(a, [r[numk] for r in rows]); A.append(1 - np.sqrt(num.sum() / rows[0]["den"].sum()))
        ch = [D[sp][m][sh]["chains"] for m in ("z", "w", "lf")]; Lc = ch[0]["L"]; ac = assign(rule, sh, Lc.mean(1)) if rolls else None
        if rolls:
            for k in range(4):
                nc = np.choose(ac, [c[numk][:, k] for c in ch]); Rk[k].append(1 - np.sqrt(nc.sum() / ch[0]["den"][:, k].sum()))
        caps = []
        for pi in range(len(D[sp]["z"][sh]["pools"])):
            pl = [D[sp][m][sh]["pools"][pi] for m in ("z", "w", "lf")]; ap = assign(rule, sh, pl[0]["L"]); vp = np.choose(ap[:, None].repeat(pl[0]["vt"].shape[1], 1), [p[vpk] for p in pl]); vt = pl[0]["vt"]
            den = vt.mean(0) - vt.min(0); ok = den > 1e-9
            if ok.any(): caps.append(np.where(ok, (vt.mean(0) - vt[vp.argmin(0), np.arange(vt.shape[1])]) / np.where(ok, den, 1), np.nan))
        S.append(np.nanmean(np.nanmean(np.stack(caps), 0)))
    r = dict(acc1=float(np.mean(A)), slateN=float(np.mean(S)))
    if rolls: r.update({f"roll_{k+1}": float(np.mean(Rk[k])) for k in range(4)})
    return r


always = lambda m: (lambda sh, L: np.full(len(L), m))
def Lgate(t, lo=0, hi=1): return lambda sh, L: np.where(L < t, lo, hi)
def nrule(N, lo, hi): return lambda sh, L: np.full(len(L), lo if nobj(sh) <= N else hi)
def scene_rule(assn): return lambda sh, L: np.full(len(L), assn[scene(sh)])
def per_shard(assn): return lambda sh, L: np.full(len(L), assn[sh])
def per_shard_bin(assn): return lambda sh, L: np.array([assn[sh][int(np.searchsorted(EDGES, l, side="right"))] for l in L])
J = lambda r: r["acc1"] + r["slateN"]
res = {"variant": variant, "rules": {}}


def report(name, rule_val, rule_test=None, rolls=True):
    rt = rule_test or rule_val; v = metrics("val", rule_val, rolls=rolls); t = metrics("test", rt, rolls=rolls); res["rules"][name] = dict(val=v, test=t)
    print(f"{name:46s} val J {J(v):.3f} | test acc1 {t['acc1']:.3f} slateN {t['slateN']:.3f}" + (f" roll4 {t['roll_4']:.3f}" if rolls else ""), flush=True)


for m in range(3): report(f"always {NAMES[m]}", always(m))
# push-length gates (per candidate; rolls use the chain's mean push length)
best = None
for t in (25, 30, 35, 40, 45, 50, 55, 60, 65):
    for lo, hi in ((0, 1), (1, 0)):
        nm = f"L<{t}: {NAMES[lo]} else {NAMES[hi]}"; report(nm, Lgate(t, lo, hi))
# object-count gates
for N in (20, 50):
    for lo, hi in ((0, 1), (1, 0)): report(f"n<={N}: {NAMES[lo]} else {NAMES[hi]}", nrule(N, lo, hi))
# scene gates: best of 3^3 assignments (selected on val)
cands = {a: scene_rule(dict(zip(("scattered", "inbetween", "piled"), a))) for a in itertools.product(range(3), repeat=3)}
bs = max(cands, key=lambda a: J(metrics("val", cands[a], rolls=False))); report(f"scene gate (val-best) {dict(zip(('scat','inb','pile'), [NAMES[i] for i in bs]))}", cands[bs])
# per-shard gates (8 shards, pick best of 3 per shard on val)
def shard_best(spl="val"):
    a = {}
    for sh in SH:
        sc = [J(metrics("val", always(m), shards=[sh], rolls=False)) for m in range(3)]; a[sh] = int(np.argmax(sc))
    return a
ps = shard_best(); report(f"per-shard gate (val-best) {[NAMES[ps[s]] for s in SH]}", per_shard(ps))
# per-(shard, push-length-bin) gate: exhaustive 3^3 per shard on val
pb = {}
for sh in SH:
    bestc, bj = None, -9
    for c in itertools.product(range(3), repeat=3):
        j = J(metrics("val", per_shard_bin({sh: c}), shards=[sh], rolls=False))
        if j > bj: bj, bestc = j, c
    pb[sh] = bestc
report("per-(shard x push-length-bin) gate (val-best)", per_shard_bin(pb), rolls=False)
res["per_shard_choice"] = {s: NAMES[ps[s]] for s in SH}; res["per_shard_bin_choice"] = {s: [NAMES[i] for i in pb[s]] for s in SH}
# where does each model win on TEST (descriptive): per division (n, scene, L bin), acc1 and slateN of each member
div = {}
for sh in SH:
    for m in range(3):
        r = metrics("test", always(m), shards=[sh], rolls=True); div.setdefault(sh, {})[NAMES[m]] = r
res["test_per_shard"] = div
json.dump(res, open(R0 + f"results/gating_{variant}.json", "w"), indent=1)
