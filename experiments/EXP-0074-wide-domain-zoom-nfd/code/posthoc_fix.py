"""Honest re-test of the post-hoc mass fix (zero |delta|<t, balance pos/neg parts of the predicted CHANGE to equal mass), selected on VAL sets only.
Fixed(model,thr,mode).rollout applies the fix per step to the pasted-frame delta (raw_k - raw_{k-1}); internal model state (canvas / raster) is NOT altered.
Modes: none | thr | balance | thr+balance.  Selection rule (per model TYPE, zoom / vanilla): max over settings of mean over that type's models (z128,z64 / w128,w64)
of [val MEAN acc1 + val MEAN slateN]. Then evaluate on test sets. Usage: PYTHONPATH=. python -u posthoc_fix.py val MK (MK in z128 w128 z64 w64) ; ... select ; ... test MK"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as E
from eval_wide import *  # noqa
R0 = "experiments/EXP-0074-wide-domain-zoom-nfd/"
VS = R0 + "artifacts/val_sets.pt"; VCF = R0 + "artifacts/val_canvases.pt"

def val_sets(n_rows=1000, n_pools=60):
    if os.path.exists(VS): return torch.load(VS, weights_only=False)
    out = {}; D = load_split("val"); rng = np.random.default_rng(1)
    for sh in SHARDS:
        t = rows_table(D[sh]); N = len(t["S"]); rows = np.sort(rng.permutation(N)[:n_rows]); ch = chains_of(t, minlen=4, maxlen=4); ch = [ch[i] for i in rng.permutation(len(ch))[:400]]
        pl = pools_of(t); pl = [pl[i][:32] for i in rng.permutation(len(pl))[:n_pools]]
        out[sh] = dict(t={k: v for k, v in t.items() if k != "file"}, rows=rows, chains=ch, pools=pl)
    torch.save(out, VS); return out

class Memo:
    def __init__(self, m): self.m, self.c = m, {}
    def rollout(self, S0, P0s, P1s, canvas0=None):
        k = (tuple(S0.shape), tuple(P0s.shape), float(S0.double().sum()), float(P0s.double().sum()), float(P1s.double().sum()), float(S0.flatten()[::97].double().sum()))
        if k not in self.c: self.c[k] = self.m.rollout(S0, P0s, P1s, canvas0)
        return self.c[k]

def fix_delta(d, thr, mode):
    if mode in ("thr", "thr+balance"): d = d * (d.abs() >= thr)
    if mode in ("balance", "thr+balance"):
        pos = d.clamp_min(0); neg = (-d).clamp_min(0); ps, ns = pos.sum((-1, -2), keepdim=True), neg.sum((-1, -2), keepdim=True); tg = (ps + ns) / 2
        d = pos * tg / ps.clamp_min(1e-9) - neg * tg / ns.clamp_min(1e-9)
    return d

class Fixed:
    def __init__(self, model, thr=0., mode="none"): self.m, self.thr, self.mode = model, thr, mode
    def rollout(self, S0, P0s, P1s, canvas0=None):
        raw = self.m.rollout(S0, P0s, P1s, canvas0)
        if self.mode == "none": return raw
        prev_raw = prev = occ_from_particles(S0); out = []
        for r in raw:
            prev = (prev + fix_delta(r - prev_raw, self.thr, self.mode)).clamp(0, 1); prev_raw = r; out.append(prev)
        return out

MODELS = {"z128": ("z128_f8_ms4", "zoom", 128, (8, 16, 32)), "w128": ("w128_f8_ms4", "world", 128, (8, 16, 32)),
          "z64": ("z64_f8_ms4", "zoom", 64, (8, 16, 32)), "w64": ("w64_s0_ms4", "world", 64, (4, 8, 16))}
KEYS = ["acc1", "slateN", "roll_1", "roll_2", "roll_3", "roll_4"]
def sn(s): return f"{s[0]}@{s[1]}"

SETTINGS = [("none", 0.), ("balance", 0.)] + [("thr+balance", t) for t in (.05, .1, .2)]   # thr-only dropped for time (a pure threshold did nothing in the first val pass: z128 thr@.05 == none)
TYPE = {"z128": "zoom", "z64": "zoom", "w128": "vanilla", "w64": "vanilla"}
RES = R0 + "results/"

def one(split, mk):
    """val: full grid for model mk -> results/posthoc_fix_<split>_<mk>.json ; test: none + balance + selected(type) setting."""
    torch.set_num_threads(2)
    nm, kind, r, feats = MODELS[mk]; sets = val_sets() if split == "val" else test_sets()
    if split == "val": E.CV = torch.load(VCF, weights_only=False); E.CVF = VCF
    net = make_net(R0 + f"runs/{nm}/unet_best.pth", feats); base = Memo(Zoom(net, r) if kind == "zoom" else World(net, r))
    if split == "val": todo = SETTINGS
    else:
        sel = json.load(open(RES + "posthoc_fix_selected.json"))[TYPE[mk]]; todo = list(dict.fromkeys([("none", 0.), ("balance", 0.), (sel["mode"], sel["thr"])]))
    d = {}
    for s in todo:
        r_ = run(Fixed(base, s[1], s[0]), sets); d[sn(s)] = {k: r_["MEAN"][k] for k in KEYS}; d[sn(s)]["per_shard"] = {sh: {k: r_[sh][k] for k in KEYS} for sh in SHARDS}
        d[sn(s)]["MEAN_n"] = {nb: r_[f"MEAN_n{nb}"] for nb in (20, 50, 100)}
        print("RESULT", split, mk, sn(s), {k: round(d[sn(s)][k], 3) for k in KEYS}, flush=True)
        json.dump(d, open(RES + f"posthoc_fix_{split}_{mk}.json", "w"), indent=1)

def select():
    sel = {}
    for typ, mks in (("zoom", ["z128", "z64"]), ("vanilla", ["w128", "w64"])):
        v = {m: json.load(open(RES + f"posthoc_fix_val_{m}.json")) for m in mks}
        sc = {sn(s): float(np.mean([v[m][sn(s)]["acc1"] + v[m][sn(s)]["slateN"] for m in mks])) for s in SETTINGS}
        b = max(sc, key=sc.get); mode, thr_ = b.split("@"); sel[typ] = dict(mode=mode, thr=float(thr_), score=sc[b], all_scores=sc)
    json.dump(sel, open(RES + "posthoc_fix_selected.json", "w"), indent=1); print("SELECTED", {k: (v["mode"], v["thr"]) for k, v in sel.items()})

if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "select": select()
    else: one(a[0], a[1])
