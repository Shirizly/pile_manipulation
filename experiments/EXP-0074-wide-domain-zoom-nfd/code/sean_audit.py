"""Audit of the Sean corpus: shards, files, chains (successor rows by exact state match), push-length and count distributions, failed files."""
import sys, glob, re, json, collections, numpy as np, torch
fs = sorted(glob.glob("Genesis/data/Sean/**/*_data.pt", recursive=True))
def key(f):
    m = re.search(r"/(scattered|piled|inbetween)-.*?/cube/n(\d+)/", f); return f"{m.group(1)}_n{m.group(2)}"
by = collections.defaultdict(list)
for f in fs: by[key(f)].append(f)
out = {}
rng = np.random.default_rng(0)
for k, ff in sorted(by.items()):
    nfail = len(glob.glob(ff[0].rsplit("/", 1)[0] + "/_*_failed.pt"))
    samp = list(rng.choice(ff, min(8, len(ff)), replace=False)); rows = succ = 0; L = []; pl = []; zs = []
    for f in samp:
        d = torch.load(f, map_location="cpu", weights_only=False); s, s_ = d["states"][:, :, :2], d["states_"][:, :, :2]; n = len(s)
        D = torch.cdist(s_.reshape(n, -1), s.reshape(n, -1)); m, idx = D.min(1)
        nxt = {i: int(idx[i]) for i in range(n) if m[i] < 1e-4 and int(idx[i]) != i}; prev = set(nxt.values()); rows += n; succ += len(nxt)
        for i in range(n):
            if i not in prev:
                l, j = 1, i
                while j in nxt: j = nxt[j]; l += 1
                L.append(l)
        pl += list((d["p_stops"] - d["p_starts"])[:, :2].norm(dim=1) * 1000); zs.append(float(d["states"][:, :, 2].max()))
    out[k] = dict(files=len(ff), rows_per_file=rows // len(samp), failed_files_in_dir=nfail, frac_with_successor=succ / rows, chain_len_hist=dict(collections.Counter(L)),
                  push_mm_pct=[round(float(x), 1) for x in np.percentile(pl, [0, 10, 50, 90, 100])], max_z=max(zs))
    print(k, out[k], flush=True)
json.dump(out, open("experiments/EXP-0074-wide-domain-zoom-nfd/results/sean_audit.json", "w"), indent=1)
