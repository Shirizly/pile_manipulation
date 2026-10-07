"""Recover the TEST file set that results/../artifacts/test_sets.pt was built from. sean_data.split_files() uses python's per-process-salted hash(str) in its rng seed, so
the file-level split differs between processes (different PYTHONHASHSEED) -- only the cached test_sets.pt is a fixed object. We identify the test files by fingerprinting the
cached test rows' start states, then take val (~5 % of the rest, seed 0) / train (rest) from the remaining files. Mapping cached in results/lf_split_recovered.json."""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import sean_data as sd
MAP = "experiments/EXP-0074-wide-domain-zoom-nfd/results/lf_split_recovered.json"; TS = "experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/test_sets.pt"


def fp(S, P0, P1): return ((S[:, :4, :2].double().sum((1, 2)) + 3.1 * P0.double().sum(1) + 7.3 * P1.double().sum(1)) * 1e6).round().long()


def build():
    sets = torch.load(TS, weights_only=False); fs = sorted(glob.glob("Genesis/data/Sean/**/_*_data.pt", recursive=True)); mp = {}; rng = np.random.default_rng(0); by = {}
    for f in fs: by.setdefault(sd.shard_of(f), []).append(f)
    for sh, ff in by.items():
        if sh not in sets: continue
        tf = set(fp(sets[sh]["t"]["S"], sets[sh]["t"]["P0"], sets[sh]["t"]["P1"]).tolist()); test = []; ntest_rows = 0
        for f in ff:
            d = torch.load(f, map_location="cpu", weights_only=False); h = fp(d["states"].float(), d["p_starts"][:, :2].float(), d["p_stops"][:, :2].float()); m = np.mean([int(x) in tf for x in h.tolist()])
            if m > 0.5: test.append(f)
            elif m > 0.05: print("  partial match", f, m)
        rest = [f for f in ff if f not in test]; p = rng.permutation(len(rest)); nv = max(1, round(0.055 * len(rest))); val = {rest[i] for i in p[:nv]}
        for f in ff: mp[f] = "test" if f in test else ("val" if f in val else "train")
        print(sh, "files", len(ff), "test", len(test), "val", len(val), flush=True)
    json.dump(mp, open(MAP, "w"), indent=0)


def load_split(split, shards):
    mp = json.load(open(MAP)); out = {}
    for sh in shards:
        out[sh] = [sd.load_file(f, s, sh) for f, s in sorted(mp.items()) if s == split and sd.shard_of(f) == sh]
    return out


if __name__ == "__main__": build()
