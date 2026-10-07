"""Sean corpus loader for EXP-0074 (canonical wide-domain training corpus): all `_k_data.pt` files, per-shard (spawn mode x object count),
null rows dropped (max cube displacement < 1 mm), exact successor links (states_[i] == states[j] within the file, non-null rows only: a null
push in between leaves the state unchanged, so the link skips it), chains, and a FILE-level split (test 10 % / val 5 % / train rest per shard)."""
import glob, re, os, collections, numpy as np, torch
FILES = None


def shard_of(f):
    m = re.search(r"/(scattered|piled|inbetween)-.*?/cube/n(\d+)/", f); return f"{m.group(1)}_n{m.group(2)}"


def split_files(seed=0):
    fs = sorted(glob.glob("Genesis/data/Sean/**/_*_data.pt", recursive=True)); by = collections.defaultdict(list)
    for f in fs: by[shard_of(f)].append(f)
    sp = {}
    for k, ff in sorted(by.items()):
        r = np.random.default_rng(seed + hash(k) % 1000); p = r.permutation(len(ff)); nt = max(1, round(0.10 * len(ff))); nv = max(1, round(0.05 * len(ff)))
        for j, i in enumerate(p): sp[ff[i]] = "test" if j < nt else ("val" if j < nt + nv else "train")
    return sp


def load_file(f, split, shard):
    d = torch.load(f, map_location="cpu", weights_only=False); S, S_ = d["states"].float(), d["states_"].float(); n = len(S)
    disp = (S_[:, :, :2] - S[:, :, :2]).norm(dim=-1).max(1).values; ok = disp > 1e-3
    A, B = S_[:, :, :2].reshape(n, -1), S[:, :, :2].reshape(n, -1); D = torch.cdist(A, B); D[:, ~ok] = 9; D[torch.arange(n), torch.arange(n)] = 9
    m, idx = D.min(1); succ = torch.where((m < 1e-4) & ok, idx, torch.full_like(idx, -1))
    return dict(S=S, S_=S_, P0=d["p_starts"][:, :2].float(), P1=d["p_stops"][:, :2].float(), ok=ok, succ=succ, file=f, shard=shard, split=split)


def load_split(split, shards=None, max_files=None):
    sp = split_files(); out = collections.defaultdict(list)
    for f, s in sp.items():
        sh = shard_of(f)
        if s != split or (shards and sh not in shards): continue
        out[sh].append(load_file(f, s, sh))
    return out


def rows_table(files):
    """list of per-file dicts (same shard) -> stacked arrays of the NON-NULL rows + chain bookkeeping (global successor index within the stack)."""
    S, S_, P0, P1, nxt, fid = [], [], [], [], [], []; base = 0; keep_ix = []
    for k, d in enumerate(files):
        ok = d["ok"]; new = torch.cumsum(ok.long(), 0) - 1
        S.append(d["S"][ok]); S_.append(d["S_"][ok]); P0.append(d["P0"][ok]); P1.append(d["P1"][ok]); fid.append(torch.full((int(ok.sum()),), k))
        sc = d["succ"][ok]; nxt.append(torch.where(sc >= 0, new[sc.clamp_min(0)] + base, torch.full_like(sc, -1))); base += int(ok.sum())
    return dict(S=torch.cat(S), S_=torch.cat(S_), P0=torch.cat(P0), P1=torch.cat(P1), succ=torch.cat(nxt), file=torch.cat(fid))


def chains_of(tab, minlen=2, maxlen=4):
    """starting rows with >= minlen consecutive linked rows -> (idx list of rows per chain, truncated to maxlen)."""
    succ = tab["succ"].numpy(); has_prev = np.zeros(len(succ), bool); has_prev[succ[succ >= 0]] = True; out = []
    for i in range(len(succ)):
        if has_prev[i]: continue
        c = [i]
        while succ[c[-1]] >= 0 and len(c) < 64: c.append(int(succ[c[-1]]))
        if len(c) >= minlen:
            for s in range(0, len(c) - minlen + 1, maxlen): out.append(c[s:s + maxlen])   # non-overlapping segments of up to maxlen
    return out


def pools_of(tab, minpush=6):
    """same-state pools (identical start state to 0.1 mm, >= minpush rows) -> list of row index arrays."""
    key = collections.defaultdict(list)
    for i in range(len(tab["S"])):
        key[tuple(np.round(tab["S"][i][:6, :2].numpy().ravel() * 1e4).astype(np.int64))].append(i)
    return [np.array(v) for v in key.values() if len(v) >= minpush]


if __name__ == "__main__":
    import json; rep = {}
    for split in ("train", "val", "test"):
        D = load_split(split)
        for sh, files in sorted(D.items()):
            t = rows_table(files); ch = chains_of(t); L = (t["P1"] - t["P0"]).norm(dim=1) * 1000
            rep[f"{split}/{sh}"] = dict(files=len(files), rows=len(t["S"]), chains_ge2=len(ch), chain_len=dict(collections.Counter(len(c) for c in ch)),
                                       push_mm=[round(float(x), 1) for x in np.percentile(L.numpy(), [0, 25, 50, 75, 100])], pools=len(pools_of(t)) if split == "test" else None)
            print(split, sh, rep[f"{split}/{sh}"], flush=True)
    json.dump(rep, open("experiments/EXP-0074-wide-domain-zoom-nfd/results/sean_splits.json", "w"), indent=1)
