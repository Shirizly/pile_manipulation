"""Fit the switched-linear foresight (6 equal-width push-length bins over [0,70] mm, res 64, ridge toward identity) on Sean TRAIN rows;
ridge lambda chosen on Sean VAL files (one-step swept-region accuracy, pooled mean over shards). Reuses fit_linear_foresight.canonicalise/fit_operator maths
(G, C accumulated once per bin, one solve per lambda)."""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
import eval_lf_wide as el
from sean_data import rows_table
from lf_split import load_split as _ls
load_split = lambda split, shards: _ls(split, shards)   # recovered, process-independent split (see lf_split.py)
from simple_mpc.adapters import occ_from_particles
from fit_linear_foresight import canonicalise, actions_to_pixels
from Baselines.LinearForesight.model import bin_index, push_length_m
dev = "cuda"; RES = 64; NB = 6; MAXM = 0.070; MIN_ROWS = 50; NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SH = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
LAMS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]; OUT = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/lf_wide"
edges = torch.linspace(0, MAXM, NB + 1); rng = np.random.default_rng(0)
G = [torch.zeros(RES * RES, RES * RES, device=dev) for _ in range(NB)]; C = [torch.zeros_like(G[0]) for _ in range(NB)]; cnt = [0] * NB
D = load_split("train", SH); t0 = time.time(); used = {}
for sh in SH:
    t = rows_table(D[sh]); N = len(t["S"]); ix = torch.from_numpy(np.sort(rng.permutation(N)[:NROWS])); used[sh] = [int(N), len(ix)]
    for i in range(0, len(ix), 250):
        j = ix[i:i + 250]; o0 = occ_from_particles(t["S"][j], dev); o1 = occ_from_particles(t["S_"][j], dev); act = torch.cat([t["P0"][j], t["P1"][j]], 1).float().to(dev)
        s, e = actions_to_pixels(act, el.WS[0], el.WS[1], (64, 64)); b = bin_index(push_length_m(act), edges.to(dev))
        Y0 = canonicalise(o0, s, e, RES).reshape(len(j), -1); Y1 = canonicalise(o1, s, e, RES).reshape(len(j), -1)
        for k in range(NB):
            m = b == k
            if m.any(): G[k] += Y0[m].T @ Y0[m]; C[k] += Y1[m].T @ Y0[m]; cnt[k] += int(m.sum())
    print(sh, used[sh], f"{time.time()-t0:.0f}s", flush=True); del t
del D; G = [g.cpu() for g in G]; C = [c.cpu() for c in C]; torch.cuda.empty_cache()   # GPU is shared and tight: keep G,C on CPU, move one bin at a time
print("bin counts", cnt, [f"{float(edges[k])*1000:.0f}-{float(edges[k+1])*1000:.0f}mm" for k in range(NB)], flush=True)
assert all(c >= MIN_ROWS for c in cnt)
# val sets: val split files, 600 random non-null rows per shard + 4-chain-free (one-step only)
NV = 600; V = load_split("val", SH); vs = {}
for sh in SH:
    t = rows_table(V[sh]); r = np.sort(rng.permutation(len(t["S"]))[:NV]); vs[sh] = {k: v[r] for k, v in t.items() if k in ("S", "S_", "P0", "P1")}
del V
I = torch.eye(RES * RES, device=dev)
def val_acc(model):
    a = []
    for sh in SH:
        v = vs[sh]; P, T, O, R = [], [], [], []
        for i in range(0, NV, 200):
            S0, S1, P0, P1 = v["S"][i:i+200], v["S_"][i:i+200], v["P0"][i:i+200], v["P1"][i:i+200]
            P.append(model.rollout(S0, P0[:, None], P1[:, None])[0]); T.append(occ_from_particles(S1)); O.append(occ_from_particles(S0)); R.append(ew.swept_region(torch.cat([P0, P1], 1), "cpu").cpu())
        a.append(ew.acc(*map(torch.cat, (P, T, O, R))))
    return float(np.mean(a)), a
res = {}; best = None
for lam in LAMS:
    ops = [torch.linalg.solve((G[k].to(dev) + lam * I).T, (C[k].to(dev) + lam * I).T).T.cpu() for k in range(NB)]   # float32 GPU solve per bin (repo's fit_switched does the same)
    ck = dict(operators=ops, bin_edges=edges, counts=cnt, res=RES, crop=1.0, constraint="ridge", ridge=lam, n_bins=NB)
    m = el.LF(ck, dev); mv, per = val_acc(m); res[lam] = dict(val_acc1=mv, per_shard=per); print("lambda", lam, "val acc1", round(mv, 4), flush=True)
    if best is None or mv > best[0]: best = (mv, lam, ck)
ck = best[2]; ck["train_rows"] = used; ck["val_acc1_by_lambda"] = {str(k): v["val_acc1"] for k, v in res.items()}
torch.save(ck, f"{OUT}/operators.pt")
json.dump(dict(best_lambda=best[1], val_acc1=best[0], bins_mm=[float(x) * 1000 for x in edges], counts=cnt, res=RES, rows_per_shard=NROWS, train_rows=used, val_rows_per_shard=NV,
               shards=SH, lambdas=res, min_rows_per_bin=MIN_ROWS), open(f"{OUT}/hparams.json", "w"), indent=1)
print("best", best[1], best[0])
