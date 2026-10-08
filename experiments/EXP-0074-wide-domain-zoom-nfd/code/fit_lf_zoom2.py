"""v2: per-bin ridge lambda (chosen on VAL one-step acc1 per bin, pooled over shards) and more rows.  Fit LF-zoom (see lf_zoom.py) on Sean TRAIN rows (recovered pinned split, lf_split), 10k random rows per shard x 8 shards (arg1 rows/shard), ridge lambda selected on VAL rows (600/shard, one-step acc1 in the pasted 64x64 frame).
Window pairs are cached per bin in fp16 on the CPU, then per bin: G = W0^T W0, C = W1^T W0 (GPU), A = (C + lam I)(G + lam I)^-1.  usage: fit_lf_zoom.py [rows_per_shard] [shards comma list]"""
import sys, json, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from lf_split import load_split
from sean_data import rows_table
from lf_zoom import LFZoom, BIN_RES, EDGES, dev
from model.zoom_nfd.window_var import extract_windows_v
from model.zoom_nfd.rollout import canvas_from_particles
from simple_mpc.adapters import occ_from_particles
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000; SH = sys.argv[2].split(",") if len(sys.argv) > 2 else ew.SHARDS; LAMS = [1.0, 10.0, 100.0, 1000.0, 10000.0]; OUT = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/lf_zoom2"
NB = len(BIN_RES); rng = np.random.default_rng(0); W0 = [[] for _ in range(NB)]; W1 = [[] for _ in range(NB)]; t0 = time.time(); used = {}; edges = EDGES.to(dev)
D = load_split("train", SH)
for sh in SH:
    t = rows_table(D[sh]); N = len(t["S"]); ix = torch.from_numpy(np.sort(rng.permutation(N)[:NROWS])); used[sh] = [int(N), len(ix)]; n = t["S"].shape[1]
    for i in range(0, len(ix), 250):
        j = ix[i:i + 250]; c0 = canvas_from_particles(t["S"][j], [[0.005] * 3] * n, 300, 0.2).float().to(dev); c1 = canvas_from_particles(t["S_"][j], [[0.005] * 3] * n, 300, 0.2).float().to(dev)
        P0, P1 = t["P0"][j].to(dev), t["P1"][j].to(dev); b = torch.bucketize((P1 - P0).norm(dim=-1), edges[1:-1])
        for k in range(NB):
            m = torch.nonzero(b == k)[:, 0]
            if len(m) == 0: continue
            side = torch.full((len(m),), BIN_RES[k] / 1000.0, device=dev)
            W0[k].append(extract_windows_v(c0[m], P0[m], P1[m], side, BIN_RES[k]).reshape(len(m), -1).half().cpu()); W1[k].append(extract_windows_v(c1[m], P0[m], P1[m], side, BIN_RES[k]).reshape(len(m), -1).half().cpu())
    print(sh, used[sh], f"{time.time() - t0:.0f}s", flush=True); del t
del D; cnt = [int(sum(len(x) for x in W0[k])) for k in range(NB)]; print("bin counts", cnt, flush=True)
# val rows
V = load_split("val", SH); vs = {}
for sh in SH:
    t = rows_table(V[sh]); r = np.sort(rng.permutation(len(t["S"]))[:600]); vs[sh] = {k: v[r] for k, v in t.items() if k in ("S", "S_", "P0", "P1")}
del V
def val_acc(model):
    P, T, O, R, L = [], [], [], [], []
    for sh in SH:
        v = vs[sh]
        for i in range(0, 600, 200):
            S0, S1, P0, P1 = v["S"][i:i + 200], v["S_"][i:i + 200], v["P0"][i:i + 200], v["P1"][i:i + 200]
            P.append(model.rollout(S0, P0[:, None], P1[:, None])[0]); T.append(occ_from_particles(S1)); O.append(occ_from_particles(S0)); R.append(ew.swept_region(torch.cat([P0, P1], 1), "cpu").cpu()); L.append((P1 - P0).norm(dim=1))
    P, T, O, R, L = map(torch.cat, (P, T, O, R, L)); bi = torch.bucketize(L, EDGES[1:-1])
    return [float(ew.acc(P[bi == k], T[bi == k], O[bi == k], R[bi == k])) for k in range(NB)], float(ew.acc(P, T, O, R))
# accumulate G, C per bin on the GPU one bin at a time; keep the solves per lambda on CPU (fp32)
Gs, Cs = [], []
for k in range(NB):
    n = BIN_RES[k] ** 2; G = torch.zeros(n, n, device=dev); C = torch.zeros(n, n, device=dev); X0 = torch.cat(W0[k]); X1 = torch.cat(W1[k])
    for i in range(0, len(X0), 2000):
        a0 = X0[i:i + 2000].to(dev).float(); a1 = X1[i:i + 2000].to(dev).float(); G += a0.T @ a0; C += a1.T @ a0
    Gs.append(G.cpu()); Cs.append(C.cpu()); del G, C; torch.cuda.empty_cache(); print("bin", k, "res", BIN_RES[k], "G,C done", f"{time.time() - t0:.0f}s", flush=True)
del W0, W1
res = {}; allops = {}
for lam in LAMS:
    ops = []
    for k in range(NB):
        n = BIN_RES[k] ** 2; I = torch.eye(n, device=dev); G = Gs[k].to(dev); C = Cs[k].to(dev); ops.append(torch.linalg.solve((G + lam * I).T, (C + lam * I).T).T.cpu()); del G, C, I; torch.cuda.empty_cache()
    ck = dict(operators=ops, bin_edges=EDGES, counts=cnt, bin_res=BIN_RES, ridge=lam, n_bins=NB); m = LFZoom(ck); pb, a = val_acc(m); res[lam] = dict(per_bin=pb, pooled=a); allops[lam] = ops; print("lambda", lam, "val acc1 per bin", [round(x, 3) for x in pb], "pooled", round(a, 4), f"{time.time() - t0:.0f}s", flush=True); del m
bestlam = [max(LAMS, key=lambda l: res[l]["per_bin"][k]) for k in range(NB)]; ops = [allops[bestlam[k]][k] for k in range(NB)]
ck = dict(operators=ops, bin_edges=EDGES, counts=cnt, bin_res=BIN_RES, ridge=bestlam, n_bins=NB, train_rows=used); m = LFZoom(ck); pb, a = val_acc(m); print("per-bin lambda", bestlam, "val per bin", [round(x, 3) for x in pb], "pooled", round(a, 4), flush=True)
torch.save(ck, f"{OUT}/operators.pt"); json.dump(dict(best_lambda_per_bin=bestlam, val_pooled=a, val_per_bin=pb, counts=cnt, bin_res=BIN_RES, rows_per_shard=NROWS, lambdas=res), open(f"{OUT}/hparams.json", "w"), indent=1); print("best", bestlam, a)
