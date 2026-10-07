"""Training/val caches from Sean train/val files: per shard up to NPER random non-null rows -> uint8 windows (zoom, variable side) and world rasters.
x = state, y = label rasters; meta: shard id, P0, P1, L. python build_cache.py KIND RES [NPER]; KIND in {zoom, world}; output artifacts/{kind}_r{RES}.pt"""
import os, sys, time, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
from sean_data import *
from model.zoom_nfd.window_var import window_batch_var
from model.zoom_nfd.world_res import raster_res
SHARDS = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
ONLY = [x for x in os.environ.get("ONLY", "").split(",") if x]; LMAX = float(os.environ.get("LMAX", 1e9)) / 1000; LMIN = float(os.environ.get("LMIN", 0)) / 1000
kind, res = sys.argv[1], int(sys.argv[2]); nper = int(sys.argv[3]) if len(sys.argv) > 3 else 10000


def work(a):
    S, S_, P0, P1, n = a; sz = [[0.005] * 3] * n
    if kind == "zoom": return window_batch_var(S, sz, P0, P1, res).numpy() > 0.5, window_batch_var(S_, sz, P0, P1, res).numpy() > 0.5
    return raster_res(S, sz, res).numpy() > 0.5, raster_res(S_, sz, res).numpy() > 0.5


if __name__ == "__main__":
    out = {}
    for split, cap in (("train", nper), ("val", 600)):
        X, Y, M = [], [], []; D = load_split(split)
        for si, sh in enumerate(SHARDS):
            if ONLY and sh not in ONLY: continue
            t = rows_table(D[sh]); N = len(t["S"]); okL = np.nonzero((((t["P1"] - t["P0"]).norm(dim=1) <= LMAX) & ((t["P1"] - t["P0"]).norm(dim=1) >= LMIN)).numpy())[0]; ix = np.sort(np.random.default_rng(si).permutation(okL)[:cap]); n = t["S"].shape[1]; t0 = time.time()
            chunks = [(t["S"][c], t["S_"][c], t["P0"][c].numpy(), t["P1"][c].numpy(), n) for c in np.array_split(ix, 40)]
            with Pool(18) as p: r = p.map(work, chunks)
            X.append(torch.from_numpy(np.concatenate([a for a, _ in r]).astype(np.uint8))); Y.append(torch.from_numpy(np.concatenate([b for _, b in r]).astype(np.uint8)))
            M.append(dict(shard=torch.full((len(ix),), si), P0=t["P0"][ix], P1=t["P1"][ix])); print(split, sh, len(ix), f"{time.time()-t0:.0f}s", flush=True)
        out[split] = dict(x=torch.cat(X), y=torch.cat(Y), shard=torch.cat([m["shard"] for m in M]), P0=torch.cat([m["P0"] for m in M]), P1=torch.cat([m["P1"] for m in M]))
    torch.save(out, f"experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/{kind}_r{res}{os.environ.get('ZTAG', '')}.pt"); print("saved")
