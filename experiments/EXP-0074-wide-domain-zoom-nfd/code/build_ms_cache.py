"""Multi-step (unrolled) training caches: 4-push chains (non-overlapping segments of linked rows) from Sean train/val files.
zoom: canvas0 (N,300,300) uint8 of the first state + Y (N,4,res,res) label windows (own window per push); world: x0 (N,res,res) + Y (N,4,res,res) state rasters after each push.
python build_ms_cache.py KIND RES [MAXPER]  -> artifacts/ms_{kind}_r{res}.pt"""
import sys, time, numpy as np, torch
from multiprocessing import Pool
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
from sean_data import *
from model.zoom_nfd.window_var import window_batch_var
from model.zoom_nfd.world_res import raster_res
from model.zoom_nfd.rollout import canvas_from_particles
SHARDS = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
kind, res = sys.argv[1], int(sys.argv[2]); maxper = int(sys.argv[3]) if len(sys.argv) > 3 else 2500


def work(a):
    S0, Sk, P0, P1, n = a; sz = [[0.005] * 3] * n; B = len(S0)
    if kind == "zoom":
        Y = [window_batch_var(Sk[:, k], sz, P0[:, k], P1[:, k], res).numpy() > 0.5 for k in range(4)]
        return canvas_from_particles(S0, sz, 300, 0.2).numpy(), np.stack(Y, 1)
    return raster_res(S0, sz, res).numpy() > 0.5, np.stack([raster_res(Sk[:, k], sz, res).numpy() > 0.5 for k in range(4)], 1)


if __name__ == "__main__":
    out = {}
    for split, cap in (("train", maxper), ("val", 120)):
        A, Y, P0s, P1s, SH = [], [], [], [], []; D = load_split(split)
        for si, sh in enumerate(SHARDS):
            t = rows_table(D[sh]); ch = chains_of(t, minlen=4, maxlen=4); ch = [ch[i] for i in np.random.default_rng(si).permutation(len(ch))[:cap]]
            if not ch: continue
            c = torch.tensor(ch); n = t["S"].shape[1]; t0 = time.time()
            chunks = [(t["S"][c[i, 0]], t["S_"][c[i]].transpose(0, 1).transpose(0, 1) if False else t["S_"][c[i]], t["P0"][c[i]].numpy(), t["P1"][c[i]].numpy(), n) for i in np.array_split(np.arange(len(c)), 40) if len(i)]
            chunks = [(a[0], a[1], a[2], a[3], a[4]) for a in chunks]
            with Pool(16) as p: r = p.map(work, chunks)
            A.append(torch.from_numpy(np.concatenate([a for a, _ in r]).astype(np.uint8))); Y.append(torch.from_numpy(np.concatenate([b for _, b in r]).astype(np.uint8)))
            P0s.append(t["P0"][c]); P1s.append(t["P1"][c]); SH.append(torch.full((len(c),), si)); print(split, sh, len(c), f"{time.time()-t0:.0f}s", flush=True)
        out[split] = dict(x0=torch.cat(A), Y=torch.cat(Y), P0=torch.cat(P0s), P1=torch.cat(P1s), shard=torch.cat(SH))
    torch.save(out, f"experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/ms_{kind}_r{res}.pt"); print("saved")
