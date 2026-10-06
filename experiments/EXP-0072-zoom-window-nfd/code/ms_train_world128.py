"""Control for ms_train.py: the plain world-frame 128x128 NFD (no crop, no window) fine-tuned through T unrolled pushes on the same
DS-0015 chains with the same recipe (own output fed back as the next state; MSE on the whole 128x128 raster; random 90-degree rotation aug)."""
import sys, os, time, json, argparse, numpy as np, torch
sys.path.insert(0, ".")
from model.zoom_nfd.rollout import load_chains
from model.zoom_nfd.world128 import raster128, plates128
from model.UNetModels_modular import UNet
SIZES = [[0.005] * 3] * 20; dev = "cuda"
ap = argparse.ArgumentParser(); ap.add_argument("--T", type=int, default=4); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--init", default="experiments/EXP-0072-zoom-window-nfd/runs/world128_300/unet_best.pth"); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(0); np.random.seed(0)
cache = "experiments/EXP-0072-zoom-window-nfd/runs/ms_cache_world128.pt"
if os.path.exists(cache): D = torch.load(cache)
else:
    ch = load_chains("Genesis/data/narrow_l20_n20/train_v2/_*_data.pt"); E, T = ch["S"].shape[:2]
    R = torch.cat([raster128(ch["S"].reshape(E * T, 20, 7), SIZES).reshape(E, T, 128, 128), raster128(ch["S_"][:, -1], SIZES)[:, None]], 1).half()   # (E,T+1,128,128)
    Pl = torch.cat([plates128(ch["P0"].reshape(-1, 2), ch["P1"].reshape(-1, 2)).reshape(E, T, 2, 128, 128)]).half()
    D = dict(R=R, Pl=Pl); torch.save(D, cache)
Rr, Pl = D["R"].to(dev), D["Pl"].to(dev); E, Tt = Pl.shape[:2]
net = UNet({"features": [4, 8, 16], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu", "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
net.load_state_dict(torch.load(a.init, map_location="cpu")); net.to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
perm = np.random.default_rng(0).permutation(E); nval = max(16, E // 20); va, tr = perm[:nval], perm[nval:]
def unroll(e, s, T, k=0):
    st = torch.rot90(Rr[e, s].float(), k, (-2, -1)); out = []
    for j in range(T):
        pl = torch.rot90(Pl[e, s + j].float(), k, (-2, -1)); st = torch.sigmoid(net(torch.cat([st[:, None], pl], 1))).squeeze(1)
        out.append(((st - torch.rot90(Rr[e, s + j + 1].float(), k, (-2, -1))) ** 2).mean())
    return out
@torch.no_grad()
def val():
    net.eval(); o = np.zeros(4)
    for i in range(0, len(va), 32):
        e = torch.from_numpy(va[i:i + 32]).to(dev); l = unroll(e, 0, 4); o += np.array([x.item() for x in l]) * len(e)
    return o / len(va)
v0 = val(); print("init", np.round(v0, 5), flush=True); best = v0.mean(); log = []; t0 = time.time()
for ep in range(1, a.epochs + 1):
    net.train(); pm = np.random.permutation(tr); tl = n = 0
    for i in range(0, len(pm) - a.bs + 1, a.bs):
        e = torch.from_numpy(pm[i:i + a.bs]).to(dev); s = int(np.random.randint(0, Tt - a.T + 1))
        loss = sum(unroll(e, s, a.T, int(np.random.randint(0, 4)))) / a.T
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); n += 1
    sched.step(); v = val(); log.append(dict(epoch=ep, train=tl / n, val=v.tolist()))
    if v.mean() < best: best = v.mean(); torch.save(net.state_dict(), f"{a.out}/unet_best.pth")
    torch.save(net.state_dict(), f"{a.out}/unet_last.pth"); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1); print(ep, round(tl / n, 5), np.round(v, 5), round(time.time() - t0), "s", flush=True)
