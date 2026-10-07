"""Single-step training on the Sean caches (build_cache.py). kind zoom: variable-side windows (+ lateral-flip aug); kind world: plain world raster at res
(+ D4 aug). Plates are rendered on the GPU per batch. MSE on sigmoid output over the whole raster (same recipe as the narrow pilots).
Filters --shards / --lmin --lmax train on a bin only (fallback experiments)."""
import sys, os, time, json, argparse, numpy as np, torch
sys.path.insert(0, ".")
from model.UNetModels_modular import UNet
from model.zoom_nfd.window_var import plates_v, side_for
from model.zoom_nfd.world_res import plates_res
SHARDS = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--res", type=int, required=True)
ap.add_argument("--init", default=""); ap.add_argument("--features", default="4,8,16"); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shards", default=""); ap.add_argument("--lmin", type=float, default=0); ap.add_argument("--lmax", type=float, default=1e9); ap.add_argument("--cache", default="")
torch.set_num_threads(4)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
D = torch.load(a.cache or f"experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/{a.kind}_r{a.res}.pt")
def select(d):
    L = (d["P1"] - d["P0"]).norm(dim=1) * 1000; m = (L >= a.lmin) & (L <= a.lmax)
    if a.shards: m &= torch.isin(d["shard"], torch.tensor([SHARDS.index(s) for s in a.shards.split(",")]))
    return {k: v[m] for k, v in d.items()}
tr, va = select(D["train"]), select(D["val"]); print("train rows", len(tr["x"]), "val rows", len(va["x"]), flush=True)
net = UNet({"features": [int(x) for x in a.features.split(",")], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu",
            "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
if a.init: net.load_state_dict(torch.load(a.init, map_location="cpu"))
net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

def batch(d, ix, aug):
    x, y = d["x"][ix].to(dev).float(), d["y"][ix].to(dev).float(); P0, P1 = d["P0"][ix].to(dev), d["P1"][ix].to(dev)
    if a.kind == "zoom": pl = plates_v(P0, P1, side_for((P1 - P0).norm(dim=-1)), a.res)
    else: pl = plates_res(P0, P1, a.res)
    if aug:
        if a.kind == "zoom":
            fl = torch.rand(len(ix), device=dev) < 0.5; x = torch.where(fl[:, None, None], x.flip(1), x); y = torch.where(fl[:, None, None], y.flip(1), y)
        else:
            k = int(torch.randint(0, 4, (1,))); x, y, pl = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2)), torch.rot90(pl, k, (2, 3))
            if torch.rand(1) < 0.5: x, y, pl = x.flip(2), y.flip(2), pl.flip(3)
    return torch.cat([x[:, None], pl], 1), y

@torch.no_grad()
def val():
    net.eval(); per = np.zeros(len(SHARDS)); cnt = np.zeros(len(SHARDS))
    for i in range(0, len(va["x"]), 256):
        ix = torch.arange(i, min(i + 256, len(va["x"]))); x, y = batch(va, ix, False); e = ((torch.sigmoid(net(x)).squeeze(1) - y) ** 2).mean((1, 2)).cpu().numpy()
        for s, v in zip(va["shard"][ix].numpy(), e): per[s] += v; cnt[s] += 1
    return per / np.maximum(cnt, 1), float(per.sum() / cnt.sum())
pv, mv = val(); log = [dict(epoch=0, val=mv, per=pv.tolist())]; best = mv; print("init val", round(mv, 5), flush=True); t0 = time.time()
for ep in range(1, a.epochs + 1):
    net.train(); perm = torch.randperm(len(tr["x"])); tl = n = 0
    for i in range(0, len(perm) - a.bs + 1, a.bs):
        x, y = batch(tr, perm[i:i + a.bs], True); loss = ((torch.sigmoid(net(x)).squeeze(1) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); n += 1
    sched.step(); pv, mv = val(); log.append(dict(epoch=ep, train=tl / n, val=mv, per=pv.tolist(), t=round(time.time() - t0)))
    if mv < best: best = mv; torch.save(net.state_dict(), f"{a.out}/unet_best.pth")
    torch.save(net.state_dict(), f"{a.out}/unet_last.pth"); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
    print(ep, round(tl / n, 5), round(mv, 5), log[-1]["t"], "s", flush=True)
