"""Fine-tune the EXP-0022 RUN-0010 warped NFD UNet on zoom-window rasters (DS-0015 train_v2, narrow pilot).
Same UNet / MSE-on-sigmoid recipe as the warped NFD; row-flip (mirror across the push axis) augmentation."""
import sys, time, json, argparse, torch
sys.path.insert(0, ".")
from model.UNetModels_modular import UNet
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--out", default="experiments/EXP-0072-zoom-window-nfd/runs/ft1")
ap.add_argument("--init", default="Baselines/NFD/runs/nfd_warped_randlen_flipaug/unet_best.pth")
ap.add_argument("--cache", default="experiments/EXP-0072-zoom-window-nfd/runs/window_cache.pt"); ap.add_argument("--aug", default="flip")
ap.add_argument("--features", default="4,8,16"); ap.add_argument("--subset", type=int, default=0)
ap.add_argument("--scratch", action="store_true"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
import os; os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
dev = "cuda"
net = UNet({"features": [int(x) for x in a.features.split(",")], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1,
            "activation": "relu", "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
if not a.scratch:
    net.load_state_dict(torch.load(a.init, map_location="cpu"))
net.to(dev)
D = torch.load(a.cache)
import numpy as np
if a.subset:
    ix = np.sort(np.random.default_rng(0).permutation(len(D["train"]["x"]))[:a.subset]); D["train"]["x"], D["train"]["y"] = D["train"]["x"][ix], D["train"]["y"][ix]   # same 2000-row seed subset as EXP-0073
Xtr, Ytr = D["train"]["x"].to(dev), D["train"]["y"].to(dev); Xva, Yva = D["val"]["x"].to(dev).float(), D["val"]["y"].to(dev).float()
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

def fwd(x):
    o = net(x); return torch.sigmoid(o.squeeze(1) if o.dim() == 4 else o)
@torch.no_grad()
def val():
    net.eval(); l = 0
    for i in range(0, len(Xva), 128):
        l += ((fwd(Xva[i:i+128]) - Yva[i:i+128]) ** 2).mean((1, 2)).sum().item()
    return l / len(Xva)
log = [dict(epoch=0, val_mse=val())]; best = log[0]["val_mse"]; print(log[0], flush=True)
t0 = time.time()
for ep in range(1, a.epochs + 1):
    net.train(); perm = torch.randperm(len(Xtr), device=dev); tl = 0; n = 0
    for i in range(0, len(perm) - a.bs + 1, a.bs):
        ix = perm[i:i + a.bs]; x, y = Xtr[ix].float(), Ytr[ix].float()
        if a.aug == "flip":
            fl = torch.rand(len(ix), device=dev) < 0.5
            x = torch.where(fl[:, None, None, None], x.flip(2), x); y = torch.where(fl[:, None, None], y.flip(1), y)
        else:   # world frame: random rot90 x flip per batch (the D4 group the narrow NFD recipe uses)
            k = int(torch.randint(0, 4, (1,))); x = torch.rot90(x, k, (2, 3)); y = torch.rot90(y, k, (1, 2))
            if torch.rand(1) < 0.5: x, y = x.flip(3), y.flip(2)
        loss = ((fwd(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); n += 1
    sched.step(); v = val()
    log.append(dict(epoch=ep, train_mse=tl / n, val_mse=v, t=round(time.time() - t0)))
    if v < best:
        best = v; torch.save(net.state_dict(), f"{a.out}/unet_best.pth")
    torch.save(net.state_dict(), f"{a.out}/unet_last.pth")
    json.dump(log, open(f"{a.out}/log.json", "w"), indent=1); print(log[-1], flush=True)
