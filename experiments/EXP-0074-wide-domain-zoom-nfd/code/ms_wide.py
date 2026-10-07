"""Unrolled (multi-step) fine-tuning on Sean 4-push chains (build_ms_cache.py). zoom: canvas -> variable-side window -> UNet -> change pasted back
(window_var), loss = mean over T pushes of window MSE vs recorded state in that push's window; world: own raster output fed back, MSE on the raster.
Aug: random 90-degree world rotation (exact). python ms_wide.py --kind zoom|world --res R --init ckpt --T 4 --epochs N --out dir"""
import sys, os, time, json, argparse, numpy as np, torch
sys.path.insert(0, ".")
from model.UNetModels_modular import UNet
from model.zoom_nfd.window_var import *
from model.zoom_nfd.world_res import plates_res
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--res", type=int, required=True); ap.add_argument("--init", required=True)
ap.add_argument("--T", type=int, default=4); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--C", type=int, default=300); ap.add_argument("--features", default="4,8,16")
a = ap.parse_args(); torch.set_num_threads(4); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda"
D = torch.load(f"experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/ms_{a.kind}_r{a.res}.pt"); tr, va = D["train"], D["val"]; print("chains", len(tr["x0"]), len(va["x0"]), flush=True)
net = UNet({"features": [int(x) for x in a.features.split(",")], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu", "residual": True,
            "bottleneck_type": "None", "bottleneck_kwargs": {}}); net.load_state_dict(torch.load(a.init, map_location="cpu")); net.to(dev)
opt = torch.optim.Adam(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)


def rot(x, P0, P1, k):
    for _ in range(k):
        x = torch.rot90(x, 1, (-2, -1)); P0 = torch.stack([-P0[..., 1], P0[..., 0]], -1); P1 = torch.stack([-P1[..., 1], P1[..., 0]], -1)
    return x, P0, P1


def unroll(d, ix, T, aug):
    x0 = d["x0"][ix].to(dev).float(); Y = d["Y"][ix][:, :T].to(dev).float(); P0, P1 = d["P0"][ix][:, :T].to(dev), d["P1"][ix][:, :T].to(dev)
    k = int(np.random.randint(0, 4)) if aug else 0
    x0, P0, P1 = rot(x0, P0, P1, k)
    if a.kind == "world": Y = torch.rot90(Y, k, (-2, -1))
    losses = []; st = x0
    for j in range(T):
        if a.kind == "zoom":
            side = side_for((P1[:, j] - P0[:, j]).norm(dim=-1)); w = extract_windows_v(st, P0[:, j], P1[:, j], side, a.res)
            p = torch.sigmoid(net(torch.cat([w[:, None], plates_v(P0[:, j], P1[:, j], side, a.res)], 1))).squeeze(1)
            st = (st + paste_canvas_v(p - w, P0[:, j], P1[:, j], side, a.C)).clamp(0, 1)
        else:
            p = torch.sigmoid(net(torch.cat([st[:, None], plates_res(P0[:, j], P1[:, j], a.res)], 1))).squeeze(1); st = p
        losses.append(((p - Y[:, j]) ** 2).mean())
    return losses


@torch.no_grad()
def val():
    net.eval(); o = np.zeros(a.T); n = len(va["x0"])
    for i in range(0, n, 32):
        l = unroll(va, torch.arange(i, min(i + 32, n)), a.T, False); o += np.array([x.item() for x in l]) * len(l and range(i, min(i + 32, n)))
    return o / n


v0 = val(); print("init val per step", np.round(v0, 5), flush=True); best = v0.mean(); log = []; t0 = time.time(); N = len(tr["x0"])
for ep in range(1, a.epochs + 1):
    net.train(); perm = torch.randperm(N); tl = n = 0
    for i in range(0, N - a.bs + 1, a.bs):
        loss = sum(unroll(tr, perm[i:i + a.bs], a.T, True)) / a.T
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); n += 1
    sched.step(); v = val(); log.append(dict(epoch=ep, train=tl / n, val=v.tolist(), t=round(time.time() - t0)))
    if v.mean() < best: best = v.mean(); torch.save(net.state_dict(), f"{a.out}/unet_best.pth")
    torch.save(net.state_dict(), f"{a.out}/unet_last.pth"); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1); print(ep, round(tl / n, 5), np.round(v, 5), log[-1]["t"], "s", flush=True)
