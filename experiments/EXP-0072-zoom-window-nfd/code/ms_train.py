"""Multi-step (unrolled) fine-tuning of the zoom-window NFD on recorded 8-push chains (DS-0015 train_v2), training on the
model's OWN output as the next input: canvas -> window -> UNet -> predicted change pasted back into the canvas -> next window ...
(model/zoom_nfd/rollout.py; differentiable end to end). Loss = mean over T unrolled pushes of the window MSE against the recorded
state after that push, in that push's own window. Augmentation: random 90-degree world rotation (exact: windows are push-relative).
T=1 is the control (canvas-extracted inputs, no unrolling)."""
import sys, os, time, json, argparse, numpy as np, torch
sys.path.insert(0, ".")
from model.zoom_nfd.window import WindowSpec, window_batch
from model.zoom_nfd.window_gpu import plates_gpu
from model.zoom_nfd.rollout import *
from model.UNetModels_modular import UNet
SIZES = [[0.005] * 3] * 20; dev = "cuda"


def prepare(res, C, cache):
    if os.path.exists(cache): return torch.load(cache)
    ch = load_chains("Genesis/data/narrow_l20_n20/train_v2/_*_data.pt"); E, T = ch["S"].shape[:2]; spec = WindowSpec(res=res)
    t = time.time(); canv = canvas_from_particles(ch["S"].reshape(E * T, 20, 7), SIZES, C).reshape(E, T, C, C)
    Y = window_batch(ch["S_"].reshape(E * T, 20, 7), SIZES, ch["P0"].reshape(-1, 2).numpy(), ch["P1"].reshape(-1, 2).numpy(), spec).reshape(E, T, res, res).half()
    d = dict(canv=canv, Y=Y, P0=ch["P0"], P1=ch["P1"]); torch.save(d, cache); print("prepared", E, T, f"{time.time()-t:.0f}s", flush=True); return d


def rot(canvas, P0, P1, k):
    for _ in range(k):
        canvas = torch.rot90(canvas, 1, (-2, -1)); P0 = torch.stack([-P0[..., 1], P0[..., 0]], -1); P1 = torch.stack([-P1[..., 1], P1[..., 0]], -1)
    return canvas, P0, P1


def unroll(net, canvas, P0s, P1s, spec, C, T):
    ws, preds = [], []
    for j in range(T):
        P0, P1 = P0s[:, j], P1s[:, j]
        w = extract_windows_b(canvas, P0, P1, spec); x = torch.cat([w[:, None], plates_gpu(P0, P1, spec)], 1)
        p = torch.sigmoid(net(x)).squeeze(1); preds.append(p)
        canvas = (canvas + paste_canvas(p - w, P0, P1, spec, C)).clamp(0, 1)
    return preds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=64); ap.add_argument("--C", type=int, default=300); ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default="experiments/EXP-0072-zoom-window-nfd/runs/ft300/unet_best.pth"); ap.add_argument("--features", default="4,8,16")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--test-rot", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
    spec = WindowSpec(res=a.res)
    D = prepare(a.res, a.C, f"experiments/EXP-0072-zoom-window-nfd/runs/ms_cache_r{a.res}_c{a.C}.pt")
    canv, Y, P0, P1 = D["canv"].to(dev), D["Y"].to(dev), D["P0"].to(dev), D["P1"].to(dev); E, Tt = P0.shape[:2]
    if a.test_rot:   # exactness of the augmentation: windows from rotated canvas + rotated pushes == original windows
        c = canv[:4, 0].float(); w0 = extract_windows_b(c, P0[:4, 0], P1[:4, 0], spec)
        for k in (1, 2, 3):
            c2, p0, p1 = rot(c, P0[:4, 0], P1[:4, 0], k); print("rot", k, "max abs window diff", float((extract_windows_b(c2, p0, p1, spec) - w0).abs().max())); 
        sys.exit()
    net = UNet({"features": [int(x) for x in a.features.split(",")], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu",
                "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
    if a.init != "none": net.load_state_dict(torch.load(a.init, map_location="cpu"))
    net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    perm = np.random.default_rng(0).permutation(E); nval = max(16, E // 20); va, tr = perm[:nval], perm[nval:]
    @torch.no_grad()
    def val(T):
        net.eval(); out = np.zeros(T)
        for i in range(0, len(va), 32):
            e = torch.from_numpy(va[i:i + 32]).to(dev)
            pr = unroll(net, canv[e, 0].float(), P0[e], P1[e], spec, a.C, T)
            for j in range(T): out[j] += ((pr[j] - Y[e, j].float()) ** 2).mean((1, 2)).sum().item()
        return out / len(va)
    log = []; v0 = val(4); print("init val mse per step (rollout from step 0):", np.round(v0, 5), flush=True); best = v0.mean(); t0 = time.time()
    for ep in range(1, a.epochs + 1):
        net.train(); pm = np.random.permutation(tr); tl = 0; n = 0
        for i in range(0, len(pm) - a.bs + 1, a.bs):
            e = torch.from_numpy(pm[i:i + a.bs]).to(dev); s = int(np.random.randint(0, Tt - a.T + 1))
            c, p0, p1 = rot(canv[e, s].float(), P0[e, s:s + a.T], P1[e, s:s + a.T], int(np.random.randint(0, 4)))
            pr = unroll(net, c, p0, p1, spec, a.C, a.T)
            loss = sum(((pr[j] - Y[e, s + j].float()) ** 2).mean() for j in range(a.T)) / a.T
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); tl += loss.item(); n += 1
        sched.step(); v = val(4); log.append(dict(epoch=ep, train=tl / n, val=v.tolist(), t=round(time.time() - t0)))
        if v.mean() < best: best = v.mean(); torch.save(net.state_dict(), f"{a.out}/unet_best.pth")
        torch.save(net.state_dict(), f"{a.out}/unet_last.pth"); json.dump(log, open(f"{a.out}/log.json", "w"), indent=1)
        print(ep, round(tl / n, 5), np.round(v, 5), log[-1]["t"], "s", flush=True)
