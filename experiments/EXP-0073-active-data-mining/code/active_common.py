"""PILOT (EXP-0072 active plan v2): shared helpers for the retrospective active-learning study.
 - train_rows(): DS-0015 train_v2 (exclude_flagged) rows aligned 1:1 with runs/window_cache.pt (states, next states, p0, p1)
 - train_unet(): same recipe as code/train_zoom.py (MSE on sigmoid, Adam lr 2e-4 cosine, bs 32, row-flip aug), callable
 - EvalSet: precomputed window rasters / goal fields for a pools file -> window slateN, window accuracy_1, per-row e_sw
"""
import sys, os, glob, json, time
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code"); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0053-narrow-domain-models/code")
from model.zoom_nfd.window import *
from model.zoom_nfd.window import _window_pixel_world, push_axes
from model.UNetModels_modular import UNet
DEV = os.environ.get("ADEV", "cuda"); R = "experiments/EXP-0072-zoom-window-nfd/runs/"; AP = "experiments/EXP-0073-active-data-mining/runs/active_pilot/"
def rp(p): return AP + p[len("active_pilot/"):] if p.startswith("active_pilot/") else R + p   # EXP-0072 runs vs EXP-0073 active_pilot
SIZES = [[0.005] * 3] * 20; spec = WindowSpec(res=64); D = "Genesis/data/narrow_l20_n20/"
from Baselines.common.goals import dist_field_from_mask
from batched_closed_loop import goal_mask
from eval_narrow import GOALS


def train_rows():
    f = AP + "train_rows.pt"
    if os.path.exists(f): return torch.load(f, weights_only=False)
    from Baselines.NFD.nfd_lib import PileSweepData3Ch
    out = {}
    for split in ("train", "val"):
        ds = PileSweepData3Ch(paths=["narrow_l20_n20/train_v2"], split=split, resolution_scale=0.5, exclude_flagged=True, min_push_length_m=1e-4)
        S, S_, P0, P1 = [], [], [], []
        for i in range(len(ds)):
            ri = ds._resolve_idx(i); r = ds._run_lookup[ri]; k = ri - ds._offsets[r]; run = ds.runs[r]
            S.append(run["states"][k]); S_.append(run["states_"][k]); P0.append(run["p_starts"][k][:2].numpy()); P1.append(run["p_stops"][k][:2].numpy())
        out[split] = dict(S=torch.stack(S).float(), S_=torch.stack(S_).float(), P0=np.stack(P0), P1=np.stack(P1))
    torch.save(out, f); return out


def new_net():
    return UNet({"features": [4, 8, 16], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu",
                 "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})


def load_net(path, dev=DEV):
    n = new_net(); n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval()


def train_unet(X, Y, Xva, Yva, epochs, seed, out, init=None, lr=2e-4, bs=32, dev=DEV, w=None):
    """X (N,3,64,64) half, Y (N,64,64) half. Saves out/unet_best.pth (best val MSE) + log. Skips if done. w: optional per-row loss weights."""
    if os.path.exists(out + "/unet_best.pth") and os.path.exists(out + "/done"): return out + "/unet_best.pth"
    os.makedirs(out, exist_ok=True); torch.manual_seed(seed)
    net = new_net()
    if init: net.load_state_dict(torch.load(init, map_location="cpu"))
    net.to(dev); X, Y = X.to(dev), Y.to(dev); Xva, Yva = Xva.to(dev).float(), Yva.to(dev).float()
    W = None if w is None else torch.as_tensor(w, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    fwd = lambda x: torch.sigmoid(net(x).squeeze(1))
    best = 1e9; log = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(len(X), device=dev); tl = 0; n = 0
        for i in range(0, len(perm) - bs + 1, bs):
            ix = perm[i:i + bs]; x, y = X[ix].float(), Y[ix].float()
            fl = torch.rand(len(ix), device=dev) < 0.5
            x = torch.where(fl[:, None, None, None], x.flip(2), x); y = torch.where(fl[:, None, None], y.flip(1), y)
            l = ((fwd(x) - y) ** 2).mean((1, 2)); loss = l.mean() if W is None else (l * W[ix]).sum() / W[ix].sum()
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); n += 1
        sch.step()
        if ep % 5 == 0 or ep == epochs:
            net.eval()
            with torch.no_grad(): v = sum(((fwd(Xva[i:i + 256]) - Yva[i:i + 256]) ** 2).mean((1, 2)).sum().item() for i in range(0, len(Xva), 256)) / len(Xva)
            log.append(dict(epoch=ep, train=tl / n, val=v, t=round(time.time() - t0)))
            if v < best: best = v; torch.save(net.state_dict(), out + "/unet_best.pth")
    json.dump(log, open(out + "/log.json", "w")); open(out + "/done", "w").write(str(best)); return out + "/unet_best.pth"


@torch.no_grad()
def predict(net, x, dev=DEV):
    return torch.cat([torch.sigmoid(net(x[i:i + 512].to(dev).float())).squeeze(1).cpu() for i in range(0, len(x), 512)])


class EvalSet:
    """rows: S, S_ (B,20,7), P0, P1 (B,2) numpy, pool (B,) int (-1 = not part of a same-state pool)."""
    def __init__(self, S, S_, P0, P1, pool, name=""):
        self.name, self.pool, self.P0, self.P1 = name, np.asarray(pool), P0, P1; B = len(S)
        self.Xw = window_batch(S, SIZES, P0, P1, spec); self.Yw = window_batch(S_, SIZES, P0, P1, spec)
        self.x = torch.cat([self.Xw[:, None], plates_batch(P0, P1, spec)], 1)
        self.R = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(B)]).float()
        self.S, self.S_ = S, S_
        pools = [p for p in np.unique(self.pool) if p >= 0]; self.pools = {p: np.nonzero(self.pool == p)[0] for p in pools}
        Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}; self.goals = list(GOALS)
        self.TW0 = {}; self.TW1 = {}; self.Dw = {}
        for p, ix in self.pools.items():
            self.TW1[p] = torch.stack([splat_window_mass(S_[b], P0[b], P1[b], spec) for b in ix]); self.TW0[p] = torch.stack([splat_window_mass(S[b], P0[b], P1[b], spec) for b in ix])
            self.Dw[p] = torch.stack([torch.stack([sample_world_field_into_window(Dist[g], P0[b], P1[b], spec) for g in self.goals]) for b in ix]).half()  # (n,G,64,64)

    @staticmethod
    def lw(occ, Dw):   # occ (n,64,64), Dw (n,G,64,64) -> (n,G)
        return (occ[:, None] * Dw.float()).sum((2, 3)) / occ.sum((1, 2)).clamp_min(1e-6)[:, None]

    def evaluate(self, preds):
        """preds: (B,64,64) probabilities (ensemble mean or single model). Returns dict: slateN_window, acc1_window, e_sw (per row), capture per pool/goal."""
        Pw = preds.float(); se = (Pw - self.Yw) ** 2; e_sw = (se * self.R).sum((1, 2)).numpy()
        pers = (((self.Xw - self.Yw) ** 2) * self.R).sum().item(); acc = 1 - se.mul(self.R).sum().item() / pers
        caps = {}
        for p, ix in self.pools.items():
            ixt = torch.as_tensor(ix); vt = (self.lw(self.TW1[p], self.Dw[p]) - self.lw(self.TW0[p], self.Dw[p])).numpy()
            vp = (self.lw(Pw[ixt], self.Dw[p]) - self.lw(self.Xw[ixt], self.Dw[p])).numpy()
            for gi, g in enumerate(self.goals):
                den = vt[:, gi].mean() - vt[:, gi].min()
                if den > 1e-9: caps.setdefault(g, []).append(float((vt[:, gi].mean() - vt[vp[:, gi].argmin(), gi]) / den))
        sl = float(np.mean([np.mean(v) for v in caps.values()]))
        return dict(slateN=sl, acc1=float(acc), e_sw=e_sw, e_sw_mean=float(e_sw.mean()))

    def optimism(self, preds):
        """per pool x goal: model's top-1 push (lowest predicted dv); optimism = predicted dv - true dv of that push (negative = over-optimistic: predicted better than simulated).
        Also true-rank percentile of the picked push. Returns arrays over (pool, goal) and per picked row index."""
        Pw = preds.float(); out = []
        for p, ix in self.pools.items():
            ixt = torch.as_tensor(ix); vt = (self.lw(self.TW1[p], self.Dw[p]) - self.lw(self.TW0[p], self.Dw[p])).numpy()
            vp = (self.lw(Pw[ixt], self.Dw[p]) - self.lw(self.Xw[ixt], self.Dw[p])).numpy()
            for gi in range(vt.shape[1]):
                j = vp[:, gi].argmin(); out.append((ix[j], p, gi, vp[j, gi] - vt[j, gi], float((vt[:, gi] < vt[j, gi]).mean())))
        return out


def eval_set_pools(path, name):
    d = torch.load(path, map_location="cpu", weights_only=False); v = d["valid"].bool() if "valid" in d else torch.ones(len(d["states"]), dtype=torch.bool)
    return EvalSet(d["states"].float()[v], d["states_"].float()[v], d["p_starts"][v][:, :2].numpy(), d["p_stops"][v][:, :2].numpy(), d["pool_idx"][v].numpy(), name)
