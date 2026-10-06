"""Score a zoom-window NFD on DS-0016 (clean narrow test) two ways, side by side:

  PASTED : window prediction pasted onto the standard 64x64 world raster (persistence outside the
           window), then scored EXACTLY like EXP-0053 eval_narrow (world truth = occ_from_particles,
           swept_region, slateN with soft truth occ_for_scoring, lyapunov, same 13 goals).
           Comparable with every other model (also computed here for reference adapters).
  WINDOW : everything inside the window: truth = the window raster of the next state, region =
           the same swept rectangle in window pixels, goal distance field CUT into the window
           (bilinear), value = mass-normalised lyapunov on the window raster, true dv from a
           mass-conserving splat of cube centres in the window. Upper bound on what this model
           type can show (finer grid, no paste loss, window-local goal).

Window inputs are rasterised from the particle state (like training); reference adapters get the
standard raster. State that asymmetry when quoting numbers.
"""
import argparse, glob, json, sys
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code")
from batched_closed_loop import goal_mask
from Baselines.common.goals import dist_field_from_mask
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles
from simple_mpc.learned_mpc import lyap
sys.path.insert(0, "experiments/EXP-0053-narrow-domain-models/code")
from eval_narrow import swept_region, acc, GOALS
from model.UNetModels_modular import UNet
from model.zoom_nfd.window import *
from model.zoom_nfd.window import _window_pixel_world
from model.zoom_nfd.window_gpu import plates_gpu

D = "Genesis/data/narrow_l20_n20/"
SIZES = [[0.005] * 3] * 20


def load_unet(path):
    net = UNet({"features": [4, 8, 16], "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1,
                "activation": "relu", "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
    net.load_state_dict(torch.load(path, map_location="cpu")); return net.eval()


@torch.no_grad()
def predict_windows(net, S, P0, P1, spec, dev):
    X = window_batch(S, SIZES, P0, P1, spec); A = plates_batch(P0, P1, spec)
    x = torch.cat([X[:, None], A], 1)
    out = torch.cat([torch.sigmoid(net(x[i:i + 256].to(dev))).squeeze(1).cpu() for i in range(0, len(x), 256)])
    return X, out


class ZoomModel:
    def __init__(self, net, dev="cuda"): self.net = net.to(dev); self.dev = dev
    def windows(self, S, P0, P1, spec):
        Xw, Pw = predict_windows(self.net, S, P0, P1, spec, self.dev); return Xw, Pw, None
    def paste(self, ctx, Xw, Pw, b, o0, p0, p1, spec): return paste_delta(Pw[b], Xw[b], o0, p0, p1, spec)


class ZoomModelHR(ZoomModel):
    """Zoom model fed from a VISUAL hi-res whole-tray raster (HxH, binary) via rotation+zoom grid_sample (window_gpu)."""
    def __init__(self, net, H, thr=None, dev="cuda", aa=1):
        super().__init__(net, dev); self.H, self.thr, self.aa = H, thr, aa
    @torch.no_grad()
    def windows(self, S, P0, P1, spec):
        from model.zoom_nfd.window_gpu import raster_world, extract_windows
        hr = raster_world(S, SIZES, self.H, aa=self.aa).to(self.dev)
        p0, p1 = torch.from_numpy(np.asarray(P0)).float().to(self.dev), torch.from_numpy(np.asarray(P1)).float().to(self.dev)
        X = torch.cat([extract_windows(hr[b], p0[b:b+1], p1[b:b+1], spec) for b in range(len(S))])
        if self.thr is not None: X = (X > self.thr).float()
        A = plates_gpu(p0, p1, spec)
        x = torch.cat([X[:, None], A], 1)
        Pw = torch.cat([torch.sigmoid(self.net(x[i:i + 256])).squeeze(1) for i in range(0, len(x), 256)])
        return X.cpu(), Pw.cpu(), None


class World128Model:
    """Plain world-frame NFD on 128x128 rasters (1 mm/px). Prediction is resampled into the same
    window for the window-frame score, or its CHANGE is sampled onto the 64 world grid for the paste score."""
    def __init__(self, net, dev="cuda"): self.net = net.to(dev); self.dev = dev
    @torch.no_grad()
    def windows(self, S, P0, P1, spec):
        from model.zoom_nfd.world128 import raster128, plates128, LO, PX, RES
        X = raster128(S, SIZES); x = torch.cat([X[:, None], plates128(P0, P1)], 1)
        pred = torch.cat([torch.sigmoid(self.net(x[i:i + 128].to(self.dev))).squeeze(1).cpu() for i in range(0, len(x), 128)])
        Xw = window_batch(S, SIZES, P0, P1, spec); Pw = []
        for b in range(len(S)):
            P = torch.from_numpy(_window_pixel_world(P0[b], P1[b], spec)).float()
            ij = (P - LO) / PX - 0.5                               # (row, col) index coords in the 128 raster
            g = torch.stack([ij[..., 1], ij[..., 0]], -1) / (RES - 1) * 2 - 1
            Pw.append(torch.nn.functional.grid_sample(pred[b][None, None], g[None], mode="bilinear", padding_mode="border", align_corners=True)[0, 0])
        return Xw, torch.stack(Pw), (pred, X)
    def paste(self, ctx, Xw, Pw, b, o0, p0, p1, spec):
        from model.zoom_nfd.world128 import LO, PX, RES
        pred, X = ctx; d = (pred[b] - X[b])[None, None]
        pitch = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / (OCC_GRID - 1); sub = 3
        ax = np.arange(OCC_GRID) * pitch + OCC_BOUNDS["x_min"]; offs = (np.arange(sub) - (sub - 1) / 2) * pitch / sub
        acc_ = torch.zeros(OCC_GRID, OCC_GRID)
        for dx in offs:
            for dy in offs:
                Xg, Yg = np.meshgrid(ax + dx, ax + dy, indexing="ij")
                ij = torch.from_numpy(np.stack([Xg, Yg], -1)).float()
                ij = (ij - LO) / PX - 0.5
                g = torch.stack([ij[..., 1], ij[..., 0]], -1) / (RES - 1) * 2 - 1
                acc_ += torch.nn.functional.grid_sample(d, g[None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        return (o0 + acc_ / (sub * sub)).clamp(0, 1)


ZERO = torch.zeros(OCC_GRID, OCC_GRID)
def paste_delta(pred_w, state_w, o0, p0, p1, spec):
    """Paste only the predicted CHANGE (pred - state window) onto the world raster, so a
    no-change prediction reproduces the world raster exactly (box-vs-disc raster styles differ)."""
    return (o0 + paste_window_into_world(pred_w - state_w, ZERO, p0, p1, spec)).clamp(0, 1)


def rows(files, key=None):
    cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
    return [cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in sorted(glob.glob(files))]


def score(model, spec, dev="cuda", refs=("nfd_3ch_narrow_l20_v2",), want_cases=False):
    res = {}
    ch = rows(D + "test_chains_v2_clean/_*_data.pt"); pools = rows(D + "test_pools_v2/pools_*.pt")
    Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}
    adapters = {r: make_occ_adapter(r, dev, "corner") for r in refs}
    # ---------------- 1-step accuracy
    acc_rows = {"pasted": [], "window": [], "paste_oracle(true label pasted)": [], "paste_persistence(state window pasted)": []}; ref_rows = {r: [] for r in refs}; kinds = []
    for d in ch:
        S, S_ = d["states"].float(), d["states_"].float()
        P0, P1 = d["p_starts"][:, :2].numpy(), d["p_stops"][:, :2].numpy()
        act = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        Xw, Pw, ctx = model.windows(S, P0, P1, spec)
        Yw = window_batch(S_, SIZES, P0, P1, spec)
        Rw = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(len(S))])
        O0, O1 = occ_from_particles(S).cpu(), occ_from_particles(S_).cpu()
        Pp = torch.stack([model.paste(ctx, Xw, Pw, b, O0[b], P0[b], P1[b], spec) for b in range(len(S))])
        Rg = swept_region(act, "cpu").cpu()
        Po = torch.stack([paste_delta(Yw[b], Xw[b], O0[b], P0[b], P1[b], spec) for b in range(len(S))])
        Pq = torch.stack([paste_delta(Xw[b], Xw[b], O0[b], P0[b], P1[b], spec) for b in range(len(S))])
        acc_rows["paste_oracle(true label pasted)"].append((Po, O1, O0, Rg)); acc_rows["paste_persistence(state window pasted)"].append((Pq, O1, O0, Rg))
        acc_rows["pasted"].append((Pp, O1, O0, Rg)); acc_rows["window"].append((Pw, Yw, Xw, Rw)); kinds += list(d["start_kind"])
        for r, ad in adapters.items():
            with torch.no_grad():
                p = torch.cat([ad.predict_step(O0[i:i + 128].to(dev), act[i:i + 128].to(dev)) for i in range(0, len(act), 128)]).float().cpu()
            ref_rows[r].append((p, O1, O0, Rg))
    kinds = np.array(kinds)
    def pool(lst): return [torch.cat([x[j] for x in lst]) for j in range(4)]
    for name, lst in list(acc_rows.items()) + [(r, v) for r, v in ref_rows.items()]:
        P, T, O, R = pool(lst)
        res[name] = {"accuracy_1": acc(P, T, O, R)}
        for k in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(kinds == k)[0]); res[name][f"accuracy_1_{k}"] = acc(P[ix], T[ix], O[ix], R[ix])
    # ---------------- slateN
    caps = {n: {g: [] for g in GOALS} for n in ["pasted", "window"] + list(refs)}
    for d in pools:
        Sall, Sall_ = d["states"].float(), d["states_"].float()
        P0all, P1all = d["p_starts"][:, :2].numpy(), d["p_stops"][:, :2].numpy()
        act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            S0 = Sall[ix[0]][None]; S1 = Sall_[ix]
            n = len(ix); P0, P1 = P0all[ix.numpy()], P1all[ix.numpy()]
            Sx = S0.expand(n, -1, -1).contiguous()
            o0 = occ_from_particles(S0); truth_occ = occ_for_scoring(S1[:, :, :3]); t0 = occ_for_scoring(S0[:, :, :3])
            Xw, Pw, ctx = model.windows(Sx, P0, P1, spec)
            Ypp = [model.paste(ctx, Xw, Pw, b, o0[0], P0[b], P1[b], spec) for b in range(n)]
            po_p = torch.stack(Ypp)
            po_r = {r: ad.predict_step(o0.expand(n, -1, -1).contiguous().to(dev), act_all[ix].to(dev)).float().cpu() for r, ad in adapters.items()}
            # window truth: mass splat of cube centres inside each candidate's window (state and next state)
            TW1 = torch.stack([splat_window_mass(S1[b], P0[b], P1[b], spec) for b in range(n)])
            TW0 = torch.stack([splat_window_mass(S0[0], P0[b], P1[b], spec) for b in range(n)])
            for g in GOALS:
                vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                den = vt.mean() - vt.min()
                if den <= 1e-9: continue
                cap = lambda vp: float((vt.mean() - vt[vp.argmin()]) / den)
                caps["pasted"][g].append(cap((lyap(po_p, Dist[g]) - lyap(o0.expand(n, -1, -1), Dist[g])).numpy()))
                for r in refs:
                    caps[r][g].append(cap((lyap(po_r[r], Dist[g]) - lyap(o0.expand(n, -1, -1), Dist[g])).numpy()))
                # window version: goal cut into each candidate's window; truth dv and pred dv both in-window
                Dw = torch.stack([sample_world_field_into_window(Dist[g], P0[b], P1[b], spec) for b in range(n)])
                lw = lambda occ, i=None: (occ.reshape(n, -1) * Dw.reshape(n, -1)).sum(1) / occ.reshape(n, -1).sum(1).clamp_min(1e-6)
                vtw = (lw(TW1) - lw(TW0)).numpy(); denw = vtw.mean() - vtw.min()
                if denw > 1e-9:
                    vpw = (lw(Pw) - lw(Xw)).numpy()
                    caps["window"][g].append(float((vtw.mean() - vtw[vpw.argmin()]) / denw))
    for k, c in caps.items():
        res[k]["slateN"] = float(np.mean([np.mean(v) for v in c.values() if v]))
        res[k]["slateN_per_goal"] = {g: float(np.mean(v)) for g, v in c.items() if v}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--world128", action="store_true"); ap.add_argument("--hr", type=int, default=0); ap.add_argument("--res", type=int, default=64); ap.add_argument("--aa", type=int, default=1); ap.add_argument("--thr", type=float, default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    a = ap.parse_args()
    net = load_unet(a.ckpt); mdl = World128Model(net) if a.world128 else (ZoomModelHR(net, a.hr, a.thr, aa=a.aa) if a.hr else ZoomModel(net))
    r = score(mdl, WindowSpec(res=a.res), refs=tuple(a.refs))
    json.dump(r, open(a.out, "w"), indent=1)
    for k, v in r.items():
        print(f"{k:28s} acc1 {v['accuracy_1']:+.3f} (scatter {v['accuracy_1_scatter']:+.3f} clump {v['accuracy_1_clump']:+.3f})  slateN {v.get('slateN', float('nan')):.3f}")
