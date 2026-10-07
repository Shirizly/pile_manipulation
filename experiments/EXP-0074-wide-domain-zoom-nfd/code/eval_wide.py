"""Evaluate window / vanilla-world NFDs on the held-out Sean TEST files, per shard (scene type x object count) and pooled:
 acc1      one-step accuracy (pasted/world-64 frame, swept region, persistence = start state; pooled ratio of means), also by push-length bin
 slateN    same-state pools (>= 6 pushes), lyapunov, 13 goals, soft mass-conserving truth (eval_narrow recipe), capture averaged over pools x goals
 roll_k    rollout accuracy k=1..4 on 4-push chains (own output fed back, truth = recorded state after k pushes, region = union of swept regions)
Models: --zoom RES CKPT [name] (variable-side window from a 300x300 canvas, thr 0.2) ; --world RES CKPT [name] (vanilla raster NFD).
Usage: python eval_wide.py --out name --zoom 128 ckpt ... | --world 128 ckpt"""
import sys, os, json, argparse, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0053-narrow-domain-models/code")
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
from batched_closed_loop import goal_mask
from Baselines.common.goals import dist_field_from_mask
from eval_narrow import swept_region, acc, GOALS
from simple_mpc.adapters import occ_from_particles, occ_for_scoring
from simple_mpc.learned_mpc import lyap
from sean_data import load_split, rows_table, chains_of, pools_of
from model.UNetModels_modular import UNet
from model.zoom_nfd.rollout import canvas_from_particles
from model.zoom_nfd.window_var import *
from model.zoom_nfd.world_res import raster_res, plates_res
torch.set_num_threads(4)
dev = "cuda"; SHARDS = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
TS = "experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/test_sets.pt"


def test_sets(n_rows=1000, n_pools=60):
    if os.path.exists(TS): return torch.load(TS, weights_only=False)
    out = {}; D = load_split("test"); rng = np.random.default_rng(0)
    for sh in SHARDS:
        t = rows_table(D[sh]); N = len(t["S"]); rows = np.sort(rng.permutation(N)[:n_rows]); ch = chains_of(t, minlen=4, maxlen=4); ch = [ch[i] for i in rng.permutation(len(ch))[:400]]
        pl = pools_of(t); pl = [pl[i][:32] for i in rng.permutation(len(pl))[:n_pools]]
        out[sh] = dict(t={k: v for k, v in t.items() if k != "file"}, rows=rows, chains=ch, pools=pl)
    torch.save(out, TS); return out


def make_net(path, feats=None):
    feats = feats or tuple(int(x) for x in os.environ.get("EVAL_FEAT", "4,8,16").split(","))
    n = UNet({"features": list(feats), "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu", "residual": True,
              "bottleneck_type": "None", "bottleneck_kwargs": {}}); n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval()


def down_world(delta, res, grid=64, sub=3):
    pitch = 0.128 / (grid - 1); ax = torch.arange(grid, device=delta.device) * pitch - 0.064; off = (torch.arange(sub, device=delta.device) - (sub - 1) / 2) * pitch / sub
    X = (ax[:, None] + off[None, :]).reshape(-1); ij = (torch.stack(torch.meshgrid(X, X, indexing="ij"), -1) + 0.064) / (0.128 / res) - 0.5
    g = (torch.stack([ij[..., 1], ij[..., 0]], -1) / (res - 1) * 2 - 1)[None].expand(len(delta), -1, -1, -1)
    return F.avg_pool2d(F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True), sub).squeeze(1)


CVF = "experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/test_canvases.pt"
CV = torch.load(CVF, weights_only=False) if os.path.exists(CVF) else {}


def canv(sh, t, idx):
    """uint8 300x300 training-style canvases (AA raster > 0.2) of the start states t['S'][idx]; cached on disk per (shard, row)."""
    miss = [int(i) for i in idx.tolist() if (sh, int(i)) not in CV]
    if miss:
        m = torch.tensor(sorted(set(miss))); c = canvas_from_particles(t["S"][m], [[0.005] * 3] * t["S"].shape[1], 300, 0.2)
        for i, x in zip(m.tolist(), c): CV[(sh, i)] = x
        torch.save(CV, CVF)
    return torch.stack([CV[(sh, int(i))] for i in idx.tolist()])


class Zoom:
    def __init__(self, net, res, C=300, thr=0.2): self.net, self.res, self.C, self.thr = net, res, C, thr
    @torch.no_grad()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        canvas = (canvas0 if canvas0 is not None else canvas_from_particles(S0, [[0.005] * 3] * S0.shape[1], self.C, self.thr)).float().to(dev); cur = occ_from_particles(S0).to(dev); out = []
        for k in range(P0s.shape[1]):
            P0, P1 = P0s[:, k].to(dev), P1s[:, k].to(dev); side = side_for((P1 - P0).norm(dim=-1))
            w = extract_windows_v(canvas, P0, P1, side, self.res); d = torch.sigmoid(self.net(torch.cat([w[:, None], plates_v(P0, P1, side, self.res)], 1))).squeeze(1) - w
            canvas = (canvas + paste_canvas_v(d, P0, P1, side, self.C)).clamp(0, 1); cur = (cur + delta_to_world64_v(d, P0, P1, side)).clamp(0, 1); out.append(cur.cpu())
        return out


class World:
    def __init__(self, net, res): self.net, self.res = net, res
    @torch.no_grad()
    def rollout(self, S0, P0s, P1s, canvas0=None):
        n = S0.shape[1]; st = raster_res(S0, [[0.005] * 3] * n, self.res).to(dev); cur = occ_from_particles(S0).to(dev); out = []
        for k in range(P0s.shape[1]):
            P0, P1 = P0s[:, k].to(dev), P1s[:, k].to(dev); p = torch.sigmoid(self.net(torch.cat([st[:, None], plates_res(P0, P1, self.res)], 1))).squeeze(1)
            cur = (cur + down_world(p - st, self.res)).clamp(0, 1); st = p; out.append(cur.cpu())
        return out


def run(model, sets, bs=128):
    res = {}; Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}
    for sh in SHARDS:
        s = sets[sh]; t = s["t"]; r = {}
        # ---- one-step accuracy over rows
        P, T, O, R, Ls = [], [], [], [], []
        for i in range(0, len(s["rows"]), bs):
            ix = torch.from_numpy(s["rows"][i:i + bs]); S0, S1, P0, P1 = t["S"][ix], t["S_"][ix], t["P0"][ix], t["P1"][ix]
            pr = model.rollout(S0, P0[:, None], P1[:, None], canv(sh, t, ix))[0]; P.append(pr); T.append(occ_from_particles(S1)); O.append(occ_from_particles(S0))
            R.append(swept_region(torch.cat([P0, P1], 1), "cpu").cpu()); Ls.append((P1 - P0).norm(dim=1) * 1000)
        P, T, O, R, Ls = map(torch.cat, (P, T, O, R, Ls)); r["acc1"] = acc(P, T, O, R)
        for nm, lo, hi in (("L<30", 0, 30), ("L30-50", 30, 50), ("L>=50", 50, 99), ("L<35", 0, 35), ("L35-55", 35, 55), ("L>=55", 55, 99)):
            m = (Ls >= lo) & (Ls < hi); r["acc1_" + nm] = acc(P[m], T[m], O[m], R[m]) if m.sum() > 20 else None
        # ---- rollout on 4-push chains
        ch = torch.tensor(s["chains"])
        if len(ch):
            preds = [[] for _ in range(4)]
            for i in range(0, len(ch), bs):
                c = ch[i:i + bs]; o = model.rollout(t["S"][c[:, 0]], t["P0"][c], t["P1"][c], canv(sh, t, c[:, 0]))
                for k in range(4): preds[k].append(o[k])
            start = occ_from_particles(t["S"][ch[:, 0]]); reg = torch.zeros(len(ch), 64, 64, dtype=torch.bool)
            for k in range(4):
                reg = reg | swept_region(torch.cat([t["P0"][ch[:, k]], t["P1"][ch[:, k]]], 1), "cpu").cpu()
                r[f"roll_{k+1}"] = acc(torch.cat(preds[k]), occ_from_particles(t["S_"][ch[:, k]]), start, reg)
        # ---- slateN on same-state pools
        caps = {g: [] for g in GOALS}
        for pidx in s["pools"]:
            ix = torch.from_numpy(pidx); S0 = t["S"][ix]; o0 = occ_from_particles(S0[:1]); pr = model.rollout(S0, t["P0"][ix][:, None], t["P1"][ix][:, None], canv(sh, t, ix[:1]).expand(len(ix), -1, -1))[0]
            tr = occ_for_scoring(t["S_"][ix][:, :, :3]); t0 = occ_for_scoring(S0[:1][:, :, :3])
            for g in GOALS:
                vt = (lyap(tr, Dist[g]) - lyap(t0, Dist[g])).numpy(); vp = (lyap(pr, Dist[g]) - lyap(o0.expand(len(ix), -1, -1), Dist[g])).numpy(); den = vt.mean() - vt.min()
                if den > 1e-9: caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
        r["slateN"] = float(np.mean([np.mean(v) for v in caps.values() if v])); r["n_pools"] = len(s["pools"])
        res[sh] = r; print(sh, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    keys = ["acc1", "slateN", "roll_1", "roll_2", "roll_3", "roll_4"]
    res["MEAN"] = {k: float(np.mean([res[sh][k] for sh in SHARDS if res[sh].get(k) is not None])) for k in keys}
    for nb in (20, 50, 100): res[f"MEAN_n{nb}"] = {k: float(np.mean([res[sh][k] for sh in SHARDS if sh.endswith(f"_n{nb}") and res[sh].get(k) is not None])) for k in keys}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zoom", nargs=2, metavar=("RES", "CKPT")); ap.add_argument("--world", nargs=2, metavar=("RES", "CKPT"))
    ap.add_argument("--name", required=True); ap.add_argument("--thr", type=float, default=0.2); a = ap.parse_args()
    sets = test_sets()
    if a.zoom: m = Zoom(make_net(a.zoom[1]), int(a.zoom[0]), thr=a.thr)
    else: m = World(make_net(a.world[1]), int(a.world[0]))
    r = run(m, sets); json.dump(r, open(f"experiments/EXP-0074-wide-domain-zoom-nfd/results/eval_{a.name}.json", "w"), indent=1)
    print("MEAN", {k: round(v, 3) for k, v in r["MEAN"].items()})
