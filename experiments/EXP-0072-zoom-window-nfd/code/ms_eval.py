"""Multi-step (rollout) accuracy on DS-0016 test_chains_v2_clean: every model is fed its OWN previous prediction for k=1..8 pushes
(eval_narrow.py's rollout_accuracy_k: truth = recorded state after k pushes, persistence = start, region = union of the k swept
rectangles, pasted/world-64 frame, accuracy = 1 - rms(pred-truth)/rms(start-truth) pooled over chains). Zoom models keep a 300x300
whole-tray canvas and extract each push's window from it (model/zoom_nfd/rollout.py); world128 feeds its 128x128 raster back."""
import sys, json, argparse, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0043-batched-closed-loop/code"); sys.path.insert(0, "experiments/EXP-0053-narrow-domain-models/code")
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
import torch.nn.functional as F
from eval_narrow import swept_region, acc
from simple_mpc.adapters import make_occ_adapter, occ_from_particles
from model.zoom_nfd.window import WindowSpec
from model.zoom_nfd.window_gpu import plates_gpu, LO, SIDE
from model.zoom_nfd.rollout import *
from model.UNetModels_modular import UNet
from model.zoom_nfd.world128 import raster128, plates128
dev = "cuda"; SIZES = [[0.005] * 3] * 20


def make_unet(path, features=(4, 8, 16)):
    n = UNet({"features": list(features), "in_channels": 3, "out_channels": 1, "kernel_size": 3, "final_kernel_size": 1, "activation": "relu",
              "residual": True, "bottleneck_type": "None", "bottleneck_kwargs": {}})
    n.load_state_dict(torch.load(path, map_location="cpu")); return n.to(dev).eval()


@torch.no_grad()
def zoom_rollout(net, res, ch, C=300, thr=0.2, T=8):
    spec = WindowSpec(res=res)
    canvas = canvas_from_particles(ch["S"][:, 0], SIZES, C, thr).float().to(dev); cur = occ_from_particles(ch["S"][:, 0]).to(dev); out = []
    for k in range(T):
        P0, P1 = ch["P0"][:, k].to(dev), ch["P1"][:, k].to(dev)
        w = extract_windows_b(canvas, P0, P1, spec); x = torch.cat([w[:, None], plates_gpu(P0, P1, spec)], 1)
        d = torch.sigmoid(net(x)).squeeze(1) - w
        canvas = (canvas + paste_canvas(d, P0, P1, spec, C)).clamp(0, 1); cur = (cur + delta_to_world64(d, P0, P1, spec)).clamp(0, 1); out.append(cur.cpu())
    return out


def down128(delta, grid=64, sub=3):
    pitch = SIDE / (grid - 1); ax = torch.arange(grid, device=delta.device) * pitch + LO
    off = (torch.arange(sub, device=delta.device) - (sub - 1) / 2) * pitch / sub; X = (ax[:, None] + off[None, :]).reshape(-1)
    ij = (torch.stack(torch.meshgrid(X, X, indexing="ij"), -1) - LO) / (SIDE / 128) - 0.5
    g = (torch.stack([ij[..., 1], ij[..., 0]], -1) / 127 * 2 - 1)[None].expand(len(delta), -1, -1, -1)
    return F.avg_pool2d(F.grid_sample(delta[:, None], g, mode="bilinear", padding_mode="zeros", align_corners=True), sub).squeeze(1)


@torch.no_grad()
def world128_rollout(net, ch, T=8):
    st = raster128(ch["S"][:, 0], SIZES).to(dev); cur = occ_from_particles(ch["S"][:, 0]).to(dev); out = []
    for k in range(T):
        P0, P1 = ch["P0"][:, k].to(dev), ch["P1"][:, k].to(dev)
        p = torch.sigmoid(net(torch.cat([st[:, None], plates128(P0.cpu(), P1.cpu()).to(dev)], 1))).squeeze(1)
        cur = (cur + down128(p - st)).clamp(0, 1); st = p; out.append(cur.cpu())
    return out


@torch.no_grad()
def adapter_rollout(name, ch, T=8):
    ad = make_occ_adapter(name, dev, "corner"); cur = occ_from_particles(ch["S"][:, 0]).to(dev); out = []
    for k in range(T):
        act = torch.cat([ch["P0"][:, k], ch["P1"][:, k]], 1).to(dev)
        cur = ad.predict_step(cur, act).float(); out.append(cur.cpu())
    return out


def score_rollout(preds, ch, T=8):
    start = occ_from_particles(ch["S"][:, 0]).cpu(); reg = torch.zeros(len(start), 64, 64, dtype=torch.bool); res = {}
    for k in range(T):
        act = torch.cat([ch["P0"][:, k], ch["P1"][:, k]], 1); reg = reg | swept_region(act, "cpu").cpu()
        truth = occ_from_particles(ch["S_"][:, k]).cpu()
        res[k + 1] = acc(preds[k], truth, start, reg)
        for kind in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(ch["kind"] == kind)[0])
            if len(ix): res[f"{k+1}_{kind}"] = acc(preds[k][ix], truth[ix], start[ix], reg[ix])
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="experiments/EXP-0072-zoom-window-nfd/results/ms_eval.json")
    ap.add_argument("--zoom64", nargs="*", default=[]); ap.add_argument("--zoom128", nargs="*", default=[]); ap.add_argument("--world128", nargs="*", default=[])
    ap.add_argument("--adapters", nargs="*", default=[]); ap.add_argument("--feat", default="4,8,16"); ap.add_argument("--thr", type=float, default=0.2)
    a = ap.parse_args()
    ch = load_chains("Genesis/data/narrow_l20_n20/test_chains_v2_clean/_*_data.pt"); print("chains", ch["S"].shape, {k: int((ch["kind"] == k).sum()) for k in ("scatter", "clump")})
    feats = tuple(int(x) for x in a.feat.split(","))
    try: res = json.load(open(a.out))
    except Exception: res = {}
    for n in a.adapters: res[n] = score_rollout(adapter_rollout(n, ch), ch)
    for p in a.zoom64: res["zoom64:" + p.split("/")[-2]] = score_rollout(zoom_rollout(make_unet(p, feats), 64, ch, thr=a.thr), ch)
    for p in a.zoom128: res["zoom128:" + p.split("/")[-2]] = score_rollout(zoom_rollout(make_unet(p, feats), 128, ch, thr=a.thr), ch)
    for p in a.world128: res["world128:" + p.split("/")[-2]] = score_rollout(world128_rollout(make_unet(p, feats), ch), ch)
    json.dump(res, open(a.out, "w"), indent=1)
    print("model".ljust(34), *[f"k={k}".rjust(7) for k in (1, 2, 3, 4, 5, 8)])
    for n, r in res.items(): print(n.ljust(34), *[f"{r[str(k)] if str(k) in r else r[k]:+.3f}".rjust(7) for k in (1, 2, 3, 4, 5, 8)])
