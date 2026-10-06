"""Cache each model's PASTED world-64 next-state prediction on DS-0016 chains (896, load_chains order) and pools
(2048 = 32 pools x 64, ceiling_resim pool order), so metric studies need no GPU. Uses score_zoom's model classes verbatim.
Output artifacts/ceiling/preds_<model>.pt  {chains: (896,64,64) f16, pools: (2048,64,64) f16}"""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_common import *
import score_zoom as Z
from simple_mpc.adapters import make_occ_adapter, occ_from_particles
from model.zoom_nfd.window import WindowSpec
ART = "experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/"; R = "experiments/EXP-0072-zoom-window-nfd/runs/"
MODELS = {"nfd64": None, "ft100": ("zoom", 64), "scratch100": ("zoom", 64), "ft300": ("zoom", 64), "zoom128": ("zoom", 128),
          "world128": ("w128", 64), "world128_300": ("w128", 64)}
for _n in ["nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2", "nfd_3ch_narrow_l20_v2_sharp_w03", "nfd_3ch_narrow_l20_v2_sharp_w07",
           "nfd_3ch_narrow_l20_v2_soft_s1", "nfd_3ch_narrow_l20_v2_soft_s2", "linear_narrow_l20_v2_res64", "linear_narrow_l20_v2_res32"]:
    MODELS[_n] = "adapter"      # EXTRA reference adapters (EXP-0036/0063/0066 variants), world-64 frame, no paste
dev = "cuda"
c = load_chains(); S0 = c["states"].float(); P0, P1 = c["p_starts"][:, :2].numpy(), c["p_stops"][:, :2].numpy()
act = torch.cat([c["p_starts"][:, :2], c["p_stops"][:, :2]], 1).float()
pools = [torch.load(f, map_location="cpu", weights_only=False) for f in sorted(glob.glob(D + "test_pools_v2/pools_*.pt"))]
PL = []
for d in pools:
    for p in torch.unique(d["pool_idx"]):
        ix = torch.nonzero(d["pool_idx"] == p)[:, 0]; PL.append((d["states"][ix[0]].float()[None].expand(len(ix), -1, -1).contiguous(), d["p_starts"][ix, :2].numpy(), d["p_stops"][ix, :2].numpy(), ix, d))
for name, spec in MODELS.items():
    fn = ART + f"preds_{name}.pt"
    if os.path.exists(fn): continue
    if spec is None or spec == "adapter":
        try: ad = make_occ_adapter("nfd_3ch_narrow_l20_v2" if spec is None else name, dev, "corner")
        except NotImplementedError as e: print("skip", name, e); continue
        def run(S, a0, a1):
            o0 = occ_from_particles(S).cpu(); a = torch.cat([torch.from_numpy(a0), torch.from_numpy(a1)], 1).float()
            with torch.no_grad(): return torch.cat([ad.predict_step(o0[i:i+128].to(dev), a[i:i+128].to(dev)) for i in range(0, len(a), 128)]).float().cpu()
    else:
        kind, res = spec; net = Z.load_unet(R + name + "/unet_best.pth"); m = Z.World128Model(net) if kind == "w128" else Z.ZoomModel(net); ws = WindowSpec(res=res)
        def run(S, a0, a1, m=m, ws=ws):
            Xw, Pw, ctx = m.windows(S, a0, a1, ws); o0 = occ_from_particles(S).cpu()
            return torch.stack([m.paste(ctx, Xw, Pw, b, o0[b], a0[b], a1[b], ws) for b in range(len(S))])
    ch = torch.cat([run(S0[i:i+128], P0[i:i+128], P1[i:i+128]) for i in range(0, len(S0), 128)])
    pl = torch.cat([run(S, a0, a1) for S, a0, a1, _, _ in PL])
    torch.save({"chains": ch.half(), "pools": pl.half()}, fn + ".tmp"); os.replace(fn + ".tmp", fn); print(name, ch.shape, pl.shape, flush=True)
