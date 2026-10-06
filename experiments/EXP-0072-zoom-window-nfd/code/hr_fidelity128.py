"""128x128 zoom window: how well do windows extracted from an HxH whole-tray raster match the direct 128x128 window raster?"""
import sys, glob, torch
sys.path.insert(0, ".")
from model.zoom_nfd.window import WindowSpec, rasterise_window
from model.zoom_nfd.window_gpu import raster_world, extract_windows
spec = WindowSpec(res=128); SIZES = [[0.005] * 3] * 20
d = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_chains_v2_clean/_*_data.pt"))[0], map_location="cpu", weights_only=False)
S = d["states"].float()[:200]; P0 = d["p_starts"][:200, :2].float(); P1 = d["p_stops"][:200, :2].float()
direct = torch.stack([rasterise_window(S[b], SIZES, P0[b].numpy(), P1[b].numpy(), spec) for b in range(len(S))])
for ss in (2, 3):
  print("ss", ss, "H : IoU at thr .1/.2/.3/.4/.5 (vs direct 128px window)")
  for H in (128, 192, 256, 300, 400, 512):
    hr = raster_world(S, SIZES, H)
    win = torch.stack([extract_windows(hr[b].cuda(), P0[b:b+1].cuda(), P1[b:b+1].cuda(), spec, ss=ss)[0].cpu() for b in range(len(S))])
    out = []
    for thr in (.1, .2, .3, .4, .5):
        b1, b2 = win > thr, direct > .5
        out.append(float((b1 & b2).sum((1, 2)).float().div((b1 | b2).sum((1, 2)).clamp(min=1)).mean()))
    print(f"{H:4d}", " ".join(f"{v:.3f}" for v in out))
