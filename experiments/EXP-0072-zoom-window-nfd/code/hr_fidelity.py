"""How much hi-res raster does the visual zoom pipeline need? Compare windows extracted (rotation+zoom, antialiased)
from an HxH whole-tray raster against the DIRECT window raster the model was trained on."""
import sys, glob, numpy as np, torch
sys.path.insert(0, ".")
from model.zoom_nfd.window import WindowSpec, rasterise_window
from model.zoom_nfd.window_gpu import raster_world, extract_windows
spec = WindowSpec(); SIZES = [[0.005] * 3] * 20
d = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_chains_v2_clean/_*_data.pt"))[0], map_location="cpu", weights_only=False)
S = d["states"].float().cpu()[:200]; P0 = d["p_starts"][:200, :2].float().cpu(); P1 = d["p_stops"][:200, :2].float().cpu()
direct = torch.stack([rasterise_window(S[b], SIZES, P0[b].numpy(), P1[b].numpy(), spec) for b in range(len(S))])
print("HR   thr0.5-IoU  soft-MAE  mean-abs-area-diff")
for H in (64, 128, 192, 256, 320, 400, 512):
    hr = raster_world(S, SIZES, H)
    win = torch.stack([extract_windows(hr[b].cuda(), P0[b:b+1].cuda(), P1[b:b+1].cuda(), spec)[0].cpu() for b in range(len(S))])
    b1, b2 = win > .5, direct > .5
    iou = float((b1 & b2).sum(dim=(1, 2)).float().div((b1 | b2).sum(dim=(1, 2)).clamp(min=1)).mean())
    print(f"{H:4d}  {iou:.4f}      {float((win-direct).abs().mean()):.5f}   {float((win.sum((1,2))-direct.sum((1,2))).abs().mean()):.2f} px")
print("\nbest binarisation threshold per HR (IoU vs direct training raster)")
for H in (128, 192, 256, 320, 400):
    hr = raster_world(S, SIZES, H)
    win = torch.stack([extract_windows(hr[b].cuda(), P0[b:b+1].cuda(), P1[b:b+1].cuda(), spec)[0].cpu() for b in range(len(S))])
    row = []
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5):
        b1, b2 = win > thr, direct > .5
        row.append((thr, float((b1 & b2).sum(dim=(1, 2)).float().div((b1 | b2).sum(dim=(1, 2)).clamp(min=1)).mean())))
    print(H, " ".join(f"{t}:{i:.3f}" for t, i in row))
