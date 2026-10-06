"""Reference = the crop window rendered DIRECTLY from the cube poses at very high resolution (REFxREF over the window, true cube
footprints) then area-averaged down to the model resolution (soft occupancy fraction). Compared with the same window obtained by
warping an HxH full-frame raster (rotation+zoom grid_sample), for several H incl. 300. Also vs the training-style direct raster."""
import sys, glob, torch, torch.nn.functional as F
sys.path.insert(0, ".")
from model.zoom_nfd.window import WindowSpec, rasterise_window
from model.zoom_nfd.window_gpu import raster_world, extract_windows
SIZES = [[0.005] * 3] * 20
d = torch.load(sorted(glob.glob("Genesis/data/narrow_l20_n20/test_chains_v2_clean/_*_data.pt"))[0], map_location="cpu", weights_only=False)
S = d["states"].float()[:150]; P0 = d["p_starts"][:150, :2].float(); P1 = d["p_stops"][:150, :2].float()
def iou(a, b): return float((a & b).sum((1, 2)).float().div((a | b).sum((1, 2)).clamp(min=1)).mean())
import os; AA = int(os.environ.get('AA', 1))
print('source raster antialias factor', AA)
for res in (64, 128):
    spec = WindowSpec(res=res); k = 1024 // res
    ref_spec = WindowSpec(res=1024)
    ref = torch.stack([F.avg_pool2d(rasterise_window(S[b], SIZES, P0[b].numpy(), P1[b].numpy(), ref_spec)[None, None], k)[0, 0] for b in range(len(S))])
    train_style = torch.stack([rasterise_window(S[b], SIZES, P0[b].numpy(), P1[b].numpy(), spec) for b in range(len(S))])
    print(f"\n=== model window {res}x{res} ({spec.px*1e3:.2f} mm/px); reference = pose-rendered 1024-px window, area-averaged (soft) ===")
    print(f"training-style direct raster vs reference: soft MAE {float((train_style-ref).abs().mean()):.4f}  IoU@0.5 {iou(train_style>.5, ref>.5):.3f}  area ratio {float(train_style.sum()/ref.sum()):.3f}")
    print(" H    softMAE  IoU@.5  IoU@best-thr  area-ratio  (source px = %.3f mm)" % (128/300))
    for H in (128, 192, 256, 300, 400, 512, 768):
        hr = raster_world(S, SIZES, H, aa=AA)
        win = torch.stack([extract_windows(hr[b].cuda(), P0[b:b+1].cuda(), P1[b:b+1].cuda(), spec)[0].cpu() for b in range(len(S))])
        best = max((iou(win > t, ref > .5), t) for t in (.2, .3, .4, .5, .6))
        print(f"{H:4d}  {float((win-ref).abs().mean()):.4f}   {iou(win>.5, ref>.5):.3f}   {best[0]:.3f}@{best[1]}   {float(win.sum()/ref.sum()):.3f}")
