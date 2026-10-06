"""6-panel example figures for a zoom-window NFD (one PNG per DS-0016 test-chain transition),
layout follows scripts/probes/transition_panel.py, but in the push-aligned zoom window:
 (a) input state window  (b) action rasters  (c) true next state
 (d) metrics: window vs pasted-to-world accuracy, changed-cell IoU
 (e) model output        (f) error map (red pred>true, blue pred<true) + swept region."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, "."); sys.path.insert(0, "scripts/probes"); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from transition_panel import error_rgb
from score_zoom import *

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", default="zoom NFD")
ap.add_argument("--n", type=int, default=8); ap.add_argument("--world128", action="store_true")
a = ap.parse_args()
spec = WindowSpec(); model = (World128Model if a.world128 else ZoomModel)(load_unet(a.ckpt))
ch = rows(D + "test_chains_v2_clean/_*_data.pt")
S = torch.cat([d["states"] for d in ch]).float(); S_ = torch.cat([d["states_"] for d in ch]).float()
P0 = torch.cat([d["p_starts"][:, :2] for d in ch]).numpy(); P1 = torch.cat([d["p_stops"][:, :2] for d in ch]).numpy()
kinds = sum([list(d["start_kind"]) for d in ch], [])
moved = (S_[:, :, :2] - S[:, :, :2]).norm(dim=-1).gt(0.002).sum(1).numpy()
rng = np.random.default_rng(3)
order = np.argsort(moved)
picks = []
for kind in ("scatter", "clump"):
    ix = [i for i in order if kinds[i] == kind and moved[i] >= 2]
    q = np.linspace(0.1, 0.98, a.n // 2)
    picks += [ix[int(f * (len(ix) - 1))] for f in q]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ext = [-spec.margin_back * 1e3, (spec.side - spec.margin_back) * 1e3, spec.side / 2 * 1e3, -spec.side / 2 * 1e3]
def show(ax, img, **kw):
    ax.imshow(img, extent=ext, interpolation="nearest", **kw)
    ax.set_xlabel("along push [mm]  (start plate at 0)", fontsize=7); ax.set_ylabel("lateral [mm]", fontsize=7); ax.tick_params(labelsize=6)
for i in picks:
    s, s_, p0, p1 = S[i:i+1], S_[i:i+1], P0[i:i+1], P1[i:i+1]
    Xw, Pw, ctx = model.windows(s, p0, p1, spec)
    Yw = rasterise_window(s_[0], SIZES, p0[0], p1[0], spec); Pl = plate_channels_window(p0[0], p1[0], spec)
    R = swept_region_window(p0[0], p1[0], spec)
    o0, o1 = occ_from_particles(s)[0], occ_from_particles(s_)[0]
    act = torch.cat([torch.from_numpy(p0), torch.from_numpy(p1)], 1).float(); Rg = swept_region(act, "cpu")[0]
    pasted = model.paste(ctx, Xw, Pw, 0, o0, p0[0], p1[0], spec); Xw, Pw = Xw[0], Pw[0]
    def met(p, t, prev, r):
        rf = r.float(); rm = float(torch.sqrt((((p - t) ** 2) * rf).sum() / rf.sum())); rp = float(torch.sqrt((((prev - t) ** 2) * rf).sum() / rf.sum()))
        ct = ((t - prev).abs() > .5) & r; cp = ((p - prev).abs() > .5) & r
        iou = float((ct & cp & ((p > .5) == (t > .5))).sum() / max(1, (ct | cp).sum()))
        return 1 - rm / max(rp, 1e-12), iou
    aw, iw = met(Pw, Yw, Xw, R); ap_, ip = met(pasted, o1, o0, Rg)
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.4)); gk = dict(cmap="gray_r", vmin=0, vmax=1)
    show(ax[0, 0], Xw, **gk); ax[0, 0].set_title(("(a) state window (true raster; model input is the full 128x128 tray)" if a.world128 else "(a) model input: state window (64x64, 1 mm/px)"), fontsize=9)
    rgb = np.ones((64, 64, 3)); c1, c2 = np.array([.9, .45, 0.]), np.array([0., .55, .75])
    for c, chn in ((c1, Pl[0].numpy()), (c2, Pl[1].numpy())): rgb = rgb * (1 - chn[..., None]) + chn[..., None] * c
    show(ax[0, 1], np.clip(rgb, 0, 1)); ax[0, 1].legend(handles=[Patch(color=c1, label="plate at push start"), Patch(color=c2, label="plate at push end")], fontsize=7, loc="lower right")
    ax[0, 1].set_title(f"(b) action rasters, L={np.linalg.norm(p1[0]-p0[0])*1e3:.0f} mm", fontsize=9)
    show(ax[0, 2], Yw, **gk); ax[0, 2].set_title("(c) true next state (window raster)", fontsize=9)
    show(ax[1, 1], Pw, **gk); ax[1, 1].set_title(("(e) model output (128x128 prediction resampled into the window)" if a.world128 else "(e) model output"), fontsize=9)
    show(ax[1, 2], error_rgb(Pw, Yw))
    ax[1, 2].contour(np.linspace(ext[0], ext[1], 64), np.linspace(ext[3], ext[2], 64)[::-1][::-1], R.numpy().astype(float), levels=[.5], colors="0.5", linewidths=.7, linestyles="--")
    ax[1, 2].legend(handles=[Patch(color="r", label="pred > true"), Patch(color="b", label="pred < true"), Patch(color="k", label="agree, occupied"), Patch(fc="w", ec="k", label="agree, empty"),
                             plt.Line2D([], [], color="0.5", ls="--", label="swept region")], fontsize=6, loc="lower right")
    ax[1, 2].set_title("(f) error map, intensity = |pred - true|", fontsize=9)
    b = ax[1, 0]; keys = ["accuracy", "changed IoU"]
    for j, (lab, vals, col) in enumerate((("window frame (finer grid)", [aw, iw], "#2a6fbb"), ("pasted into 64x64 world raster", [ap_, ip], "#8c8c8c"))):
        bars = b.bar(np.arange(2) + (j - .5) * .38, vals, .36, color=col, label=lab)
        for bb, v in zip(bars, vals): b.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    b.axhline(0, color="k", lw=.6); b.set_xticks([0, 1]); b.set_xticklabels(keys, fontsize=8); b.legend(fontsize=6, loc="upper right"); b.set_ylim(min(-.1, aw - .1, ap_ - .1), 1.1)
    b.set_title("(d) accuracy = 1 - rms(pred-true)/rms(prev-true) in swept region", fontsize=8)
    fig.suptitle(f"{a.tag} | test chain row {i} ({kinds[i]}, {int(moved[i])} cubes moved) | window {spec.side*1e3:.0f} mm", fontsize=10)
    fig.tight_layout(); f = out / f"zoom_row{i}_{kinds[i]}.png"; fig.savefig(f, dpi=100); plt.close(fig); print("wrote", f, f"win acc {aw:.2f} pasted {ap_:.2f}")
