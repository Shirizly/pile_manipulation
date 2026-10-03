"""Mask-vs-image comparison figures for EXP-0061 (reads eval_agreement.py output).
python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/image_mask/make_figures.py"""
import json, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from image_mask import Camera, GridSpec, segment, mask_to_occupancy
from FlexData.dataset import rasterize_disk, flex_xz_to_table

OUT = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/image_mask"
D20 = REPO / "datasets/DS-0020-training-data-flex-N864/old_data"
D19 = next((REPO / "datasets").glob("DS-0019-*")) / "data"
G = GridSpec(); CAM = Camera.load()
THR = 0.0                      # recommended binarisation: any visible foreground in the cell
EXT = [-G.half_extent, G.half_extent]
R = [json.loads(l) for l in open(OUT / "agreement.jsonl")]
MAN = [json.loads(l) for l in open(D19.parent / "manifest.jsonl")]


def paths(ds, sid):
    if ds == "DS-0020":
        t, k = sid.split("/")
        return D20 / f"{t}/{k}_color.png", D20 / f"{t}/{k}_particles.npy"
    s, a = sid.split("/")
    if a == "init":
        return D19 / f"{s}/initial_color.png", D19 / f"{s}/initial_particles.npy"
    return D19 / f"{s}/{a}_after_color.png", D19 / f"{s}/{a}_after_particles.npy"


def panel_row(axs, ds, sid, label):
    cp, pp = paths(ds, sid)
    bgr = cv2.imread(str(cp)); rgb = bgr[..., ::-1]
    m = segment(bgr)
    frac = mask_to_occupancy(m, CAM, G)
    P = np.load(pp).reshape(-1, 4)[:, :3]
    P = P[np.isfinite(P).all(1) & (np.abs(P).max(1) < 10)]
    ras = rasterize_disk(flex_xz_to_table(P[:, [0, 2]]) * G.to_pxl + G.ctr, 64, 64, 1.0) > 0
    b = frac > THR
    iou = (b & ras).sum() / (b | ras).sum()
    rec = next((r for r in R if r["ds"] == ds and r["id"] == sid), None)
    hid = rec["hidden_frac"] if rec else float("nan")
    # grid boundary (+-7.2 on the table plane) drawn in the image
    c = CAM.project(np.array([[x, 0, z] for x, z in [(-7.2, -7.2), (7.2, -7.2), (7.2, 7.2), (-7.2, 7.2), (-7.2, -7.2)]]))
    axs[0].imshow(rgb); axs[0].plot(c[:, 0], c[:, 1], "k--", lw=0.6)
    axs[0].set_title(f"{label}\n{ds} {sid}  N={len(P)}", fontsize=8)
    axs[1].imshow(m, cmap="gray_r"); axs[1].set_title("colour mask (image space)\nfg = any channel != 255", fontsize=8)
    kw = dict(extent=[EXT[0], EXT[1], EXT[1], EXT[0]], interpolation="nearest")   # x = row (down), Y=-z = col (right)
    axs[2].imshow(frac, cmap="viridis", vmin=0, vmax=1, **kw)
    axs[2].set_title(f"mask -> grid (area frac)\nbinary >{THR}: {int(b.sum())} cells", fontsize=8)
    axs[3].imshow(ras, cmap="gray_r", vmin=0, vmax=1, **kw)
    axs[3].set_title(f"particle raster (1-px disc)\n{int(ras.sum())} cells, y_max={P[:, 1].max():.2f}", fontsize=8)
    ov = np.ones((64, 64, 3)); ov[b & ras] = 0.6
    ov[b & ~ras] = [0.85, 0.1, 0.1]; ov[~b & ras] = [0.1, 0.3, 0.9]
    axs[4].imshow(ov, **kw)
    axs[4].set_title(f"diff: red=mask only, blue=raster only\nIoU {iou:.3f}; hidden (stacked) {100*hid:.0f}% of particles", fontsize=8)
    for a in axs[:2]:
        a.set_xticks([]); a.set_yticks([])
        a.set_xlabel("image u (= +x_flex) ->,  v (= +z_flex) down", fontsize=6)
    for a in axs[2:]:
        a.set_xlabel("col: Y = -z_flex", fontsize=7); a.set_ylabel("row: X = x_flex", fontsize=7)
        a.tick_params(labelsize=6)


def pick():
    d20 = [r for r in R if r["ds"] == "DS-0020"]
    d19 = [r for r in R if r["ds"] == "DS-0019"]
    blob = [r for r in d19 if r["init_pos"] == "rand_blob" and r["id"].endswith("init")]
    spread = [r for r in d19 if r["init_pos"] == "rand_spread" and r["id"].endswith("init")]
    med = lambda L, k: sorted(L, key=lambda r: r[k])[len(L) // 2]
    out = [("DS-0019 blob (median IoU)", med(blob, "iou_t0.0")),
           ("DS-0019 spread (median IoU)", med(spread, "iou_t0.0")),
           ("thin pile (lowest y_max)", min(d19, key=lambda r: r["y_max"])),
           ("multi-layer (most hidden)", max(d19, key=lambda r: r["hidden_frac"])),
           ("DS-0019 worst IoU", min(d19, key=lambda r: r["iou_t0.0"])),
           ("DS-0020 (median IoU)", med(d20, "iou_t0.0"))]
    return [(lab, r["ds"], r["id"]) for lab, r in out]


def before_after():
    # the DS-0019 after-state with the largest raster change vs its init, both rows
    afters = [r for r in R if r["ds"] == "DS-0019" and not r["id"].endswith("init")]
    best = max(afters, key=lambda r: abs(r["raster_area"] - next(q for q in R if q["ds"] == "DS-0019" and q["id"] == r["id"].split("/")[0] + "/init")["raster_area"]))
    s, a = best["id"].split("/")
    act = next(m["action"] for m in MAN if m["type"] == "action" and m["state_idx"] == int(s) and m["action_idx"] == int(a))
    return s, a, act


def comparison(rows, fname, title, action=None):
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 3.5 * len(rows)))
    axes = np.atleast_2d(axes)
    for ax, (lab, ds, sid) in zip(axes, rows):
        panel_row(ax, ds, sid, lab)
    if action is not None:   # stored action = (x0, -z0, x1, -z1) -> table (X, Y) is the action itself
        x0, y0, x1, y1 = action
        for ax in axes[:, 2:].ravel():
            ax.annotate("", xy=(y1, x1), xytext=(y0, x0), arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
        for ax in axes[:, :2].ravel():
            uv = CAM.project(np.array([[x0, 0, -y0], [x1, 0, -y1]]))
            ax.annotate("", xy=uv[1], xytext=uv[0], arrowprops=dict(arrowstyle="->", color="orange", lw=1.5))
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / len(rows) / 3.5))
    fig.savefig(OUT / fname, dpi=110); plt.close(fig)
    print("wrote", OUT / fname)


def summary():
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    groups = [("DS-0020\n(train, n=200)", lambda r: r["ds"] == "DS-0020", BLUE),
              ("DS-0019 blob\n(test, n=100)", lambda r: r.get("init_pos") == "rand_blob", ORANGE),
              ("DS-0019 spread\n(test, n=100)", lambda r: r.get("init_pos") == "rand_spread", AQUA)]
    metrics = [("iou_t0.0", "IoU, binary mask (frac > 0) vs particle raster"),
               ("hidden_frac", "fraction of particles hidden under the top layer"),
               ("tv_mass", "TV distance to particle-mass splat (scoring truth)")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    rng = np.random.default_rng(0)
    for ax, (k, ttl) in zip(axs, metrics):
        for i, (lab, f, col) in enumerate(groups):
            v = np.array([r[k] for r in R if f(r)])
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=col, alpha=0.55, lw=0)
            q1, q2, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i - 0.28, i + 0.28], [q2, q2], color="#0b0b0b", lw=2)
            ax.plot([i, i], [q1, q3], color="#0b0b0b", lw=1)
            ax.text(i + 0.3, q2, f"{q2:.3f}", fontsize=8, va="center", color="#52514e")
        if k == "tv_mass":   # reference: the binary particle raster's own distance to the mass splat
            for i, (lab, f, col) in enumerate(groups):
                v = np.median([r["tv_raster_vs_mass"] for r in R if f(r)])
                ax.plot([i - 0.28, i + 0.28], [v, v], color="#52514e", lw=1.2, ls=":")
            ax.text(0.98, 0.02, "dotted = particle raster's own TV to mass", transform=ax.transAxes,
                    fontsize=7, va="bottom", ha="right", color="#52514e")
        ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.suptitle("Image-mask occupancy vs particle data, 64x64 grid +-7.2 (black bar = median, line = IQR)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "summary_iou.png", dpi=130); plt.close(fig)
    print("wrote", OUT / "summary_iou.png")


if __name__ == "__main__":
    summary()
    comparison(pick(), "mask_vs_image_states.png",
               "Image mask vs particle raster -- 6 states. Grid panels: row = X = x_flex (down), col = Y = -z_flex (right), "
               "+-7.2 units; image panels are the raw 720x720 top-down render (dashed = grid boundary)")
    s, a, act = before_after()
    comparison([("before push", "DS-0019", f"{s}/init"), ("after push", "DS-0019", f"{s}/{a}")],
               "mask_vs_image_before_after.png", f"Before/after one push (DS-0019 state {s}, action {a}; orange = push)", action=act)
