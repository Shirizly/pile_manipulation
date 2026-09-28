"""One (state, push) transition through one learned occupancy model, as a 2x3 figure.

    (a) model INPUT state raster      (b) ACTION as the model sees it     (c) TRUE next state
    (d) RMS / accuracy / changed IoU  (e) model OUTPUT raster             (f) signed error map

Everything is on the planner's 64 px slate grid (`simple_mpc.adapters.occ_from_particles`,
dim 0 = world x, dim 1 = world y; the truth is the same hard raster the models are trained
and `accuracy`-scored against, EXP-0053 `eval_narrow.py`). For NFD-family models panel (b) is
the UNet's own input channels 1 and 2, CAPTURED with a forward pre-hook on the UNet (not
re-derived): plate at push start / plate at push end (`draw_plate_soft`). Models without an
action raster (switched-linear, `linear_switched_hard` = the fitted hard length gate,
`linear_switched_soft` = its relaxed gate for gradients) show the same raster, captured from
the reference NFD for the same action.

Display orientation: world x to the right, world y DOWN (`--y-up` for the other way). This is
the orientation in which the letter goals (`Baselines/common/goals.py::letter_mask`, asset
row = world y) read correctly, and the one the demo GIFs use (EXP-0051/EXP-0055 demo_gifs.py).

Usage:
  CUDA_VISIBLE_DEVICES="" python scripts/probes/transition_panel.py \
      --model nfd_residual_worldframe_noaug_ep43 --source narrow --index 3 17 40 \
      --out experiments/temp/transition-panels
  --source narrow   = Genesis/data/narrow_l20_n20/test_chains (rows concatenated in file order)
  --source <dir>    = a binned slate corpus (BinnedSlateCorpus; index into .transitions())
"""
import argparse, glob, sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, OCC_BOUNDS, OCC_GRID
from eval_narrow import swept_region            # the accuracy region EXP-0053 reports
from cube_viz import push_arrow as _push_arrow  # shared arrow-drawing primitive

NARROW = REPO / "Genesis/data/narrow_l20_n20/test_chains"
PX_MM = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) * 1000 / (OCC_GRID - 1)   # pixel-centre pitch
LO, HI = OCC_BOUNDS["x_min"] * 1000 - PX_MM / 2, OCC_BOUNDS["x_max"] * 1000 + PX_MM / 2


def load_rows(source):
    """-> (states (N,P,>=3), states_ (N,P,>=3), act (N,4) metres [sx,sy,ex,ey])."""
    if source == "narrow":
        cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        ch = [cpu(torch.load(f, map_location="cpu", weights_only=False))
              for f in sorted(glob.glob(str(NARROW / "_*_data.pt")))]
        S = torch.cat([d["states"] for d in ch]); S_ = torch.cat([d["states_"] for d in ch])
        A = torch.cat([torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1) for d in ch])
    else:
        from Genesis.binned_slate_dataset import BinnedSlateCorpus
        t = BinnedSlateCorpus.load(REPO / source).transitions()
        S, S_ = t.states, t.states_
        A = torch.cat([t.p_starts[:, :2], t.p_stops[:, :2]], 1)
    return S.float(), S_.float(), A.float()


def show(ax, img, y_up=False, **kw):
    """Draw a (64,64) dim0=x, dim1=y grid in world mm, x right, y down (or up)."""
    img = np.asarray(img)
    ax.imshow(img.transpose(1, 0, 2) if img.ndim == 3 else img.T, origin="lower", extent=[LO, HI, LO, HI], interpolation="nearest", **kw)
    ax.set_xlim(LO, HI); ax.set_ylim((LO, HI) if y_up else (HI, LO)); ax.set_aspect("equal")
    ax.set_xlabel("world x [mm]  (grid dim 0)", fontsize=7)
    ax.set_ylabel(f"world y [mm]  (grid dim 1), {'up' if y_up else 'down'}", fontsize=7)
    ax.tick_params(labelsize=6)


def push_arrow(ax, act_mm, color="k"):
    sx, sy, ex, ey = act_mm
    _push_arrow(ax, (sx, sy), (ex, ey), color=color, lw=1.2)


def capture_input(ad, occ, act):
    """Run predict_step; return (pred, UNet input tensor or None)."""
    grab = {}
    net = getattr(getattr(ad, "predictor", None), "model", None)
    h = net.register_forward_pre_hook(lambda m, a: grab.setdefault("x", a[0].detach().cpu())) if net is not None else None
    with torch.no_grad():
        pred = ad.predict_step(occ, act).float().cpu()
    if h is not None:
        h.remove()
    return pred, grab.get("x")


def metrics(pred, truth, prev, region):
    out = {}
    for name, r in (("whole", torch.ones_like(truth, dtype=torch.bool)), ("swept", region)):
        rf = r.float(); n = rf.sum().clamp_min(1)
        rm = float(torch.sqrt((((pred - truth) ** 2) * rf).sum() / n))
        rp = float(torch.sqrt((((prev - truth) ** 2) * rf).sum() / n))
        ch_t = ((truth - prev).abs() > 0.5) & r; ch_p = ((pred - prev).abs() > 0.5) & r
        agree = ch_t & ch_p & ((pred > 0.5) == (truth > 0.5))
        union = (ch_t | ch_p).sum()
        out[name] = dict(rms_model=rm, rms_persist=rp, accuracy=1 - rm / max(rp, 1e-12),
                         changed_iou=float(agree.sum() / union) if union > 0 else float("nan"))
    return out


def error_rgb(pred, truth):
    """white = both empty, black = both occupied, red = pred > true, blue = pred < true;
    colour saturation = |pred - true| blended over the shared-occupancy grey."""
    p, t = pred.numpy().clip(0, 1), truth.numpy().clip(0, 1)
    d = p - t; a = np.abs(d)[..., None]
    base = np.repeat((1 - np.minimum(p, t))[..., None], 3, -1)          # agreement colour
    col = np.where((d > 0)[..., None], [1.0, 0, 0], [0, 0, 1.0])
    return np.clip((1 - a) * base + a * col, 0, 1)


_REF = {}


def reference_adapter():
    """NFD adapter whose UNet input channels 1, 2 are the action raster drawn in panel (b)."""
    if "ad" not in _REF:
        _REF["ad"] = make_occ_adapter("nfd_residual_worldframe_noaug_ep43", "cpu", "corner")
    return _REF["ad"]


def figure(model, ad, S, S_, A, i, out, y_up):
    occ0 = occ_from_particles(S[i:i + 1]); truth = occ_from_particles(S_[i:i + 1])[0]
    act = A[i:i + 1]; act_mm = (act[0] * 1000).tolist()
    pred, x = capture_input(ad, occ0, act)
    pred = pred[0]; prev = occ0[0]
    region = swept_region(act, "cpu")[0].cpu()
    M = metrics(pred, truth, prev, region)
    L_mm = float((act[0, 2:] - act[0, :2]).norm()) * 1000

    fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.6))
    gk = dict(cmap="gray_r", vmin=0, vmax=1)
    show(ax[0, 0], prev, y_up, **gk); ax[0, 0].set_title("(a) model input: state occupancy", fontsize=9)
    if x is None or x.shape[1] < 3:
        # models without an action raster (switched-linear): draw the SAME raster the NFD
        # models are given for this action (their UNet input channels 1, 2), for comparability
        x = capture_input(reference_adapter(), occ0, act)[1]
    rgb = np.ones((OCC_GRID, OCC_GRID, 3))
    c1, c2 = np.array([0.90, 0.45, 0.0]), np.array([0.0, 0.55, 0.75])          # orange, teal
    for c, ch in ((c1, x[0, 1].numpy()), (c2, x[0, 2].numpy())):
        rgb = rgb * (1 - ch[..., None]) + ch[..., None] * c
    show(ax[0, 1], np.clip(rgb, 0, 1), y_up)
    ax[0, 1].legend(handles=[Patch(color=c1, label="plate at push start"),
                             Patch(color=c2, label="plate at push end")], fontsize=7, loc="lower right")
    ax[0, 1].set_title(f"(b) action raster, L={L_mm:.0f} mm", fontsize=9)
    show(ax[0, 2], truth, y_up, **gk); ax[0, 2].set_title("(c) true next state (occ_from_particles)", fontsize=9)
    show(ax[1, 1], pred, y_up, **gk); ax[1, 1].set_title("(e) model output", fontsize=9)
    show(ax[1, 2], error_rgb(pred, truth), y_up)
    ax[1, 2].contour(np.linspace(LO, HI, OCC_GRID), np.linspace(LO, HI, OCC_GRID), region.numpy().T.astype(float),
                     levels=[0.5], colors="0.5", linewidths=0.7, linestyles="--")
    ax[1, 2].legend(handles=[Patch(color="r", label="pred > true"), Patch(color="b", label="pred < true"),
                             Patch(color="k", label="agree, occupied"), Patch(fc="w", ec="k", label="agree, empty"),
                             plt.Line2D([], [], color="0.5", ls="--", label="swept region")], fontsize=6, loc="lower right")
    ax[1, 2].set_title("(f) error map, intensity = |pred - true|", fontsize=9)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 1]):
        push_arrow(a, act_mm, "crimson")

    b = ax[1, 0]
    keys = [("rms_model", "RMS model"), ("rms_persist", "RMS persistence"), ("accuracy", "accuracy"),
            ("changed_iou", "changed-cell IoU")]
    xs = np.arange(len(keys))
    for j, (reg, col) in enumerate((("whole", "#8c8c8c"), ("swept", "#2a6fbb"))):
        vals = [M[reg][k] for k, _ in keys]
        bars = b.bar(xs + (j - 0.5) * 0.38, vals, 0.36, color=col, label=f"{reg} {'image' if reg == 'whole' else 'region (push rect +-24 mm, 20 mm pre-pad)'}")
        for bb, v in zip(bars, vals):
            b.text(bb.get_x() + bb.get_width() / 2, v if np.isfinite(v) else 0, f"{v:.2f}", ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=7)
    b.axhline(0, color="k", lw=0.6); b.set_xticks(xs); b.set_xticklabels([k for _, k in keys], fontsize=7)
    b.legend(fontsize=6, loc="upper left"); b.tick_params(labelsize=7)
    b.set_title("(d) accuracy = 1 - rms(pred-true)/rms(prev-true);\nchanged IoU: cells that moved (|dv|>0.5), persistence = 0", fontsize=8)
    lo = min(0, min(M[r]["accuracy"] for r in M)) - 0.1; b.set_ylim(lo, 1.12)

    fig.suptitle(f"{model}  |  transition {i}  |  action [sx,sy,ex,ey] = "
                 f"[{', '.join(f'{v:.1f}' for v in act_mm)}] mm  |  64 px grid, dim0 = world x", fontsize=10)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{model}_idx{i}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p, {r: {k: round(v, 3) for k, v in M[r].items()} for r in M})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nfd_residual_worldframe_noaug_ep43")
    ap.add_argument("--source", default="narrow", help="'narrow' or a binned corpus dir (repo-relative)")
    ap.add_argument("--index", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="experiments/temp/transition-panels")
    ap.add_argument("--y-up", action="store_true", help="world y up instead of down (letters then read upside down)")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    S, S_, A = load_rows(a.source)
    ad = make_occ_adapter(a.model, a.device, "corner")
    for i in a.index:
        figure(a.model, ad, S, S_, A, i, REPO / a.out, a.y_up)


if __name__ == "__main__":
    main()
