"""EXP-0075 diagnosis figures from results/diag_sim_arrays.npz.
fig_traj_<goal>_<start>.png : the selected H=4 plan executed open loop. Rows: TRUE state after push 0..4 (soft occupancy, goal outline, next push arrow) / MODEL-predicted state after push 1..4 (pasted raster) /
                              change error (predicted change - true change since the start; red = model adds more mass than happened, blue = less). Titles carry the true / predicted objective-value change.
fig_first_push.png          : same start state, the first push chosen by the H=1, H=2 and H=4 plans (arrows) and the TRUE state after executing only that push, with the true value change.
Display orientation: world x right, world y DOWN (letters read upright)."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import occ_from_particles, OCC_BOUNDS
R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; F = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures"; F.mkdir(exist_ok=True)
Z = np.load(R / "diag_sim_arrays.npz", allow_pickle=True); goals, starts = Z["goals"], Z["starts"]
LO, HI = OCC_BOUNDS["x_min"] * 1000, OCC_BOUNDS["x_max"] * 1000; EXT = [LO, HI, LO, HI]
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
base = {int(s): occ_from_particles(rows.states[(rows.slate_idx == int(s)).nonzero()[0, 0]].float()[None])[0].numpy() for s in set(starts.tolist())}
COL = {1: "tab:green", 2: "tab:orange", 4: "tab:red"}


def show(ax, img, cmap="gray_r", vmin=0, vmax=1):
    ax.imshow(np.asarray(img, float).T, origin="lower", extent=EXT, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"); ax.set_xlim(LO, HI); ax.set_ylim(HI, LO); ax.set_xticks([]); ax.set_yticks([])


def goal_outline(ax, g):
    m = goal_mask(g).astype(float); ax.contour(np.linspace(LO, HI, m.shape[0]), np.linspace(LO, HI, m.shape[1]), m.T, levels=[0.5], colors="tab:blue", linewidths=0.8)


def arrow(ax, a, c, lw=1.6):
    ax.annotate("", xy=(a[3] * 1000, a[2] * 1000 * 0 + a[3] * 1000) if False else (a[2] * 1000, a[3] * 1000), xytext=(a[0] * 1000, a[1] * 1000), arrowprops=dict(arrowstyle="->", color=c, lw=lw))
    ax.plot([a[0] * 1000], [a[1] * 1000], "o", ms=3, color=c)


def err_rgb(d):
    a = np.clip(np.abs(d) * 1.5, 0, 1)[..., None]; col = np.where((d > 0)[..., None], [1.0, 0, 0], [0, 0, 1.0]); return np.clip((1 - a) + a * col, 0, 1)


sel = [i for i in range(len(goals)) if int(starts[i]) in (40, 42)]
H = 4; seq, pred, true, tv, pv = Z[f"H{H}_seq"], Z[f"H{H}_pred_occ"], Z[f"H{H}_true_occ"].astype(float), Z[f"H{H}_true_val"], Z[f"H{H}_pred_val"]
for i in sel:
    g, s = str(goals[i]), int(starts[i]); fig, ax = plt.subplots(3, H + 1, figsize=(3 * (H + 1), 9.2)); pb = base[s]; ts = true[i, 0]
    for j in range(H + 1):
        show(ax[0, j], true[i, j]); goal_outline(ax[0, j], g)
        if j < H: arrow(ax[0, j], seq[i, j], COL[4])
        ax[0, j].set_title(f"TRUE after push {j}" + ("" if j == 0 else f"  dV {tv[i, j] - tv[i, 0]:+.3f}"), fontsize=9)
        if j == 0:
            show(ax[1, 0], pb); goal_outline(ax[1, 0], g); ax[1, 0].set_title("model input raster", fontsize=9); ax[2, 0].axis("off"); continue
        pr = pred[i, j - 1].astype(float); show(ax[1, j], pr); goal_outline(ax[1, j], g); ax[1, j].set_title(f"PREDICTED after push {j}  dV {pv[i, j - 1]:+.3f}", fontsize=9)
        ax[2, j].imshow(err_rgb(((pr - pb) - (true[i, j] - ts)).T), origin="lower", extent=EXT, interpolation="nearest"); ax[2, j].set_xlim(LO, HI); ax[2, j].set_ylim(HI, LO); ax[2, j].set_xticks([]); ax[2, j].set_yticks([])
        ax[2, j].set_title("pred change - true change", fontsize=8)
    fig.suptitle(f"{g}, start {s}: H=4 plan executed open loop (true dV after 4 pushes {tv[i, -1] - tv[i, 0]:+.3f}, predicted {pv[i, -1]:+.3f}); blue outline = goal", fontsize=10); fig.tight_layout(); fig.savefig(F / f"fig_traj_{g}_{s}.png", dpi=90); plt.close(fig)
# first push comparison
fig, ax = plt.subplots(len(sel), 4, figsize=(12, 3 * len(sel)))
for r, i in enumerate(sel):
    g, s = str(goals[i]), int(starts[i]); show(ax[r, 0], Z["start_occ"][i].astype(float)); goal_outline(ax[r, 0], g); ax[r, 0].set_title(f"{g} start {s}: first push chosen by H=1 / 2 / 4", fontsize=8)
    for k, Hh in enumerate((1, 2, 4)):
        a = Z[f"H{Hh}_seq"][i, 0]; arrow(ax[r, 0], a, COL[Hh], 1.4); t = Z[f"H{Hh}_true_occ"].astype(float)[i, 1]; v = Z[f"H{Hh}_true_val"][i]
        show(ax[r, k + 1], t); goal_outline(ax[r, k + 1], g); arrow(ax[r, k + 1], a, COL[Hh], 1.2); ax[r, k + 1].set_title(f"true state after the H={Hh} first push: dV {v[1] - v[0]:+.3f}", fontsize=8)
fig.tight_layout(); fig.savefig(F / "fig_first_push.png", dpi=90); plt.close(fig)
print("figures written to", F)
