"""figures for the simulator execution of the intensive sequences: TRUE state after each push (replica 0) / model-PREDICTED state in sequence / pred - true change; summary of predicted vs true value per push."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask
from simple_mpc.adapters import OCC_BOUNDS
R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; F = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures"
res = json.load(open(R / "intensive.json")); sim = json.load(open(R / "intensive_sim.json")); A = np.load(R / "intensive_arrays.npz"); S = np.load(R / "intensive_sim_arrays.npz")
NAMES = ["ref", "random", "cem10k", "greedy", "beam", "gd"]; LO, HI = OCC_BOUNDS["x_min"] * 1000, OCC_BOUNDS["x_max"] * 1000; EXT = [LO, HI, LO, HI]; gm = goal_mask("letter_T_w20").astype(float); cols = ["tab:red", "tab:orange", "tab:green", "tab:purple"]


def show(ax, img, cmap="gray_r", vmax=1.0):
    ax.imshow(np.asarray(img, float).T, origin="lower", extent=EXT, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest"); ax.set_xlim(LO, HI); ax.set_ylim(HI, LO); ax.set_xticks([]); ax.set_yticks([])
    ax.contour(np.linspace(LO, HI, 64), np.linspace(LO, HI, 64), gm.T, levels=[0.5], colors="tab:blue", linewidths=0.8)


def arrow(ax, a, c, k):
    ax.annotate("", xy=(a[2] * 1000, a[3] * 1000), xytext=(a[0] * 1000, a[1] * 1000), arrowprops=dict(arrowstyle="->", color=c, lw=1.8)); ax.plot([a[0] * 1000], [a[1] * 1000], "o", ms=3, color=c); ax.text(a[0] * 1000, a[1] * 1000, str(k), color=c, fontsize=9, weight="bold")


def err(d):
    a = np.clip(np.abs(d) * 1.5, 0, 1)[..., None]; col = np.where((d > 0)[..., None], [1.0, 0, 0], [0, 0, 1.0]); return np.clip((1 - a) + a * col, 0, 1)


for n in ("ref", "greedy", "beam", "gd"):
    i = NAMES.index(n); rep = int(np.nonzero(S["which"] == i)[0][0]); seq = np.array(res[n]["seq"]); tr = S["occs"][rep].astype(float); hd = S["hard"][rep].astype(float); pr = A[n + "_pred"]; start = A["start"]
    fig, ax = plt.subplots(3, 5, figsize=(15, 9.4)); show(ax[0, 0], tr[0]); [arrow(ax[0, 0], seq[k], cols[k], k + 1) for k in range(4)]; ax[0, 0].set_title("TRUE start + the 4 pushes", fontsize=9); show(ax[1, 0], start); ax[1, 0].set_title("model input raster", fontsize=9); ax[2, 0].axis("off")
    tv = (S["vals"][rep] - S["vals"][rep][0])
    for k in range(4):
        show(ax[0, k + 1], tr[k + 1]); arrow(ax[0, k + 1], seq[k], cols[k], k + 1); ax[0, k + 1].set_title(f"TRUE (simulator) after push {k + 1}\nvalue change {tv[k + 1]:+.3f}", fontsize=8)
        show(ax[1, k + 1], pr[k]); arrow(ax[1, k + 1], seq[k], cols[k], k + 1); ax[1, k + 1].set_title(f"PREDICTED in sequence after push {k + 1}\nvalue change {res[n]['seq_value'][k]:+.3f}", fontsize=8)
        ax[2, k + 1].imshow(err((pr[k] - hd[k + 1]).T), origin="lower", extent=EXT, interpolation="nearest"); ax[2, k + 1].set_xlim(LO, HI); ax[2, k + 1].set_ylim(HI, LO); ax[2, k + 1].set_xticks([]); ax[2, k + 1].set_yticks([]); ax[2, k + 1].set_title(f"PREDICTION ERROR after push {k + 1}: predicted - true\n(64x64 raster; red = model has mass the truth lacks; blue = reverse)", fontsize=8)
    fig.suptitle(f"letter_T_w20 start 40, '{n}': predicted terminal {res[n]['seq_value'][-1]:+.3f} vs simulator {np.mean(sim[n]['true_mean'][-1:]):+.3f} (mean of {sim[n]['n_replicas']} replicas)", fontsize=10); fig.tight_layout(); fig.savefig(F / f"intensive_sim_{n}.png", dpi=85); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for n in NAMES:
    ax[0].errorbar(range(1, 5), sim[n]["true_mean"], yerr=sim[n]["true_sd"], fmt="o-", label=f"{n}: sim {sim[n]['true_mean'][-1]:+.2f}"); ax[0].plot(range(1, 5), res[n]["seq_value"], "x--", color=ax[0].lines[-1].get_color(), alpha=0.6)
ax[0].set_title("value change after each push: simulator (solid) vs model prediction (dashed x)"); ax[0].legend(fontsize=8)
ax[1].bar(NAMES, [sim[n]["optimism_terminal"] for n in NAMES]); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("terminal optimism = predicted - simulator (negative = model over-promised)")
fig.tight_layout(); fig.savefig(F / "intensive_sim_summary.png", dpi=90); print("ok")
