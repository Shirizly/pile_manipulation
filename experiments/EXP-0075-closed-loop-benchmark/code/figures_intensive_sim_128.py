"""intensive plans (letter_T_w20 start 40) at the models' native 128 px: TRUE simulator state (exact hard raster) / PREDICTED in sequence (zoom canvas + vanilla128, balance) / PREDICTION ERROR pred - true per push. Needs results/intensive_sim_arrays.npz 'parts'."""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0074-wide-domain-zoom-nfd/code"))
from batched_closed_loop import goal_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import OCC_BOUNDS
import wide_planner as wp
from fig128 import predict128, truth128
R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; F = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures"; R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"
res = json.load(open(R / "intensive.json")); sim = json.load(open(R / "intensive_sim.json")); S = np.load(R / "intensive_sim_arrays.npz")
NAMES = ["ref", "random", "cem10k", "greedy", "beam", "gd"]; LO, HI = OCC_BOUNDS["x_min"] * 1000, OCC_BOUNDS["x_max"] * 1000; EXT = [LO, HI, LO, HI]; gm = goal_mask("letter_T_w20").astype(float); cols = ["tab:red", "tab:orange", "tab:green", "tab:purple"]
S0 = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); S0 = S0.states[(S0.slate_idx == 40).nonzero()[0, 0]].float()
model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)


def show(ax, img, vmax=1.0):
    ax.imshow(np.asarray(img, float).T, origin="lower", extent=EXT, cmap="gray_r", vmin=0, vmax=vmax, interpolation="nearest"); ax.set_xlim(LO, HI); ax.set_ylim(HI, LO); ax.set_xticks([]); ax.set_yticks([])
    ax.contour(np.linspace(LO, HI, gm.shape[0]), np.linspace(LO, HI, gm.shape[0]), gm.T, levels=[0.5], colors="tab:blue", linewidths=0.8)


def arrow(ax, a, c, k):
    ax.annotate("", xy=(a[2] * 1000, a[3] * 1000), xytext=(a[0] * 1000, a[1] * 1000), arrowprops=dict(arrowstyle="->", color=c, lw=1.8)); ax.plot([a[0] * 1000], [a[1] * 1000], "o", ms=3, color=c); ax.text(a[0] * 1000, a[1] * 1000, str(k), color=c, fontsize=9, weight="bold")


def err(d):
    a = np.clip(np.abs(d) * 1.5, 0, 1)[..., None]; col = np.where((d > 0)[..., None], [1.0, 0, 0], [0, 0, 1.0]); return np.clip((1 - a) + a * col, 0, 1)


for n in ("ref", "random", "cem10k", "greedy", "beam", "gd"):
    i = NAMES.index(n); rep = int(np.nonzero(S["which"] == i)[0][0]); seq = np.array(res[n]["seq"]); tr = truth128(torch.from_numpy(S["parts"][rep])).numpy(); pr = predict128(model, S0, torch.tensor(seq)).numpy(); tv = S["vals"][rep] - S["vals"][rep][0]
    fig, ax = plt.subplots(3, 5, figsize=(16, 10)); show(ax[0, 0], tr[0]); [arrow(ax[0, 0], seq[k], cols[k], k + 1) for k in range(4)]; ax[0, 0].set_title("TRUE start (128 px) + the 4 pushes", fontsize=9); show(ax[1, 0], pr[0]); ax[1, 0].set_title("model input (mean of member inputs)", fontsize=9); ax[2, 0].axis("off")
    for k in range(4):
        show(ax[0, k + 1], tr[k + 1]); arrow(ax[0, k + 1], seq[k], cols[k], k + 1); ax[0, k + 1].set_title(f"TRUE (simulator) after push {k + 1}\nvalue change {tv[k + 1]:+.3f}", fontsize=8)
        show(ax[1, k + 1], pr[k + 1]); arrow(ax[1, k + 1], seq[k], cols[k], k + 1); ax[1, k + 1].set_title(f"PREDICTED (128 px) after push {k + 1}\nvalue change {res[n]['seq_value'][k]:+.3f}", fontsize=8)
        ax[2, k + 1].imshow(err((pr[k + 1] - tr[k + 1]).T), origin="lower", extent=EXT, interpolation="nearest"); ax[2, k + 1].set_xlim(LO, HI); ax[2, k + 1].set_ylim(HI, LO); ax[2, k + 1].set_xticks([]); ax[2, k + 1].set_yticks([])
        ax[2, k + 1].set_title(f"PREDICTION ERROR after push {k + 1}: pred - true\n(red: model has mass the truth lacks)", fontsize=8)
    fig.suptitle(f"letter_T_w20 start 40, '{n}' at 128 px: predicted terminal {res[n]['seq_value'][-1]:+.3f} vs simulator {sim[n]['true_mean'][-1]:+.3f}", fontsize=10); fig.tight_layout(); fig.savefig(F / f"intensive_sim128_{n}.png", dpi=100); plt.close(fig)
print("ok")
