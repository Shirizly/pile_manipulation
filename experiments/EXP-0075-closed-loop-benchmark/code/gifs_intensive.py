"""EXP-0075: GIFs of the intensive-optimisation transitions.
(1) intensive_T40_methods.gif : letter_T_w20 start 40, the six intensive plans (ref, random, cem10k, greedy, beam, gd) executed open loop in the simulator; top row = TRUE simulator states (cubes), bottom row = model-PREDICTED state (128 px ensemble prediction, truth overlaid by the goal outline) with the push drawn.
(2) gdbudget_<goal>_s<start>.gif : per task of the GD-budget study (ens128), plans ref / gd_ref (1,280-eval CEM + GD) / gd5 / gd10, true simulator states.
Frames = state after k pushes with the next push drawn (planned push; legaliser shifts were 0-3 mm).  usage: gifs_intensive.py"""
import json, sys, os
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import importlib.util
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code")); sys.path.insert(0, str(REPO / "scripts/probes")); sys.path.insert(0, str(REPO / "experiments/EXP-0074-wide-domain-zoom-nfd/code"))
spec = importlib.util.spec_from_file_location("exp0051_gifs", REPO / "experiments/EXP-0051-task-success-completion-time/code/demo_gifs.py"); d51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d51)
os.chdir(REPO)
R = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures/gifs"; R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"
NAMES = ["ref", "random", "cem10k", "greedy", "beam", "gd"]
res = json.load(open(R / "intensive.json")); S = np.load(R / "intensive_sim_arrays.npz"); mask = d51.goal_mask("letter_T_w20")
import wide_planner as wp
from fig128 import predict128
from batched_closed_loop import goal_mask as gm
from Genesis.binned_slate_dataset import BinnedSlateCorpus
S0 = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0); S0 = S0.states[(S0.slate_idx == 40).nonzero()[0, 0]].float()
model = wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True)
EXT = [d51.LO, d51.HI, d51.LO, d51.HI]
data = {}
for n in NAMES:
    rep = int(np.nonzero(S["which"] == NAMES.index(n))[0][0]); seq = np.array(res[n]["seq"]); data[n] = dict(parts=S["parts"][rep], seq=seq, vals=S["vals"][rep], pred=predict128(model, S0, torch.tensor(seq)).numpy(), pv=res[n]["seq_value"])
fig, axs = plt.subplots(2, 6, figsize=(24, 9.2))


def frame(k):
    k = min(k, 4)
    for c, n in enumerate(NAMES):
        d = data[n]; act = d["seq"][k] * 1000.0 if k < 4 else None
        d51.draw_frame(axs[0, c], mask, d["parts"][k][:, :2] * 1000.0, act, title=f"{n}: TRUE (simulator)\npush {k}/4   true dV {d['vals'][k] - d['vals'][0]:+.3f}")
        ax = axs[1, c]; ax.clear(); ax.imshow(d["pred"][k].T, origin="lower", extent=EXT, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.contour(np.linspace(d51.LO, d51.HI, mask.shape[0]), np.linspace(d51.LO, d51.HI, mask.shape[0]), mask.T, levels=[0.5], colors="tab:green", linewidths=1)
        if act is not None: ax.annotate("", xy=(act[2], act[3]), xytext=(act[0], act[1]), arrowprops=dict(arrowstyle="->", color="royalblue", lw=2))
        ax.set_xlim(d51.LO, d51.HI); ax.set_ylim(d51.HI, d51.LO); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{n}: model PREDICTED (128 px)\n" + (f"predicted dV {d['pv'][k - 1]:+.3f}" if k else "model input"), fontsize=8)
    fig.suptitle("letter_T_w20 start 40, intensive optimisation plans (H=4) executed open loop", fontsize=12)


fig.tight_layout(rect=(0, 0, 1, 0.96)); anim = FuncAnimation(fig, frame, frames=7, interval=800); anim.save(OUT / "intensive_T40_methods.gif", writer=PillowWriter(fps=1.1), dpi=90); plt.close(fig); print("wrote intensive_T40_methods.gif", flush=True)
# (2) gd-budget study plans per task
parts, keys = [], []
for f in sorted(R.glob("gd_budget_sim_parts_*.npz")):
    z = np.load(f); parts += list(z["parts"]); keys += [str(x) for x in z["specs"]]
PL = ["ref", "gd_ref", "gd5", "gd10"]; TASKS = sorted({tuple(k.split("|")[:2]) for k in keys})
for g, s in TASKS:
    m = d51.goal_mask(g); items = []
    for p in PL:
        i = keys.index(f"{g}|{s}|{p}"); j = json.load(open(R / f"gd_budget/{g}_{s}.json"))[p]; items.append((p, parts[i], np.array(j["seq"]), j["cost"]))
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.9))
    def fr(k, items=items, m=m, axs=axs, fig=fig, g=g, s=s):
        k = min(k, 4)
        for ax, (p, pt, sq, c) in zip(axs, items):
            lab = {"ref": "benchmark CEM 1,280", "gd_ref": "CEM 1,280 + GD", "gd5": "CEM 10,000 + GD", "gd10": "CEM 20,000 + GD"}[p]
            d51.draw_frame(ax, m, pt[k][:, :2] * 1000.0, sq[k] * 1000.0 if k < 4 else None, title=f"{lab}\npush {k}/4   predicted terminal {c:+.2f}")
        fig.suptitle(f"{g} start {s}: GD-budget plans (ens128) executed open loop", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); FuncAnimation(fig, fr, frames=7, interval=800).save(OUT / f"gdbudget_{g}_s{s}.gif", writer=PillowWriter(fps=1.1), dpi=100); plt.close(fig); print("wrote", g, s, flush=True)
