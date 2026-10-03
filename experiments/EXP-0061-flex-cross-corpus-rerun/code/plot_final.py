"""EXP-0061 summary figure: slateN per model x goal x value fn (slate-bootstrap 95% CI of the
family mean; NFD/GNN = mean over 3 training / node-sampling seeds, seed range as a tick),
and accuracy DS-0020 val vs DS-0019 test (cluster-bootstrap 95% CI).

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/plot_final.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

E = Path(__file__).resolve().parents[1]
o = json.load(open(E / "results/final_eval/summary.json"))
fam = o["slateN_family"]
FAMS = [("NFD (3 seeds)", "NFD (3 training seeds)", "#2a78d6"),
        ("GNN (3 sampling seeds)", "GNN dyn-res, N=200 (3 sampling seeds)", "#eb6834"),
        ("lf_flex_switched", "Linear foresight, switched", "#1baf7a"),
        ("lf_flex_single", "Linear foresight, single", "#eda100")]
GOALS = ["random_quadrant", "ring_O", "T", "avg_goals"]
GL = ["random\nquadrant", "ring O", "T", "mean of\n3 goals"]
VFS = ["lyapunov", "mass_in_region", "signed_mass"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(15, 4.6), facecolor="#fcfcfb")
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.3, 0.95], wspace=0.28)
w = 0.19
for k, v in enumerate(VFS):
    ax = fig.add_subplot(gs[0, k], facecolor="#fcfcfb")
    for j, (key, lab, col) in enumerate(FAMS):
        x = np.arange(4) + (j - 1.5) * (w + 0.015)
        m = np.array([fam[key][g][v]["mean"] for g in GOALS])
        lo = np.array([fam[key][g][v]["slate_boot_ci"][0] for g in GOALS])
        hi = np.array([fam[key][g][v]["slate_boot_ci"][1] for g in GOALS])
        ax.bar(x, m, w, color=col, label=lab, zorder=2)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        if fam[key][GOALS[0]][v]["seed_sd"] is not None:   # seed range ticks
            smin = [fam[key][g][v]["seed_min"] for g in GOALS]
            smax = [fam[key][g][v]["seed_max"] for g in GOALS]
            ax.vlines(x + w * 0.32, smin, smax, color="#fcfcfb", lw=1.5, zorder=4)
    rnd = [fam["random"][g][v]["mean"] for g in GOALS]
    ax.plot(np.arange(4), rnd, ls="none", marker="_", ms=22, color=INK2, zorder=4,
            label="random (slateN 0 by construction)" if k == 0 else None)
    ax.axvline(2.5, color=GRID, lw=1, zorder=1)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(GL)
    ax.set_ylim(-0.05, 1.02); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(v, color=INK, fontsize=10, loc="left")
    if k == 0:
        ax.set_ylabel("slateN on DS-0019 (100 slates)")
fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="upper left", ncol=5, frameon=False,
           bbox_to_anchor=(0.06, 1.0), fontsize=8.5)

ax = fig.add_subplot(gs[0, 4], facecolor="#fcfcfb")
vt = o["val_vs_test_accuracy"]
rows = [("nfd_s0", "NFD seed 0", 0), ("nfd_s1", "NFD seed 1", 0), ("nfd_s2", "NFD seed 2", 0),
        ("lf_flex_switched", "LF switched", 2), ("lf_flex_single", "LF single", 3),
        ("gnn_flex_drp_samp0", "GNN (samp 0)", 1)]
for i, (m, lab, c) in enumerate(rows):
    d = vt[m]; y = len(rows) - 1 - i; col = FAMS[c][2]
    ax.plot([d["test"], d["val"]], [y, y], color=GRID, lw=2, zorder=1)
    ax.errorbar(d["val"], y, xerr=[[d["val"] - d["val_traj_ci"][0]], [d["val_traj_ci"][1] - d["val"]]],
                fmt="o", mfc="#fcfcfb", mec=col, mew=2, ms=8, ecolor=col, zorder=3)
    ax.errorbar(d["test"], y, xerr=[[d["test"] - d["test_slate_ci"][0]], [d["test_slate_ci"][1] - d["test"]]],
                fmt="o", color=col, ms=8, ecolor=col, zorder=3)
    ax.text(d["test"] - 0.025, y, f"{d['test']:.2f}", ha="right", va="center", fontsize=8, color=INK2)
    ax.text(d["val"] + 0.025, y, f"{d['val']:.2f}", ha="left", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1])
ax.set_xlim(0, 0.72); ax.axvline(0, color=INK2, lw=1)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("swept-region accuracy (0 = persistence)")
ax.set_title("accuracy: filled = DS-0019 test, hollow = DS-0020 val", color=INK, fontsize=9.5, loc="left")
fig.text(0.06, -0.04, "Bars: family mean; black whiskers: slate-bootstrap 95% CI; white tick: range over the 3 seeds. "
         "Truth = binary top-down image mask. Accuracy CIs: cluster bootstrap over slates (test) / trajectories (val).",
         fontsize=8, color=INK2)
out = E / "figures/final_eval"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "slateN_and_accuracy.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("wrote", out / "slateN_and_accuracy.png")
