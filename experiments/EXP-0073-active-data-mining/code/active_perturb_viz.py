"""PILOT: draw original vs perturbed+re-settled states from runs/active_pilot/perturbed_scatter_pilot.npz."""
import sys, numpy as np; sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scripts.probes.cube_viz import cube_patch
from active_perturb import yaw_of, TRAY, CUBE
Z = np.load("experiments/EXP-0073-active-data-mining/runs/active_pilot/perturbed_scatter_pilot.npz")
ops = ["jitter_large", "relocate3", "relocate6", "cluster_shift", "rigid"]; idx = [0, 1, 2]
fig, ax = plt.subplots(len(idx), 1 + len(ops), figsize=(2.2 * (1 + len(ops)), 2.2 * len(idx)))
for r, i in enumerate(idx):
    for c, nm in enumerate(["orig"] + ops):
        a = ax[r, c]; S = Z["src"][i] if nm == "orig" else Z[nm + "_post"][i]
        pre = None if nm == "orig" else Z[nm + "_pre"][i]
        for j in range(len(S)):
            a.add_patch(cube_patch(S[j, :2], yaw_of(S[j:j + 1, 3:7])[0], CUBE, facecolor="tab:blue" if nm == "orig" else "tab:orange", edgecolor="k", alpha=.8, lw=.4))
        a.plot([-TRAY, TRAY, TRAY, -TRAY, -TRAY], [-TRAY, -TRAY, TRAY, TRAY, -TRAY], "k", lw=.8); a.set_xlim(-.07, .07); a.set_ylim(-.07, .07); a.set_aspect("equal"); a.axis("off")
        if r == 0: a.set_title(nm if nm != "orig" else "source (DS-0015 scatter)", fontsize=8)
fig.suptitle("PILOT: legal perturbations, shown after in-sim re-settle (row = one source state)", fontsize=9); fig.tight_layout()
fig.savefig("experiments/EXP-0073-active-data-mining/figures/active_perturb_examples.png", dpi=110)
