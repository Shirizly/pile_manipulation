"""EXP-0062 combined three-family summary (no new scoring).

Reads the per-family analysis files already written by RUN-0001/0002/0003
(results/{nfd_seed0,lf,gnn}_ds0019.json, results/timing_{nfd,lf,gnn}.json) and
writes results/combined_v2.{json,md} + figures/combined_v2.png:
slateN per value function (mean of 3 goals, slate-bootstrap CI), accuracy
(slate-cluster CI), CUDA/CPU ms per slate, and the cross-family paired slateN
differences (copied from the files that computed them; Holm families differ by
file -- every cross-family p < 1e-4 regardless).

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/combined_summary.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
R = EXP / "results"
VFS = ["lyapunov", "mass_in_region", "signed_mass"]


def load(name):
    return json.loads((R / name).read_text())


nfd, lf, gnn = load("nfd_seed0_ds0019.json"), load("lf_ds0019.json"), load("gnn_ds0019.json")
tn, tl, tg = load("timing_nfd.json"), load("timing_lf.json"), load("timing_gnn.json")

# (label, family, generation, slateN dict, accuracy dict)
ROWS = [
    ("NFD v2 (MODEL-0008)", "NFD", "v2", nfd["slateN"]["nfd_v2_s0"], nfd["accuracy_ds0019"]["nfd_v2_s0"]),
    ("NFD v1 seed 0 (MODEL-0005)", "NFD", "v1", nfd["slateN"]["nfd_v1_s0 (EXP-0061 file)"], nfd["accuracy_ds0019"]["nfd_v1_s0"]),
    ("LF switched v2 (MODEL-0009)", "LF", "v2", lf["slateN"]["lf_v2_switched"], lf["accuracy_ds0019"]["lf_v2_switched"]),
    ("LF switched v1 (MODEL-0004)", "LF", "v1", lf["slateN"]["lf_v1_switched (MODEL-0004, EXP-0061 file)"], lf["accuracy_ds0019"]["lf_v1_switched"]),
    ("LF single v2 (MODEL-0009)", "LF single", "v2", lf["slateN"]["lf_v2_single"], lf["accuracy_ds0019"]["lf_v2_single"]),
    ("GNN v2 N30 (MODEL-0010)", "GNN", "v2", gnn["slateN"]["gnn_v2_n30 (MODEL-0010)"], gnn["accuracy_ds0019"]["gnn_v2_n30"]),
    ("GNN original ckpt N200 (EXP-0061)", "GNN", "v1", gnn["slateN"]["gnn_old_n200 (orig ckpt, EXP-0061 file)"], gnn["accuracy_ds0019"]["gnn_old_n200"]),
    ("GNN original ckpt N30", "GNN", "v1b", gnn["slateN"]["gnn_old_n30 (orig ckpt, N 30)"], gnn["accuracy_ds0019"]["gnn_old_n30"]),
    ("GNN render cap N30 (true node motion)", "cap", "-", gnn["slateN"]["cap_n30 (true node motion)"], gnn["accuracy_ds0019"]["cap_n30"]),
]


def tmed(cfg):
    return cfg.get("aggregate", cfg)["per_slate_ms_median"]["median"]


gk = list(tg["configs"])
TIMING = {
    "NFD v2": {"cuda": tn["aggregate"]["per_slate_ms_median"]["median"], "cpu": tmed(tg["configs"][gk[4]])},
    "LF switched v2": {"cuda": tmed(tl["configs"]["switched_cuda"]), "cpu": tmed(tl["configs"]["switched_cpu"])},
    "LF single v2": {"cuda": tmed(tl["configs"]["single_cuda"]), "cpu": tmed(tl["configs"]["single_cpu"])},
    "GNN v2 incl. perception": {"cuda": tmed(tg["configs"][gk[0]]), "cpu": tmed(tg["configs"][gk[2]])},
    "GNN v2 perception cached": {"cuda": tmed(tg["configs"][gk[1]]), "cpu": tmed(tg["configs"][gk[3]])},
}


def pick(pairs, a, b):
    for p in pairs:
        if p["a"] == a and p["b"] == b:
            return {k: p[k] for k in ("mean_diff", "ci_lo", "ci_hi", "wins_a", "wins_b", "p_holm")}
    raise KeyError((a, b))


PAIRS = {}
for scope in ["all_9_cells"] + VFS:
    PAIRS[scope] = {
        "nfd_v2 - lf_v2_switched": {k: (-v if k in ("mean_diff",) else v) for k, v in pick(lf["paired_slateN"][scope], "lf_v2_switched", "nfd_v2_s0").items()},
        "lf_v2_switched - gnn_v2": {k: (-v if k == "mean_diff" else v) for k, v in pick(gnn["paired_slateN"][scope], "gnn_v2_n30", "lf_v2_switched").items()},
        "nfd_v2 - gnn_v2": {k: (-v if k == "mean_diff" else v) for k, v in pick(gnn["paired_slateN"][scope], "gnn_v2_n30", "nfd_v2_s0").items()},
    }
    # sign-flipped pairs: swap CI ends and win counts
    for name in PAIRS[scope]:
        d = PAIRS[scope][name]
        d["ci_lo"], d["ci_hi"] = -d["ci_hi"], -d["ci_lo"]
        d["wins_a"], d["wins_b"] = d["wins_b"], d["wins_a"]

out = {
    "note": "Derived from RUN-0001/0002/0003 analysis files; no new scoring. Pairs copied from lf_ds0019.json (NFD-LF) and gnn_ds0019.json (GNN pairs), sign-flipped so the better model is first.",
    "rows": [{"label": l, "family": f, "gen": g,
              "slateN": {vf: s["avg_goals"][vf] for vf in VFS} | {"avg_9": s["avg_9"]},
              "accuracy": a} for l, f, g, s, a in ROWS],
    "timing_ms_per_slate": TIMING,
    "paired_slateN": PAIRS,
}
tmp = R / "combined_v2.json.tmp"
tmp.write_text(json.dumps(out, indent=1))
os.replace(tmp, R / "combined_v2.json")

# ---- markdown -------------------------------------------------------------
L = ["# EXP-0062 combined summary (DS-0019, 100 slates; derived by code/combined_summary.py)", "",
     "| model | slateN lyapunov | slateN mass_in_region | slateN signed_mass | slateN all 9 | accuracy (suspect across families) |",
     "|---|---|---|---|---|---|"]
for r in out["rows"]:
    c = [f"{r['slateN'][k]['mean']:.3f} [{r['slateN'][k]['ci'][0]:.3f}, {r['slateN'][k]['ci'][1]:.3f}]" for k in VFS + ["avg_9"]]
    a = r["accuracy"]
    L.append(f"| {r['label']} | " + " | ".join(c) + f" | {a['accuracy']:.3f} [{a['slate_ci'][0]:.3f}, {a['slate_ci'][1]:.3f}] |")
L += ["", "Paired slateN, better model first (diff [95 % slate-bootstrap CI], wins/losses, Holm p in its source file):", "",
      "| pair | all 9 | lyapunov | mass_in_region | signed_mass |", "|---|---|---|---|---|"]
for name in PAIRS["all_9_cells"]:
    cells = []
    for scope in ["all_9_cells"] + VFS:
        d = PAIRS[scope][name]
        cells.append(f"{d['mean_diff']:+.3f} [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] {d['wins_a']}/{d['wins_b']}")
    L.append(f"| {name} | " + " | ".join(cells) + " |")
L += ["", "Inference, ms per DS-0019 slate (whole pool, 86-191 candidates, one batch; median of 3 processes). Perception (PNG -> binary mask, CPU) adds 10.3 ms per image for NFD/LF; the GNN's own perception is inside its 'incl.' row.", "",
      "| model | CUDA (RTX 4070 Laptop) | CPU (4 threads) |", "|---|---|---|"]
for k, v in TIMING.items():
    L.append(f"| {k} | {v['cuda']:.1f} | {v['cpu']:.1f} |")
(R / "combined_v2.md").write_text("\n".join(L) + "\n")

# ---- figure ---------------------------------------------------------------
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"NFD": "#2a78d6", "GNN": "#eb6834", "LF": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axs = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=BG,
                        gridspec_kw={"width_ratios": [2.2, 1.1, 1.3]})
fams = [("NFD", "NFD v2 (MODEL-0008)", "NFD v1 seed 0 (MODEL-0005)"),
        ("LF", "LF switched v2 (MODEL-0009)", "LF switched v1 (MODEL-0004)"),
        ("GNN", "GNN v2 N30 (MODEL-0010)", "GNN original ckpt N200 (EXP-0061)")]
byl = {r["label"]: r for r in out["rows"]}
ax = axs[0]
for i, vf in enumerate(VFS):
    for j, (f, v2, v1) in enumerate(fams):
        x = i + (j - 1) * 0.26
        for lab, dx, filled in ((v2, 0.05, True), (v1, -0.05, False)):
            s = byl[lab]["slateN"][vf]
            ax.errorbar(x + dx, s["mean"], yerr=[[s["mean"] - s["ci"][0]], [s["ci"][1] - s["mean"]]],
                        fmt="o", ms=7, color=COL[f], mfc=COL[f] if filled else BG, mew=2, elinewidth=2, capsize=0, zorder=3)
cap = byl["GNN render cap N30 (true node motion)"]["slateN"]
for i, vf in enumerate(VFS):
    ax.hlines(cap[vf]["mean"], i + 0.26 - 0.1, i + 0.26 + 0.1, color=INK2, lw=1.5, linestyles=":", zorder=2)
ax.set_xticks(range(3)); ax.set_xticklabels(VFS)
ax.set_ylabel("slateN, mean of 3 goals (95 % slate CI)")
ax.set_title("slateN on DS-0019 (filled = trained on DS-0020 v2, hollow = v1 / original)", color=INK, fontsize=9.5, loc="left")
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=COL[f], ms=7, label=n) for f, n in
     (("NFD", "NFD"), ("LF", "LF switched"), ("GNN", "GNN (v1 = original ckpt N200)"))]
h.append(Line2D([], [], color=INK2, ls=":", label="GNN render cap, N30"))
ax.legend(handles=h, frameon=False, loc="lower left", fontsize=8)

ax = axs[1]
for j, (f, v2, v1) in enumerate(fams):
    for lab, dx, filled in ((v2, 0.08, True), (v1, -0.08, False)):
        a = byl[lab]["accuracy"]
        ax.errorbar(j + dx, a["accuracy"], yerr=[[a["accuracy"] - a["slate_ci"][0]], [a["slate_ci"][1] - a["accuracy"]]],
                    fmt="o", ms=7, color=COL[f], mfc=COL[f] if filled else BG, mew=2, elinewidth=2, capsize=0, zorder=3)
ca = byl["GNN render cap N30 (true node motion)"]["accuracy"]["accuracy"]
ax.hlines(ca, 2 - 0.2, 2 + 0.2, color=INK2, lw=1.5, linestyles=":")
ax.set_xticks(range(3)); ax.set_xticklabels(["NFD", "LF sw", "GNN"])
ax.set_ylim(0, 0.65); ax.set_ylabel("swept-region accuracy (slate CI)")
ax.set_title("accuracy (diagnostic; suspect across families)", color=INK, fontsize=9.5, loc="left")

ax = axs[2]
names = ["NFD v2", "LF switched v2", "GNN v2 perception cached", "GNN v2 incl. perception"]
short = ["NFD", "LF switched", "GNN (cached\nperception)", "GNN (incl.\nperception)"]
cols = [COL["NFD"], COL["LF"], COL["GNN"], COL["GNN"]]
for k, (n, c) in enumerate(zip(names, cols)):
    v = TIMING[n]["cuda"]
    ax.barh(k, v, 0.6, color=c, zorder=2)
    ax.text(v * 1.03, k, f"{v:.1f} ms", va="center", color=INK, fontsize=8.5)
ax.set_yticks(range(4)); ax.set_yticklabels(short); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("CUDA ms per slate (RTX 4070 Laptop)")
ax.set_title("inference, whole pool per slate", color=INK, fontsize=9.5, loc="left")
ax.text(0.98, 0.97, "+10.3 ms/image PNG->mask (CPU)\nfor NFD/LF, shared", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.5, color=INK2)
for a in axs:
    a.set_facecolor(BG)
    a.grid(axis="y" if a is not axs[2] else "x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
(EXP / "figures").mkdir(exist_ok=True)
fig.savefig(EXP / "figures" / "combined_v2.png", dpi=150, facecolor=BG)
print((R / "combined_v2.md").read_text())
