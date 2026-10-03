"""EXP-0062 RUN-0003 -- GNN training curves (val loss / persistence val loss, per target) -> figures/gnn_v2_n30_convergence.png
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/plot_gnn_convergence.py"""
import json, os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parents[3]; os.chdir(REPO)
E = Path("experiments/EXP-0062-flex-v2-train-rerun")
pers = json.load(open(E / "results/gnn_persistence_val_loss.json"))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, (t, lab) in enumerate([("chamfer_carry", "visual target (Chamfer, PRIMARY)"), ("particle", "particle-supervised (alt)")]):
    h = json.load(open(f"Baselines/GNN/runs/flex_v2_n30_{t}_s0/history.json"))
    ep = [r["epoch"] for r in h]
    ax[i].plot(ep, [r["val"] / pers[t] for r in h], "o-", ms=3, label="val")
    ax[i].plot(ep, [r["train"] / pers[t] for r in h], "-", alpha=.6, label="train (sampled)")
    ax[i].plot(ep, [r["best"] / pers[t] for r in h], "k--", lw=1, label="best val")
    ax[i].set_title(lab, fontsize=10); ax[i].set_xlabel("epoch"); ax[i].set_ylabel("loss / persistence val loss")
    ax[i].grid(alpha=.3); ax[i].legend(fontsize=8)
fig.suptitle("dyn-res GNN, DS-0020 v2, N = 30, rollout 5, batch 4 (CPU)", fontsize=10)
fig.tight_layout(); (E / "figures").mkdir(exist_ok=True)
fig.savefig(E / "figures/gnn_v2_n30_convergence.png", dpi=120)
print("saved")
