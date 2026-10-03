"""EXP-0062 -- NFD v2 seed-0 convergence curve from the training log (+ EXP-0061 v1 seed 0
for reference, whose val loss is on a DIFFERENT val split -- DS-0020 v1 -- so only the shape
is comparable). Writes figures/nfd_v2_seed0_convergence.png and results/nfd_v2_seed0_curve.json.

    python -u experiments/EXP-0062-flex-v2-train-rerun/code/plot_convergence.py
"""
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
os.chdir(REPO)
E = Path("experiments/EXP-0062-flex-v2-train-rerun")
PAT = re.compile(r"Epoch\s+(\d+): trn=([\d.]+)\s+val=([\d.]+)")


def parse(path):
    txt = Path(path).read_text().replace("\r", "\n")
    d = {}
    for m in PAT.finditer(txt):
        d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))   # a resume re-logs nothing twice; last wins
    ep = sorted(d)
    return ep, [d[e][0] for e in ep], [d[e][1] for e in ep]


def main():
    ep, trn, val = parse(E / "logs/train_nfd_seed0.log")
    best = []
    b = float("inf")
    for v in val:
        b = min(b, v); best.append(b)
    # relative improvement of best-val over trailing 20 epochs
    rel20 = [None if i < 20 else (best[i - 20] - best[i]) / best[i - 20] for i in range(len(best))]
    out = dict(epochs=ep, train=trn, val=val, best_val=best, rel_best_improvement_20ep=rel20)
    v1 = "experiments/EXP-0061-flex-cross-corpus-rerun/logs/nfd_flex_mask_seed0.log"
    if Path(v1).exists():
        e1, t1, v1v = parse(v1)
        out["v1_reference"] = dict(epochs=e1, train=t1, val=v1v, note="DS-0020 v1 val split -- different rows")
    tmp = E / "results/nfd_v2_seed0_curve.json.tmp"
    tmp.write_text(json.dumps(out)); os.replace(tmp, E / "results/nfd_v2_seed0_curve.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, trn, color="#4a6fa5", label="train (v2)")
    ax[0].plot(ep, val, color="#c8553d", label="val (v2, 1,665 rows)")
    ax[0].plot(ep, best, color="#c8553d", ls=":", lw=1, label="best val")
    if "v1_reference" in out:
        ax[0].plot(out["v1_reference"]["epochs"], out["v1_reference"]["val"], color="#888888", lw=1,
                   label="EXP-0061 v1 val (other split)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE loss"); ax[0].legend(frameon=False)
    ax[0].set_title("NFD seed 0, DS-0020 v2")
    r = [(e, 100 * x) for e, x in zip(ep, rel20) if x is not None]
    if r:
        ax[1].plot([a for a, _ in r], [b for _, b in r], color="#4a6fa5")
    ax[1].axhline(0.5, color="#c8553d", ls="--", lw=1, label="plateau threshold 0.5 %")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("best-val improvement over last 20 epochs (%)")
    ax[1].set_ylim(0, 5); ax[1].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (E / "figures").mkdir(exist_ok=True)
    fig.savefig(E / "figures/nfd_v2_seed0_convergence.png", dpi=130)
    print("epochs", len(ep), "best", min(val), "at", ep[val.index(min(val))])


if __name__ == "__main__":
    main()
