"""EXP-0061 -- per-seed NFD train/val loss curves from the training logs.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/nfd_val_curves.py

Parses `logs/nfd_flex_mask_seed{s}.log` "Epoch N: trn= val= ..." lines (a resumed
run appends to the same log; for a repeated epoch the LAST line wins) ->
figures/nfd/val_curves.png + results/nfd_val_curves.json.
"""
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path(__file__).resolve().parents[1]
PAT = re.compile(r"Epoch\s+(\d+): trn=([\d.]+)\s+val=([\d.]+)\s+iou=([\d.]+)\s+chg_mse=([\d.]+)")


def parse(path):
    rows = {}
    for line in path.read_text(errors="ignore").replace("\r", "\n").splitlines():
        m = PAT.search(line)
        if m:
            e = int(m.group(1))
            rows[e] = dict(epoch=e, train=float(m.group(2)), val=float(m.group(3)),
                           hard_iou=float(m.group(4)), changed_mse=float(m.group(5)))
    return [rows[k] for k in sorted(rows)]


def main():
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in (0, 1, 2):
        p = E / f"logs/nfd_flex_mask_seed{s}.log"
        if not p.exists():
            continue
        r = parse(p)
        out[f"seed{s}"] = r
        if not r:
            continue
        ep = [x["epoch"] for x in r]
        ax[0].plot(ep, [x["val"] for x in r], label=f"seed {s} val")
        ax[0].plot(ep, [x["train"] for x in r], "--", alpha=0.5, label=f"seed {s} train")
        ax[1].plot(ep, [x["changed_mse"] for x in r], label=f"seed {s}")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE loss (sigmoid vs binary mask)")
    ax[0].legend(fontsize=7); ax[0].set_title("NFD flex-mask: train / val loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val changed-pixel MSE"); ax[1].legend(fontsize=7)
    ax[1].set_title("val MSE on changed pixels")
    fig.tight_layout()
    (E / "figures/nfd").mkdir(parents=True, exist_ok=True)
    fig.savefig(E / "figures/nfd/val_curves.png", dpi=100)
    (E / "results").mkdir(exist_ok=True)
    tmp = E / "results/nfd_val_curves.json.tmp"
    tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, E / "results/nfd_val_curves.json")
    for k, r in out.items():
        if r:
            b = min(r, key=lambda x: x["val"])
            print(k, "epochs", len(r), "best", b["epoch"], b["val"], "last", r[-1]["val"])


if __name__ == "__main__":
    main()
