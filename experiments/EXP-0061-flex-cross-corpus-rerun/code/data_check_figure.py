"""EXP-0061 phase 1 -- visual + numeric sanity check of the FleX loader.

Per dataset: a few (before + plate channels, after, change) triples straight
out of ``FlexPileData.__getitem__`` (the exact tensors the trainer sees), and
over a random sample of rows the fraction of REMOVED occupancy (occ0 & !occ1)
and of ADDED occupancy that lies inside ``swept_region_mask`` built from the
action with the configured plate width (the region eval_report scores) --
if the plate/axis convention were wrong, removal would not concentrate there.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/data_check_figure.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from FlexData.dataset import FlexPileData  # noqa: E402
from fit_linear_foresight import actions_to_pixels, plate_width_px, swept_region_mask  # noqa: E402

OUT = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/data_check"
CFGS = {"DS-0020": "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/config.yaml",
        "DS-0019": "datasets/DS-0019-slates-flex-pile-varN/config.yaml"}


def region_for(ds, idxs):
    acts = torch.stack([ds.get_raw_action(i) for i in idxs])
    s, e = actions_to_pixels(acts, *ds.workspace_bounds, (ds.H, ds.W))
    p = plate_width_px(ds, ds.W)
    return swept_region_mask(s, e, (ds.H, ds.W), 0.5 * p + 2.0, 0.5 * p), s, e


def main():
    rng = np.random.default_rng(0)
    stats = {}
    for name, cfg in CFGS.items():
        ds = FlexPileData(cfg, "all")
        idx = rng.choice(len(ds), size=min(1000, len(ds)), replace=False)
        reg, _, _ = region_for(ds, idx)
        rem_in = rem = add_in = add = 0.0
        for k, i in enumerate(idx):
            (x, _), y = ds[int(i)]
            removed = ((x[0] > 0.5) & (y < 0.5)).float()
            added = ((x[0] < 0.5) & (y > 0.5)).float()
            rem += float(removed.sum()); rem_in += float((removed * reg[k]).sum())
            add += float(added.sum()); add_in += float((added * reg[k]).sum())
        stats[name] = dict(n_rows=int(len(idx)), removed_px_inside_swept_region=rem_in / max(rem, 1),
                           added_px_inside_swept_region=add_in / max(add, 1),
                           swept_region_frac_of_grid=float(reg.mean()),
                           to_pxl=ds.to_pxl, plate_width_px=ds.plate_width_px)
        print(name, stats[name], flush=True)

        show = rng.choice(len(ds), size=4, replace=False)
        fig, ax = plt.subplots(4, 4, figsize=(14, 14))
        regs, s_px, e_px = region_for(ds, show)
        for r, i in enumerate(show):
            (x, _), y = ds[int(i)]
            a = ds.get_raw_action(int(i)).numpy()
            rgb = np.stack([x[0].numpy() * 0.6, x[0].numpy() * 0.6, x[0].numpy() * 0.6], -1)
            rgb[..., 0] = np.maximum(rgb[..., 0], x[1].numpy() / float(x[1].max()))   # start plate red (display-normalised)
            rgb[..., 2] = np.maximum(rgb[..., 2], x[2].numpy() / float(x[2].max()))   # stop plate blue
            panels = [(x[0], "before occ0"), (rgb, "occ0 + plate start(red)/stop(blue)"),
                      (y, "after occ1"), ((y - x[0]), "occ1 - occ0 (swept region outlined)")]
            for c, (im, title) in enumerate(panels):
                A = ax[r, c]
                if c == 3:
                    A.imshow(im, cmap="bwr", vmin=-1, vmax=1)
                    A.contour(regs[r].numpy(), levels=[0.5], colors="k", linewidths=0.8)
                else:
                    A.imshow(im if c == 1 else im.numpy(), cmap=None if c == 1 else "gray", vmin=0, vmax=1)
                # actions_to_pixels gives (col, row) with a -0.5 px pixel-centre shift
                A.annotate("", xy=(e_px[r, 0], e_px[r, 1]), xytext=(s_px[r, 0], s_px[r, 1]),
                           arrowprops=dict(color="lime", width=1.2, headwidth=6))
                A.set_title(f"{title}\nrow {int(i)}: grp {ds.get_run_index(int(i))} "
                            f"a=[{a[0]:.1f},{a[1]:.1f}]->[{a[2]:.1f},{a[3]:.1f}]", fontsize=7)
                A.set_xlabel("col = Y = -z_flex (action 2nd comp)", fontsize=7)
                A.set_ylabel("row = X = x_flex", fontsize=7)
        fig.suptitle(f"{name}: FlexPileData.__getitem__ tensors, grid +-{ds.half_extent} "
                     f"({ds.to_pxl:.3f} px/unit), plate {ds.plate_size[0]} wide", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{name}_triples.png", dpi=90)
        plt.close(fig)
    tmp = OUT / "data_check.json.tmp"
    tmp.write_text(json.dumps(stats, indent=1)); os.replace(tmp, OUT / "data_check.json")


if __name__ == "__main__":
    main()
