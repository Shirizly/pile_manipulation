"""Visual check of what the transition-gate encoder actually sees.

Loads transitions through the EXACT path the gate uses -- `data.load_transitions`
on `randlen_files("train")`, then `raster.rasterise` -- and draws, per sampled
transition and per resolution: occ0, occ1, their difference, with the push drawn
on top. Point of the plot is to confirm by eye that

  * the pile is where the particles say it is (no axis transposition),
  * the push actually moves material, and moves it in the direction drawn,
  * a grain is still resolvable at 32x32 (pitch 4 mm vs a 5 mm cube).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0019-lejepa-encoder-pushlen-switched/code"))
from raster import rasterise, BOUNDS, CUBE_SIZE            # noqa: E402
from data import randlen_files, load_transitions           # noqa: E402

OUT = REPO / "experiments/EXP-0021-five-way-ranking-comparison/results"
OUT.mkdir(parents=True, exist_ok=True)
SIDE = BOUNDS["x_max"] - BOUNDS["x_min"]


def rad(grid):
    return 0.5 * CUBE_SIZE / (SIDE / grid)


def to_px(xy, grid):
    """World metres -> voxel coords, the SAME mapping rasterise uses:
    p = (xy - lo) / (hi - lo) * (grid - 1), with dim0 = world x (row)."""
    lo = np.array([BOUNDS["x_min"], BOUNDS["y_min"]])
    hi = np.array([BOUNDS["x_max"], BOUNDS["y_max"]])
    return (np.asarray(xy) - lo) / (hi - lo) * (grid - 1)


def main(n_show=3, grids=(64, 32), seed=0):
    tr = load_transitions(randlen_files("train")[:4])
    n = len(tr["pts0"])
    length = (tr["p_stop"] - tr["p_start"]).norm(dim=1)
    # pick transitions that actually move something: longest pushes
    idx = torch.argsort(length, descending=True)[:200]
    idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))[:n_show]]
    print(f"{n} transitions loaded; showing rows {idx.tolist()} "
          f"(push length {[round(float(length[i])*1000,1) for i in idx]} mm)")

    fig, axes = plt.subplots(len(idx) * len(grids), 3,
                             figsize=(9.5, 3.2 * len(idx) * len(grids)))
    axes = np.atleast_2d(axes)
    r = 0
    for i in idx:
        i = int(i)
        for g in grids:
            p0 = tr["pts0"][i:i + 1]
            p1 = tr["pts1"][i:i + 1]
            nr = tr["n_real"][i:i + 1]
            valid = torch.arange(p0.shape[1])[None, :] < nr[:, None]
            o0 = rasterise(p0, rad(g), g, BOUNDS, valid=valid)[0].numpy()
            o1 = rasterise(p1, rad(g), g, BOUNDS, valid=valid)[0].numpy()
            s_px = to_px(tr["p_start"][i].numpy(), g)
            e_px = to_px(tr["p_stop"][i].numpy(), g)

            for c, (img, ttl) in enumerate((
                    (o0, f"occ0  {g}x{g}"), (o1, f"occ1  {g}x{g}"),
                    (o1 - o0, "occ1 - occ0"))):
                ax = axes[r, c]
                if c == 2:
                    m = max(abs(img.min()), abs(img.max()), 1e-6)
                    ax.imshow(img.T, origin="lower", cmap="bwr", vmin=-m, vmax=m)
                else:
                    ax.imshow(img.T, origin="lower", cmap="gray_r", vmin=0, vmax=1)
                # imshow(.T) so the horizontal axis is world x (dim 0) and the
                # vertical is world y -- matching the arrow drawn below.
                ax.annotate("", xy=(e_px[0], e_px[1]), xytext=(s_px[0], s_px[1]),
                            arrowprops=dict(arrowstyle="->", color="tab:green", lw=2))
                ax.set_title(f"row {i}  {ttl}", fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[r, 0].set_ylabel(f"{float(length[i])*1000:.0f} mm", fontsize=8)
            r += 1
    fig.suptitle("transition-gate input: world-frame raster, x=horizontal, y=vertical, "
                 "green = push start->stop", fontsize=9)
    fig.tight_layout()
    p = OUT / "transition_raster_check.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    print("wrote", p)

    # numeric companions -- what the eye cannot check
    for g in grids:
        valid = torch.arange(tr["pts0"].shape[1])[None, :] < tr["n_real"][:256, None]
        a = rasterise(tr["pts0"][:256], rad(g), g, BOUNDS, valid=valid)
        b = rasterise(tr["pts1"][:256], rad(g), g, BOUNDS, valid=valid)
        print(f"grid {g:>3}: mass/frame occ0 {a.sum(dim=(1,2)).mean():7.1f} "
              f"occ1 {b.sum(dim=(1,2)).mean():7.1f} | "
              f"mean |occ1-occ0| {float((b-a).abs().mean()):.5f} | "
              f"frac frames identical {(a==b).all(dim=(1,2)).float().mean():.3f} | "
              f"px per cube {2*rad(g):.2f}")


if __name__ == "__main__":
    main()
