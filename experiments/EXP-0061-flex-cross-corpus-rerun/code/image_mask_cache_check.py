"""EXP-0061 -- validation of the image-mask cache (FlexData/image_mask.py output).

1. Reproduce the prototype's IoU (binary mask frac > 0 vs particle disk raster)
   on the SAME 400 states as figures/image_mask/agreement.jsonl, reading the
   masks from cache/image_masks.npz (checks the cache, not a recomputation).
2. Frame agreement with the loader's particle path: IoU(mask occ0, particle occ0)
   for 500 rows of each corpus through FlexPileData(occ_source="image_mask"/
   "particles"), over integer shifts -3..3 px in row and col (peak must be at 0,0).
3. Action convention in the image frame: fraction of REMOVED mask occupancy
   (occ0 & !occ1) inside swept_region_mask for actions AS STORED vs with the 2nd/4th
   components negated (the literal +z reading). The plate must sweep the mask.
4. Figure: input occ0 + plate start/stop + target for a few DS-0020 val rows and
   DS-0019 rows from occ_source="image_mask" -> figures/data_check/image_mask_triples_*.png

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/image_mask_cache_check.py
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
from FlexData.dataset import FlexPileData, rasterize_disk, flex_xz_to_table  # noqa: E402
from FlexData.image_mask import GridSpec  # noqa: E402
from fit_linear_foresight import actions_to_pixels, plate_width_px, swept_region_mask  # noqa: E402

E = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun"
OUT = E / "figures/data_check"
D20 = REPO / "datasets/DS-0020-training-data-flex-N864"
D19 = REPO / "datasets/DS-0019-slates-flex-pile-varN"
CFGS = {"DS-0020": str(D20 / "old_data/_ported_v1/config.yaml"), "DS-0019": str(D19 / "config.yaml")}


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def check_iou():
    G = GridSpec()
    recs = [json.loads(l) for l in open(E / "figures/image_mask/agreement.jsonl")]
    z20, z19 = np.load(D20 / "old_data/_ported_v1/cache/image_masks.npz"), np.load(D19 / "cache/image_masks.npz")
    m20, t20 = z20["masks"], {int(t): i for i, t in enumerate(z20["traj_ids"])}
    init19, after19 = z19["init"], z19["after"]
    s19 = {int(s): i for i, s in enumerate(z19["state_ids"])}
    a19 = {(int(a), int(b)): i for i, (a, b) in enumerate(z19["after_keys"])}
    out = {"DS-0020": [], "DS-0019": []}
    diffs = []
    for r in recs:
        a, b = r["id"].split("/")
        if r["ds"] == "DS-0020":
            m = m20[t20[int(a)], int(b)] > 0
            pp = D20 / f"old_data/{a}/{b}_particles.npy"
        elif b == "init":
            m = init19[s19[int(a)]] > 0
            pp = D19 / f"data/{a}/initial_particles.npy"
        else:
            m = after19[a19[(int(a), int(b))]] > 0
            pp = D19 / f"data/{a}/{b}_after_particles.npy"
        P = np.load(pp).reshape(-1, 4)[:, :3]
        P = P[np.isfinite(P).all(1) & (np.abs(P).max(1) < 10)]
        px = flex_xz_to_table(P[:, [0, 2]]) * G.to_pxl + G.ctr
        ras = rasterize_disk(px, 64, 64, 1.0) > 0
        v = iou(m, ras)
        out[r["ds"]].append(v)
        diffs.append(abs(v - r["iou_t0.0"]))
    res = {k: dict(n=len(v), mean=float(np.mean(v)), min=float(np.min(v)), p5=float(np.percentile(v, 5)))
           for k, v in out.items()}
    proto = {k: float(np.mean([r["iou_t0.0"] for r in recs if r["ds"] == k])) for k in out}
    res["prototype_mean_iou_t0.0"] = proto
    res["max_abs_diff_vs_prototype"] = float(max(diffs))
    return res


def shift_iou(a, b, dr, dc):
    b2 = torch.roll(b, shifts=(dr, dc), dims=(0, 1))
    return iou(a.bool().numpy(), b2.bool().numpy())


def check_frame_and_sweep(rng):
    res = {}
    for name, cfg in CFGS.items():
        split = "val" if name == "DS-0020" else "all"
        dm = FlexPileData(cfg, split, occ_source="image_mask", verbose=False)
        dp = FlexPileData(cfg, split, occ_source="particles", verbose=False)
        assert len(dm) == len(dp)
        idx = rng.choice(len(dm), size=min(500, len(dm)), replace=False)
        sh = np.zeros((7, 7))
        for i in idx[:200]:
            (xm, _), _ = dm[int(i)]
            (xp, _), _ = dp[int(i)]
            for a, dr in enumerate(range(-3, 4)):
                for b, dc in enumerate(range(-3, 4)):
                    sh[a, b] += shift_iou(xm[0], xp[0], dr, dc) / 200
        pk = np.unravel_index(sh.argmax(), sh.shape)
        acts = torch.stack([dm.get_raw_action(int(i)) for i in idx])
        p = plate_width_px(dm, dm.W)
        sweep = {}
        for conv, A in (("as_stored(Y=-z)", acts), ("negated_2nd_4th(Y=+z)", acts * torch.tensor([1., -1., 1., -1.]))):
            s, e = actions_to_pixels(A, *dm.workspace_bounds, (dm.H, dm.W))
            reg = swept_region_mask(s, e, (dm.H, dm.W), 0.5 * p + 2.0, 0.5 * p)
            rem_in = rem = 0.0
            for k, i in enumerate(idx):
                (x, _), y = dm[int(i)]
                removed = ((x[0] > 0.5) & (y < 0.5)).float()
                rem += float(removed.sum()); rem_in += float((removed * reg[k]).sum())
            sweep[conv] = dict(removed_px_inside_swept_region=rem_in / max(rem, 1),
                               swept_region_frac_of_grid=float(reg.mean()))
        res[name] = dict(n_rows_shift=200, iou_at_zero_shift=float(sh[3, 3]),
                         peak_shift_row_col=[int(pk[0]) - 3, int(pk[1]) - 3], peak_iou=float(sh.max()),
                         shift_iou_grid=sh.round(4).tolist(), n_rows_sweep=int(len(idx)), sweep=sweep,
                         mean_occ_mask=float(np.mean([float(dm[int(i)][0][0][0].mean()) for i in idx[:200]])),
                         mean_occ_particles=float(np.mean([float(dp[int(i)][0][0][0].mean()) for i in idx[:200]])))
        print(name, {k: v for k, v in res[name].items() if k != "shift_iou_grid"}, flush=True)
        figure(name, dm, dp, rng)
    return res


def figure(name, dm, dp, rng):
    show = rng.choice(len(dm), size=4, replace=False)
    acts = torch.stack([dm.get_raw_action(int(i)) for i in show])
    s_px, e_px = actions_to_pixels(acts, *dm.workspace_bounds, (dm.H, dm.W))
    fig, ax = plt.subplots(4, 4, figsize=(14, 14))
    for r, i in enumerate(show):
        (x, _), y = dm[int(i)]
        (xp, _), _ = dp[int(i)]
        g = x[0].numpy()
        rgb = np.stack([g * 0.6] * 3, -1)
        rgb[..., 0] = np.maximum(rgb[..., 0], x[1].numpy() / float(x[1].max()))
        rgb[..., 2] = np.maximum(rgb[..., 2], x[2].numpy() / float(x[2].max()))
        ov = np.stack([g, xp[0].numpy(), np.zeros_like(g)], -1)
        panels = [(rgb, "mask occ0 + plate start(red)/stop(blue)", None),
                  (y.numpy(), "target: mask occ1", "gray"),
                  ((y - x[0]).numpy(), "mask occ1 - occ0", "bwr"),
                  (ov, "mask occ0 (red) vs particle occ0 (green)", None)]
        for c, (im, title, cm) in enumerate(panels):
            A = ax[r, c]
            A.imshow(im, cmap=cm, vmin=-1 if cm == "bwr" else 0, vmax=1)
            A.annotate("", xy=(e_px[r, 0], e_px[r, 1]), xytext=(s_px[r, 0], s_px[r, 1]),
                       arrowprops=dict(color="lime", width=1.0, headwidth=6))
            A.set_title(f"{title}\nrow {int(i)} grp {dm.get_run_index(int(i))}", fontsize=7)
            A.set_xlabel("col = Y = -z_flex", fontsize=7); A.set_ylabel("row = X = x_flex", fontsize=7)
    fig.suptitle(f"{name}: FlexPileData(occ_source='image_mask').__getitem__ tensors (binary mask, frac > 0)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"image_mask_triples_{name}.png", dpi=90)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {"iou_vs_particle_raster": check_iou()}
    print(json.dumps(res, indent=1), flush=True)
    res["frame_and_sweep"] = check_frame_and_sweep(rng)
    tmp = OUT / "image_mask_check.json.tmp"
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, OUT / "image_mask_check.json")


if __name__ == "__main__":
    main()
