"""DS-0020 v2 data check (2026-10-02; payload replaced by the user, see
datasets/DS-0020-*/DATASET.md). Reads the v2 caches (FlexData/build_cache.py ds0020,
FlexData/image_mask ds0020) and DS-0019's caches READ-ONLY. Writes
figures/data_check_v2/{data_check_v2.json, triples.png, distributions.png, leakage.png}.

1. Loader flags (nan / escaped / out_of_grid / null) on all / train / val.
2. Chaining: step k's after-state is step k+1's before-state. No before-state file
   is stored, so test it by locality: with before = after_{k-1} (hypothesis H_chain)
   the particles moved by push k must lie in push k's swept region; with
   before = initial (H_reset) particles moved by pushes 0..k-1 show up outside it.
   Also image level: (after_{k-1} mask XOR after_k mask) inside the swept region.
3. Action -z convention on v2: removed mask pixels (occ0 & !occ1) inside the swept
   region with the actions AS STORED vs 2nd/4th negated (image_mask_cache_check.py's
   test), plus the particle version (fraction of in-path particles that move).
4. Leakage scan: every v2 initial mask vs all 100 DS-0019 initial masks (IoU) +
   piece / particle count; list IoU > 0.7 for trajectories >= 100.
5. Distributions vs DS-0019: push length, piece count, particle count, blob/spread,
   occupied fraction of the initial mask.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/data_check_v2.py
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

E = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun"
OUT = E / "figures/data_check_v2"
D20 = REPO / "datasets/DS-0020-training-data-flex-N864"
D19 = REPO / "datasets/DS-0019-slates-flex-pile-varN"
CFG = str(D20 / "config.yaml")
RNG = np.random.default_rng(0)


def _dump(obj):
    tmp = OUT / "data_check_v2.json.tmp"
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, OUT / "data_check_v2.json")


def push_frame(xy, a):
    """(N,2) table-frame points -> (u along push from start, v lateral), push length L."""
    s, e = np.asarray(a[:2], np.float64), np.asarray(a[2:], np.float64)
    d = e - s
    L = float(np.hypot(*d))
    ux = d / max(L, 1e-9)
    rel = xy.astype(np.float64) - s
    return rel @ ux, rel @ np.array([-ux[1], ux[0]]), L


def in_swept(xy, a, pad_u=(1.0, 2.5), half_w=2.5):
    u, v, L = push_frame(xy, a)
    return (u > -pad_u[0]) & (u < L + pad_u[1]) & (np.abs(v) < half_w)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}

    # ---------------------------------------------------------------- 1. flags
    for split in ("all", "train", "val"):
        ds = FlexPileData(CFG, split, exclude_flagged=True, verbose=True)
        res[f"flags_{split}"] = dict(rows=int(len(ds.flags["any"])), kept=int(len(ds)),
                                     **{k: int(v.sum()) for k, v in ds.flags.items()},
                                     n_traj=int(len(ds.src.traj_ids)))
    _dump(res)
    dall = FlexPileData(CFG, "all", exclude_flagged=False, verbose=False, occ_source="particles")
    src = dall.src
    T = len(src.traj_ids)
    pos = {int(t): i for i, t in enumerate(src.traj_ids)}

    # ---------------------------------------------------------------- 2. chaining
    MOVE = 0.25
    chain = dict(H_chain=[], H_reset=[], step0_given=[], retry_steps_H_chain=[], retry_steps_H_reset=[])
    retries = np.concatenate([np.load(f)["retries"] for f in sorted((D20 / "cache").glob("v2_chunk_*.npz"))])
    for t in range(T):
        init = src.state(t, 0).astype(np.float32) * [1, -1]
        for k in range(10):
            a = src.actions[t, k]
            aft = src.state(t, k + 1).astype(np.float32) * [1, -1]
            befs = {"step0_given": init} if k == 0 else {
                "H_chain": src.state(t, k).astype(np.float32) * [1, -1], "H_reset": init}
            for h, b in befs.items():
                mv = np.hypot(*(aft - b).T) > MOVE
                mv &= np.isfinite(aft).all(1) & np.isfinite(b).all(1)
                if mv.sum() == 0:
                    continue
                out = float((mv & ~in_swept(b, a)).sum() / mv.sum())
                chain[h].append(out)
                if k > 0 and retries[t, k] > 0:
                    chain[f"retry_steps_{h}"].append(out)
    res["chaining"] = {h: dict(n=len(v), mean_frac_moved_outside_swept=float(np.mean(v)) if v else None,
                               median=float(np.median(v)) if v else None,
                               p99=float(np.percentile(v, 99)) if v else None,
                               frac_steps_over_0p05=float(np.mean(np.asarray(v) > 0.05)) if v else None)
                       for h, v in chain.items()}
    res["chaining"]["note"] = ("fraction of particles with |disp| > 0.25 that lie OUTSIDE push k's padded swept "
                               "box (u in [-1, L+2.5], |v| < 2.5, table frame) of the BEFORE state; H_chain: "
                               "before = after-state of push k-1 (k>=1), H_reset: before = initial state")
    _dump(res)
    print("chaining", json.dumps(res["chaining"], indent=0), flush=True)

    # image-level chaining + -z check on masks
    dm = FlexPileData(CFG, "all", exclude_flagged=True, verbose=False, occ_source="image_mask")
    idx = RNG.choice(len(dm), 3000, replace=False)
    acts = torch.stack([dm.get_raw_action(int(i)) for i in idx])
    p = plate_width_px(dm, dm.W)
    sweep = {}
    for conv, A in (("as_stored(Y=-z)", acts), ("negated_2nd_4th(Y=+z)", acts * torch.tensor([1., -1., 1., -1.]))):
        s, e = actions_to_pixels(A, *dm.workspace_bounds, (dm.H, dm.W))
        reg = swept_region_mask(s, e, (dm.H, dm.W), 0.5 * p + 2.0, 0.5 * p)
        rem_in = rem = ch_in = ch = 0.0
        for k, i in enumerate(idx):
            (x, _), y = dm[int(i)]
            removed = ((x[0] > 0.5) & (y < 0.5)).float()
            changed = ((x[0] > 0.5) ^ (y > 0.5)).float()
            rem += float(removed.sum()); rem_in += float((removed * reg[k]).sum())
            ch += float(changed.sum()); ch_in += float((changed * reg[k]).sum())
        sweep[conv] = dict(removed_px_inside_swept_region=rem_in / max(rem, 1),
                           changed_px_inside_swept_region=ch_in / max(ch, 1),
                           swept_region_frac_of_grid=float(reg.mean()))
    # particle version: fraction of in-path particles that move
    pm = {}
    sub = RNG.choice(len(dall), 1500, replace=False)
    for conv, sg in (("as_stored(Y=-z)", 1.0), ("negated_2nd_4th(Y=+z)", -1.0)):
        fr = []
        for i in sub:
            a = dall.get_raw_action(int(i)).numpy() * np.array([1, sg, 1, sg])
            b, af = dall.particles_before(int(i)).numpy(), dall.particles_after(int(i)).numpy()
            u, v, L = push_frame(b, a)
            path = (u > 0.15 * L) & (u < 0.85 * L) & (np.abs(v) < 1.0)
            if L > 1.5 and path.sum() > 20:
                fr.append(float((np.hypot(*(af - b)[path].T) > 0.25).mean()))
        pm[conv] = dict(n_rows=len(fr), mean_frac_in_path_moved=float(np.mean(fr)))
    res["neg_z_check"] = dict(n_rows_mask=len(idx), mask_sweep=sweep, particle_in_path=pm)
    _dump(res)
    print("neg_z", json.dumps(res["neg_z_check"], indent=0), flush=True)

    # ---------------------------------------------------------------- 4. leakage
    z20 = np.load(D20 / "cache/image_masks.npz")
    with np.load(D19 / "cache/image_masks.npz") as z:
        m19, s19 = z["init"].astype(bool), z["state_ids"]
    m20, t20 = z20["masks"][:, 0].astype(bool), z20["traj_ids"]
    man19 = json.loads((D19 / "cache/manifest.json").read_text())["states"]
    rig19 = np.array([man19[f"state_{s:03d}.npz"]["n_rigids"] for s in s19])
    par19 = np.array([man19[f"state_{s:03d}.npz"]["n_particles"] for s in s19])
    ip19 = [man19[f"state_{s:03d}.npz"]["init_pos"] for s in s19]
    a = m20.reshape(len(t20), -1).astype(np.float32); b = m19.reshape(len(s19), -1).astype(np.float32)
    inter = a @ b.T
    iou = inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter)
    rig20 = np.array([src.n_rigids[pos[int(t)]] for t in t20])
    par20 = np.array([src.n_particles[pos[int(t)]] for t in t20])
    diag = np.array([iou[i, list(s19).index(int(t))] for i, t in enumerate(t20) if t < 100])
    low = t20 < 100
    off = iou[~low]
    best = off.max(1); arg = s19[off.argmax(1)]
    hi = np.nonzero(best > 0.7)[0]
    near = [dict(traj=int(t20[~low][j]), ds0019_state=int(arg[j]), iou=round(float(best[j]), 4),
                 pieces=int(rig20[~low][j]), ds0019_pieces=int(rig19[list(s19).index(int(arg[j]))]),
                 particles=int(par20[~low][j]), ds0019_particles=int(par19[list(s19).index(int(arg[j]))]))
            for j in hi[np.argsort(-best[hi])]]
    same_rig = (rig20[~low][:, None] == rig19[None]) & (par20[~low][:, None] == par19[None])
    res["leakage"] = dict(
        traj_0_99_vs_same_index=dict(iou_median=float(np.median(diag)), iou_min=float(diag.min()),
                                     same_piece_count=int(sum(rig20[i] == rig19[list(s19).index(int(t))]
                                                              for i, t in enumerate(t20) if t < 100))),
        traj_ge_100=dict(n=int((~low).sum()), max_iou_over_ds0019_median=float(np.median(best)),
                         max_iou_p99=float(np.percentile(best, 99)), max_iou_max=float(best.max()),
                         n_over_0p7=int((best > 0.7).sum()), n_over_0p6=int((best > 0.6).sum()),
                         n_exact_piece_and_particle_count_match=int(same_rig.any(1).sum()),
                         near_duplicates_iou_gt_0p7=near),
        unrelated_pairs_iou_median=float(np.median(off)))
    _dump(res)
    print("leakage", json.dumps({k: v for k, v in res["leakage"].items()}, indent=0)[:2000], flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(diag, 40, alpha=.7, label="traj 0-99 vs DS-0019 same index")
    ax[0].hist(best, 40, alpha=.7, label="traj >=100: max over 100 DS-0019 states")
    ax[0].axvline(0.7, c="k", ls="--"); ax[0].set_xlabel("initial image-mask IoU"); ax[0].legend(fontsize=8)
    ax[1].scatter(rig20[~low], best, s=4); ax[1].set_xlabel("pieces (v2 traj >= 100)"); ax[1].set_ylabel("max IoU vs DS-0019")
    fig.tight_layout(); fig.savefig(OUT / "leakage.png", dpi=110); plt.close(fig)

    # ---------------------------------------------------------------- 5. distributions
    sp = json.loads((D20 / "splits.json").read_text())["splits"]
    keep_ids = set(sp["train"]) | set(sp["val"])
    kmask = np.array([int(t) in keep_ids for t in src.traj_ids])
    pl20 = src.push_length[kmask].ravel()
    recs19 = [json.loads(l) for l in open(D19 / "manifest.jsonl")]
    pl19 = np.array([r["push_length"] for r in recs19 if r["type"] == "action" and r["valid"]])
    occ20 = m20[np.isin(t20, list(keep_ids))].mean((1, 2)); occ19 = m19.mean((1, 2))
    q = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 3) for p in (0, 5, 25, 50, 75, 95, 100)}
    ip20 = src.init_pos[kmask]
    res["distributions"] = dict(
        kept_traj=int(kmask.sum()), push_length_v2=q(pl20), push_length_ds0019=q(pl19),
        pieces_v2=q(src.n_rigids[kmask]), pieces_ds0019=q(rig19),
        particles_v2=q(src.n_particles[kmask]), particles_ds0019=q(par19),
        init_pos_v2=dict(rand_blob=int((ip20 == 0).sum()), rand_spread=int((ip20 == 1).sum())),
        init_pos_ds0019=dict(rand_blob=ip19.count("rand_blob"), rand_spread=ip19.count("rand_spread")),
        init_mask_occupied_frac_v2=q(occ20), init_mask_occupied_frac_ds0019=q(occ19),
        bins_v2=np.bincount(np.concatenate([np.load(f)["bin"][np.isin(np.load(f)["traj_ids"], list(keep_ids))].ravel()
                                            for f in sorted((D20 / "cache").glob("v2_chunk_*.npz"))]), minlength=6).tolist())
    _dump(res)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
    for a_, x20, x19, lab in ((ax[0], pl20, pl19, "push length (FleX units)"),
                              (ax[1], src.n_rigids[kmask], rig19, "pieces (rigids)"),
                              (ax[2], src.n_particles[kmask], par19, "particles"),
                              (ax[3], occ20, occ19, "initial mask occupied fraction")):
        bins = np.histogram_bin_edges(np.concatenate([x20, x19]).astype(float), 40)
        a_.hist(x20, bins, density=True, alpha=.6, label=f"DS-0020 v2 kept (n={len(x20)})")
        a_.hist(x19, bins, density=True, alpha=.6, label=f"DS-0019 (n={len(x19)})")
        a_.set_xlabel(lab); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "distributions.png", dpi=110); plt.close(fig)

    # ---------------------------------------------------------------- 6. triples
    dv = FlexPileData(CFG, "val", exclude_flagged=True, verbose=False, occ_source="image_mask")
    rows = RNG.choice(len(dv), 8, replace=False)
    fig, ax = plt.subplots(len(rows), 3, figsize=(7.5, 2.5 * len(rows)))
    for r, i in enumerate(rows):
        (x, _), y = dv[int(i)]
        t, k = dv.get_run_index(int(i)), dv.get_step_index(int(i))
        ax[r, 0].imshow(x[0], cmap="gray_r", vmin=0, vmax=1)
        ax[r, 1].imshow(x[0] * 0.35 + 0 * x[1], cmap="gray_r", vmin=0, vmax=1)
        ax[r, 1].imshow(np.ma.masked_less(x[1].numpy(), 0.2), cmap="Blues", vmin=0, vmax=1, alpha=.9)
        ax[r, 1].imshow(np.ma.masked_less(x[2].numpy(), 0.2), cmap="Reds", vmin=0, vmax=1, alpha=.9)
        ax[r, 2].imshow(y, cmap="gray_r", vmin=0, vmax=1)
        ax[r, 0].set_ylabel(f"traj {t} push {k}", fontsize=8)
        for c, ttl in enumerate(("before mask (occ0)", "plate start (blue) / stop (red)", "after mask (target)")):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(ttl, fontsize=8)
    fig.suptitle("DS-0020 v2 val rows, image-mask input (row = X = x_flex, col = Y = -z_flex)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "triples.png", dpi=100); plt.close(fig)
    res["figures"] = ["triples.png", "distributions.png", "leakage.png"]
    _dump(res)
    print("done", flush=True)


if __name__ == "__main__":
    main()
