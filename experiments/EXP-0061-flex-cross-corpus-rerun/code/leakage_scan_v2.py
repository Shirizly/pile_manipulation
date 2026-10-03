"""DS-0020 v2 vs DS-0019 near-duplicate scan, particle level (complements the
mask-IoU scan in data_check_v2.py, which is NOT discriminative on its own: two
unrelated rand_spread piles both cover most of the workspace, so their masks
overlap at IoU ~0.8).

For every v2 trajectory: symmetric Chamfer distance (mean nearest-neighbour
distance, FleX units, XZ) between its INITIAL particle cloud and each of its
top-5 DS-0019 initial states by mask IoU plus any DS-0019 state with the same
piece count. Calibrated on traj 0-99 vs the same-index DS-0019 state (the known
leak) and on traj 0-99 vs a different DS-0019 state of the same init_pos.
Particles are used here for this audit only (never as model input).

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/leakage_scan_v2.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from FlexData.dataset import FlexPileData  # noqa: E402

OUT = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/data_check_v2"
D20 = REPO / "datasets/DS-0020-training-data-flex-N864"
D19 = REPO / "datasets/DS-0019-slates-flex-pile-varN"


def chamfer(a, b, ta=None, tb=None):
    ta = ta or cKDTree(a); tb = tb or cKDTree(b)
    return 0.5 * (ta.query(b)[0].mean() + tb.query(a)[0].mean())


def main():
    src = FlexPileData(str(D20 / "config.yaml"), "all", exclude_flagged=False, verbose=False).src
    t20 = src.traj_ids
    man19 = json.loads((D19 / "cache/manifest.json").read_text())["states"]
    s19 = list(range(100))
    c19 = {s: np.load(D19 / f"cache/state_{s:03d}.npz")["init_xz"].astype(np.float32) for s in s19}
    k19 = {s: cKDTree(c19[s]) for s in s19}
    rig19 = np.array([man19[f"state_{s:03d}.npz"]["n_rigids"] for s in s19])
    ip19 = np.array([man19[f"state_{s:03d}.npz"]["init_pos"] for s in s19])
    with np.load(D20 / "cache/image_masks.npz") as z:
        m20 = z["masks"][:, 0].reshape(len(z["traj_ids"]), -1).astype(np.float32)
        assert np.array_equal(z["traj_ids"], t20)
    with np.load(D19 / "cache/image_masks.npz") as z:
        m19 = z["init"].reshape(100, -1).astype(np.float32)
    inter = m20 @ m19.T
    iou = inter / (m20.sum(1)[:, None] + m19.sum(1)[None] - inter)
    rows = []
    for i, t in enumerate(t20):
        a = src.state(i, 0).astype(np.float32)
        ta = cKDTree(a)
        cand = set(np.argsort(-iou[i])[:5].tolist()) | set(np.nonzero(rig19 == src.n_rigids[i])[0].tolist())
        if t < 100:
            cand.add(int(t))
        best = min((chamfer(a, c19[s], ta, k19[s]), s) for s in cand)
        rec = dict(traj=int(t), pieces=int(src.n_rigids[i]), init_pos=int(src.init_pos[i]),
                   best_state=int(best[1]), best_chamfer=float(best[0]), iou_best_state=float(iou[i, best[1]]),
                   max_iou=float(iou[i].max()))
        if t < 100:
            rec["same_index_chamfer"] = float(chamfer(a, c19[int(t)], ta, k19[int(t)]))
            other = [s for s in s19 if s != t and ip19[s] == ip19[int(t)] and rig19[s] == rig19[int(t)]]
            if other:
                rec["other_same_type_chamfer"] = float(min(chamfer(a, c19[s], ta, k19[s]) for s in other))
        rows.append(rec)
        if i % 200 == 0:
            print(i, rec, flush=True)
    leak = [r for r in rows if r["traj"] < 100]
    rest = [r for r in rows if r["traj"] >= 100]
    same = np.array([r["same_index_chamfer"] for r in leak])
    oth = np.array([r["other_same_type_chamfer"] for r in leak if "other_same_type_chamfer" in r])
    restc = np.array([r["best_chamfer"] for r in rest])
    thr = float(max(same.max(), 0.0))
    q = lambda x: {f"p{p}": round(float(np.percentile(x, p)), 4) for p in (0, 1, 5, 50, 95, 100)}
    flagged = sorted([r for r in rest if r["best_chamfer"] <= thr], key=lambda r: r["best_chamfer"])
    res = dict(
        method=__doc__.strip().splitlines()[0],
        leak_same_index_chamfer=q(same), leak_vs_other_same_type_chamfer=q(oth),
        traj_ge_100_best_chamfer=q(restc),
        threshold_used="max chamfer of the known leaked pairs (traj 0-99 vs same index)", threshold=thr,
        n_traj_ge_100_within_threshold=len(flagged), flagged=flagged[:50],
        closest_20_traj_ge_100=sorted(rest, key=lambda r: r["best_chamfer"])[:20],
    )
    tmp = OUT / "leakage_chamfer.json.tmp"
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, OUT / "leakage_chamfer.json")
    print(json.dumps({k: v for k, v in res.items() if k not in ("flagged", "closest_20_traj_ge_100")}, indent=1))
    print("closest:", [(r["traj"], r["best_state"], round(r["best_chamfer"], 3), round(r["iou_best_state"], 3))
                       for r in res["closest_20_traj_ge_100"][:10]])


def piece_scan():
    """Piece-level scan (the DISCRIMINATIVE one): ``rigid_translations`` (x, z) of each
    initial_state.npz (raw files, read-only). ordered = mean |c_i - c'_i| over piece
    index (same piece count only -- the same collector seed generates pieces in the
    same order); centroid_chamfer = symmetric mean NN distance between centroid sets.
    -> figures/data_check_v2/leakage_pieces.json"""
    import glob
    R20 = D20 / "data/true_action_transitions_carrots"
    R19 = D19 / "data"
    t19 = {s: np.load(R19 / f"{s}/initial_state.npz")["rigid_translations"][:, [0, 2]] for s in range(100)}
    ids = sorted(int(Path(p).name) for p in glob.glob(str(R20 / "*"))
                 if Path(p).name.isdigit() and (Path(p) / "initial_state.npz").exists())
    t20 = {t: np.load(R20 / f"{t}/initial_state.npz")["rigid_translations"][:, [0, 2]] for t in ids}
    bad = [t for t in ids if not np.isfinite(t20[t]).all()]
    fin = lambda a: a[np.isfinite(a).all(1)]

    def sym(a, b):
        a, b = fin(a), fin(b)
        return 0.5 * (cKDTree(a).query(b)[0].mean() + cKDTree(b).query(a)[0].mean())

    def ordd(a, b):
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        return np.linalg.norm(a[ok] - b[ok], axis=1).mean()
    leak_o = [ordd(t20[t], t19[t]) for t in range(100)]
    leak_c = [sym(t20[t], t19[t]) for t in range(100)]
    rest = []
    for t in ids:
        if t < 100:
            continue
        cc = min((sym(t20[t], t19[s]), s) for s in range(100))
        oo = min([(ordd(t20[t], t19[s]), s) for s in range(100) if len(t20[t]) == len(t19[s])] or [(np.inf, None)])
        rest.append(dict(traj=t, best_cc=float(cc[0]), cc_state=cc[1], best_ordered=float(oo[0]),
                         ordered_state=oo[1], pieces=len(t20[t])))
    out = dict(note=piece_scan.__doc__, leak_ordered=list(map(float, leak_o)),
               leak_centroid_chamfer=list(map(float, leak_c)), nonfinite_traj=bad, rest=rest)
    (OUT / "leakage_pieces.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
    piece_scan()
