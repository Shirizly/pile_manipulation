"""EXP-0063 pilot: soft-occupancy and condensed-output NFD variants vs the
3-seed nfd_3ch_narrow_l20_v2 baseline, on EXP-0059's clean narrow harness.

Reuses EXP-0059 `eval_extended.eval_occ_model` (accuracy_1 / rollout / slateN
/ slateN_tough, unchanged) on the SAME corpora and the same per-call split as
EXP-0059 `test_v2.py`:
  * accuracy_1:   DS-0016 test_chains_v2 (--exclude-flagged)
  * rollout_1..4: DS-0016 test_chains_v2_clean
  * slateN:       DS-0016 test_pools_v2 (32 pools)  [primary]
                  DS-0017 val_pools_v2  (32 pools)  [second, independent pool set]
Nothing is selected on either pool set (checkpoints are each run's own
best-val-LOSS, as for the baseline), so both are held out.

Added here only:
  * output-blur control `nfd_v2_outblur{1,2}`: the BASELINE seed-0 model with
    its prediction blurred at eval time -- separates "trained on soft
    occupancy" from "ranks with a smoothed image";
  * per-model prediction binariness on the test pools (mean p(1-p), fraction
    of pixels in (0.05, 0.95), predicted/raster mass ratio);
  * paired pool-bootstrap deltas vs EACH baseline seed and vs the per-pool
    mean of the 3 seeds (resampling pools once per draw, shared across goals).

Usage:
    python -u eval_pilot.py [--models ...] [--ckpt NAME=PATH ...] [--out PATH]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
EXP59 = REPO / "experiments/EXP-0059-retrieval-transition-model/code"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(REPO / "experiments/EXP-0060-metric-correlation-study/code"))
sys.path.insert(0, str(EXP59))
import eval_extended as EE  # noqa: E402
import correlate as CO  # noqa: E402  (pool_bootstrap_ci)
from bootstrap_ci import _align_capture_matrix  # noqa: E402

import simple_mpc.adapters as A  # noqa: E402
from simple_mpc.adapters import occ_from_particles  # noqa: E402
from transforms.functional import gaussian_blur_occ  # noqa: E402

D = REPO / "Genesis/data/narrow_l20_n20"
RES = Path(__file__).resolve().parents[1] / "results" / "pilot_eval.json"
BASE = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2"]
ARMS = ["nfd_3ch_narrow_l20_v2_soft_s1", "nfd_3ch_narrow_l20_v2_soft_s2",
        "nfd_3ch_narrow_l20_v2_sharp_w03", "nfd_3ch_narrow_l20_v2_sharp_w07"]
CONTROLS = ["nfd_v2_outblur1", "nfd_v2_outblur2"]
DEFAULT_MODELS = ["persistence"] + BASE + CONTROLS + ARMS


def _register_outblur(sigma):
    base = A.OCC_ADAPTERS["nfd_3ch_narrow_l20_v2"]

    def f(device, goal_shape):
        ad = base(device, goal_shape)
        inner = ad.predict_step
        ad.predict_step = lambda occ, act: gaussian_blur_occ(inner(occ, act), sigma)
        return ad
    A.OCC_ADAPTERS[f"nfd_v2_outblur{sigma}"] = f


def _atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj)); os.replace(tmp, path)


def binariness(model_id, dev, pools, max_pools=32):
    """Prediction statistics on the test pools' candidate pushes."""
    if model_id == "persistence":
        return None
    ad = A.make_occ_adapter(model_id, dev, "corner")
    gini, mid, mass = [], [], []
    seen = 0
    for d in pools:
        act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
        for pi in torch.unique(d["pool_idx"]):
            if seen >= max_pools:
                break
            seen += 1
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            o0 = occ_from_particles(d["states"][ix[0]][None].float(), dev)
            with torch.no_grad():
                p = ad.predict_step(EE._encode(ad, o0).expand(len(ix), -1, -1).contiguous(),
                                    act_all[ix].to(dev)).float()
            gini.append(float((p * (1 - p)).mean()))
            mid.append(float(((p > 0.05) & (p < 0.95)).float().mean()))
            mass.append(float(p.sum(dim=(1, 2)).mean() / o0.sum()))
    return dict(mean_p1mp=float(np.mean(gini)), frac_mid=float(np.mean(mid)),
                mass_ratio=float(np.mean(mass)))


def _cap_matrix(raw):
    CAP, pids = _align_capture_matrix(raw)
    return CAP, pids


def paired(cap_a, cap_b, n_boot=2000, seed=0):
    """b - a on goal x pool capture matrices of the SAME pools; pools resampled once per draw."""
    n = cap_a.shape[1]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    ba = np.nanmean(np.nanmean(cap_a[:, idx], axis=2), axis=0)
    bb = np.nanmean(np.nanmean(cap_b[:, idx], axis=2), axis=0)
    dlt = bb - ba
    point = float(np.nanmean(np.nanmean(cap_b, 1)) - np.nanmean(np.nanmean(cap_a, 1)))
    lo, hi = np.percentile(dlt, [2.5, 97.5])
    return dict(point=point, ci95=[float(lo), float(hi)], frac_gt0=float((dlt > 0).mean()))


def comparisons(out, split, key="slateN_tough_raw"):
    """Every arm/control vs each baseline seed and vs the per-pool 3-seed mean."""
    res = {}
    caps = {m: _cap_matrix(out[m][split]["raw"][key]) for m in out
            if not m.startswith("_") and split in out[m]}
    if not all(b in caps for b in BASE):
        return res
    pids = caps[BASE[0]][1]
    base_mean = np.nanmean(np.stack([caps[b][0] for b in BASE]), axis=0)
    for m, (cap, p) in caps.items():
        if m in BASE or m == "persistence":
            continue
        assert p == pids, f"{m}: pool ids differ from the baseline's"
        res[m] = {"vs_seed_mean": paired(base_mean, cap)}
        for b in BASE:
            res[m][f"vs_{b}"] = paired(caps[b][0], cap)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--ckpt", nargs="*", default=[], metavar="NAME=PATH",
                    help="register an extra NFD checkpoint under NAME (e.g. a smoke run)")
    ap.add_argument("--splits", nargs="*", default=["test", "val"])
    ap.add_argument("--out", default=str(RES))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res_path = Path(a.out)
    out = json.loads(res_path.read_text()) if res_path.exists() else {}
    for s in (1, 2):
        _register_outblur(s)
    for kv in a.ckpt:
        name, path = kv.split("=", 1)
        A.OCC_ADAPTERS[name] = A._nfd(path)

    corp = {}
    ch_acc, pools_test = EE._load_corpus(chains_dir=str(D / "test_chains_v2"),
                                         pools_dir=str(D / "test_pools_v2"), exclude_flagged=True)
    ch_roll, _ = EE._load_corpus(chains_dir=str(D / "test_chains_v2_clean"),
                                 pools_dir=str(D / "__no_pools__"), exclude_flagged=False)
    corp["test"] = (ch_acc, ch_roll, pools_test)
    if "val" in a.splits:
        _, pools_val = EE._load_corpus(pools_dir=str(D / "val_pools_v2"),
                                       chains_dir=str(D / "__no_chains__"), exclude_flagged=True)
        corp["val"] = ([], [], pools_val)
    Dist13 = EE._dist_fields(EE.GOALS_13); DistTough = EE._dist_fields(EE.GOALS_TOUGH)
    Masks13 = EE._masks(EE.GOALS_13); MasksTough = EE._masks(EE.GOALS_TOUGH)

    for m in a.models:
        if m not in A.OCC_ADAPTERS and m != "persistence":
            print("skip (not registered -- not trained yet?):", m, flush=True); continue
        entry = out.setdefault(m, {})
        for split in a.splits:
            if split in entry:
                continue
            ch, ch_r, pools = corp[split]
            t0 = time.time()
            r, raw = EE.eval_occ_model(m, dev, ch, pools, Dist13, DistTough, Masks13, MasksTough)
            if ch_r:
                r_roll, _ = EE.eval_occ_model(m, dev, ch_r, [], Dist13, DistTough, Masks13, MasksTough)
                for k in range(1, 5):
                    if f"rollout_accuracy_{k}" in r_roll:
                        r[f"rollout_accuracy_{k}"] = r_roll[f"rollout_accuracy_{k}"]
            mean, lo, hi = CO.pool_bootstrap_ci(raw["slateN_tough_raw"])
            r["slateN_tough_ci95"] = [lo, hi]
            r["wall_s"] = time.time() - t0
            entry[split] = {"metrics": r, "raw": raw}
            _atomic(out, res_path)
            print(f"{m:34s} [{split}] slateN_tough {r['slateN_tough']:.3f} [{lo:.3f},{hi:.3f}] "
                  f"slateN {r['slateN']:.3f} acc1 {r.get('accuracy_1')} "
                  f"rollout4 {r.get('rollout_accuracy_4')} ({r['wall_s']:.0f}s)", flush=True)
        if "binariness" not in entry:
            entry["binariness"] = binariness(m, dev, pools_test)
            _atomic(out, res_path)

    for split in a.splits:
        out[f"_paired_{split}"] = comparisons(out, split)
    # pooled: 64 pools (test + val are disjoint pool sets) -- pool ids offset to keep them apart
    pooled = {}
    for m, e in out.items():
        if m.startswith("_") or not all(s in e for s in ("test", "val")):
            continue
        raw = {}
        for s, off in (("test", 0), ("val", 10_000)):
            for g, rows in e[s]["raw"]["slateN_tough_raw"].items():
                raw.setdefault(g, []).extend({**r, "pool": r["pool"] + off} for r in rows)
        pooled[m] = {"pooled": {"raw": {"slateN_tough_raw": raw}}}
    if pooled:
        out["_paired_pooled"] = comparisons(pooled, "pooled")
    _atomic(out, res_path)
    print("wrote", res_path)


if __name__ == "__main__":
    main()
