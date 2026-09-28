"""EXP-0029: does a model's per-state advantage survive a fresh, independent
action pool from the same state? (DESIGN.md has the pre-registered plan.)

Stages (each checkpointed to artifacts/RUN-0001-split/, rerunnable):
  A  predicted dv per (action, goal, value fn) for every model  -> pred_<model>.pt
  B  true dv per (action, goal, value fn), soft ground-truth scoring -> truth.pt
  C  split analysis -> results/split_test.json (rewritten when done)

dv = value(after) - value(before). lyapunov is a COST, mass_in_region a VALUE;
senses come from goals.higher_is_better_for and captures read higher = better.
Predicted dv is computed on the model's own images (input occ0 = the hard
`occ_from_particles` image the models expect); true dv on `occ_for_scoring`.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.common.goals import (dist_field_from_mask, higher_is_better_for, letter_mask,
                                    quadrant_mask)
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles, OCC_GRID

CORPUS = "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"
ART = REPO / "experiments/EXP-0029-state-vs-pool-split/artifacts/RUN-0001-split"
RES = REPO / "experiments/EXP-0029-state-vs-pool-split/results/split_test.json"
MODELS = ["nfd_3ch_randlen", "nfd_3ch_finetuned", "nfd_warped_randlen",
          "nfd_residual_warped", "linear_switched_hard"]
GOALS = [f"letter_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"] + [f"quadrant_{q}" for q in range(4)]
VFS = ["lyapunov", "mass_in_region"]


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    if str(path).endswith(".json"):
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def goal_tensors(dev):
    masks, dists = [], []
    for g in GOALS:
        m = quadrant_mask(OCC_GRID, OCC_GRID, int(g[-1])) if g.startswith("quadrant_") \
            else letter_mask(g[-1], OCC_GRID, OCC_GRID)
        masks.append(torch.from_numpy(m.astype(np.float32)))
        dists.append(torch.from_numpy(dist_field_from_mask(m)).float())
    return torch.stack(masks).to(dev), torch.stack(dists).to(dev)      # (G,H,W)


def values(occ, masks, dists):
    """occ (B,H,W) -> (B, G, 2): [lyapunov, mass_in_region] per goal."""
    f = occ.reshape(occ.shape[0], -1)
    lyap = (f @ dists.reshape(len(dists), -1).T) / f.sum(1, keepdim=True).clamp_min(1e-6)
    mass = f @ masks.reshape(len(masks), -1).T
    return torch.stack([lyap, mass], -1)


def capture(vp, vt, hib):
    """vp, vt (P, G) numpy -> (G,) slateN capture, higher = better."""
    if not hib:
        vp, vt = -vp, -vt
    pick = vp.argmax(0)
    chosen = vt[pick, np.arange(vt.shape[1])]
    mean, best = vt.mean(0), vt.max(0)
    den = best - mean
    out = np.full(vt.shape[1], np.nan)
    ok = np.abs(den) > 1e-9
    out[ok] = (chosen[ok] - mean[ok]) / den[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    RES.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    slate = rows.slate_idx.long()
    acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
    S = int(slate.max()) + 1
    masks, dists = goal_tensors(dev)

    # ---- stage B: truth (soft scoring) --------------------------------------
    tpath = ART / "truth.pt"
    if not tpath.exists():
        v0 = values(occ_for_scoring(rows.states.float(), dev), masks, dists)
        v1 = torch.cat([values(occ_for_scoring(rows.states_[i:i + 2000].float(), dev), masks, dists)
                        for i in range(0, len(slate), 2000)])
        save_atomic({"dv": (v1 - v0).cpu(), "scoring": "soft (occ_for_scoring)"}, tpath)
        print("truth done", flush=True)
    dvt = torch.load(tpath, weights_only=False)["dv"].numpy()               # (N, G, 2)

    # ---- stage A: predictions ------------------------------------------------
    dvp = {}
    for m in MODELS:
        p = ART / f"pred_{m}.pt"
        if not p.exists():
            t0 = time.time()
            ad = make_occ_adapter(m, dev, "corner")
            out = torch.zeros(len(slate), len(GOALS), 2)
            for s in range(S):
                idx = (slate == s).nonzero(as_tuple=True)[0]
                occ0 = occ_from_particles(rows.states[idx[0]].float()[None], dev)
                v0 = values(occ0, masks, dists)                                # (1,G,2)
                with torch.no_grad():
                    for i in range(0, len(idx), 100):
                        j = idx[i:i + 100]
                        pred = ad.predict_step(occ0.expand(len(j), -1, -1).contiguous(), acts[j].to(dev))
                        out[j] = (values(pred.float(), masks, dists) - v0).cpu()
            save_atomic({"dv": out, "model": m}, p)
            print(f"pred {m} done ({time.time() - t0:.0f}s)", flush=True)
            del ad
            torch.cuda.empty_cache()
        dvp[m] = torch.load(p, weights_only=False)["dv"].numpy()

    # ---- stage C: splits ------------------------------------------------------
    rng = np.random.default_rng(a.seed)
    sl = slate.numpy()
    idx_by_s = [np.nonzero(sl == s)[0] for s in range(S)]
    M = len(MODELS)
    results = {"models": MODELS, "goals": GOALS, "n_states": S, "splits": a.splits,
               "truth_scoring": "soft", "per_K": {}}
    for K in (128, 500):
        for vi, vf in enumerate(VFS):
            hib = higher_is_better_for(vf)
            # cap[r, pool, m, s] goal-averaged capture
            cap = np.zeros((a.splits, 2, M, S))
            for r in range(a.splits):
                for s in range(S):
                    perm = rng.permutation(idx_by_s[s])
                    pools = (perm[:K], perm[K:2 * K])
                    for h, pool in enumerate(pools):
                        vt = dvt[pool, :, vi]
                        for mi, m in enumerate(MODELS):
                            cap[r, h, mi, s] = np.nanmean(capture(dvp[m][pool, :, vi], vt, hib))
            # S1: split-half reliability of per-state pairwise advantage
            rel = {}
            for i, j in itertools.combinations(range(M), 2):
                d1, d2 = cap[:, 0, i] - cap[:, 0, j], cap[:, 1, i] - cap[:, 1, j]     # (R, S)
                rs = [np.corrcoef(d1[r], d2[r])[0, 1] for r in range(a.splits)]
                rel[f"{MODELS[i]}|{MODELS[j]}"] = dict(r_mean=float(np.nanmean(rs)),
                                                       r_lo=float(np.nanquantile(rs, .025)),
                                                       r_hi=float(np.nanquantile(rs, .975)),
                                                       mean_adv=float(np.mean(d1)))
            # S2: cross-fitted switching gain (choose on pool 1, score on pool 2)
            m1, m2 = cap[:, 0], cap[:, 1]                                              # (R, M, S)
            single = m1.mean(2).argmax(1)                                              # (R,)
            single_score = m2[np.arange(a.splits), single].mean(1)
            pick = m1.argmax(1)                                                        # (R, S)
            switch_score = np.take_along_axis(m2, pick[:, None, :], 1)[:, 0].mean(1)
            gain = switch_score - single_score
            # state bootstrap of the gain (per split, resample states)
            bs = []
            for r in range(0, a.splits, 10):
                per_s = (np.take_along_axis(m2[r], pick[r][None], 0)[0] - m2[r, single[r]])
                for _ in range(100):
                    bs.append(per_s[rng.integers(0, S, S)].mean())
            key = f"K{K}_{vf}"
            results["per_K"][key] = dict(
                model_means={m: float(cap[:, :, mi].mean()) for mi, m in enumerate(MODELS)},
                reliability=rel,
                median_r=float(np.median([v["r_mean"] for v in rel.values()])),
                gain_mean=float(gain.mean()), gain_lo=float(np.quantile(gain, .025)),
                gain_hi=float(np.quantile(gain, .975)), frac_gain_pos=float((gain > 0).mean()),
                gain_state_boot_lo=float(np.quantile(bs, .025)), gain_state_boot_hi=float(np.quantile(bs, .975)),
                single_score=float(single_score.mean()), switch_score=float(switch_score.mean()),
                hindsight=float(m2.max(1).mean()))
            save_atomic(results, RES)
            r_ = results["per_K"][key]
            print(f"[{key}] median split-half r {r_['median_r']:+.2f}; switching gain {r_['gain_mean']:+.4f} "
                  f"[{r_['gain_lo']:+.4f},{r_['gain_hi']:+.4f}] over splits, state-bootstrap "
                  f"[{r_['gain_state_boot_lo']:+.4f},{r_['gain_state_boot_hi']:+.4f}]; single {r_['single_score']:.3f} "
                  f"-> switch {r_['switch_score']:.3f} (hindsight {r_['hindsight']:.3f})", flush=True)
            for k, v in sorted(rel.items(), key=lambda kv: -kv[1]["r_mean"]):
                print(f"     {k:50s} r {v['r_mean']:+.2f} [{v['r_lo']:+.2f},{v['r_hi']:+.2f}]  mean adv {v['mean_adv']:+.3f}")
    print("wrote", RES)


if __name__ == "__main__":
    main()
