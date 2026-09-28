"""EXP-0030: state-dependent model superiority on DS-0005 (DESIGN.md = the plan).

Stages, each checkpointed under artifacts/RUN-0001/ (rerun skips finished ones):
  truth.pt, pred_<model>.pt, descriptors.pt, then results/analysis.json
  rewritten after every analysis block.
Scoring functions are imported from EXP-0029's split_test.py so the two
experiments score identically (soft truth, image predictions, slateN).
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import torch
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0029-state-vs-pool-split/code"))
from split_test import capture, goal_tensors, save_atomic, values, VFS
from Baselines.common.goals import higher_is_better_for
from Baselines.common.paired_stats import holm
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles

HERE = REPO / "experiments/EXP-0030-state-superiority-ds0005"
ART, RES = HERE / "artifacts/RUN-0001", HERE / "results/analysis.json"
MODELS = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
          "nfd_warped_randlen_flipaug_epoch30", "nfd_residual_warped_flipaug_randlen",
          "nfd_residual_worldframe_noaug_ep43", "nfd_3ch_finetuned", "linear_switched_hard"]
NFD = [m for m in MODELS if m.startswith("nfd")]
DESCRIPTORS = ["radius_of_gyration", "mean_nn_dist", "mean_wall_dist", "frac_with_neighbour_12mm",
               "centroid_offset", "hull_area"]


def descriptors(state, half_width=0.064, near=0.012):
    """Short, interpretable descriptors of one start state (n, >=2) in metres."""
    from scipy.spatial import ConvexHull
    xy = state[:, :2].double().numpy()
    c = xy.mean(0)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    frac_near = (d.min(1) < near).mean()
    wall = (half_width - np.abs(xy)).min(1)
    try:
        hull = ConvexHull(xy).volume
    except Exception:
        hull = 0.0
    return [float(np.sqrt(((xy - c) ** 2).sum(1).mean())), float(d.min(1).mean()),
            float(wall.mean()), float(frac_near), float(np.linalg.norm(c)), float(hull)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--resplits", type=int, default=100)
    ap.add_argument("--art", default=None, help="override artifact dir (smoke tests)")
    ap.add_argument("--res", default=None, help="override results json (smoke tests)")
    a = ap.parse_args()
    global ART, RES
    if a.art: ART = Path(a.art)
    if a.res: RES = Path(a.res)
    ART.mkdir(parents=True, exist_ok=True); RES.parent.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.corpus.endswith(".pt"):                     # DS-0007-style flat-row shard
        from types import SimpleNamespace
        d = torch.load(REPO / a.corpus, weights_only=False)
        rows = SimpleNamespace(states=d["states"], states_=d["states_"], p_starts=d["p_starts"],
                               p_stops=d["p_stops"], slate_idx=d["pool_idx"])
    else:
        rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
    slate = rows.slate_idx.long().numpy()
    S = int(slate.max()) + 1
    idx_by_s = [np.nonzero(slate == s)[0] for s in range(S)]
    acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
    masks, dists = goal_tensors(dev)

    tp = ART / "truth.pt"
    if not tp.exists():
        v0 = values(occ_for_scoring(rows.states.float(), dev), masks, dists)
        v1 = torch.cat([values(occ_for_scoring(rows.states_[i:i + 2000].float(), dev), masks, dists)
                        for i in range(0, len(slate), 2000)])
        save_atomic({"dv": (v1 - v0).cpu(), "scoring": "soft"}, tp)
    dvt = torch.load(tp, weights_only=False)["dv"].numpy()

    dvp = {}
    for m in MODELS:
        p = ART / f"pred_{m}.pt"
        if not p.exists():
            t0 = time.time(); ad = make_occ_adapter(m, dev, "corner")
            out = torch.zeros(len(slate), dvt.shape[1], 2)
            for s in range(S):
                ix = torch.from_numpy(idx_by_s[s])
                occ0 = occ_from_particles(rows.states[ix[0]].float()[None], dev)
                v0 = values(occ0, masks, dists)
                with torch.no_grad():
                    for i in range(0, len(ix), 128):
                        j = ix[i:i + 128]
                        pred = ad.predict_step(occ0.expand(len(j), -1, -1).contiguous(), acts[j].to(dev))
                        out[j] = (values(pred.float(), masks, dists) - v0).cpu()
            save_atomic({"dv": out, "model": m}, p); del ad; torch.cuda.empty_cache()
            print(f"pred {m} ({time.time() - t0:.0f}s)", flush=True)
        dvp[m] = torch.load(p, weights_only=False)["dv"].numpy()
    dvp["ensemble_nfd"] = np.mean([dvp[m] for m in NFD], 0)        # A3: average predictions

    dp = ART / "descriptors.pt"
    if not dp.exists():
        D = np.array([descriptors(rows.states[idx_by_s[s][0]]) for s in range(S)])
        save_atomic({"D": torch.from_numpy(D), "names": DESCRIPTORS}, dp)
    D = torch.load(dp, weights_only=False)["D"].numpy()

    names = MODELS + ["ensemble_nfd"]
    results = {"n_states": S, "models": names, "pool_size": len(idx_by_s[0]) // 2, "per_vf": {}}

    def caps_for(perms, vi, hib):
        """(n_perm, 2 pools, M, S) goal-averaged capture."""
        out = np.zeros((len(perms), 2, len(names), S))
        for r, perm in enumerate(perms):
            for s in range(S):
                half = len(perm[s]) // 2
                for h, pool in enumerate((perm[s][:half], perm[s][half:])):
                    vt = dvt[pool, :, vi]
                    for mi, m in enumerate(names):
                        out[r, h, mi, s] = np.nanmean(capture(dvp[m][pool, :, vi], vt, hib))
        return out

    rng = np.random.default_rng(0)
    perms = [[rng.permutation(ix) for ix in idx_by_s] for _ in range(a.resplits)]
    for vi, vf in enumerate(VFS):
        hib = higher_is_better_for(vf)
        C = caps_for(perms, vi, hib)                     # (R, 2, M, S); r=0 is the primary split
        res = {}
        # A1 reliability (primary split + mean over re-splits)
        rel = {}
        for i, j in itertools.combinations(range(len(MODELS)), 2):
            d1, d2 = C[:, 0, i] - C[:, 0, j], C[:, 1, i] - C[:, 1, j]
            def _r(x, y):
                ok = ~(np.isnan(x) | np.isnan(y))
                return np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 3 else np.nan
            rs = np.array([_r(d1[r], d2[r]) for r in range(len(perms))])
            rel[f"{MODELS[i]}|{MODELS[j]}"] = dict(r_primary=float(rs[0]), r_mean=float(np.nanmean(rs)),
                                                   mean_adv=float(np.nanmean(d1)))
        nfd_pairs = [k for k in rel if all(x in NFD for x in k.split("|"))]
        res["A1"] = dict(pairs=rel, median_r_nfd=float(np.median([rel[k]["r_mean"] for k in nfd_pairs])),
                         median_r_all=float(np.median([v["r_mean"] for v in rel.values()])))
        # A2 cross-fitted switching gain within the NFD family, and A3 ensemble
        nfd_i = [names.index(m) for m in NFD]; ens_i = names.index("ensemble_nfd")
        gains, ens_gain = [], []
        for r in range(len(perms)):
            m1, m2 = C[r, 0][nfd_i], C[r, 1][nfd_i]      # (Mn, S)
            best = int(np.nanmean(m1, 1).argmax())
            gains.append(m2[np.nan_to_num(m1, nan=-np.inf).argmax(0), np.arange(S)] - m2[best])
            ens_gain.append(C[r, 1, ens_i] - m2[best])
        gains, ens_gain = np.array(gains), np.array(ens_gain)   # (R, S)

        def boot(x, n=5000):
            x = x[~np.isnan(x)]
            b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
            return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
        res["n_undefined_state_pool_cells"] = int(np.isnan(C).any(2).sum())
        res["A2"] = dict(gain=float(np.nanmean(gains)), state_boot_ci_primary=boot(gains[0]),
                         resplit_range=[float(np.quantile(np.nanmean(gains, 1), .025)), float(np.quantile(np.nanmean(gains, 1), .975))])
        res["A3"] = dict(ensemble_minus_best=float(np.nanmean(ens_gain)), state_boot_ci_primary=boot(ens_gain[0]),
                         model_means={m: float(np.nanmean(C[:, :, mi])) for mi, m in enumerate(names)})
        # A4 descriptors, for pairs with reliable advantage
        adv_pairs = [k for k, v in rel.items() if v["r_mean"] > 0.2]
        tests, sel = [], {}
        for k in adv_pairs:
            i, j = (MODELS.index(x) for x in k.split("|"))
            adv = np.nanmean(C[:, :, i] - C[:, :, j], (0, 1))   # (S,) pooled over pools and re-splits
            okS = ~np.isnan(adv)
            for di, dn in enumerate(DESCRIPTORS):
                rho = stats.spearmanr(D[okS, di], adv[okS]).statistic
                null = np.array([stats.spearmanr(D[okS, di][rng.permutation(okS.sum())], adv[okS]).statistic for _ in range(2000)])
                tests.append(dict(pair=k, descriptor=dn, rho=float(rho), p=float((np.abs(null) >= abs(rho)).mean())))
            # linear selector: ridge on standardised descriptors, choose i vs j, CV by state (16 folds)
            from sklearn.linear_model import RidgeCV
            a1 = (C[0, 0, i] - C[0, 0, j]); a2 = (C[0, 1, i] - C[0, 1, j])   # primary split
            ok = ~(np.isnan(a1) | np.isnan(C[0, 1, i]) | np.isnan(C[0, 1, j]))
            Zall = (D - D.mean(0)) / D.std(0).clip(1e-12)
            Z, a1o = Zall[ok], a1[ok]
            So = int(ok.sum())
            fold = rng.permutation(So) % 16
            pred = np.zeros(So)
            for f in range(16):
                tr, te = fold != f, fold == f
                pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Z[tr], a1o[tr]).predict(Z[te])
            better = i if np.nanmean(a1) > 0 else j
            chosen = np.where(pred > 0, C[0, 1, i][ok], C[0, 1, j][ok])
            g = chosen - C[0, 1, better][ok]
            sel[k] = dict(gain=float(g.mean()), ci=boot(g), frac_choose_i=float((pred > 0).mean()))
        for t, ph in zip(tests, holm([t["p"] for t in tests]) if tests else []):
            t["p_holm"] = ph
        res["A4"] = dict(pairs_tested=adv_pairs, descriptor_tests=tests, linear_selector=sel,
                         n_holm_05=sum(t.get("p_holm", 1) < .05 for t in tests))
        results["per_vf"][vf] = res
        save_atomic(results, RES)
        print(f"\n=== {vf} ===\nA1 median r (NFD pairs) {res['A1']['median_r_nfd']:+.2f}, all {res['A1']['median_r_all']:+.2f}")
        for k2, v in sorted(rel.items(), key=lambda kv: -kv[1]["r_mean"])[:8]:
            print(f"   {k2:62s} r {v['r_mean']:+.2f} (primary {v['r_primary']:+.2f}) adv {v['mean_adv']:+.3f}")
        print(f"undefined (state, pool) cells: {res['n_undefined_state_pool_cells']} of {C.shape[0] * 2 * S}")
        print(f"A2 switch-within-NFD gain {res['A2']['gain']:+.4f} state-boot {res['A2']['state_boot_ci_primary']}")
        print(f"A3 ensemble - best single {res['A3']['ensemble_minus_best']:+.4f} state-boot {res['A3']['state_boot_ci_primary']}")
        print("   model means: " + " ".join(f"{m}={v:.3f}" for m, v in res['A3']['model_means'].items()))
        print(f"A4 pairs tested {len(adv_pairs)}; Holm<.05 descriptor links {res['A4']['n_holm_05']}")
        for t in sorted(tests, key=lambda t: t["p"])[:6]:
            print(f"   {t['pair']:62s} {t['descriptor']:18s} rho {t['rho']:+.2f} p {t['p']:.4f} holm {t.get('p_holm', 1):.3f}")
        for k2, v in sel.items():
            print(f"   linear selector {k2:50s} gain {v['gain']:+.4f} {v['ci']}")
    print("wrote", RES)


if __name__ == "__main__":
    main()
