"""EXP-0036 P2, like for like: seed ensemble vs architecture ensembles on the SAME
whole pools (EXP-0035's +0.014..+0.058 was on half pools, so not comparable).

Arms per corpus (cached predictions: EXP-0030 / EXP-0035 + this experiment's seeds):
  best single NFD architecture (of the 7 'nfd*'), each seed, seed_ens4 (4 seeds),
  arch_ens4 (nfd_3ch_randlen, nfd_warped_randlen, nfd_residual_warped_flipaug_randlen,
  nfd_residual_worldframe_noaug_ep43 -- 4 architectures, member count matched to
  seed_ens4), ensemble_nfd5, ensemble_nfd (7).
Differences are paired over states (state bootstrap). -> results/ensemble_compare.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_seeds as ss

NFD7 = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug", "nfd_warped_randlen_flipaug_epoch30",
        "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43", "nfd_3ch_finetuned"]
ENS = {"seed_ens4": ss.SEEDS,
       "arch_ens4": ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_residual_warped_flipaug_randlen",
                     "nfd_residual_worldframe_noaug_ep43"],
       "ensemble_nfd5": ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
                         "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43"],
       "ensemble_nfd": NFD7}
RES = ss.HERE / "results/ensemble_compare.json"


def main():
    rng = np.random.default_rng(0)
    res = {}
    for cname, (path, src) in ss.CORPORA.items():
        rows = ss.load_rows(path)
        slate = rows.slate_idx.long().numpy()
        idx = [np.nonzero(slate == s)[0] for s in range(int(slate.max()) + 1)]
        idx = [i for i in idx if len(i) > 1]
        vt = torch.load(src / "truth.pt", weights_only=False)["dv"].numpy()
        dvp = {m: torch.load(src / f"pred_{m}.pt", weights_only=False)["dv"].numpy() for m in NFD7}
        for m in ss.SEEDS[1:]:
            dvp[m] = torch.load(ss.ART / f"pred_{cname}_{m}.pt", weights_only=False)["dv"].numpy()
        for e, mem in ENS.items():
            dvp[e] = np.mean([dvp[m] for m in mem], 0)
        res[cname] = {}
        for vi, vf in enumerate(ss.VFS):
            hib = ss.higher_is_better_for(vf)
            C = {m: ss.per_state_capture(dvp[m][..., vi], vt[..., vi], idx, hib) for m in dvp}
            means = {m: float(np.nanmean(c)) for m, c in C.items()}
            best = max(NFD7, key=lambda m: means[m])
            d = lambda a, b: dict(diff=float(np.nanmean(C[a] - C[b])), ci=ss.boot(C[a] - C[b], rng))
            r = dict(means=means, best_single_arch=best,
                     seed_ens4_minus_best_arch=d("seed_ens4", best),
                     arch_ens4_minus_best_arch=d("arch_ens4", best),
                     ensemble_nfd_minus_best_arch=d("ensemble_nfd", best),
                     seed_ens4_minus_arch_ens4=d("seed_ens4", "arch_ens4"),
                     seed_ens4_minus_ensemble_nfd=d("seed_ens4", "ensemble_nfd"))
            res[cname][vf] = r
            print(f"[{cname} / {vf}] best arch {best} {means[best]:.3f}; seed_ens4 {means['seed_ens4']:.3f} "
                  f"arch_ens4 {means['arch_ens4']:.3f} ens5 {means['ensemble_nfd5']:.3f} ens7 {means['ensemble_nfd']:.3f}; "
                  f"seed4-arch4 {r['seed_ens4_minus_arch_ens4']['diff']:+.3f} {np.round(r['seed_ens4_minus_arch_ens4']['ci'], 3)}",
                  flush=True)
        ss.save_atomic(res, RES)
    print("wrote", RES)


if __name__ == "__main__":
    main()
