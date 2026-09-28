"""EXP-0030 A5 (DESIGN.md addendum 3): slateN ranking at a MATCHED wall-clock budget.

Stage 1 (GPU): each model's per-candidate cost at throughput (2048 candidates in
  chunks of 128); ensembles cost the sum of their members (run member by member, as
  learned_mpc.ModelObjective does). -> artifacts/RUN-0002/timings.json
Stage 2 (CPU): from A3's cached predictions (artifacts/RUN-0001/pred_*.pt) and soft
  truth, each model with budget B picks the best predicted push among the first
  n_m(B) pushes of 200 shared random permutations of each state's 128-push pool;
  capture is normalised by the full pool. -> results/budget_matched.json
Both stages checkpoint atomically; a rerun skips finished timings.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0029-state-vs-pool-split/code"))
from split_test import goal_tensors, save_atomic, values, VFS
from Baselines.common.goals import higher_is_better_for
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_from_particles

HERE = REPO / "experiments/EXP-0030-state-superiority-ds0005"
A3_ART, ART = HERE / "artifacts/RUN-0001", HERE / "artifacts/RUN-0002"
RES = HERE / "results/budget_matched.json"
CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys"
SINGLES = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
           "nfd_warped_randlen_flipaug_epoch30", "nfd_residual_warped_flipaug_randlen",
           "nfd_residual_worldframe_noaug_ep43", "nfd_3ch_finetuned", "linear_switched_hard"]
ENSEMBLES = {"ensemble_nfd": [m for m in SINGLES if m.startswith("nfd")],
             "ensemble_nfd5": ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
                               "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43"]}
REF, REF_M = "nfd_3ch_randlen", [16, 32, 64, 128]
A3_BEST = {"lyapunov": "nfd_residual_warped_flipaug_randlen",
           "mass_in_region": "nfd_residual_worldframe_noaug_ep43"}
N_TIME, CHUNK = 2048, 128
N_PERM, N_BOOT = 200, 5000


def time_models(rows, slate):
    """Per-candidate cost c_m (s): 2048 candidates of one state in chunks of 128
    (as learned_mpc.ModelObjective), median of 7 timed repeats after 2 warm-ups.
    (Addendum 4: a single call's time is launch overhead, not throughput.)"""
    tp = ART / "timings.json"
    T = json.loads(tp.read_text()) if tp.exists() else {}
    dev = "cuda"
    masks, dists = goal_tensors(dev)
    acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
    for m in SINGLES:
        if m in T:
            continue
        ad = make_occ_adapter(m, dev, "corner")
        reps = []
        for k in range(9):
            s = k % 4
            i = np.nonzero(slate == s)[0]
            occ0 = occ_from_particles(rows.states[i[:1]].float(), dev)
            a_all = acts[i].to(dev).repeat(N_TIME // len(i), 1)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                for j in range(0, N_TIME, CHUNK):
                    a = a_all[j:j + CHUNK]
                    v = values(ad.predict_step(occ0.expand(len(a), -1, -1).contiguous(), a), masks, dists)
                    float(v.sum())
            torch.cuda.synchronize()
            if k >= 2:
                reps.append(time.perf_counter() - t0)
        T[m] = float(np.median(reps)) / N_TIME
        save_atomic(T, tp)
        print(f"timed {m}: {1e6 * T[m]:.1f} us/candidate ({1 / T[m]:.0f}/s)", flush=True)
        del ad; torch.cuda.empty_cache()
    return T


def cost_of(T, arm):
    return sum(T[m] for m in ENSEMBLES.get(arm, [arm]))


def boot(x, rng):
    x = x[~np.isnan(x)]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    ART.mkdir(parents=True, exist_ok=True); RES.parent.mkdir(parents=True, exist_ok=True)
    rows = BinnedSlateCorpus.load(str(REPO / CORPUS)).step(0)
    slate = rows.slate_idx.long().numpy()
    S = int(slate.max()) + 1
    idx_by_s = [np.nonzero(slate == s)[0] for s in range(S)]
    assert all(len(i) == 128 for i in idx_by_s)
    T = time_models(rows, slate)
    truth = torch.load(A3_ART / "truth.pt", weights_only=False)
    assert truth["scoring"] == "soft"
    vt_all = truth["dv"].numpy()                                         # (N, G, 2)
    dvp = {m: torch.load(A3_ART / f"pred_{m}.pt", weights_only=False)["dv"].numpy() for m in SINGLES}
    for e, mem in ENSEMBLES.items():
        dvp[e] = np.mean([dvp[m] for m in mem], 0)
    arms = SINGLES + list(ENSEMBLES)
    cost = {a: cost_of(T, a) for a in arms}
    budgets = [m * cost[REF] for m in REF_M]
    nb = {a: [int(min(128, max(1, np.floor(B / cost[a] + 1e-9)))) for B in budgets] for a in arms}
    perms = np.stack([np.random.default_rng(s).permuted(np.tile(np.arange(128), (N_PERM, 1)), axis=1)
                      for s in range(S)])                                  # (S, R, 128)
    out = dict(budgets_s=budgets, ref=REF, ref_m=REF_M, n_candidates=nb, pool=128, n_perm=N_PERM,
               cost_s_per_candidate={a: float(cost[a]) for a in arms}, per_vf={})
    rng = np.random.default_rng(0)
    for vi, vf in enumerate(VFS):
        hib = higher_is_better_for(vf)
        C = np.full((len(arms), len(budgets), S), np.nan)                   # per-state capture
        for s in range(S):
            vt = vt_all[idx_by_s[s], :, vi]
            if not hib:
                vt = -vt
            mean, best = vt.mean(0), vt.max(0)
            den = best - mean
            ok = np.abs(den) > 1e-9
            if not ok.any():
                continue
            for ai, a in enumerate(arms):
                vp = dvp[a][idx_by_s[s], :, vi]
                vp = -vp if not hib else vp
                for bi, n in enumerate(nb[a]):
                    sub = perms[s, :, :n]                                   # (R, n)
                    pick = sub[np.arange(N_PERM)[:, None], vp[sub].argmax(1)]   # (R, G)
                    chosen = vt[pick, np.arange(vt.shape[1])[None]]
                    cap = (chosen[:, ok] - mean[ok]) / den[ok]
                    C[ai, bi, s] = cap.mean()
        res = dict(means={a: [float(np.nanmean(C[ai, bi])) for bi in range(len(budgets))]
                          for ai, a in enumerate(arms)}, comparisons={})
        for e in ENSEMBLES:
            ei = arms.index(e)
            comp = {}
            for bi in range(len(budgets)):
                ref_i = arms.index(A3_BEST[vf])
                post = max(SINGLES, key=lambda a: np.nanmean(C[arms.index(a), bi]))
                d_pre = C[ei, bi] - C[ref_i, bi]
                d_post = C[ei, bi] - C[arms.index(post), bi]
                comp[str(REF_M[bi])] = dict(
                    n_ensemble=nb[e][bi], n_a3_best=nb[A3_BEST[vf]][bi],
                    minus_a3_best=float(np.nanmean(d_pre)), ci_a3_best=boot(d_pre, rng),
                    best_single_here=post, n_best_here=nb[post][bi],
                    minus_best_here=float(np.nanmean(d_post)), ci_best_here=boot(d_post, rng))
            res["comparisons"][e] = comp
        out["per_vf"][vf] = res
        save_atomic(out, RES)
        print(f"\n== {vf}: budgets (ref {REF} m) {REF_M} = {[f'{1e3 * b:.2f}ms' for b in budgets]}")
        for a in sorted(arms, key=lambda a: -res["means"][a][-1]):
            print(f"  {a:40s} n={nb[a]}  capture {[round(x, 3) for x in res['means'][a]]}")
        for e, comp in res["comparisons"].items():
            for m, c in comp.items():
                print(f"  {e} m={m}: n_ens {c['n_ensemble']} vs {A3_BEST[vf]} n {c['n_a3_best']}: "
                      f"{c['minus_a3_best']:+.3f} {np.round(c['ci_a3_best'], 3)}; best here {c['best_single_here']} "
                      f"{c['minus_best_here']:+.3f} {np.round(c['ci_best_here'], 3)}", flush=True)
    print("wrote", RES)


if __name__ == "__main__":
    main()
