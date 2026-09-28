"""EXP-0036: score the 4 training seeds of nfd_3ch_randlen (DESIGN.md) with slateN.

Seed 0 = the existing `nfd_3ch_randlen` run; its predictions are reused from
EXP-0030 (DS-0006) / EXP-0035 (DS-0007 n20, n50), and seeds 1-3 are predicted here
by the same code path (analyse.py's block: occ_from_particles -> predict_step in
chunks of 128 -> per-goal lyapunov / mass_in_region dv), with the same soft truth.
CPU only (CUDA hidden by the caller) so EXP-0039's timed planning is not slowed.

Per corpus: per-state slateN (whole pool, 30 goals averaged), for each seed and the
seed ensemble (mean predicted dv); the seed sd; every seed pair's paired difference
(state bootstrap); seed ensemble minus mean / best single seed.
Checkpoints: one pred file per (corpus, seed); results json after every corpus.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0029-state-vs-pool-split/code"))
from split_test import goal_tensors, save_atomic, values, VFS
from Baselines.common.goals import higher_is_better_for
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_from_particles

HERE = REPO / "experiments/EXP-0036-seed-noise-floor"
ART, RES = HERE / "artifacts/RUN-0002", HERE / "results/slaten_seeds.json"
E30 = REPO / "experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001"
E35 = REPO / "experiments/EXP-0035-state-superiority-ds0007/artifacts"
CORPORA = {
    "DS-0006": ("Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys", E30),
    "DS-0007_scattered_n20": ("datasets/DS-0007-sean-same-state-pools/data/scattered_n20.pt", E35 / "RUN-n20"),
    "DS-0007_scattered_n50": ("datasets/DS-0007-sean-same-state-pools/data/scattered_n50.pt", E35 / "RUN-n50"),
}
SEEDS = ["nfd_3ch_randlen", "nfd_3ch_randlen_seed1", "nfd_3ch_randlen_seed2", "nfd_3ch_randlen_seed3"]
N_BOOT = 5000


def load_rows(path):
    if path.endswith(".pt"):
        d = torch.load(REPO / path, weights_only=False)
        return SimpleNamespace(states=d["states"], p_starts=d["p_starts"], p_stops=d["p_stops"],
                               slate_idx=d["pool_idx"])
    return BinnedSlateCorpus.load(str(REPO / path)).step(0)


def predict(m, rows, idx_by_s, acts, masks, dists, G):
    ad = make_occ_adapter(m, "cpu", "corner")
    out = torch.zeros(len(acts), G, 2)
    for ix in idx_by_s:
        ix = torch.from_numpy(ix)
        occ0 = occ_from_particles(rows.states[ix[0]].float()[None], "cpu")
        v0 = values(occ0, masks, dists)
        with torch.no_grad():
            for i in range(0, len(ix), 128):
                j = ix[i:i + 128]
                pred = ad.predict_step(occ0.expand(len(j), -1, -1).contiguous(), acts[j])
                out[j] = values(pred.float(), masks, dists) - v0
    return out


def per_state_capture(vp, vt, idx_by_s, hib):
    """(N, G) predicted / true dv -> (S,) slateN averaged over goals (nan if undefined)."""
    if not hib:
        vp, vt = -vp, -vt
    out = np.full(len(idx_by_s), np.nan)
    for s, ix in enumerate(idx_by_s):
        p, t = vp[ix], vt[ix]
        mean, best = t.mean(0), t.max(0)
        den = best - mean
        ok = np.abs(den) > 1e-9
        if ok.any():
            ch = t[p.argmax(0), np.arange(t.shape[1])]
            out[s] = np.mean((ch[ok] - mean[ok]) / den[ok])
    return out


def boot(x, rng):
    x = x[~np.isnan(x)]
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def main():
    torch.set_num_threads(4)
    ART.mkdir(parents=True, exist_ok=True); RES.parent.mkdir(parents=True, exist_ok=True)
    masks, dists = goal_tensors("cpu")
    res = json.loads(RES.read_text()) if RES.exists() else {}
    rng = np.random.default_rng(0)
    for cname, (path, src) in CORPORA.items():
        if cname in res:
            continue
        rows = load_rows(path)
        slate = rows.slate_idx.long().numpy()
        S = int(slate.max()) + 1
        idx_by_s = [np.nonzero(slate == s)[0] for s in range(S)]
        idx_by_s = [i for i in idx_by_s if len(i) > 1]
        acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
        dvt = torch.load(src / "truth.pt", weights_only=False)
        assert dvt["scoring"] == "soft"
        dvt = dvt["dv"].numpy()
        dvp = {SEEDS[0]: torch.load(src / f"pred_{SEEDS[0]}.pt", weights_only=False)["dv"].numpy()}
        for m in SEEDS[1:]:
            p = ART / f"pred_{cname}_{m}.pt"
            if not p.exists():
                t0 = time.time()
                save_atomic({"dv": predict(m, rows, idx_by_s, acts, masks, dists, dvt.shape[1]), "model": m}, p)
                print(f"[{cname}] pred {m} ({time.time() - t0:.0f}s)", flush=True)
            dvp[m] = torch.load(p, weights_only=False)["dv"].numpy()
        # consistency check: seed-0 predicted here on CPU must match the cached (GPU) file
        chk = predict(SEEDS[0], rows, idx_by_s[:3], acts, masks, dists, dvt.shape[1]).numpy()
        ix3 = np.concatenate(idx_by_s[:3])
        max_diff = float(np.abs(chk[ix3] - dvp[SEEDS[0]][ix3]).max())
        dvp["seed_ensemble"] = np.mean([dvp[m] for m in SEEDS], 0)
        out = dict(n_states=len(idx_by_s), pool_sizes=sorted({len(i) for i in idx_by_s}),
                   cpu_vs_cached_seed0_max_abs_diff=max_diff, per_vf={})
        for vi, vf in enumerate(VFS):
            hib = higher_is_better_for(vf)
            C = {m: per_state_capture(dvp[m][..., vi], dvt[..., vi], idx_by_s, hib) for m in dvp}
            means = {m: float(np.nanmean(c)) for m, c in C.items()}
            singles = np.array([means[m] for m in SEEDS])
            pairs = {f"{a}-{b}": dict(diff=float(np.nanmean(C[a] - C[b])), ci=boot(C[a] - C[b], rng))
                     for a, b in itertools.combinations(SEEDS, 2)}
            mean_single = np.nanmean(np.stack([C[m] for m in SEEDS]), 0)
            best = max(SEEDS, key=lambda m: means[m])
            out["per_vf"][vf] = dict(
                means=means, seed_sd=float(singles.std(ddof=1)), seed_range=float(singles.max() - singles.min()),
                pairs=pairs, n_pairs_ci_excl_0=sum(p["ci"][0] > 0 or p["ci"][1] < 0 for p in pairs.values()),
                ens_minus_mean_single=float(np.nanmean(C["seed_ensemble"] - mean_single)),
                ens_minus_mean_single_ci=boot(C["seed_ensemble"] - mean_single, rng),
                best_seed=best, ens_minus_best_seed=float(np.nanmean(C["seed_ensemble"] - C[best])),
                ens_minus_best_seed_ci=boot(C["seed_ensemble"] - C[best], rng))
            r = out["per_vf"][vf]
            print(f"[{cname} / {vf}] seeds {np.round(singles, 4)} sd {r['seed_sd']:.4f}; pairs CI!=0 "
                  f"{r['n_pairs_ci_excl_0']}/6; ens {means['seed_ensemble']:.4f} (-mean single "
                  f"{r['ens_minus_mean_single']:+.4f} {np.round(r['ens_minus_mean_single_ci'], 4)}; -best "
                  f"{r['ens_minus_best_seed']:+.4f} {np.round(r['ens_minus_best_seed_ci'], 4)})", flush=True)
        res[cname] = out
        save_atomic(res, RES)
        print(f"[{cname}] cpu vs cached seed0 max |diff| {max_diff:.2e}", flush=True)
    print("wrote", RES)


if __name__ == "__main__":
    main()
