"""Coordinator follow-up (2026-09-28, hard stop 08:10, offline/CPU only):
confidence model deliverable -- "given a state, which actions does the
dataset predict least well?" For `k5_cube_median@25k` (task 1's winning
bank), test whether `RetrievalPredictor.predict_particles_with_confidence`'s
two FREE signals (`top1_dist`, `knn_disagreement`) predict per-row error, on
DS-0009 test_pools (all 32 pools x 64 candidate actions = 2048 rows -- a
POOL gives 64 different actions from the SAME start state, exactly the
"given a state, which actions" framing, and each row also carries its own
recorded true outcome, so per-row error is directly available):

  * row-level Spearman(confidence signal, moved-cube mm error) and
    Spearman(confidence signal, swept-region proxy error), with a row-level
    bootstrap 95% CI;
  * WITHIN-POOL Spearman (does the signal rank a given pool's 64 actions by
    their own error, averaged over the 32 pools);
  * lift: does the top-10%-least-confident decile carry disproportionate
    error (mean error in that decile / mean error overall).

Usage:
    CUDA_VISIBLE_DEVICES= python -u confidence_model.py
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0053-narrow-domain-models/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_narrow as EN  # noqa: E402
from model.retrieval.bank import TransitionBank  # noqa: E402
from model.retrieval.distance import DistanceConfig  # noqa: E402
from model.retrieval.predictor import RetrievalPredictor  # noqa: E402
from model.retrieval.frame import world_to_push_frame, push_frame_basis  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
RES = Path(__file__).resolve().parents[1] / "results" / "confidence_model.json"
D = EN.D
_CAPPED = dict(cap=0.02, mismatch_penalty=0.04, corridor_weight=3.0)
BEST_CFG = DistanceConfig(**_CAPPED)
MOVED_THRESH = 0.001


def in_swept_rect(xy_world, p_start, p_stop):
    uv = world_to_push_frame(xy_world[None], p_start, p_stop)[0]
    _, _, L = push_frame_basis(p_start, p_stop)
    return (uv[:, 0] >= -0.02) & (uv[:, 0] <= float(L[0])) & (uv[:, 1].abs() <= 0.024)


def swept_proxy_error(true_xy1_in_swept, pred_xy1):
    if true_xy1_in_swept.shape[0] == 0:
        return None
    d = torch.cdist(true_xy1_in_swept, pred_xy1)
    return float((d.min(dim=1).values ** 2).mean())


def bootstrap_spearman(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rng = np.random.default_rng(seed)
    rho0, _ = spearmanr(x, y)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[ix], y[ix])
        boots[i] = r if r == r else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rho0), float(lo), float(hi)


def main():
    assert not torch.cuda.is_available(), "expected CUDA hidden; refusing to touch the GPU per coordinator instruction"
    bank = TransitionBank.load(str(ARTIFACTS / "bank_25k.pt"))
    predictor = RetrievalPredictor(bank, cfg=BEST_CFG, k=5, aggregation="cube_median", device="cpu")
    print(f"bank_25k: {len(bank)} rows, predictor {predictor.name}")

    pools = [torch.load(f, map_location="cpu", weights_only=False)
            for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]

    rows = []  # dicts: pool_id, top1_dist, knn_disagreement, mm_error(nan-able), swept_error
    t0 = time.time()
    pool_id = 0
    for d in pools:
        for pi in torch.unique(d["pool_idx"]):
            ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
            states0 = d["states"][ix].float()
            states1 = d["states_"][ix].float()
            p_start = d["p_starts"][ix].float(); p_stop = d["p_stops"][ix].float()
            pred_states, conf = predictor.predict_particles_with_confidence(states0, p_start, p_stop)
            top1_dist = conf["top1_dist"].numpy()
            knn_dis = conf["knn_disagreement"].numpy()
            for b in range(states0.shape[0]):
                true_disp = (states1[b, :, :2] - states0[b, :, :2]).norm(dim=-1)
                moved_mask = true_disp > MOVED_THRESH
                if bool(moved_mask.any()):
                    err_mm = (pred_states[b, :, :2] - states1[b, :, :2])[moved_mask].norm(dim=-1).mean().item() * 1000.0
                else:
                    err_mm = float("nan")
                in_sw = in_swept_rect(states1[b, :, :2], p_start[b:b + 1], p_stop[b:b + 1])
                sw_err = swept_proxy_error(states1[b, :, :2][in_sw], pred_states[b, :, :2])
                rows.append(dict(pool=pool_id, top1_dist=float(top1_dist[b]),
                                 knn_disagreement=float(knn_dis[b]), mm_error=err_mm,
                                 swept_error=sw_err if sw_err is not None else float("nan")))
            pool_id += 1
    print(f"scored {len(rows)} rows over {pool_id} pools in {time.time()-t0:.1f}s")

    top1 = np.array([r["top1_dist"] for r in rows])
    knn = np.array([r["knn_disagreement"] for r in rows])
    mm = np.array([r["mm_error"] for r in rows])
    sw = np.array([r["swept_error"] for r in rows])
    pool_ids = np.array([r["pool"] for r in rows])

    out = {"n_rows": len(rows), "n_pools": pool_id}
    mask_mm = ~np.isnan(mm)
    mask_sw = ~np.isnan(sw)
    print(f"non-nan mm rows: {mask_mm.sum()}/{len(rows)}  non-nan swept rows: {mask_sw.sum()}/{len(rows)}")

    for sig_name, sig in (("top1_dist", top1), ("knn_disagreement", knn)):
        rho_mm, lo_mm, hi_mm = bootstrap_spearman(sig[mask_mm], mm[mask_mm])
        rho_sw, lo_sw, hi_sw = bootstrap_spearman(sig[mask_sw], sw[mask_sw])
        out[f"{sig_name}_vs_mm_error"] = dict(spearman=rho_mm, ci95=[lo_mm, hi_mm], n=int(mask_mm.sum()))
        out[f"{sig_name}_vs_swept_error"] = dict(spearman=rho_sw, ci95=[lo_sw, hi_sw], n=int(mask_sw.sum()))
        print(f"{sig_name:18s} vs mm_error:     rho={rho_mm:+.3f} [{lo_mm:+.3f},{hi_mm:+.3f}] (n={int(mask_mm.sum())})")
        print(f"{sig_name:18s} vs swept_error: rho={rho_sw:+.3f} [{lo_sw:+.3f},{hi_sw:+.3f}] (n={int(mask_sw.sum())})")

        # within-pool Spearman (mean over pools with enough non-degenerate spread)
        within_rhos_mm, within_rhos_sw = [], []
        for p in np.unique(pool_ids):
            m = pool_ids == p
            mm_p, sw_p, sig_p = mm[m], sw[m], sig[m]
            mmask = ~np.isnan(mm_p)
            if mmask.sum() >= 5 and np.std(sig_p[mmask]) > 1e-9 and np.std(mm_p[mmask]) > 1e-9:
                r, _ = spearmanr(sig_p[mmask], mm_p[mmask])
                if r == r: within_rhos_mm.append(r)
            smask = ~np.isnan(sw_p)
            if smask.sum() >= 5 and np.std(sig_p[smask]) > 1e-9 and np.std(sw_p[smask]) > 1e-9:
                r, _ = spearmanr(sig_p[smask], sw_p[smask])
                if r == r: within_rhos_sw.append(r)
        out[f"{sig_name}_within_pool_mean_rho_mm"] = float(np.mean(within_rhos_mm)) if within_rhos_mm else None
        out[f"{sig_name}_within_pool_mean_rho_swept"] = float(np.mean(within_rhos_sw)) if within_rhos_sw else None
        out[f"{sig_name}_within_pool_n_pools_mm"] = len(within_rhos_mm)
        out[f"{sig_name}_within_pool_n_pools_swept"] = len(within_rhos_sw)
        print(f"  within-pool mean rho (mm): {out[f'{sig_name}_within_pool_mean_rho_mm']} "
             f"(n_pools={len(within_rhos_mm)})")
        print(f"  within-pool mean rho (swept): {out[f'{sig_name}_within_pool_mean_rho_swept']} "
             f"(n_pools={len(within_rhos_sw)})")

        # lift: top-10% LEAST confident (highest signal value) vs overall mean error
        n = len(sig)
        thresh = np.percentile(sig, 90)
        least_conf = sig >= thresh
        for err_name, err, mask in (("mm_error", mm, mask_mm), ("swept_error", sw, mask_sw)):
            grp = least_conf & mask
            if grp.sum() > 0 and mask.sum() > 0:
                lift = float(np.mean(err[grp]) / np.mean(err[mask]))
            else:
                lift = None
            out[f"{sig_name}_lift_top10pct_{err_name}"] = lift
            print(f"  lift (top-10% least confident by {sig_name}, {err_name}): {lift}")

    RES.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(RES) + ".tmp"; Path(tmp).write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)
    print("wrote", RES)


if __name__ == "__main__":
    main()
