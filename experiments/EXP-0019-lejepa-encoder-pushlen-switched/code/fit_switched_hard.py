"""EXP-0019 Stage 2b: residual switched-linear latent dynamics with a HARD gate
on PUSH LENGTH.

    z' = z + (A_b z + B_b a + c_b),   b = bin_index(push_length(a))

The bin is a KNOWN function of the action -- nothing about the gating is
learned.  This deliberately REPLACES the design doc's section-10 soft softmax
gate (and EXP-0016's `SwitchedLinearDynamics`, which implements it).  The bin
scheme is the repo's existing one: `Baselines/LinearForesight/model.py::bin_index`
(bucketize on INTERIOR edges only, so out-of-range clamps to the end bins) with
`fit_switched.py`'s 6 equal-width bins over [0, max] and `MIN_ROWS_PER_BIN = 50`.

Fit is closed-form ridge (toward zero) of dz on [z, a, 1], per bin, plus the
mandatory single global operator (one bin) as reference, plus the mandatory
"do nothing" baseline dz = 0.

METRIC: latent R^2 = 1 - mse(dz_pred - dz_true) / mse(dz_true), i.e. skill
against a `Delta z = 0` persistence-in-latent baseline IN THE SAME LATENT
SPACE.  0 = no better than assuming the push did nothing.  COMPARABLE ONLY
WITHIN ONE ENCODER -- a different encoder moves both the target and the
denominator.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from Baselines.LinearForesight.model import bin_index       # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MIN_ROWS_PER_BIN = 50      # Baselines/LinearForesight/fit_switched.py


def ridge_fit(X, Y, lam):
    """X (n,f) -> Y (n,d), ridge toward zero.  Returns W (f,d)."""
    XtX = X.T @ X
    XtX += lam * torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    return torch.linalg.solve(XtX, X.T @ Y)


def feats(z, a):
    return torch.cat([z, a, torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)], 1)


def r2(pred, true):
    return 1.0 - float(((pred - true) ** 2).sum() / (true ** 2).sum())


def fit_bins(ztr, atr, dztr, bins_tr, n_bins, lam, W_global):
    Ws, counts = [], []
    for b in range(n_bins):
        m = bins_tr == b
        n_b = int(m.sum())
        counts.append(n_b)
        if n_b < MIN_ROWS_PER_BIN:
            Ws.append(W_global.clone())     # fall back to the global operator
            continue
        Ws.append(ridge_fit(feats(ztr[m], atr[m]), dztr[m], lam))
    return Ws, counts


def predict(Ws, z, a, bins):
    out = torch.zeros(z.shape[0], z.shape[1], device=z.device, dtype=z.dtype)
    F = feats(z, a)
    for b, W in enumerate(Ws):
        m = bins == b
        if int(m.sum()) == 0:
            continue
        out[m] = F[m] @ W
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    ap.add_argument("--no-centre", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    L = Path(args.latents)
    tr = torch.load(L / "latents_train.pt"); te = torch.load(L / "latents_test.pt")
    ztr, atr = tr["z0"].double().to(DEV), tr["a"].double().to(DEV)
    zte, ate = te["z0"].double().to(DEV), te["a"].double().to(DEV)
    dztr = (tr["z1"] - tr["z0"]).double().to(DEV)
    dzte = (te["z1"] - te["z0"]).double().to(DEV)
    assert ztr.device.type == DEV.split(":")[0], ztr.device
    Ltr, Lte = tr["length_m"].to(DEV), te["length_m"].to(DEV)

    mu = ztr.mean(0, keepdim=True)
    stats = dict(mean_norm=float(mu.norm()),
                 centred_rms=float((ztr - mu).pow(2).mean().sqrt()),
                 perdim_std_mean=float(ztr.std(0).mean()),
                 perdim_std_min=float(ztr.std(0).min()),
                 perdim_std_max=float(ztr.std(0).max()),
                 dz_rms=float(dztr.pow(2).mean().sqrt()))
    sv = torch.linalg.svdvals(ztr - mu)
    p = (sv ** 2) / (sv ** 2).sum(); p = p[p > 0]
    stats["effective_rank"] = float(torch.exp(-(p * p.log()).sum()))
    print("latent stats:", json.dumps(stats, indent=2), flush=True)
    if not args.no_centre:
        ztr = ztr - mu; zte = zte - mu

    # bin edges: 6 equal-width bins over [0, max push length], from TRAIN
    hi = float(Ltr.max())
    edges = torch.linspace(0.0, hi, args.n_bins + 1)
    btr = bin_index(Ltr, edges.to(DEV)); bte = bin_index(Lte, edges.to(DEV))
    print(f"push length train 0-{hi*1000:.2f} mm -> {args.n_bins} equal-width bins", flush=True)

    # inner split for lambda (rows of overnight_randlen are INDEPENDENT
    # transitions, not a slate corpus, so row-random folds are safe here --
    # the slate-aware-fold trap applies to DS-0001, not this corpus)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(ztr.shape[0], generator=g).to(DEV)
    nv = ztr.shape[0] // 5
    vi, fi = perm[:nv], perm[nv:]

    lams = [float(x) for x in args.lams.split(",")]
    best = {}
    for name in ("global", "switched"):
        sc = []
        for lam in lams:
            if name == "global":
                W = ridge_fit(feats(ztr[fi], atr[fi]), dztr[fi], lam)
                pv = feats(ztr[vi], atr[vi]) @ W
            else:
                Wg = ridge_fit(feats(ztr[fi], atr[fi]), dztr[fi], lam)
                Ws, _ = fit_bins(ztr[fi], atr[fi], dztr[fi], btr[fi], args.n_bins, lam, Wg)
                pv = predict(Ws, ztr[vi], atr[vi], btr[vi])
            sc.append((r2(pv, dztr[vi]), lam))
        best[name] = max(sc)[1]
        print(f"  lambda sweep {name}: " + ", ".join(f"{l:g}:{s:+.4f}" for s, l in sc)
              + f"  -> lambda={best[name]:g}", flush=True)

    rows = [dict(cell="dz=0 (do nothing, in-latent persistence)",
                 latent_r2_test=r2(torch.zeros_like(dzte), dzte),
                 latent_r2_train=r2(torch.zeros_like(dztr), dztr))]

    W_global = ridge_fit(feats(ztr, atr), dztr, best["global"])
    rows.append(dict(cell="global (single operator, one bin)", lam=best["global"],
                     latent_r2_test=r2(feats(zte, ate) @ W_global, dzte),
                     latent_r2_train=r2(feats(ztr, atr) @ W_global, dztr)))

    Wg_s = ridge_fit(feats(ztr, atr), dztr, best["switched"])
    Ws, counts = fit_bins(ztr, atr, dztr, btr, args.n_bins, best["switched"], Wg_s)
    te_counts = [int((bte == b).sum()) for b in range(args.n_bins)]
    per_bin = []
    for b in range(args.n_bins):
        m = bte == b
        lo_, hi_ = float(edges[b]) * 1000, float(edges[b + 1]) * 1000
        rec = dict(bin=b, mm=[round(lo_, 2), round(hi_, 2)], n_train=counts[b],
                   n_test=te_counts[b],
                   fell_back_to_global=counts[b] < MIN_ROWS_PER_BIN,
                   frob_vs_global=float((Ws[b] - Wg_s).norm()),
                   frob_self=float(Ws[b].norm()))
        if int(m.sum()) > 0:
            rec["latent_r2_test_this_bin"] = r2(feats(zte[m], ate[m]) @ Ws[b], dzte[m])
            rec["latent_r2_test_this_bin_global_op"] = r2(feats(zte[m], ate[m]) @ W_global, dzte[m])
        per_bin.append(rec)
        print(f"  bin {b} [{lo_:.1f},{hi_:.1f}) mm: n_train {counts[b]} n_test {te_counts[b]}"
              + ("  <-- STARVED, fell back to global" if rec["fell_back_to_global"] else "")
              + (f"  R2 {rec.get('latent_r2_test_this_bin', float('nan')):+.4f}"
                 f" (global op {rec.get('latent_r2_test_this_bin_global_op', float('nan')):+.4f})"
                 if int(m.sum()) > 0 else "  (no test rows)"), flush=True)

    rows.append(dict(cell=f"switched HARD, {args.n_bins} push-length bins", lam=best["switched"],
                     latent_r2_test=r2(predict(Ws, zte, ate, bte), dzte),
                     latent_r2_train=r2(predict(Ws, ztr, atr, btr), dztr)))

    # are the per-bin operators actually DIFFERENT from each other?
    pair = [[round(float((Ws[i] - Ws[j]).norm() / Ws[i].norm()), 4) for j in range(args.n_bins)]
            for i in range(args.n_bins)]
    ident = [[bool(torch.equal(Ws[i], Ws[j])) for j in range(args.n_bins)] for i in range(args.n_bins)]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"W_global": W_global.cpu(), "W_bins": [w.cpu() for w in Ws],
                "bin_edges": edges, "mu": mu.cpu(), "centred": not args.no_centre,
                "lam": best, "counts_train": counts, "counts_test": te_counts,
                "args": vars(args), "latents": str(L),
                "encoder_config": tr.get("encoder_config"), "encoder_ckpt": tr.get("ckpt")},
               out / f"operators{args.tag}.pt")
    res = dict(metric="latent_r2 = 1 - mse(dz_pred - dz_true)/mse(dz_true), vs Delta z = 0 "
                      "in the SAME latent space; comparable ONLY within one encoder",
               centred=not args.no_centre, latent_stats=stats, cells=rows, per_bin=per_bin,
               pairwise_rel_frob=pair, pairwise_byte_identical=ident,
               bin_edges_mm=[round(float(e) * 1000, 3) for e in edges], args=vars(args))
    json.dump(res, open(out / f"switched_metrics{args.tag}.json", "w"), indent=2)
    print("\n== latent R2 (test, vs dz=0) ==", flush=True)
    for r in rows:
        print(f"  {r['cell']:48s} {r['latent_r2_test']:+.4f}  (train {r['latent_r2_train']:+.4f})",
              flush=True)


if __name__ == "__main__":
    main()
