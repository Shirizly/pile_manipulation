"""EXP-0020: lambda selection for the latent switched-linear fit using a
FILE-DISJOINT inner validation split, instead of EXP-0019's row-random one.

EXP-0019's `fit_switched_hard.py` selects the ridge lambda on a row-random 20%
inner split of `overnight_randlen_train`.  That inner score read +0.359 where
the true held-out (file-disjoint test corpus) score was +0.060, and it picked
lambda=1 where 10-100 was better on held-out data.  Rows of one `_data.pt`
file share a spawn configuration and a simulation batch, so they are NOT
independent across files.

This script imports the fit/predict/metric functions from
`fit_switched_hard.py` UNCHANGED and only replaces the inner split: folds are
whole files.  `load_transitions` concatenates files in sorted order with a
fixed rows-per-file, so the group id of row i is i // rows_per_file, asserted
against the file count.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
E19 = HERE.parents[1] / "EXP-0019-lejepa-encoder-pushlen-switched/code"
sys.path.insert(0, str(E19))
sys.path.insert(0, str(HERE.parents[2]))
import fit_switched_hard as F                                  # noqa: E402
from data import randlen_files                                 # noqa: E402
from Baselines.LinearForesight.model import bin_index          # noqa: E402

DEV = F.DEV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=6)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    L = Path(args.latents)
    tr = torch.load(L / "latents_train.pt"); te = torch.load(L / "latents_test.pt")
    ztr, atr = tr["z0"].double().to(DEV), tr["a"].double().to(DEV)
    zte, ate = te["z0"].double().to(DEV), te["a"].double().to(DEV)
    dztr = (tr["z1"] - tr["z0"]).double().to(DEV)
    dzte = (te["z1"] - te["z0"]).double().to(DEV)
    Ltr, Lte = tr["length_m"].to(DEV), te["length_m"].to(DEV)
    n = ztr.shape[0]

    files = randlen_files("train")
    nf = len(files)
    assert n % nf == 0, (n, nf)
    rpf = n // nf
    groups = torch.arange(n, device=DEV) // rpf
    print(f"train rows {n}, files {nf}, rows/file {rpf} -> {int(groups.max())+1} groups", flush=True)

    mu = ztr.mean(0, keepdim=True)
    ztr = ztr - mu; zte = zte - mu
    hi = float(Ltr.max())
    edges = torch.linspace(0.0, hi, args.n_bins + 1)
    btr = bin_index(Ltr, edges.to(DEV)); bte = bin_index(Lte, edges.to(DEV))

    # file-disjoint folds: file f -> fold f % n_folds
    fold_of_row = groups % args.n_folds
    lams = [float(x) for x in args.lams.split(",")]
    best, sweeps = {}, {}
    for name in ("global", "switched"):
        sc = []
        for lam in lams:
            tot_num = tot_den = 0.0
            for k in range(args.n_folds):
                vm = fold_of_row == k; fm = ~vm
                Wg = F.ridge_fit(F.feats(ztr[fm], atr[fm]), dztr[fm], lam)
                if name == "global":
                    pv = F.feats(ztr[vm], atr[vm]) @ Wg
                else:
                    Ws, _ = F.fit_bins(ztr[fm], atr[fm], dztr[fm], btr[fm], args.n_bins, lam, Wg)
                    pv = F.predict(Ws, ztr[vm], atr[vm], btr[vm])
                tot_num += float(((pv - dztr[vm]) ** 2).sum())
                tot_den += float((dztr[vm] ** 2).sum())
            sc.append((1.0 - tot_num / tot_den, lam))
        best[name] = max(sc)[1]
        sweeps[name] = [dict(lam=l, grouped_cv_latent_r2=s) for s, l in sc]
        print(f"  GROUPED lambda sweep {name}: " + ", ".join(f"{l:g}:{s:+.4f}" for s, l in sc)
              + f"  -> lambda={best[name]:g}", flush=True)

    rows = [dict(cell="dz=0 (do nothing, in-latent persistence)",
                 latent_r2_test=F.r2(torch.zeros_like(dzte), dzte))]
    Wg = F.ridge_fit(F.feats(ztr, atr), dztr, best["global"])
    rows.append(dict(cell="global (single operator, one bin)", lam=best["global"],
                     latent_r2_test=F.r2(F.feats(zte, ate) @ Wg, dzte)))
    Wg_s = F.ridge_fit(F.feats(ztr, atr), dztr, best["switched"])
    Ws, counts = F.fit_bins(ztr, atr, dztr, btr, args.n_bins, best["switched"], Wg_s)
    rows.append(dict(cell=f"switched HARD, {args.n_bins} push-length bins", lam=best["switched"],
                     latent_r2_test=F.r2(F.predict(Ws, zte, ate, bte), dzte)))
    per_bin = []
    for b in range(args.n_bins):
        m = bte == b
        per_bin.append(dict(bin=b, mm=[round(float(edges[b])*1000, 2), round(float(edges[b+1])*1000, 2)],
                            n_train=counts[b], n_test=int(m.sum()),
                            latent_r2_test_this_bin=F.r2(F.feats(zte[m], ate[m]) @ Ws[b], dzte[m]) if int(m.sum()) else None))
        print(f"  bin {b} n_train {counts[b]} n_test {per_bin[-1]['n_test']} "
              f"latent_r2 {per_bin[-1]['latent_r2_test_this_bin']}", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(W_global=Wg.cpu(), W_bins=[w.cpu() for w in Ws], bin_edges=edges,
                    mu=mu.cpu(), centred=True, lam=best, counts_train=counts,
                    args=vars(args), latents=str(L),
                    encoder_config=tr.get("encoder_config"), encoder_ckpt=tr.get("ckpt"),
                    note="lambda selected by FILE-DISJOINT grouped inner CV (EXP-0020)"),
               out / "operators_grouped.pt")
    json.dump(dict(metric="latent_r2 vs Delta z = 0, same latent space; comparable ONLY within one encoder",
                   lambda_selection="file-disjoint grouped inner CV, 5 folds over train files",
                   lam=best, sweeps=sweeps, cells=rows, per_bin=per_bin),
              open(out / "switched_metrics_grouped.json", "w"), indent=2)
    print("\n== latent R2 (test, vs dz=0), GROUPED lambda selection ==", flush=True)
    for r in rows:
        print(f"  {r['cell']:48s} {r['latent_r2_test']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
