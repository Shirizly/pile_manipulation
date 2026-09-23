"""EXP-0019 diagnostics:
 (a) latent R^2 on the FILE-DISJOINT test split as a function of ridge lambda,
     to check whether the row-random inner val split used for lambda selection
     leaks (train/val R^2 ~0.32 vs test ~0.03 says it does);
 (b) the SAME closed-form hard-gate fit run on EXP-0016's FROZEN RANDOM
     encoder latents, so the fit method is held fixed and only the encoder
     varies.  Latent R^2 is NOT comparable across encoders (the target and the
     denominator both move) -- this is reported as a diagnostic of whether the
     pipeline is broken, not as a ranking of encoders.
"""
import sys, json
from pathlib import Path
import torch
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_switched_hard import ridge_fit, feats, r2, fit_bins, predict   # noqa
from Baselines.LinearForesight.model import bin_index                    # noqa
DEV = "cuda"
LAMS = [1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5]

def run(tag, d):
    tr = torch.load(d/"latents_train.pt"); te = torch.load(d/"latents_test.pt")
    ztr, atr = tr["z0"].double().to(DEV), tr["a"].double().to(DEV)
    zte, ate = te["z0"].double().to(DEV), te["a"].double().to(DEV)
    dztr = (tr["z1"]-tr["z0"]).double().to(DEV); dzte = (te["z1"]-te["z0"]).double().to(DEV)
    assert ztr.device.type == "cuda", ztr.device
    mu = ztr.mean(0, keepdim=True)
    if "length_m" in tr:
        Ltr, Lte = tr["length_m"].to(DEV), te["length_m"].to(DEV)
    else:
        Ltr = (atr[:, 4:6]*0.064).norm(dim=-1); Lte = (ate[:, 4:6]*0.064).norm(dim=-1)
    edges = torch.linspace(0., float(Ltr.max()), 7)
    btr, bte = bin_index(Ltr, edges.to(DEV)), bin_index(Lte, edges.to(DEV))
    out = {"tag": tag, "n_train": ztr.shape[0], "n_test": zte.shape[0],
           "mean_norm": float(mu.norm()), "centred_rms": float((ztr-mu).pow(2).mean().sqrt()),
           "dz_rms": float(dztr.pow(2).mean().sqrt()), "rows": []}
    for centred in (True, False):
        Z, Zt = (ztr-mu, zte-mu) if centred else (ztr, zte)
        for lam in LAMS:
            Wg = ridge_fit(feats(Z, atr), dztr, lam)
            Ws, cnt = fit_bins(Z, atr, dztr, btr, 6, lam, Wg)
            out["rows"].append(dict(centred=centred, lam=lam,
                g_tr=r2(feats(Z,atr)@Wg, dztr), g_te=r2(feats(Zt,ate)@Wg, dzte),
                s_tr=r2(predict(Ws,Z,atr,btr), dztr), s_te=r2(predict(Ws,Zt,ate,bte), dzte),
                counts=cnt))
    print(f"\n== {tag}  n_train {out['n_train']} ||mean z|| {out['mean_norm']:.3f} "
          f"centred_rms {out['centred_rms']:.4f} ==", flush=True)
    print("  centred  lambda    global_train global_test  switched_train switched_test")
    for r in out["rows"]:
        print(f"  {str(r['centred']):5s}  {r['lam']:>8g}   {r['g_tr']:+.4f}      {r['g_te']:+.4f}"
              f"       {r['s_tr']:+.4f}        {r['s_te']:+.4f}", flush=True)
    print("  bin counts (train):", out["rows"][0]["counts"], flush=True)
    return out

res = [run("EXP-0019 LeJEPA-TRAINED encoder", Path(sys.argv[1])),
       run("EXP-0016 FROZEN RANDOM encoder", Path(sys.argv[2]))]
json.dump(res, open(sys.argv[3], "w"), indent=2)
