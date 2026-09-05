"""EXP-0019 probes for C-007 / C-009.

C-007 ("essentially all the nonlinearity is one variable: how much material
the blade meets") and C-009 ("84% predictable, 58% linearly") were computed by
`variance_decomposition.py` / `density_stratified.py` and never re-checked.
This script runs three adversarial tests against them, reusing those scripts'
own feature/target construction so numbers are directly comparable:

  1. trivial-baseline: is C-009's headline R^2 dominated by the near-
     deterministic zero-contact tail (~4.6% of transitions where the band is
     empty and displacement is exactly 0)?
  2. booster-tuning: does C-007's "linear matches/beats boosted within a
     contact stratum" survive a far more heavily tuned booster (4x the trees,
     depth 6, early stopping), and does it hold in absolute RMSE terms (not
     just the R^2 ratio, which is unstable when the denominator shrinks)?
  3. target-sensitivity: does the same within-stratum match hold for a
     threshold/outlier-sensitive target (max per-particle displacement in the
     band) rather than the mean, which averages across ~15-25 particles and
     could smooth away exactly the contact-decision nonlinearity the granular
     literature predicts?

Usage: python -u scripts/probes/exp0019_contact_linearity.py [--glob ...]
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from variance_decomposition import DEFAULT_GLOB, load, features, r2, MM

WEAK = dict(max_iter=250, learning_rate=0.06)
STRONG = dict(max_iter=1000, learning_rate=0.03, max_depth=6, l2_regularization=0.0,
              early_stopping=True, n_iter_no_change=30, validation_fraction=0.15)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit(X, yy, groups, booster_kwargs, n_splits=5, return_sd=False):
    gkf = GroupKFold(n_splits=n_splits)
    rl, rg, el, eg = [], [], [], []
    for tr, te in gkf.split(X, yy, groups=groups):
        sc = X[tr].std(0); sc[sc == 0] = 1
        mu = X[tr].mean(0)
        A, B = (X[tr] - mu) / sc, (X[te] - mu) / sc
        pl = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(A, yy[tr]).predict(B)
        pg = HistGradientBoostingRegressor(random_state=0, **booster_kwargs).fit(A, yy[tr]).predict(B)
        rl.append(r2(yy[te], pl)); rg.append(r2(yy[te], pg))
        el.append(rmse(yy[te], pl)); eg.append(rmse(yy[te], pg))
    if return_sd:
        return (np.mean(rl), np.mean(rg), np.mean(el), np.mean(eg),
                np.std(rl), np.std(rg))
    return np.mean(rl), np.mean(rg), np.mean(el), np.mean(eg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    a = ap.parse_args()

    s0, s1, quat, ps, pe, run, nf = load(a.glob)
    Xo, Xp, Xq, inband = features(s0, quat, ps, pe)
    runs = run.numpy()
    cnt = inband.float().sum(1).clamp_min(1)
    disp = (s1 - s0)[..., :2].norm(dim=-1)
    y_mean = ((disp * inband.float()).sum(1) / cnt * MM).numpy()
    y_max = ((disp * inband.float()).max(dim=1).values * MM).numpy()
    n_in_band = inband.float().sum(1).numpy()
    print(f"n={len(y_mean)} from {nf} runs")

    print("\n=== TEST 1: trivial-baseline (C-009) ===")
    frac0 = (n_in_band == 0).mean()
    print(f"fraction n_in_band==0: {frac0:.3f}; y|contact==0 mean/sd: "
          f"{y_mean[n_in_band==0].mean():.3f}/{y_mean[n_in_band==0].std():.3f}")
    L, G, _, _ = fit(n_in_band.reshape(-1, 1), y_mean, runs, WEAK)
    print(f"n_in_band alone (1 feature): linear R2={L:.3f} boosted R2={G:.3f}")
    m = n_in_band > 0
    Lf, Gf, _, _ = fit(Xo, y_mean, runs, WEAK)
    Lc, Gc, _, _ = fit(Xo[m], y_mean[m], runs[m], WEAK)
    print(f"OCC full set,  all n={len(y_mean)}: linear R2={Lf:.3f} boosted R2={Gf:.3f} "
          f"(share {100*Lf/Gf:.0f}%)")
    print(f"OCC full set, contact>0 n={m.sum()}: linear R2={Lc:.3f} boosted R2={Gc:.3f} "
          f"(share {100*Lc/Gc:.0f}%)")

    print("\n=== TEST 2+3: booster-tuning and target-sensitivity (C-007) ===")
    qs = np.quantile(n_in_band, np.linspace(0, 1, 5))
    for tname, y in [("mean disp (original C-007 target)", y_mean),
                     ("max disp (threshold-sensitive target)", y_max)]:
        print(f"\n--- target: {tname} ---")
        print(f"{'stratum':>8s} {'n':>5s} {'sd(y)':>6s} | {'Lr2':>6s} {'(sd)':>5s} {'Gr2(weak)':>10s} {'(sd)':>5s} "
              f"{'share':>6s} {'RMSE_L':>7s} {'RMSE_G':>7s} | {'Gr2(strong)':>11s} {'share_s':>8s}")
        for i in range(4):
            mm = (n_in_band >= qs[i]) & ((n_in_band <= qs[i+1]) if i == 3 else (n_in_band < qs[i+1]))
            Xm, ym, rm = Xo[mm], y[mm], runs[mm]
            Lw, Gw, El, Eg, sdL, sdG = fit(Xm, ym, rm, WEAK, return_sd=True)
            Ls, Gs, _, _ = fit(Xm, ym, rm, STRONG)
            sw = 100 * Lw / Gw if Gw > 0.02 else float('nan')
            ss = 100 * Ls / Gs if Gs > 0.02 else float('nan')
            print(f"{i:8d} {mm.sum():5d} {ym.std():6.2f} | {Lw:6.3f} {sdL:5.3f} {Gw:10.3f} {sdG:5.3f} "
                  f"{sw:5.0f}% {El:7.3f} {Eg:7.3f} | {Gs:11.3f} {ss:7.0f}%")


if __name__ == "__main__":
    main()
