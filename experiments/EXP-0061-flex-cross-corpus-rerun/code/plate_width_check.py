"""EXP-0061 phase 1 -- validate the FleX pusher plate geometry from the data.

Stated (user, from DS-0019 run_config.json): plate width 2.4 workspace units
(0.1 x global_scale 24), perpendicular to the push. Here every transition is
put into its own push frame (origin = push start, +u along the push, +v
lateral) and we measure, as a function of the INITIAL lateral offset |v0|:

  moved fraction   = P(du > 0.25 | |v0|, 0.15 L < u0 < 0.85 L)   (du = along-push displacement)
  carried fraction = P(du > 0.5 (L - u0) | same)                  (rode along with the plate)

for particles initially inside the swept corridor's along-push extent. The plate
half-width shows up as the knee of these curves. Fitted with a logistic edge
f(|v|) = a / (1 + exp((|v| - h) / w)); h is the apparent half-width, 95 % CI from
a bootstrap over TRANSITIONS (the replication unit). Also measures the stop:
for carried particles with |v0| < 0.8, the distribution of u1 - L (where they
end relative to the push end point; the plate face + one particle radius).

FRAME (found by this check, see DATASET.md): the stored actions' 2nd/4th
components are -z of the particle files. Everything here is in the table
frame X = x_flex, Y = -z_flex (= the action's own frame, right-handed with y
up), via FlexData.dataset.flex_xz_to_table.

Inputs: FlexData caches (FlexData/build_cache.py). DS-0020 uses only push 0 of
every trajectory (all 2000 share one identical uniformly-spread start state,
so this is the cleanest probe); DS-0019 uses every valid row.

    python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/plate_width_check.py
Outputs: figures/data_check/plate_width.png + plate_width.json
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from FlexData.dataset import flex_xz_to_table  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/data_check"
VB = np.arange(0.0, 4.0 + 1e-9, 0.1)          # |v0| bin edges
VC = 0.5 * (VB[1:] + VB[:-1])
R_P = 0.125
STATED_HALF = 1.2


def _tbl(xz16):
    return flex_xz_to_table(xz16.astype(np.float32))


def per_transition_counts(x0, x1, act):
    """Binned (n_all, n_moved, n_carried) over |v0| for one transition, plus
    the stop samples u1 - L of carried central particles, plus signed-v edge
    counts (to check symmetry)."""
    s, e = act[:2], act[2:]
    d = e - s
    L = float(np.hypot(*d))
    if L < 1.5:
        return None
    u_hat = d / L
    v_hat = np.array([-u_hat[1], u_hat[0]])
    r0 = x0 - s
    u0, v0 = r0 @ u_hat, r0 @ v_hat
    dd = x1 - x0
    du = dd @ u_hat
    sel = (u0 > 0.15 * L) & (u0 < 0.85 * L) & np.isfinite(du)
    av = np.abs(v0[sel])
    moved = du[sel] > 0.25
    carried = du[sel] > 0.5 * (L - u0[sel])
    n_all = np.histogram(av, VB)[0]
    n_mov = np.histogram(av[moved], VB)[0]
    n_car = np.histogram(av[carried], VB)[0]
    # signed lateral: moved counts for v>0 vs v<0
    vs = v0[sel]
    n_pos = np.histogram(vs[(vs > 0) & moved], VB)[0]
    n_neg = np.histogram(-vs[(vs < 0) & moved], VB)[0]
    a_pos = np.histogram(vs[vs > 0], VB)[0]
    a_neg = np.histogram(-vs[vs < 0], VB)[0]
    u1 = (x1 - s) @ u_hat
    cen = sel & (np.abs(v0) < 0.8) & (du > 0.5 * (L - u0))
    stop = u1[cen] - L
    return n_all, n_mov, n_car, (n_pos, a_pos, n_neg, a_neg), stop


def logistic(v, a, h, w):
    return a / (1.0 + np.exp((v - h) / w))


def fit_edge(n_all, n_hit):
    f = n_hit / np.maximum(n_all, 1)
    ok = n_all > 20
    p, _ = curve_fit(logistic, VC[ok], f[ok], p0=[1.0, 1.2, 0.1],
                     bounds=([0, 0, 0.005], [1.5, 4, 2]), maxfev=20000)
    return p, f


def collect(name):
    rows = []
    if name == "DS-0020":
        for fn in sorted(glob.glob(str(REPO / "datasets/DS-0020-training-data-flex-N864/old_data/_ported_v1/cache/chunk_*.npz"))):
            z = np.load(fn)
            xz, act = z["xz"], z["actions"]
            for t in range(xz.shape[0]):
                rows.append(per_transition_counts(_tbl(xz[t, 0]), _tbl(xz[t, 1]), act[t, 0]))
    else:
        for fn in sorted(glob.glob(str(REPO / "datasets/DS-0019-slates-flex-pile-varN/cache/state_*.npz"))):
            z = np.load(fn)
            x0 = _tbl(z["init_xz"])
            for a in range(len(z["action_idx"])):
                rows.append(per_transition_counts(x0, _tbl(z["after_xz"][a]), z["actions"][a]))
    return [r for r in rows if r is not None]


def analyse(rows, rng, n_boot=500):
    A = np.stack([r[0] for r in rows]); M = np.stack([r[1] for r in rows]); C = np.stack([r[2] for r in rows])
    p_m, f_m = fit_edge(A.sum(0), M.sum(0))
    p_c, f_c = fit_edge(A.sum(0), C.sum(0))
    bm, bc = [], []
    n = len(rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            bm.append(fit_edge(A[idx].sum(0), M[idx].sum(0))[0][1])
            bc.append(fit_edge(A[idx].sum(0), C[idx].sum(0))[0][1])
        except RuntimeError:
            pass
    # 50 % crossing of the moved fraction (relative to its plateau), interpolated
    plateau = f_m[VC < 0.8].mean()
    cross = float(np.interp(-0.5 * plateau, -f_m, VC))
    sym = [np.stack([r[3][i] for r in rows]).sum(0) for i in range(4)]
    f_pos, f_neg = sym[0] / np.maximum(sym[1], 1), sym[2] / np.maximum(sym[3], 1)
    p_pos, _ = fit_edge(sym[1], sym[0]); p_neg, _ = fit_edge(sym[3], sym[2])
    stop = np.concatenate([r[4] for r in rows])
    return dict(
        n_transitions=n,
        moved=dict(h=float(p_m[1]), w=float(p_m[2]), a=float(p_m[0]),
                   h_ci95=[float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                   half_plateau_crossing=cross, plateau=float(plateau)),
        carried=dict(h=float(p_c[1]), w=float(p_c[2]), a=float(p_c[0]),
                     h_ci95=[float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5))]),
        symmetry=dict(h_pos_v=float(p_pos[1]), h_neg_v=float(p_neg[1])),
        stop_u1_minus_L=dict(n=int(stop.size),
                             pct=dict(zip(["p1", "p5", "p25", "p50", "p75", "p95"],
                                          [float(x) for x in np.percentile(stop, [1, 5, 25, 50, 75, 95])]))),
        _curves=dict(f_moved=f_m, f_carried=f_c, f_pos=f_pos, f_neg=f_neg, stop=stop),
    )


def main():
    rng = np.random.default_rng(0)
    res = {}
    for name in ("DS-0020", "DS-0019"):
        rows = collect(name)
        res[name] = analyse(rows, rng)
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "_curves"}, indent=1), flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, ls in (("DS-0020", "-"), ("DS-0019", "--")):
        cv = res[name]["_curves"]
        ax[0].plot(VC, cv["f_moved"], ls, label=f"{name} moved (du>0.25)")
        ax[0].plot(VC, cv["f_carried"], ls, label=f"{name} carried (du>0.5(L-u0))")
        ax[1].plot(VC, cv["f_pos"], ls, label=f"{name} +v")
        ax[1].plot(VC, cv["f_neg"], ls, label=f"{name} -v")
        ax[2].hist(cv["stop"], bins=np.arange(-1, 3, 0.05), histtype="step", density=True, label=name)
    for a in ax[:2]:
        a.axvline(STATED_HALF, color="k", lw=1, label="stated half-width 1.2")
        a.axvline(STATED_HALF + R_P, color="gray", lw=1, ls=":", label="1.2 + particle r")
        a.set_xlabel("|v0| initial lateral offset (FleX units)"); a.legend(fontsize=7)
    ax[0].set_ylabel("fraction of particles"); ax[0].set_title("moved / carried fraction vs |v0|")
    ax[1].set_title("moved fraction, +v vs -v side")
    ax[2].set_title("carried central particles: u1 - L (stop)"); ax[2].set_xlabel("u1 - L"); ax[2].legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "plate_width.png", dpi=110)
    out = {k: {kk: vv for kk, vv in v.items() if kk != "_curves"} for k, v in res.items()}
    out["stated"] = dict(width=2.4, half_width=STATED_HALF, particle_r=R_P)
    out["method"] = __doc__.split("\n\n")[1]
    tmp = OUT / "plate_width.json.tmp"
    tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, OUT / "plate_width.json")


if __name__ == "__main__":
    main()
