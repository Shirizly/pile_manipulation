"""Does the model you'd pick for MPC differ from the one you'd pick for accuracy?

EXP-0008 found that control-relevant ranking is destroyed by VARIANCE (noise
drawn independently per candidate) and barely touched by BIAS (amplitude, blur
and displacement errors, which perturb every candidate alike and cancel when
candidates are compared). If that mechanism is right, then shrinking the
operator harder -- which trades variance for bias -- should keep helping control
after it has started hurting per-pixel accuracy.

So: sweep the ridge strength, and read off the two optima.

    PYTHONPATH=. python scripts/probes/shrinkage_vs_control.py
"""
from __future__ import annotations

import argparse

import torch

from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (actions_to_pixels, canonicalise, fit_operator,
                                  metrics, swept_region_mask)
from transforms.functional import (blend_push_prediction, from_push_frame,
                                   push_frame_validity_mask)
from control_utility_test import lyapunov_weights, lyapunov, rank_metrics

H = W = 64


def hf_fraction(delta):
    """Share of the predicted delta's energy above the pixel scale.

    The mechanism diagnostic. A 4-neighbour Laplacian is the cheapest
    high-pass there is, and its ratio to total energy says how much of what the
    model predicts lives in the band that does not survive being compared
    across candidates.
    """
    lap = (4 * delta[:, 1:-1, 1:-1] - delta[:, :-2, 1:-1] - delta[:, 2:, 1:-1]
           - delta[:, 1:-1, :-2] - delta[:, 1:-1, 2:])
    return float(lap.pow(2).sum() / delta.pow(2).sum().clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="Genesis/data/cube_spectrum/n20/*_data.pt")
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--crop", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--cube-size", type=float, default=0.005)
    ap.add_argument("--goals", default="center,corner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambdas", default="1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1e3,1e4,1e5")
    ap.add_argument("--quick", action="store_true", help="one lambda, the kill-check")
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64,128,256,512")
    a = ap.parse_args()

    o0, o1, act, ep, _, _ = load_transition_fields(
        a.glob, 64, a.blur, "mean", 19.9, "cpu", view="mask",
        min_grains=1.0, cube_size=a.cube_size)
    M = o0.shape[0]
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(a.seed)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps) // 4)].tolist())
    te = torch.tensor([int(e) in val for e in ep]); tr = ~te
    n_tr, n_te = int(tr.sum()), int(te.sum())
    print(f"{n_tr} train / {n_te} val ({len(eps)} episodes, {len(val)} held out)")

    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(n_tr, -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(n_tr, -1).T
    D = Y0.shape[0]
    print(f"operator {D}x{D}, M/D = {n_tr / D:.2f}")

    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5 * plate + 2.0, 0.5 * plate)
    base = metrics(ote, o1te, ote, region=region)["rms"]
    Y0te = canonicalise(ote, ste, ete, a.res, a.crop).reshape(n_te, -1).T
    msk = push_frame_validity_mask(ste, ete, (H, W), (a.res, a.res), a.crop)

    def to_world(Yp):
        back = from_push_frame(Yp.T.reshape(-1, a.res, a.res), ste, ete, (H, W), a.crop)
        return blend_push_prediction(back, ote, msk).clamp_min(0.0)

    goals = [s.strip() for s in a.goals.split(",") if s.strip()]
    dws = {gl: lyapunov_weights((H, W), gl, "cpu") for gl in goals}
    v0 = {gl: lyapunov(ote, dws[gl]) for gl in goals}
    dv_true = {gl: lyapunov(o1te, dws[gl]) - v0[gl] for gl in goals}
    # Control variables for the partial correlation: the state's own cost, and
    # how much pile is in the blade's path. Candidates come from different
    # states, so without this a model that only tracked "which state was easy"
    # would score well while being useless for choosing an action.
    contact = (region * ote).sum(dim=(1, 2))

    rows = []

    def score(name, pred, extra=""):
        r = {"name": name,
             "rms_pct": 100 * metrics(pred, o1te, ote, region=region)["rms"] / base,
             "hf": hf_fraction(pred - ote), "extra": extra}
        for gl in goals:
            dvp = lyapunov(pred, dws[gl]) - v0[gl]
            pear, spear, sign, slate, part = rank_metrics(
                dvp, dv_true[gl], control=[v0[gl], contact])
            r[f"spear_{gl}"] = spear
            r[f"slate4_{gl}"] = slate[4]
            r[f"part_{gl}"] = part
        rows.append(r)
        return r

    score("persistence", ote)   # ranking columns are meaningless: dV_pred == 0
    score("oracle", o1te)
    bmd = (Y1 - Y0).mean(1, keepdim=True)
    score("mean-delta", to_world(Y0te + bmd))
    score("identity (warp only)", to_world(Y0te))

    lams = [float(x) for x in a.lambdas.split(",")]
    if a.quick:
        lams = [1.0]
    # The Gram matrices do not depend on lambda; recomputing them per lambda is
    # the whole cost of a sweep. Hoisting them makes 13 shrinkage levels cost
    # one Gram plus 13 solves.
    import time
    t0 = time.time()
    G0 = Y0 @ Y0.T
    C0 = Y1 @ Y0.T
    eye = torch.eye(D)
    print(f"gram {time.time() - t0:.1f}s")
    for lam in lams:
        t0 = time.time()
        A = torch.linalg.solve((G0 + lam * eye).T, (C0 + lam * eye).T).T
        dev = float((A - eye).norm() / eye.norm())
        score(f"ridge {lam:g}", to_world(A @ Y0te), extra=f"|A-I|/|I|={dev:.3f}")
        print(f"  lambda={lam:g} in {time.time() - t0:.1f}s")

    # Rank truncation is the OTHER knob, and on EXP-0008's mechanism it is the
    # RIGHT one: shrinking toward identity rescales the predicted delta, i.e. it
    # is an AMPLITUDE change, and EXP-0008 measured amplitude error as free for
    # control. Truncating rank removes prediction VARIANCE, which is the thing
    # that mechanism says destroys ranking.
    if not a.quick and a.ranks:
        lam0 = 10.0
        A = torch.linalg.solve((G0 + lam0 * eye).T, (C0 + lam0 * eye).T).T
        P = A @ Y0
        U, S, _ = torch.linalg.svd(P @ P.T)
        for r in [int(x) for x in a.ranks.split(",")]:
            if r >= D:
                continue
            Ur = U[:, :r]
            Ar = Ur @ (Ur.T @ A)
            dev = float((Ar - eye).norm() / eye.norm())
            score(f"rank {r} (lam {lam0:g})", to_world(Ar @ Y0te),
                  extra=f"|A-I|/|I|={dev:.3f}")

    hdr = f"{'model':22s} {'rms%':>7s} {'HF':>6s} {'|A-I|':>12s}"
    for gl in goals:
        hdr += f" {'sp_' + gl[:4]:>9s} {'sl4_' + gl[:4]:>10s} {'pa_' + gl[:4]:>9s}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        # A blank cell here shifted the columns and produced a mistranscribed
        # table in EXP-0013; always print something.
        extra = r["extra"][-12:] if r["extra"] else "-"
        line = f"{r['name']:22s} {r['rms_pct']:7.1f} {r['hf']:6.3f} {extra:>12s}"
        for gl in goals:
            line += (f" {r[f'spear_{gl}']:9.3f} {r[f'slate4_{gl}']:10.3f}"
                     f" {r[f'part_{gl}']:9.3f}")
        print(line)

    rk = [r for r in rows if r["name"].startswith("rank")]
    if rk:
        print("\n=== optima over the RANK sweep (the variance knob) ===")
        b = min(rk, key=lambda r: r["rms_pct"])
        print(f"rank* by rms            : {b['name']:20s} ({b['rms_pct']:.1f}%)")
        for gl in goals:
            b4 = max(rk, key=lambda r: r[f"slate4_{gl}"])
            print(f"rank* by slate4  [{gl:6s}]: {b4['name']:20s} "
                  f"({b4[f'slate4_{gl}']:.3f}, rms {b4['rms_pct']:.1f}%)")

    ridge = [r for r in rows if r["name"].startswith("ridge")]
    if len(ridge) > 1:
        print("\n=== optima over the ridge sweep ===")
        best_rms = min(ridge, key=lambda r: r["rms_pct"])
        print(f"lambda* by rms          : {best_rms['name']:14s} ({best_rms['rms_pct']:.1f}%)")
        for gl in goals:
            b_sp = max(ridge, key=lambda r: r[f"spear_{gl}"])
            b_sl = max(ridge, key=lambda r: r[f"slate4_{gl}"])
            print(f"lambda* by dV Spearman  [{gl:6s}]: {b_sp['name']:14s} "
                  f"({b_sp[f'spear_{gl}']:.3f}, rms {b_sp['rms_pct']:.1f}%)")
            print(f"lambda* by slate4       [{gl:6s}]: {b_sl['name']:14s} "
                  f"({b_sl[f'slate4_{gl}']:.3f}, rms {b_sl['rms_pct']:.1f}%)")


if __name__ == "__main__":
    main()
