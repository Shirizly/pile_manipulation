"""EXP-0023 gate 1: does the value objective actually carry gradient back to
the action parameters, for EVERY arm?

A silently-zero gradient produces plausible-looking optimisation numbers that
mean nothing, so this runs BEFORE any optimisation and fails loudly.

Per arm it asserts:
  * `dv` is finite and `assert_dv_convention` holds (sign fixed once);
  * d(dv)/d(action) is finite, non-zero, and non-zero in EVERY one of the four
    action components (a model that ignores, say, the push length would give a
    zero there -- that is a finding, not a pass);
  * the analytic gradient agrees in SIGN with a central finite difference on
    each component (a cheap check that it is the gradient of the thing we
    think it is, not of some detached surrogate).

Usage: PYTHONPATH=. python -u experiments/EXP-0023-*/code/verify_gradients.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import (OCC_ADAPTERS, make_occ_adapter,
                                 assert_dv_convention, occ_from_particles)

CORPUS = "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm"
COMPONENTS = ["sx", "sy", "ex", "ey"]


def check(adapter, occ0, act, eps=2e-4):
    a = act.clone().requires_grad_(True)
    dv = adapter.dv(occ0, a)
    assert torch.isfinite(dv).all(), "dv is not finite"
    g, = torch.autograd.grad(dv.sum(), a)
    assert torch.isfinite(g).all(), "gradient is not finite"
    per_comp = g.abs().mean(dim=0)                      # (4,)
    # central finite differences on the batch mean, per component
    fd = []
    with torch.no_grad():
        for c in range(4):
            ap, am = act.clone(), act.clone()
            ap[:, c] += eps; am[:, c] -= eps
            fd.append(float((adapter.dv(occ0, ap) - adapter.dv(occ0, am)).sum() / (2 * eps)))
    ana = [float(g[:, c].sum()) for c in range(4)]
    # Directional check: the right FD test is along the analytic gradient of
    # EACH ROW, not a per-component sum over rows (which cancels).
    with torch.no_grad():
        d = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
        h = 1e-4
        fd_dir = (adapter.dv(occ0, act + h * d) - adapter.dv(occ0, act - h * d)) / (2 * h)
        ana_dir = (g * d).sum(dim=1)
        dir_rel = ((fd_dir - ana_dir).abs() / ana_dir.abs().clamp_min(1e-12))
        dir_sign = ((fd_dir * ana_dir) > 0).float().mean()
    return {
        "dv_mean": float(dv.mean().detach()),
        "grad_abs_mean_per_component": [float(v) for v in per_comp],
        "grad_norm_mean": float(g.norm(dim=1).mean()),
        "frac_rows_zero_grad": float((g.norm(dim=1) == 0).float().mean()),
        "analytic_sum": ana,
        "finite_diff_sum": fd,
        "sign_agree": [bool((a_ * f_) > 0) or (abs(a_) < 1e-9 and abs(f_) < 1e-9)
                       for a_, f_ in zip(ana, fd)],
        "rel_err": [abs(a_ - f_) / max(abs(f_), 1e-12) for a_, f_ in zip(ana, fd)],
        "dir_fd_sign_agree_frac": float(dir_sign),
        "dir_fd_median_rel_err": float(dir_rel.median()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=sorted(OCC_ADAPTERS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--goal", default="corner")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = BinnedSlateCorpus.load(str(REPO / CORPUS))
    rows = corpus.step(0)
    sel = torch.arange(args.n) * 997                    # spread across slates
    occ0 = occ_from_particles(rows.states[sel].float(), dev)
    act = torch.cat([rows.p_starts[sel, :2], rows.p_stops[sel, :2]], dim=1).float().to(dev)

    report, ok = {}, True
    for arm in args.arms:
        try:
            ad = make_occ_adapter(arm, dev, args.goal)
            assert_dv_convention(ad)
            r = check(ad, occ0, act)
            zero = [c for c, v in zip(COMPONENTS, r["grad_abs_mean_per_component"]) if v == 0.0]
            r["zero_grad_components"] = zero
            r["status"] = ("PASS" if (not zero and r["frac_rows_zero_grad"] == 0.0
                           and r["dir_fd_sign_agree_frac"] == 1.0) else "DEGENERATE")
        except Exception as e:                           # noqa: BLE001
            r = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
        ok &= r["status"] == "PASS"
        report[arm] = r
        print(f"[{r['status']:10s}] {arm}", flush=True)
        if r["status"] != "FAIL":
            print(f"    dv_mean {r['dv_mean']:+.6f}  |grad| {r['grad_norm_mean']:.4g}  "
                  f"zero-grad rows {r['frac_rows_zero_grad']:.2f}")
            for c, a_, f_, s in zip(COMPONENTS, r["analytic_sum"], r["finite_diff_sum"],
                                    r["sign_agree"]):
                print(f"    d/d{c:<2s} analytic {a_:+.5g}  fd {f_:+.5g}  "
                      f"sign {'ok' if s else 'MISMATCH'}")
        else:
            print(f"    {r['error']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
