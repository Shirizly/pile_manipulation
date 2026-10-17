"""EXP-0075 planner-fix cells (2026-10-11 bottleneck review: fetch proposals, legalise-before-scoring, cumulative / spill objective for H>1).
Per cell: rel (in-goal mass / optimum, analyse.py) at 10 and mean over pushes 1-10, completion at 0.9 / 0.8 x optimum, cube counts (in goal at push 10,
entering / leaving per episode), wasted late pushes (pushes 7-10 that add no cube), skeleton coverage (METRICS.md `skeleton_coverage_r5`: fraction of the
goal mask's skeleton pixels with a cube centre within 5 mm) at push 10 and episodes with coverage >= 0.8, fallbacks / shifted pushes, optimism of the EXECUTED
push (pred_cost_exec - true single-push value change; negative = over-promised), plan time. Paired differences b - a over common (goal, start) episodes,
bootstrap 95 % CI (2,000 resamples).
usage: python -u analyse_fixes.py --out fixes_analysis --pairs a:b,a:b ... results/X.json ..."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "code")); sys.path.insert(0, str(HERE.parents[1]))
from analyse import rel_curve, transition_diag, _goal_tensors, wilson
from Baselines.common.goals import named_goal_mask as letter_goal_mask
from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID
from skimage.morphology import skeletonize

LO = np.array([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]]); HI = np.array([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]])
_M, _SK = {}, {}


def mask(g):
    if g not in _M:
        _M[g] = letter_goal_mask(g, OCC_GRID, OCC_GRID).astype(bool)
        _SK[g] = LO + np.argwhere(skeletonize(_M[g])) / (OCC_GRID - 1) * (HI - LO)   # skeleton pixel centres (mask[x_px, y_px] indexing)
    return _M[g]


def inside(xy, g):
    uv = np.rint((xy - LO) / (HI - LO) * (OCC_GRID - 1)).astype(int).clip(0, OCC_GRID - 1)
    return mask(g)[uv[:, 0], uv[:, 1]]


def coverage(xy, g, r=0.005):
    mask(g); d = np.linalg.norm(_SK[g][:, None] - xy[None], axis=-1).min(1)
    return float((d <= r).mean())


def episode(e):
    g = e["goal"]; S = np.array(e["states"])[:, :, :2]; K = len(e["actions"])
    dg = transition_diag(e["states"], *_goal_tensors(g)); ins = list(dg["inside"])     # shared with the harness's per-push log (T0b, 2026-10-15)
    ent = int(dg["entering"][:K].sum()); lea = int(dg["leaving"][:K].sum())
    wasted_late = [int(ins[k + 1].sum() - ins[k].sum() <= 0) for k in range(6, K)]
    true_dv = dg["true_dv"][:K]
    pe = np.array(e.get("pred_cost_exec", [np.nan] * K))
    return dict(rel=rel_curve(e["states"], g), cubes10=int(ins[K].sum()), entering=ent, leaving=lea, wasted_late=float(np.mean(wasted_late)) if wasted_late else np.nan,
                cov=np.array([coverage(S[k], g) for k in range(K + 1)]), fb=int(sum(e.get("legal_fallback", []))), shift=sum(1 for s in e.get("legal_shift_m", []) if s > 0),
                ill=sum(1 for o in e.get("legal_ok", []) if not o), t=float(np.mean(e["plan_time_s"])), opt_exec=float(np.nanmean(pe - true_dv)) if np.isfinite(pe).any() else np.nan,
                fb_opt=[float(pe[k] - true_dv[k]) for k in range(K) if e.get("legal_fallback", [0] * K)[k]])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--pairs", default=""); ap.add_argument("--out", default="fixes_analysis")
    a = ap.parse_args(); cells = {}
    for p in a.paths:
        for e in json.loads(Path(p).read_text())["episodes"]:
            if e.get("complete") and e.get("states"):
                cells.setdefault(e["cell"], {})[(e["goal"], e["start"])] = episode(e)
    out = {}; L = ["| cell | n | rel@10 | mean rel 1-10 | done@0.9 | done@0.8 | cubes in @10 | entering / leaving per ep | wasted pushes 7-10 | coverage@10 | cov>=0.8 @10 | fallbacks | shifted | optimism exec | plan s |", "|" + "---|" * 15]
    for c, eps in cells.items():
        E = list(eps.values()); R = np.stack([x["rel"] for x in E]); n = len(E); C = np.stack([x["cov"] for x in E])
        d9, d8 = int((R.max(1) >= 0.9).sum()), int((R.max(1) >= 0.8).sum()); c8 = int((C[:, -1] >= 0.8).sum())
        m = lambda k: float(np.nanmean([x[k] for x in E]))
        fbo = [v for x in E for v in x["fb_opt"]]
        out[c] = dict(n=n, rel10=float(R[:, -1].mean()), mean_1_10=float(R[:, 1:].mean()), done09=d9, done08=d8, wilson09=wilson(d9, n), wilson08=wilson(d8, n), cubes10=m("cubes10"),
                      entering=m("entering"), leaving=m("leaving"), wasted_late=m("wasted_late"), cov10=float(C[:, -1].mean()), cov_curve=[round(float(x), 3) for x in C.mean(0)], cov08=c8,
                      fallbacks=int(sum(x["fb"] for x in E)), shifted=int(sum(x["shift"] for x in E)), illegal=int(sum(x["ill"] for x in E)), opt_exec=m("opt_exec"),
                      fallback_opt=float(np.mean(fbo)) if fbo else None, plan_s=m("t"))
        o = out[c]
        L.append(f"| {c} | {n} | {o['rel10']:.3f} | {o['mean_1_10']:.3f} | {d9}/{n} {wilson(d9, n)} | {d8}/{n} | {o['cubes10']:.1f} | {o['entering']:.1f} / {o['leaving']:.1f} | {100 * o['wasted_late']:.0f} % | {o['cov10']:.3f} | {c8}/{n} | {o['fallbacks']} | {o['shifted']} | {o['opt_exec']:+.4f} | {o['plan_s']:.2f} |")
    rng = np.random.default_rng(0); P = ["| a | b | n | d rel@10 [CI] | d mean rel 1-10 [CI] | d coverage@10 [CI] | d leaving [CI] |", "|---|---|---|---|---|---|---|"]; out["paired"] = []
    for pr in [x for x in a.pairs.split(",") if x]:
        ca, cb = pr.split(":"); common = sorted(set(cells.get(ca, {})) & set(cells.get(cb, {})))
        if not common:
            continue
        row = dict(a=ca, b=cb, n=len(common)); txt = []
        for nm, f in (("rel10", lambda x: x["rel"][-1]), ("mean_1_10", lambda x: x["rel"][1:].mean()), ("cov10", lambda x: x["cov"][-1]), ("leaving", lambda x: x["leaving"])):
            dl = np.array([f(cells[cb][k]) - f(cells[ca][k]) for k in common]); bs = [rng.choice(dl, len(dl)).mean() for _ in range(2000)]
            ci = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]; row[nm] = dict(diff=float(dl.mean()), ci95=ci, better=int((dl > 0).sum()), worse=int((dl < 0).sum()))
            txt.append(f"{dl.mean():+.3f} {ci}")
        out["paired"].append(row); P.append(f"| {ca} | {cb} | {len(common)} | " + " | ".join(txt) + " |")
    md = ["# EXP-0075 planner-fix cells (letters O/T/S/X w20, DS-0006 starts 40-47, ens2bal, 10 pushes, planner seed 0)\n"] + L + ["\n## Paired differences (b - a), bootstrap 95 % CI over (goal, start)\n"] + P
    (HERE / f"results/{a.out}.md").write_text("\n".join(md) + "\n"); (HERE / f"results/{a.out}.json").write_text(json.dumps(out, indent=1)); print("\n".join(md))


if __name__ == "__main__":
    main()
