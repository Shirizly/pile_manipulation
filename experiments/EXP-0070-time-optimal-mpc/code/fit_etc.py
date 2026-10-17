"""EXP-0070: fit etc_g(V), the expected remaining task time (s, new clock = planning + physical
action time) as a monotone non-decreasing function of the TRUE lyapunov V of the current state,
per goal g, from recorded closed-loop episodes (results/retro_episodes.json, or a list of them).

Censoring: an episode that never completes contributes remaining time >= what was observed.
Pass 1 fits on completed episodes only; pass 2 adds censored states with remaining time imputed as
(observed remaining) + etc_pass1(V at the episode's last state), then refits. Both are kept.

Map: V quantile bins -> mean remaining time -> pool-adjacent-violators (monotone) -> knots.
Below the first knot etc is linear to 0 at V = 0; above the last knot it extends with the overall
least-squares slope. -> results/<out>.json {goal: {knots_V, knots_etc, slope_hi, n_*}}.
Usage: fit_etc.py OUT_NAME EPISODES_JSON [EPISODES_JSON ...] [--goals g1 g2 ...]
"""
import json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
GOALS = ["quadrant_0", "quadrant_3", "two_squares", "letter_O", "letter_T", "letter_S", "letter_X"]


def remaining(r):
    """[(V_j, remaining_s, completed)] for states s_0..s_{k*-1} (or all states if censored)."""
    V = np.array(r["V"]); step = np.array(r["plan"]) + np.array(r["t_act"]["total"])
    k = r["kstar"]; last = k if k else len(step)
    return [(V[j], step[j:last].sum(), k is not None, V[last]) for j in range(last)]


def pav_fit(x, y, n_bins=10):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    c, m, w = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (x >= lo) & (x <= hi)
        if s.sum():
            c.append(x[s].mean()); m.append(y[s].mean()); w.append(s.sum())
    i = 0
    while i < len(m) - 1:                     # non-decreasing in V
        if m[i] > m[i + 1]:
            t = w[i] + w[i + 1]
            m[i] = (m[i] * w[i] + m[i + 1] * w[i + 1]) / t; c[i] = (c[i] * w[i] + c[i + 1] * w[i + 1]) / t
            w[i] = t; del m[i + 1], w[i + 1], c[i + 1]; i = max(i - 1, 0)
        else:
            i += 1
    slope = np.polyfit(x, y, 1)[0] if len(x) > 2 else 0.0
    # smooth strictly-increasing alternative for planning (a one-step objective needs etc' > 0
    # everywhere; PAV plateaus make progress look worthless): etc = c * V^p, least squares on
    # the raw points, p on a grid, c closed form
    best = None
    for pw in np.linspace(0.2, 2.0, 37):
        z = np.clip(x, 1e-6, None) ** pw; cc = float((z * y).sum() / (z * z).sum())
        sse = float(((y - cc * z) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, cc, float(pw))
    r2 = 1 - best[0] / float(((y - y.mean()) ** 2).sum())
    return dict(knots_V=[float(v) for v in c], knots_etc=[float(v) for v in m], slope_hi=float(max(slope, 0.0)),
                power=dict(c=best[1], p=best[2], r2=float(r2)))


def etc_eval(f, V):
    kv, ke = np.array(f["knots_V"]), np.array(f["knots_etc"])
    V = np.asarray(V, float)
    out = np.interp(V, kv, ke)
    lo = V < kv[0]; out[lo] = ke[0] * np.clip(V[lo], 0, None) / max(kv[0], 1e-9)
    hi = V > kv[-1]; out[hi] = ke[-1] + f["slope_hi"] * (V[hi] - kv[-1])
    return out


def main():
    args = sys.argv[1:]
    goals = GOALS
    if "--goals" in args:
        i = args.index("--goals"); goals = args[i + 1:]; args = args[:i]
    out_name, srcs = args[0], args[1:]
    rows = []
    for s in srcs:
        rows += json.loads(Path(s).read_text())
    res = {}
    for g in goals:
        R = [r for r in rows if r["goal"] == g]
        pts = [p for r in R for p in remaining(r)]
        if not pts:
            continue
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        done = np.array([p[2] for p in pts]); vlast = np.array([p[3] for p in pts])
        n_ep = len(R); n_done = sum(r["kstar"] is not None for r in R)
        if done.sum() < 10:
            res[g] = dict(n_episodes=n_ep, n_completed=n_done, skipped="too few completed states")
            continue
        f1 = pav_fit(x[done], y[done])
        y2 = y.copy(); y2[~done] = y[~done] + etc_eval(f1, vlast[~done])
        f2 = pav_fit(x, y2)
        rho = np.corrcoef(x[done], y[done])[0, 1]
        res[g] = dict(pass1=f1, pass2=f2, n_episodes=n_ep, n_completed=n_done, n_states=int(len(x)),
                      n_states_completed=int(done.sum()), pearson_completed=float(rho),
                      theta=R[0]["theta"], sources=sorted({r["src"] for r in R}))
        print(g, f"eps {n_ep} done {n_done} | pass1 knots", np.round(f1["knots_V"], 3).tolist(), np.round(f1["knots_etc"], 1).tolist(),
              f"| pass2", np.round(f2["knots_V"], 3).tolist(), np.round(f2["knots_etc"], 1).tolist(), "| power1", f1["power"], "power2", f2["power"], flush=True)
    out = HERE / f"results/{out_name}.json"; tmp = Path(str(out) + ".tmp")
    tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, out)


if __name__ == "__main__":
    main()
