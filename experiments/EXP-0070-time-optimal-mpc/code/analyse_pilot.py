"""EXP-0070: analyse a time_mpc.py results file. Per episode: completion push k* (in-goal mass /
EXP-0046 optimum >= theta: 0.9 quadrants/two_squares, 0.7 letters), task time = sum over pushes
1..k* of (planning time + physical action time, action_time.py), censored at K pushes (all K pushes'
time) when never completed; per-push physical time and its transit part; in-goal mass rel at k = 4, 8, 16.
Per cell vs `lyap`: paired (goal, start) differences, bootstrap 95 % CI over pairs, sign-flip p.
Usage: analyse_pilot.py TAG [TAG ...] -> results/analysis_<first tag>.json; also writes
results/<tag>_episodes.json (retro_time row format) so fit_etc.py can refit on these runs.
"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from retro_time import episode_rows

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
rng = np.random.default_rng(0)


def boot(d, n=5000):
    d = np.asarray(d, float)
    if len(d) == 0:
        return dict(mean=None)
    bs = rng.choice(d, (n, len(d))).mean(1)
    signs = rng.choice([-1, 1], (n, len(d)))
    p = float(np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12))
    return dict(mean=float(d.mean()), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), p_signflip=p, n=len(d))


def ep_metrics(r):
    p = np.array(r["plan"]); t = np.array(r["t_act"]["total"]); tr = np.array(r["t_act"]["transit"])
    K = len(p); k = r["kstar"]; kk = k if k else K
    mr = np.array(r["mass_rel"])
    return dict(done=k is not None, pushes=kk, time=float(p[:kk].sum() + t[:kk].sum()),
                exec_time=float(t[:kk].sum()), plan_time=float(p[:kk].sum()),
                t_push=float(t.mean()), transit_push=float(tr.mean()), sweep_len_push=None,
                m4=float(mr[min(4, K)]), m8=float(mr[min(8, K)]), m16=float(mr[min(16, K)]),
                fallback=float(np.mean(r.get("legal_fallback", [0]))))


def main():
    tags = sys.argv[1:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for tag in tags:
        d = json.loads((HERE / f"results/{tag}.json").read_text())
        R = [episode_rows(f"EXP-0070_{tag}", e, dev) for e in d["episodes"] if e.get("complete")]
        for r, e in zip(R, [e for e in d["episodes"] if e.get("complete")]):
            r["legal_fallback"] = e.get("legal_fallback", [])
            r["n_evals"] = e.get("n_evals", [])
        rows += R
        p = HERE / f"results/{tag}_episodes.json"; tmp = Path(str(p) + ".tmp")
        tmp.write_text(json.dumps(R)); os.replace(tmp, p)
    for r in rows:
        r["m"] = ep_metrics(r); r["m"]["evals"] = float(np.mean(r["n_evals"]))
    cells = sorted({r["cell"] for r in rows}, key=lambda c: (c != "lyap", c))
    fams = ["ALL", "quadrant", "two_squares", "letter"]
    out = dict(tags=tags, per_cell={}, paired_vs_lyap={})
    keys = ["done", "pushes", "time", "exec_time", "plan_time", "t_push", "transit_push", "m4", "m8", "m16", "fallback", "evals"]
    for c in cells:
        for f in fams:
            R = [r for r in rows if r["cell"] == c and (f == "ALL" or r["fam"] == f)]
            if R:
                out["per_cell"][f"{c}|{f}"] = dict(n=len(R), **{k: float(np.mean([r["m"][k] for r in R])) for k in keys})
    base = {(r["goal"], r["start"]): r for r in rows if r["cell"] == "lyap"}
    for c in cells:
        if c == "lyap":
            continue
        for f in fams:
            P = [(r, base[(r["goal"], r["start"])]) for r in rows if r["cell"] == c and (f == "ALL" or r["fam"] == f)
                 and (r["goal"], r["start"]) in base]
            if P:
                out["paired_vs_lyap"][f"{c}|{f}"] = {k: boot([a["m"][k] - b["m"][k] for a, b in P])
                                                     for k in ("time", "pushes", "t_push", "transit_push", "m8", "m16", "done")}
    dst = HERE / f"results/analysis_{tags[0]}.json"; tmp = Path(str(dst) + ".tmp")
    tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, dst)
    print(f"{'cell|family':24s} {'n':>3s} {'done':>5s} {'time':>6s} {'push':>5s} {'t/push':>6s} {'trans':>5s} {'m8':>5s} {'m16':>5s} {'evals':>6s}")
    for k, v in out["per_cell"].items():
        print(f"{k:24s} {v['n']:3d} {v['done']:5.2f} {v['time']:6.1f} {v['pushes']:5.1f} {v['t_push']:6.2f} {v['transit_push']:5.2f} {v['m8']:5.2f} {v['m16']:5.2f} {v['evals']:6.0f}")
    print("\npaired vs lyap: mean [95% CI] (sign-flip p)")
    for k, v in out["paired_vs_lyap"].items():
        print(f"{k:24s} " + "  ".join(f"{m} {s['mean']:+.2f} [{s['lo']:+.2f},{s['hi']:+.2f}] p{s['p_signflip']:.2f}"
                                       for m, s in v.items() if m in ("time", "pushes", "t_push", "m16", "done")))


if __name__ == "__main__":
    main()
