"""Push-by-push GIFs of EXP-0055/EXP-0056 closed-loop episodes (recorded particle states, no
re-simulation). Drawing is EXP-0051's `draw_frame`/`write_gif` (world x right, y DOWN, so the
letter goals read correctly -- see that file's docstring).

  best-per-goal : per (cell, goal) the episode completed (in-goal mass >= 0.9 x optimum) in the
                  fewest pushes, else the one with the highest final mass_rel.
  --vs lyap     : side-by-side <base> | <cell> on the same (goal, start); goals = --goals, or the
                  --top N with the largest mean final mass_rel gain of <cell> over <base>; start =
                  the one with the largest paired final mass_rel gain.
Usage (from anywhere; paths default to this experiment):
  python demo_gifs.py --cells crowd                          # best per goal
  python demo_gifs.py --cells crowd --vs lyap --top 3        # side by side, 3 largest-gain goals
  python demo_gifs.py --cells s_misplaced s_deposit s_otmix --vs lyap --top 3 --best
"""
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; REPO = HERE.parents[1]
import importlib.util                                                               # noqa: E402
_spec = importlib.util.spec_from_file_location(         # same basename as this file: load by path
    "exp0051_demo_gifs", REPO / "experiments/EXP-0051-task-success-completion-time/code/demo_gifs.py")
_d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_d)
goal_mask, write_gif = _d.goal_mask, _d.write_gif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--tag", default="main", help="results/<tag>.json + results/analysis_<tag>.json")
    ap.add_argument("--vs", default=None, help="base cell for side-by-side GIFs (e.g. lyap)")
    ap.add_argument("--goals", nargs="*", default=None, help="side-by-side goals (default: --top by gain)")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--best", action="store_true", help="with --vs: also write best-per-goal GIFs")
    ap.add_argument("--out", default=str(HERE / "artifacts/demos"))
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)                                     # letter_mask reads its asset repo-relatively
    E = {(e["cell"], e["goal"], e["start"]): e for e in
         json.loads((HERE / f"results/{a.tag}.json").read_text())["episodes"] if e.get("complete")}
    R = {(r["cell"], r["goal"], r["start"]): r for r in
         json.loads((HERE / f"results/analysis_{a.tag}.json").read_text())["episodes"]}
    miss = sorted({c for c in a.cells + ([a.vs] if a.vs else []) if not any(k[0] == c for k in R)})
    if miss:
        sys.exit(f"cells {miss} not in analysis_{a.tag}.json -- rerun: CUDA_VISIBLE_DEVICES='' python "
                 f"experiments/EXP-0055-capacity-aware-value/code/analyse.py {a.tag}")

    def panel(key):
        e, r = E[key], R[key]
        S = np.array(e["states"]) * 1000.0; acts = np.array(e["actions"]) * 1000.0
        kd = r["completion"]["mass"]["pushes"]; m = goal_mask(key[1])

        def f(k):
            k = min(k, len(S) - 1)
            done = " (done)" if kd and k >= kd else ""
            return dict(mask=m, pts_mm=S[k], act_mm=acts[k] if k < len(acts) else None,
                        title=f"{key[0]}  {key[1]}  push {k}/{len(S) - 1}{done}\n"
                              f"in-goal mass {r['mass_frac'][k]:.2f} (rel opt {r['mass_rel'][k]:.2f})  "
                              f"cover rel {r['cover_rel'][k]:.2f}  lyap {e['values'][k]:.3f}",
                        xlabel=f"{e['model'][:24]} / {e['planner']}  start {key[2]}")
        return f, len(S), kd

    def score(key):
        r = R[key]; k = r["completion"]["mass"]["pushes"]
        return (0 if k else 1, k or 99, -r["mass_rel"][-1])

    for c in a.cells:
        goals = sorted({k[1] for k in R if k[0] == c and k in E})
        if not a.vs or a.best:
            for g in goals:
                key = min((k for k in R if k[0] == c and k[1] == g and k in E), key=score)
                f, n, kd = panel(key)
                p = out / f"{c}_{g}_s{key[2]}.gif"
                write_gif(p, [f], min(n, kd + 3) if kd else n)
                print("wrote", p, "completion(mass 0.9):", kd, "final mass_rel", round(R[key]["mass_rel"][-1], 2))
        if a.vs:
            gain = defaultdict(list)                   # goal -> [(paired final mass_rel gain, start)]
            for k in R:
                b = (a.vs, k[1], k[2])
                if k[0] == c and b in R and k in E and b in E:
                    gain[k[1]].append((R[k]["mass_rel"][-1] - R[b]["mass_rel"][-1], k[2]))
            sel = a.goals or sorted(gain, key=lambda g: -np.mean([x for x, _ in gain[g]]))[:a.top]
            print(c, "vs", a.vs, "mean final mass_rel gain per goal:",
                  {g: round(float(np.mean([x for x, _ in v])), 3) for g, v in gain.items()})
            for g in sel:
                d, s = max(gain[g])
                fb, nb, _ = panel((a.vs, g, s)); fc, nc, _ = panel((c, g, s))
                p = out / f"{a.vs}_vs_{c}_{g}_s{s}.gif"
                write_gif(p, [fb, fc], max(nb, nc))
                print("wrote", p, f"paired final mass_rel gain {d:+.2f}")


if __name__ == "__main__":
    main()
