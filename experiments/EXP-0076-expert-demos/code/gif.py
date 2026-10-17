"""EXP-0076 GIF of a stored demonstration (simulator truth): frame k = state after push k with the goal contour (blue), cubes (rotated squares;
2nd-layer cubes orange), the NEXT push as an arrow + blade, title = push k / mass / cumulative physical time. World x right, y down.
usage: python gif.py <demo json> [...] [--out DIR] ; or --best-per-stratum PIPELINE to render the fastest demo of every (n, init, kind)."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); REPO = HERE.parents[2]; sys.path.insert(0, str(REPO / "scripts/probes"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from cube_viz import cube_patch, push_arrow
import tasks as T


def yaw_of(q):
    w, x, y, z = q; return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def render(d, out):
    mask = T.goal_mask(d["goal"]) if "goal" in d and isinstance(d["goal"], dict) else __import__("Baselines.common.goals", fromlist=["x"]).named_goal_mask(d["goal_name"], 64, 64)
    S, Q, A = np.array(d["states"]), np.array(d["quats"]) if d.get("quats") else None, np.array(d["actions"])
    tcum = np.concatenate([[0], np.cumsum(d.get("push_times") or T.push_times(d["actions"]))]); M = d["masses"]
    fig, ax = plt.subplots(figsize=(4.6, 4.9)); ext = np.array([T.LO, T.HI]) * 1000; C = T.CENTRES * 1000

    def frame(k):
        ax.clear(); ax.contour(C, C, mask.T, levels=[0.5], colors="tab:blue", linewidths=1.2)
        for i in range(S.shape[1]):
            yw = yaw_of(Q[k, i]) if Q is not None else 0.0
            ax.add_patch(cube_patch(S[k, i, :2] * 1000, yw, 5.0, facecolor="tab:orange" if S[k, i, 2] > 0.0145 else "0.25", edgecolor="k", lw=0.3))
        if k < len(A):
            a = A[k] * 1000; push_arrow(ax, a[:2], a[2:], color="tab:red", lw=1.5)
            u = (a[2:] - a[:2]) / max(np.linalg.norm(a[2:] - a[:2]), 1e-9); v = np.array([-u[1], u[0]])
            ax.plot(*np.stack([a[:2] - 20 * v, a[:2] + 20 * v]).T, color="tab:red", lw=2.5, alpha=0.6)
        ax.set_xlim(*ext); ax.set_ylim(ext[1], ext[0]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{d['id'][:38]}\npush {k}/{len(A)}  mass {M[k]:.3f}  t {tcum[k]:.1f}s", fontsize=8)
    anim = FuncAnimation(fig, frame, frames=len(S), interval=700); anim.save(out, writer=PillowWriter(fps=1.4)); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--out", default=str(HERE.parent / "figures/gifs"))
    ap.add_argument("--best-per-stratum", default=None); ap.add_argument("--pick", default="fastest", help="fastest | median demo of each stratum"); a = ap.parse_args(); Path(a.out).mkdir(parents=True, exist_ok=True)
    files = list(a.files)
    if a.best_per_stratum:
        best = {}
        pipes = ["expert", "feasible"] if a.best_per_stratum == "all" else [a.best_per_stratum]
        for f in [g for pp in pipes for g in glob.glob(str(REPO / f"datasets/DS-0024-expert-demos/data/{pp}/*.json"))]:
            d = json.load(open(f)); key = (d["n"], d["init"], d["kind"])
            best.setdefault(key, []).append((d["time_s"], f))
        files += [sorted(v)[len(v) // 2 if a.pick == "median" else 0][1] for v in best.values()]
    for f in files:
        d = json.load(open(f)); out = Path(a.out) / (Path(f).stem + ".gif"); render(d, str(out)); print("wrote", out)


if __name__ == "__main__":
    main()
