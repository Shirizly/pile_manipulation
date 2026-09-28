"""Push-by-push GIFs of the best (fastest-completing / highest in-goal mass) episodes -- the
user's "expert demonstration" request, first version: top-down frames from the RECORDED
particle states (no re-simulation), goal mask underneath, each push drawn as an arrow with
the 40 mm blade at its start. Frame titles: push k, in-goal mass fraction, lyapunov.

ORIENTATION (fixed 2026-09-25): world x to the right, world y DOWN. The letter goals come from
`Baselines/common/goals.py::letter_mask`, whose glyph asset is read with asset row = world y,
so a letter stands in the world with its top at LOW y; drawn y-up (the first version) every
letter, the cubes and the pushes were flipped vertically (upside-down "L", mirrored-looking
"S"). Mask, cubes and pushes are all drawn in the same world-mm data coordinates and the
flip is one `set_ylim(HI, LO)` on the axes, so they can never disagree with each other.

The drawing (`draw_frame`, `write_gif`) is shared with EXP-0055 `code/demo_gifs.py`.
Usage: python demo_gifs.py <results.json> <analysis.json> <out_dir> [n_per_goal]
  e.g. python demo_gifs.py results/success_states.json results/analysis.json artifacts/demos
"""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(REPO / "scripts/probes"))
from batched_closed_loop import goal_mask                                        # noqa: E402
from cube_viz import push_arrow as _push_arrow, blade_endpoints                  # noqa: E402

CUBE, BLADE = 5.0, 40.0
PX = 128.0 / 63                                      # occ grid pixel-centre pitch, mm (64 px over +-64 mm)
LO, HI = -64.0 - PX / 2, 64.0 + PX / 2               # mask extent: pixel i centred at -64 + i*PX


def draw_frame(ax, mask, pts_mm, act_mm=None, title="", xlabel=""):
    """One top-down frame. mask: (64,64) goal mask, dim0 = world x (letter_mask convention);
    pts_mm: (n,>=2) cube centres; act_mm: [sx,sy,ex,ey] of the NEXT push or None."""
    ax.clear()
    ax.imshow(np.asarray(mask, float).T, origin="lower", extent=[LO, HI, LO, HI], cmap="Greens",
              alpha=0.35, vmin=0, vmax=1, interpolation="nearest")
    for p in pts_mm:
        ax.add_patch(plt.Rectangle((p[0] - CUBE / 2, p[1] - CUBE / 2), CUBE, CUBE, color="saddlebrown"))
    if act_mm is not None:
        sx, sy, ex, ey = act_mm
        b0, b1 = blade_endpoints((sx, sy), (ex - sx, ey - sy), BLADE / 2)
        ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color="royalblue", lw=3)
        _push_arrow(ax, (sx, sy), (ex, ey), color="royalblue", lw=1.5)
    ax.set_xlim(LO, HI); ax.set_ylim(HI, LO)          # y DOWN: letters read correctly (see docstring)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8); ax.set_xlabel(xlabel, fontsize=7)


def write_gif(path, panels, n_frames, fps=1.5):
    """panels: list of callables f(k) -> dict(mask, pts_mm, act_mm, title, xlabel) (side by side)."""
    fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.6), squeeze=False)
    anim = FuncAnimation(fig, lambda k: [draw_frame(a, **f(k)) for a, f in zip(axs[0], panels)],
                         frames=n_frames, interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps)); plt.close(fig)


def main():
    res_p, ana_p, out_dir = (Path(a).resolve() for a in sys.argv[1:4])
    os.chdir(REPO)                                   # letter_mask loads its glyph asset by a repo-relative path
    n_per_goal = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out_dir.mkdir(parents=True, exist_ok=True)
    E = [e for e in json.loads(res_p.read_text())["episodes"] if e.get("complete") and "states" in e]
    ana = {(r["model"], r["planner"], r["goal"], r["start"]): r for r in json.loads(ana_p.read_text())["episodes"]}

    def score(e):
        r = ana[(e["model"], e["planner"], e["goal"], e["start"])]
        k = r["completion"]["0.9"]["pushes"]
        return (0 if k else 1, k or 99, -max(r["mass_frac"]))  # completed first, fewest pushes, then best mass

    picked = []
    for g in sorted({e["goal"] for e in E}):
        picked += sorted([e for e in E if e["goal"] == g], key=score)[:n_per_goal]
    for e in picked:
        r = ana[(e["model"], e["planner"], e["goal"], e["start"])]
        m = goal_mask(e["goal"])
        S = np.array(e["states"]) * 1000.0
        acts = np.array(e["actions"]) * 1000.0
        k_done = r["completion"]["0.9"]["pushes"]
        n_frames = min(len(S), (k_done + 3) if k_done else len(S))

        def frame(k, e=e, r=r, m=m, S=S, acts=acts, k_done=k_done):
            done = " (done)" if k_done and k >= k_done else ""
            return dict(mask=m, pts_mm=S[k], act_mm=acts[k] if k < len(acts) else None,
                        title=f"{e['goal']}  push {k}/{len(S) - 1}{done}\nin-goal mass {r['mass_frac'][k]:.2f} "
                              f"(opt {r['mass_opt']:.2f})  lyap {e['values'][k]:.3f}",
                        xlabel=f"{e['model'][:22]} / {e['planner']}  start {e['start']}")
        name = f"{e['goal']}_{e['model'][:12]}_{e['planner']}_s{e['start']}.gif"
        write_gif(out_dir / name, [frame], n_frames)
        print("wrote", out_dir / name, "completion@0.9:", k_done, "pushes; final in-goal", round(r["mass_frac"][-1], 2))


if __name__ == "__main__":
    main()
