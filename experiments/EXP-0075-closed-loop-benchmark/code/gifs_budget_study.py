"""EXP-0075: per-task multi-panel GIFs of the budget-study trajectories (recorded particle states, no re-simulation; drawing = EXP-0051 draw_frame, world x right / y DOWN).
One GIF per task and budget: columns = the 5 models (ens128, zoom128, vanilla128, zoom64, vanilla64), rows = (top) H=4 plan executed open loop (cem_gd), (bottom) H=1 closed loop re-planned every push (budget B/4 per push).
Frame k = state after k pushes with the NEXT (legalised) push drawn; title carries the true value change so far.  usage: gifs_budget_study.py [B ...]   (default 10 1)"""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import importlib.util
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code")); sys.path.insert(0, str(REPO / "scripts/probes"))
spec = importlib.util.spec_from_file_location("exp0051_gifs", REPO / "experiments/EXP-0051-task-success-completion-time/code/demo_gifs.py"); d51 = importlib.util.module_from_spec(spec); spec.loader.exec_module(d51)
import os; os.chdir(REPO)
BS = REPO / "experiments/EXP-0075-closed-loop-benchmark/results/budget_study"; OUT = REPO / "experiments/EXP-0075-closed-loop-benchmark/figures/gifs"
MODS = ["ens128", "zoom128", "vanilla128", "zoom64", "vanilla64"]
TASKS = [("letter_T_w20", 40), ("letter_O_w20", 41), ("letter_S_w20", 42), ("letter_X_w20", 43), ("letter_T_w20", 44), ("letter_O_w20", 45), ("letter_S_w20", 46), ("letter_X_w20", 47), ("letter_O_w20", 40), ("letter_X_w20", 41)]
# H=4 plans (cem_gd): keys inside states_g*.npz
H4 = {}
for f in sorted(glob.glob(str(BS / "states_g*.npz"))):
    z = np.load(f)
    for i, k in enumerate(z["keys"]):
        g, s, m, pl, B = str(k).split("|"); H4[(g, int(s), m, pl, B)] = (z["parts"][i], z["legalised"][i], z["vals"][i])
# H=1 closed loop: groups of 3 configs (key, B) in the order of bs_closed.py, 10 tasks each
cfgs = [(k, B) for B in (1, 3, 10) for k in MODS]; H1 = {}
for gi in range(5):
    z = np.load(BS / f"states_closed_eq_{gi}.npz")
    for ci, (m, B) in enumerate(cfgs[gi * 3:gi * 3 + 3]):
        for ti, (g, s) in enumerate(TASKS):
            i = ci * 10 + ti; H1[(g, s, m, B)] = (z["parts"][i], z["legalised"][i], z["vals"][i])


def panel(data, title, mask):
    parts, act, vals = data
    def f(k):
        k = min(k, 4)
        return dict(mask=mask, pts_mm=parts[k][:, :2] * 1000.0, act_mm=act[k] * 1000.0 if k < 4 else None, title=f"{title}\npush {k}/4   true dV {vals[k] - vals[0]:+.3f}", xlabel="")
    return f


for B in [int(b) for b in sys.argv[1:]] or [10, 1]:
    for g, s in TASKS:
        mask = d51.goal_mask(g)
        rows = [[panel(H4[(g, s, m, "cem_gd", str(B))], f"{m}  H=4 plan (open loop)", mask) for m in MODS], [panel(H1[(g, s, m, B)], f"{m}  H=1 closed loop", mask) for m in MODS]]
        fig, axs = plt.subplots(2, 5, figsize=(21, 9.6))
        anim = FuncAnimation(fig, lambda k: [d51.draw_frame(axs[r, c], **rows[r][c](k)) for r in range(2) for c in range(5)] + [fig.suptitle(f"{g}  start {s}   planning budget {B} s per 4 pushes   (blue blade = next push, green = goal)", fontsize=12)], frames=7, interval=700)
        fig.tight_layout(rect=(0, 0, 1, 0.96)); p = OUT / f"{g}_s{s}_B{B}.gif"; anim.save(p, writer=PillowWriter(fps=1.2), dpi=100); plt.close(fig); print("wrote", p, flush=True)
