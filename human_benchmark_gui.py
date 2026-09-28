#!/usr/bin/env python3
"""Human demonstrations on the closed-loop BENCHMARK tasks (EXP-0058).

Same tasks, physics and push rules as the oracle / learned-model benchmark (EXP-0055, EXP-0057):
  * start states: DS-0006 corpus slates (default 40, 41), goals as in EXP-0057;
  * simulator set up exactly like `experiments/EXP-0050-oracle-ceiling/code/oracle.py`
    (oracle_config_with_physics + apply_physics + real settle/clearance steps), one env;
  * pushes: blade 40 mm, perpendicular to the push direction, length projected to 20-70 mm
    and endpoints kept in the tray (`simple_mpc.learned_mpc.project_push`);
  * success: in-goal mass fraction (soft scoring image) >= 0.9 x the goal's EXP-0046 optimum;
    the episode stops when solved or after 20 pushes.
PURE HUMAN: the drawn push executes as drawn (after the legality projection), no simulator
refinement, no undo (an undo would let the person use the simulator as a model).

Display: world x to the right, world y DOWN -- the orientation in which the letter goals read
correctly (see docs/CODEMAP.md, demo GIFs). Drag from the push START (blade centre) to its END.

Recording (atomic, after every push): one JSON per task in
experiments/EXP-0058-human-demonstrations/results/<operator>/<goal>_s<start>.json with
actions (executed, metres), particle states after every push, in-goal mass fraction,
lyapunov, think time per push (state shown -> Execute), simulation time, solved_at.
A finished task is marked done in the list; reopening it asks before overwriting.

Usage:  python human_benchmark_gui.py --operator alon [--goals ...] [--starts 40 41]
        (Genesis start-up ~30 s; each push simulates in ~6 s. Select a task, Open, drag, Space.)
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask  # noqa: E402
from Baselines.common.goals import dist_field_from_mask  # noqa: E402
from simple_mpc.adapters import OCC_BOUNDS, occ_for_scoring  # noqa: E402
from simple_mpc.learned_mpc import lyap, project_push  # noqa: E402

GOALS = ["letter_O", "letter_T", "letter_S", "letter_X", "letter_L", "letter_I", "two_squares", "quadrant_0"]
LO, HI = OCC_BOUNDS["x_min"], OCC_BOUNDS["x_max"]
PITCH = (HI - LO) / 63                     # pixel-centre spacing of the 64 px grid
E_LO, E_HI = LO - PITCH / 2, HI + PITCH / 2  # world extent covered by the 64 px image
CANVAS = 640
CUBE, BLADE = 0.005, 0.040
MAX_PUSHES, THETA = 20, 0.9
OUT_ROOT = REPO / "experiments/EXP-0058-human-demonstrations/results"


def w2c(x, y):
    s = CANVAS / (E_HI - E_LO)
    return (x - E_LO) * s, (y - E_LO) * s     # y DOWN on screen = world y increasing downward


def c2w(cx, cy):
    s = (E_HI - E_LO) / CANVAS
    return E_LO + cx * s, E_LO + cy * s


class Sim:
    """One-env Genesis simulator, configured exactly like EXP-0050/0057 oracle.py."""

    def __init__(self):
        from Genesis.binned_slate_dataset import BinnedSlateCorpus
        from simple_mpc.genesis_oracle import GenesisOracleEnv
        from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
        from simple_mpc.oracle_mpc import load_oracle_config
        import genesis as gs
        self.gs = gs
        self.rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
        cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
        cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = 1
        self.env = GenesisOracleEnv(cfg, n_envs=1); apply_physics(self.env); self.sim = self.env._sim
        self.sim._settle_steps = self.env._real_settle_steps
        self.sim._clearance_ctrl_steps = self.env._real_clearance_steps

    def reset(self, start):
        st = self.rows.states[(self.rows.slate_idx == start).nonzero()[0, 0]].float()[None]
        self.sim.set_particle_state(st[:, :, :3].to(self.gs.device), st[:, :, 3:7].to(self.gs.device))
        self.sim.update_material_state()
        return self.state()

    def state(self):
        return self.sim._particle_state[0].detach().cpu().float().clone()

    def push(self, act):
        """Execute one push from the current state. As oracle.py's simulate(): re-set the particle
        state first (zeroes velocities), then action_to_pose + execute_action + update_material_state."""
        from transforms.functional import action_to_pose
        st = self.state()[None]
        self.sim.set_particle_state(st[:, :, :3].to(self.gs.device), st[:, :, 3:7].to(self.gs.device))
        sx, sy, ex, ey, ang = action_to_pose(act[None].to(self.gs.device).float())
        z = torch.full_like(sx, float(self.sim._operation_height))
        self.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
        self.sim.update_material_state()
        return self.state()


class GUI:
    def __init__(self, root, sim, a):
        self.root, self.sim, self.a = root, sim, a
        self.out = OUT_ROOT / a.operator; self.out.mkdir(parents=True, exist_ok=True)
        self.opt = {g: v.get("mass_frac_best_placement", 1.0) for g, v in
                    json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text()).items()}
        self.tasks = [(g, s) for g in a.goals for s in a.starts]
        self.drag, self.act, self.rec, self.busy = None, None, None, False
        self._build()
        self._refresh_list()

    # ---------------- UI ----------------
    def _build(self):
        self.root.title("Human demonstrations -- benchmark tasks (pure human, no refinement)")
        left = tk.Frame(self.root); left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        tk.Label(left, text="Tasks (goal, start)").pack(anchor="w")
        self.lb = tk.Listbox(left, width=26, height=len(self.tasks) + 1, exportselection=False)
        self.lb.pack(); self.lb.bind("<Double-Button-1>", lambda e: self.open_task())
        tk.Button(left, text="Open task", command=self.open_task).pack(fill=tk.X, pady=(4, 0))
        tk.Button(left, text="Give up (mark unsolved)", command=self.give_up).pack(fill=tk.X, pady=(4, 0))
        self.info = tk.Label(left, text="", justify=tk.LEFT, font=("TkFixedFont", 11)); self.info.pack(anchor="w", pady=10)
        tk.Label(left, justify=tk.LEFT, fg="#555", text=(
            "Drag from push START (blade centre)\nto push END. Blade = 40 mm, square to\n"
            "the push. Length is projected to\n20-70 mm. Then Execute (or Space).\n\n"
            "Solved: in-goal mass >= 0.9 x optimum.\nMax 20 pushes. No undo.")).pack(anchor="w")
        right = tk.Frame(self.root); right.pack(side=tk.LEFT, padx=6, pady=6)
        self.cv = tk.Canvas(right, width=CANVAS, height=CANVAS, bg="white", highlightthickness=1)
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self._press); self.cv.bind("<B1-Motion>", self._move)
        bar = tk.Frame(right); bar.pack(fill=tk.X, pady=4)
        self.exe = tk.Button(bar, text="Execute push", command=self.execute, state=tk.DISABLED, width=16)
        self.exe.pack(side=tk.LEFT)
        self.status = tk.Label(bar, text="Open a task.", anchor="w"); self.status.pack(side=tk.LEFT, padx=8)
        self.root.bind("<space>", lambda e: self.execute())

    def _task_file(self, g, s):
        return self.out / f"{g}_s{s}.json"

    def _refresh_list(self):
        self.lb.delete(0, tk.END)
        for g, s in self.tasks:
            p = self._task_file(g, s); tag = "    "
            if p.exists():
                r = json.loads(p.read_text())
                tag = (f"OK{r['solved_at']:>2d}" if r.get("solved_at") else
                       ("  x " if r.get("finished") else " .. "))
            self.lb.insert(tk.END, f"{tag} {g:12s} s{s}")

    # ---------------- task flow ----------------
    def open_task(self):
        if self.busy or not self.lb.curselection():
            return
        g, s = self.tasks[self.lb.curselection()[0]]
        p = self._task_file(g, s)
        if p.exists() and not messagebox.askyesno("Overwrite?", f"{p.name} exists. Start this task again and overwrite it?"):
            return
        self.goal, self.start = g, s
        m = goal_mask(g); self.mask = torch.from_numpy(m).float()
        self.dist = torch.from_numpy(dist_field_from_mask(m)).float()
        self._mask_img(m)
        self.status.config(text="Resetting simulator..."); self.root.update()
        st = self.sim.reset(s)
        self.rec = dict(operator=self.a.operator, goal=g, start=s, opt=self.opt.get(g, 1.0), theta=THETA,
                        max_pushes=MAX_PUSHES, refinement=False, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                        actions=[], states=[st[:, :3].tolist()], mass_frac=[self._mass(st)], lyap=[self._lyap(st)],
                        think_s=[], sim_s=[], solved_at=None, finished=False)
        self._save(); self._refresh_list()
        self.act = None; self.t_shown = time.time()
        self._draw(); self._update_info()
        self.status.config(text="Drag a push.")

    def give_up(self):
        if self.rec and not self.rec["finished"]:
            self.rec["finished"] = True; self.rec["gave_up"] = True; self._save(); self._refresh_list()
            self.exe.config(state=tk.DISABLED); self.status.config(text="Marked unsolved. Open another task.")

    def execute(self):
        if self.busy or self.rec is None or self.rec["finished"] or self.act is None:
            return
        self.busy = True; self.exe.config(state=tk.DISABLED)
        think = time.time() - self.t_shown
        self.status.config(text="Simulating push..."); self.root.update()
        t0 = time.time(); st = self.sim.push(self.act); sim_s = time.time() - t0
        r = self.rec
        r["actions"].append(self.act.tolist()); r["states"].append(st[:, :3].tolist())
        r["mass_frac"].append(self._mass(st)); r["lyap"].append(self._lyap(st))
        r["think_s"].append(think); r["sim_s"].append(sim_s)
        k = len(r["actions"])
        if r["mass_frac"][-1] >= THETA * r["opt"]:
            r["solved_at"] = k; r["finished"] = True
        elif k >= MAX_PUSHES:
            r["finished"] = True
        self._save(); self._refresh_list()
        self.act = None; self.busy = False; self.t_shown = time.time()
        self._draw(); self._update_info()
        if r["solved_at"]:
            self.status.config(text=f"SOLVED in {k} pushes. Open the next task.")
        elif r["finished"]:
            self.status.config(text=f"Not solved in {MAX_PUSHES} pushes. Open the next task.")
        else:
            self.status.config(text=f"Push {k} done ({sim_s:.1f} s sim). Drag the next push.")

    # ---------------- scoring / saving ----------------
    def _mass(self, st):
        f = occ_for_scoring(st[None, :, :3]).reshape(-1)
        return float((f * self.mask.reshape(-1)).sum() / f.sum().clamp_min(1e-6))

    def _lyap(self, st):
        return float(lyap(occ_for_scoring(st[None, :, :3]), self.dist)[0])

    def _save(self):
        p = self._task_file(self.goal, self.start); tmp = Path(str(p) + ".tmp")
        tmp.write_text(json.dumps(self.rec)); os.replace(tmp, p)

    def _update_info(self):
        r = self.rec; k = len(r["actions"]); mf = r["mass_frac"][-1]
        self.info.config(text=(f"{r['goal']}  start {r['start']}\n"
                               f"pushes      {k:2d} / {MAX_PUSHES}\n"
                               f"in-goal     {mf:.2f}\n"
                               f"rel. opt    {mf / r['opt']:.2f}  (need {THETA:.2f})\n"
                               f"lyapunov    {r['lyap'][-1]:.3f}"))

    # ---------------- drawing ----------------
    def _mask_img(self, m):
        from PIL import Image, ImageTk
        img = np.full((64, 64, 3), 255, np.uint8)
        img[m.T.astype(bool)] = (190, 225, 190)           # rows = world y (down), cols = world x
        self._img = ImageTk.PhotoImage(Image.fromarray(img).resize((CANVAS, CANVAS), Image.NEAREST))

    def _draw(self):
        cv = self.cv; cv.delete("all")
        cv.create_image(0, 0, image=self._img, anchor="nw")
        x0, y0 = w2c(LO - 0.0005, LO - 0.0005); x1, y1 = w2c(HI + 0.0005, HI + 0.0005)
        cv.create_rectangle(x0, y0, x1, y1, outline="#999")
        st = np.array(self.rec["states"][-1])
        for x, y, _ in st:
            a0, b0 = w2c(x - CUBE / 2, y - CUBE / 2); a1, b1 = w2c(x + CUBE / 2, y + CUBE / 2)
            cv.create_rectangle(a0, b0, a1, b1, fill="#8b4513", outline="#5a2d0c")
        if self.rec["actions"]:                                       # previous push, faint
            self._draw_push(np.array(self.rec["actions"][-1]), "#bbbbdd", 1)
        if self.act is not None:
            self._draw_push(self.act.numpy(), "royalblue", 3)

    def _draw_push(self, act, color, w):
        sx, sy, ex, ey = act
        d = np.array([ex - sx, ey - sy]); L = np.linalg.norm(d) + 1e-12; u = d / L; n = np.array([-u[1], u[0]])
        b0 = np.array([sx, sy]) - n * BLADE / 2; b1 = np.array([sx, sy]) + n * BLADE / 2
        self.cv.create_line(*w2c(*b0), *w2c(*b1), fill=color, width=w + 2)
        self.cv.create_line(*w2c(sx, sy), *w2c(ex, ey), fill=color, width=w, arrow=tk.LAST)
        if w > 1:
            self.cv.create_text(*w2c(ex, ey), text=f" {L * 1000:.0f} mm", anchor="w", fill=color)

    def _press(self, e):
        if self.rec is None or self.rec["finished"] or self.busy:
            return
        self.drag = c2w(e.x, e.y)

    def _move(self, e):
        if self.drag is None:
            return
        ex, ey = c2w(e.x, e.y)
        raw = torch.tensor([[self.drag[0], self.drag[1], ex, ey]], dtype=torch.float32, device="cpu")
        if float((raw[0, 2:] - raw[0, :2]).norm()) < 1e-4:
            return
        self.act = project_push(raw)[0][0].cpu()
        self.exe.config(state=tk.NORMAL); self._draw()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", required=True, help="name for the output folder")
    ap.add_argument("--goals", nargs="*", default=GOALS)
    ap.add_argument("--starts", type=int, nargs="*", default=[40, 41])
    a = ap.parse_args()
    os.chdir(REPO)                      # letter_mask loads its glyph asset by a repo-relative path
    print("starting Genesis (about a minute)...", flush=True)
    sim = Sim()
    root = tk.Tk()
    GUI(root, sim, a)
    root.lift(); root.attributes("-topmost", True)          # Genesis start-up takes a while: surface the window
    root.after(1000, lambda: root.attributes("-topmost", False))
    root.mainloop()
    sim.env.destroy()


if __name__ == "__main__":
    main()
