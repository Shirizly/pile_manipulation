"""EXP-0076 replay check: execute stored demos' action lists OPEN LOOP in one continuous env each (state set once at the start, never
restored), compare with the recorded chain (which search pipelines produce by restoring states between pushes).
Envs whose demo is shorter execute an 'idle' push in an empty part of the tray (swept rectangle >= 12 mm from every cube) after their last
action, so lockstep stepping does not disturb them (checked: their cubes' max displacement is recorded).
usage: python replay.py --n 20 --pipeline expert [--ids ...] [--max 32] -> results/replay_<pipeline>_n<n>.json"""
import argparse, json, glob, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim, legalize
from wide_planner import swept_hits

DATA = T.REPO / "datasets/DS-0024-expert-demos/data"


def idle_push(state, rng, tries=4000):
    xy = torch.as_tensor(state)[:, :2].float()
    for _ in range(tries):
        s = torch.tensor(rng.uniform(-0.05, 0.05, 2), device="cpu").float(); th = rng.uniform(0, 2 * np.pi)
        a = torch.cat([s, s + 0.02 * torch.tensor([np.cos(th), np.sin(th)], device="cpu").float()])[None]
        if (a[0, 2:].abs() > 0.058).any():
            continue
        if not swept_hits(a, xy, 0.012, 0.012).any():
            return a[0]
    return None


def replay_chunk(sim, D, rng):
    E = len(D)
    S0 = torch.stack([torch.cat([torch.tensor(d["states"][0], device="cpu"), torch.tensor(d["quats"][0], device="cpu")], 1).float() for d in D])
    sim.set(S0); cur = S0.clone(); Tm = max(len(d["actions"]) for d in D); traj = [[S0[i]] for i in range(E)]; idle_disp = [0.0] * E; no_idle = [0] * E
    for t in range(Tm):
        acts = []
        for i, d in enumerate(D):
            if t < len(d["actions"]):
                acts.append(torch.tensor(d["actions"][t], device="cpu").float())
            else:
                ip = idle_push(cur[i], rng)
                if ip is None:
                    no_idle[i] += 1; ip = torch.tensor(d["actions"][-1], device="cpu").float()
                acts.append(ip)
        A = torch.stack(acts)
        if E < sim.E:                       # padded envs (set() copied the last state) replay the last env's push
            A = torch.cat([A, A[-1:].expand(sim.E - E, -1)])
        nxt = sim.execute(A)[:E]
        for i, d in enumerate(D):
            if t < len(d["actions"]):
                traj[i].append(nxt[i])
            else:
                idle_disp[i] = max(idle_disp[i], float((nxt[i, :, :2] - cur[i, :, :2]).norm(dim=1).max()))
        cur = nxt
    out = []
    for i, d in enumerate(D):
        M = T.mass_frac(torch.stack(traj[i]), T.goal_mask(d["goal"]))
        fin_rec = torch.tensor(d["states"][-1], device="cpu")[:, :2]; fin_rep = traj[i][-1][:, :2]
        out.append(dict(id=d["id"], pushes=len(d["actions"]), rec_final_mass=d["masses"][-1], rep_final_mass=float(M[-1]), rep_solved=bool(M[-1] >= T.SUCCESS),
                        rep_masses=[float(x) for x in M], final_pos_err_mm_mean=float((fin_rec - fin_rep).norm(dim=1).mean() * 1000),
                        final_pos_err_mm_max=float((fin_rec - fin_rep).norm(dim=1).max() * 1000), idle_disturb_mm=idle_disp[i] * 1000, no_idle_spot=no_idle[i],
                        rep_final_xy=fin_rep.tolist()))
        print(out[-1]["id"], f"rec {out[-1]['rec_final_mass']:.3f} rep {out[-1]['rep_final_mass']:.3f} err mean {out[-1]['final_pos_err_mm_mean']:.2f} mm", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--pipeline", default="expert")
    ap.add_argument("--ids", nargs="*"); ap.add_argument("--max", type=int, default=100000); ap.add_argument("--tag", default=""); ap.add_argument("--chunk", type=int, default=128); ap.add_argument("--run", default=None); a = ap.parse_args()
    if a.run:                              # replay the solved records of one run file (results/runs/<run>.json) instead of the store
        D = [r for r in json.load(open(T.REPO / f"experiments/EXP-0076-expert-demos/results/runs/{a.run}.json"))["scenarios"].values() if r.get("solved")]
        a.pipeline = "run-" + a.run
    else:
        D = [json.load(open(f)) for f in sorted(glob.glob(str(DATA / a.pipeline / "*.json")))]
    D = [d for d in D if d["n"] == a.n and (not a.ids or d["id"] in a.ids)][: a.max]
    CH = a.chunk; sim = Sim(a.n, min(max(len(D), 1), CH)); rng = np.random.default_rng(0); out = []
    for c0 in range(0, len(D), CH):
        out += replay_chunk(sim, D[c0:c0 + CH], rng)
    p = T.REPO / f"experiments/EXP-0076-expert-demos/results/replay_{a.pipeline}_n{a.n}{a.tag}.json"; tmp = str(p) + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, p)
    print(f"replayed {len(out)}: still solved {sum(o['rep_solved'] for o in out)}; mean final pos err {np.mean([o['final_pos_err_mm_mean'] for o in out]):.2f} mm", flush=True)
    sim.destroy()


if __name__ == "__main__":
    main()
