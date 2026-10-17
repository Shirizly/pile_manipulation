"""EXP-0070 RUN-0009: one-step decision test of the v2 maps with Genesis truth.

For each of RUN-0008's 270 visited states: the lyap planner (CEM on model dV, 0.5 s, pop 1024, elite 0.25) with
its 64 best evaluated pushes kept as the pool. Selection rules over that pool:
  lyap               argmin model dV (pool[0]; what the baseline executes)
  J:<map>[+off]      argmin J = t(a) + f(r') - f(r), r' = r + dr_model - b_r(r) (+off) or r + dr_model
  blend:<map>+off:L  argmin (1-L) z(dV_model) + L z(J), z = standardisation over the pool
for maps lyap|mc, lyap|td, gap|td (etc_map_m0, f(r) = c r^p per goal). Every distinct pick is then executed
in Genesis from the restored state (standard execute_action) -> TRUE dV, true in-goal mass change, physical time;
reported: true progress per second of (t_plan + t), true one-step J under the lyap|td map (common yardstick),
stall share (min J over the pool > 0), push length / transit.
-> results/decision_test.json (+ _rows.json); checkpointed per Genesis batch.
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_dist, goal_mask
from Baselines.common.goals import mass_in_region
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, occ_for_scoring
import simple_mpc.learned_mpc as LM
from action_time import action_time
from diag_small import states

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
T_PLAN = 0.5
MAPS = ("lyap|mc", "lyap|td", "gap|td")
LAMS = (0.25, 0.5, 0.75, 1.0)
dev = "cuda"


def f_of(m, r):
    return m["c"] * np.clip(r, 0, None) ** m["p"]


def offset(cal, r):
    return np.interp(r, cal["r"], cal["bias"]) if cal else 0.0


def readouts(occ, g, gm):
    dist = goal_dist(g).to(occ.device); m = torch.from_numpy(goal_mask(g)).to(occ.device).float()
    V = LM.lyap(occ, dist); fin = (occ * m).reshape(len(occ), -1).sum(1) / occ.reshape(len(occ), -1).sum(1).clamp_min(1e-6)
    return V, fin


def plan_phase(ad, emap):
    S = states(); rows = []
    torch.manual_seed(0); gen = torch.Generator().manual_seed(0)
    from diag_small import bank
    for i, s in enumerate(S):
        g = s["goal"]; gm = emap["goals"][g]; dist = goal_dist(g)
        obj = LM.ModelObjective(ad, s["parts"], dist)
        out = LM.plan(obj, bank(s["parts"], gen)[:1024], "cem", T_PLAN, cem_pop=1024, cem_elite_frac=0.25, topk=64)
        A = out["topk_actions"].to(dev)
        occ_in = occ_from_particles(s["parts"][None, :, :3], dev)
        with torch.no_grad():
            pr = ad.predict_step(occ_in.expand(len(A), -1, -1).contiguous(), A)
        V_in, M_in = readouts(occ_in, g, gm); V_pr, M_pr = readouts(pr, g, gm)
        dV = (V_pr - V_in).cpu().numpy(); dM = ((M_pr - M_in) / gm["mopt"]).cpu().numpy()
        occ_s = occ_for_scoring(s["parts"][None, :, :3].to(dev))
        Vt, Mt = readouts(occ_s, g, gm); Vt = float(Vt); mrel = float(Mt) / gm["mopt"]
        t = action_time(A, torch.tensor(s["p_prev"], device=dev), False).cpu().numpy()
        picks = {"lyap": 0}
        stall = {}
        for mp in MAPS:
            name = mp.split("|")[0]
            r0 = Vt if name == "lyap" else max(0.0, gm["theta"] - mrel)
            dr = dV if name == "lyap" else -dM
            for off in (False, True):
                cal = gm.get(f"{name}|offset") if off else None
                r1 = np.maximum(0.0, r0 + dr - offset(cal, r0))
                J = t + f_of(gm[mp], r1) - f_of(gm[mp], r0)
                key = f"J:{mp}{'+off' if off else ''}"
                picks[key] = int(J.argmin()); stall[key] = bool(J.min() > 0)
                if off:
                    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
                    for L in LAMS[:-1]:
                        picks[f"blend:{mp}+off:{L}"] = int(((1 - L) * z(dV) + L * z(J)).argmin())
        rows.append(dict(i=i, goal=g, start=s["start"], k=s["k"], cell=s["cell"], src=s["src"], Vs=Vt, mrel=mrel,
                         p_prev=s["p_prev"], pool=A.cpu().tolist(), t=t.tolist(), dV_model=dV.tolist(), picks=picks, stall=stall))
        if i % 50 == 0:
            print(f"plan {i}/{len(S)}", flush=True)
    return S, rows


def sim_phase(S, rows):
    from chain_probe import Sim, K
    sim = Sim()
    jobs = []
    for r in rows:
        for j in sorted(set(r["picks"].values())):
            jobs.append((r["i"], j))
    true = {}
    for b0 in range(0, len(jobs), K):
        J = jobs[b0:b0 + K]; Jp = J + [J[-1]] * (K - len(J))
        xyz = torch.stack([S[i]["parts"][:, :3] for i, _ in Jp]); q = torch.stack([S[i]["parts"][:, 3:7] for i, _ in Jp])
        A = torch.tensor([rows[i]["pool"][j] for i, j in Jp])
        sim.set_state(xyz, q); before = sim.parts()
        p0, p1, ang = sim.pose(A); sim.sim.execute_action(p0, p1, ang); sim.sim.update_material_state(); after = sim.parts()
        for e, (i, j) in enumerate(J):
            g = rows[i]["goal"]; gm = EMAP["goals"][g]
            ob = occ_for_scoring(before[e:e + 1, :, :3].to(dev)); oa = occ_for_scoring(after[e:e + 1, :, :3].to(dev))
            Vb, Mb = readouts(ob, g, gm); Va, Ma = readouts(oa, g, gm)
            true[(i, j)] = (float(Va - Vb), float((Ma - Mb) / gm["mopt"]), float(Va), float(Ma) / gm["mopt"])
        print(f"sim {b0 + len(J)}/{len(jobs)}", flush=True)
    sim.env.destroy()
    return true


def main():
    global EMAP
    EMAP = json.loads((HERE / "results/etc_map_m0.json").read_text())
    ad = make_occ_adapter("nfd_3ch_randlen", dev, "corner")
    LM.plan(LM.ModelObjective(ad, torch.zeros(20, 3), goal_dist("quadrant_0")), torch.zeros(8, 4) + 0.01, "cem", 0.1)
    S, rows = plan_phase(ad, EMAP)
    p = HERE / "results/decision_test_rows.json"; p.write_text(json.dumps(rows))
    true = sim_phase(S, rows)
    for r in rows:
        r["true"] = {str(j): true[(r["i"], j)] for j in set(r["picks"].values())}
    p.write_text(json.dumps(rows))
    # summary
    ref = EMAP["goals"]
    out = {}
    rules = list(rows[0]["picks"].keys())
    for band, lo, hi in (("ALL", -1, 9), ("quad_V<0.08", -1, 0.08), ("quad_V>=0.08", 0.08, 9), ("letter+two", -1, 9)):
        R = [r for r in rows if lo <= r["Vs"] < hi and (band == "ALL" or (band.startswith("quad") == r["goal"].startswith("quad")))]
        if not R:
            continue
        o = {}
        for rule in rules:
            dVt = np.array([r["true"][str(r["picks"][rule])][0] for r in R]); dMt = np.array([r["true"][str(r["picks"][rule])][1] for r in R])
            t = np.array([r["t"][r["picks"][rule]] for r in R])
            L = np.array([math.hypot(r["pool"][r["picks"][rule]][2] - r["pool"][r["picks"][rule]][0],
                                     r["pool"][r["picks"][rule]][3] - r["pool"][r["picks"][rule]][1]) for r in R])
            # common yardstick: one-step true J under lyap|td
            Jt = np.array([t[n] + f_of(ref[r["goal"]]["lyap|td"], r["true"][str(r["picks"][rule])][2])
                           - f_of(ref[r["goal"]]["lyap|td"], r["Vs"]) for n, r in enumerate(R)])
            o[rule] = dict(n=len(R), true_dV=float(dVt.mean()), true_dmass=float(dMt.mean()), t=float(t.mean()), L_mm=float(L.mean() * 1e3),
                           rate_V=float((-dVt).sum() / (T_PLAN + t).sum()), rate_mass=float(dMt.sum() / (T_PLAN + t).sum()),
                           trueJ_lyaptd=float(Jt.mean()), same_as_lyap=float(np.mean([r["picks"][rule] == 0 for r in R])),
                           stall=float(np.mean([r["stall"].get(rule, False) for r in R])) if rule.startswith("J:") else None)
        out[band] = o
    p = HERE / "results/decision_test.json"; tmp = Path(str(p) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, p)
    for band, o in out.items():
        print(f"== {band} (n={next(iter(o.values()))['n']})")
        for rule, v in o.items():
            print(f"  {rule:26s} trueJ {v['trueJ_lyaptd']:+.2f}s  true dV {v['true_dV']:+.4f} dmass {v['true_dmass']:+.4f}  rateV {v['rate_V']:.4f} rateM {v['rate_mass']:.4f}  t {v['t']:.2f} L {v['L_mm']:4.1f}  same {v['same_as_lyap']:.2f}"
                  + (f"  stall {v['stall']:.2f}" if v["stall"] is not None else ""))


if __name__ == "__main__":
    main()
