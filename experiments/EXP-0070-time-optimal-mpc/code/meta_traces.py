"""EXP-0070 RUN-0013 (meta-planner step 1): how does the incumbent improve with deliberation time, in prediction
and in truth?

For RUN-0008's 270 visited states: CEM on model lyapunov dV, exactly learned_mpc.plan's CEM branch (pool 1024 from
diag_small.bank, pop 1024, elite 0.25, running top-64), run for 1.0 s with snapshots of the running top-64 at
deliberation times CK. At each snapshot two incumbents: `lyap` (argmin model dV) and `mass` (the session's selection
rule = blendnt1.0: argmin over the top-64 of the mass gap read through the model with the calibrated offset, ties ->
lyap order). Every distinct incumbent is executed in Genesis from the restored state (standard execute_action):
true dV, true in-goal mass change (relative to the EXP-0046 optimum), physical action time t_act.
-> results/meta_traces.json (one row per state; checkpointed after the planning phase and per Genesis batch)
"""
import json, math, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_dist, goal_mask
from simple_mpc.adapters import make_occ_adapter, occ_from_particles, occ_for_scoring
import simple_mpc.learned_mpc as LM
from action_time import action_time
from diag_small import states, bank

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
CK = (0.03, 0.06, 0.1, 0.2, 0.35, 0.5, 1.0)
dev = "cuda"


def cem_trace(obj, cand0, pop=1024, elite_frac=0.25, topk=64):
    """learned_mpc.plan's CEM branch with snapshots of the running top-k at the CK times."""
    lo, hi = LM.L_MIN, LM.L_MAX
    proj = lambda x: LM.project_push(x, lo, hi)[0]
    t0 = time.time()
    cand = proj(cand0.to(dev).float())
    with torch.no_grad():
        dv = obj(cand)
    tk = [cand, dv]

    def keep(a, c):
        A = torch.cat([tk[0], a]); C = torch.cat([tk[1], c]); i = C.topk(min(topk, len(C)), largest=False).indices
        tk[0], tk[1] = A[i].clone(), C[i].clone()
    keep(cand[:0], dv[:0])
    n_elite = max(2, int(elite_frac * pop))
    snaps, ci = [], 0
    with torch.no_grad():
        elite = cand[dv.argsort()[:n_elite]]
        mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
        n_it, n_ev = 0, len(cand)
        while ci < len(CK):
            el = time.time() - t0
            while ci < len(CK) and el >= CK[ci]:
                snaps.append(dict(tau=CK[ci], elapsed=el, iters=n_it, evals=n_ev, A=tk[0].clone(), dv=tk[1].clone())); ci += 1
            if ci >= len(CK):
                break
            p = proj(mean + std * torch.randn(pop, 4, device=dev))
            c = obj(p); keep(p, c); n_it += 1; n_ev += pop
            elite = p[c.argsort()[:n_elite]]; mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
    return snaps


def main():
    ad = make_occ_adapter("nfd_3ch_randlen", dev, "corner")
    emap = json.loads((HERE / "results/etc_map_m1.json").read_text())["goals"]
    LM.plan(LM.ModelObjective(ad, torch.zeros(20, 3), goal_dist("quadrant_0")), torch.zeros(8, 4) + 0.01, "cem", 0.1)
    S = states(); gen = torch.Generator().manual_seed(0); torch.manual_seed(0)
    rows = []
    for i, s in enumerate(S):
        g = s["goal"]; gm = emap[g]; dist = goal_dist(g)
        obj = LM.ModelObjective(ad, s["parts"], dist)
        snaps = cem_trace(obj, bank(s["parts"], gen)[:1024])
        m = torch.from_numpy(goal_mask(g)).to(dev).float()
        fin = lambda o: (o * m).reshape(len(o), -1).sum(1) / o.reshape(len(o), -1).sum(1).clamp_min(1e-6)
        occ_in = occ_from_particles(s["parts"][None, :, :3], dev)
        occ_s = occ_for_scoring(s["parts"][None, :, :3].to(dev))
        mrel = float(fin(occ_s)) / gm["mopt"]; r0 = max(0.0, gm["theta"] - mrel)
        cal = gm.get("gap|offset"); off = float(np.interp(r0, cal["r"], cal["bias"])) if cal else 0.0
        cps = []
        for sn in snaps:
            A = sn["A"]
            with torch.no_grad():
                pr = ad.predict_step(occ_in.expand(len(A), -1, -1).contiguous(), A)
            dM = ((fin(pr) - fin(occ_in)) / gm["mopt"]).cpu().numpy()
            r1 = np.maximum(0.0, r0 - dM - off)
            jm = int(np.argmin(r1))                       # first index among ties = lyap order
            t = action_time(A[[0, jm]].float(), torch.tensor(s["p_prev"], device=dev), False).cpu().numpy()
            cps.append(dict(tau=sn["tau"], elapsed=sn["elapsed"], iters=sn["iters"], evals=sn["evals"],
                            lyap=dict(a=A[0].cpu().tolist(), dv=float(sn["dv"][0]), dM=float(dM[0]), t=float(t[0])),
                            mass=dict(a=A[jm].cpu().tolist(), dv=float(sn["dv"][jm]), dM=float(dM[jm]), t=float(t[1]), rank=jm)))
        rows.append(dict(i=i, goal=g, start=s["start"], k=s["k"], cell=s["cell"], src=s["src"], mrel=mrel, r0=r0, off=off,
                         p_prev=s["p_prev"], cps=cps))
        if i % 50 == 0:
            print(f"trace {i}/{len(S)}", flush=True)
    out = HERE / "results/meta_traces.json"; out.write_text(json.dumps(rows))
    # ---- truth for every distinct incumbent
    from chain_probe import Sim, K
    sim = Sim()
    jobs, key = [], {}
    for r in rows:
        for c in r["cps"]:
            for kind in ("lyap", "mass"):
                k2 = (r["i"], tuple(round(x, 7) for x in c[kind]["a"]))
                if k2 not in key:
                    key[k2] = len(jobs); jobs.append((r["i"], c[kind]["a"]))
    true = [None] * len(jobs)
    for b0 in range(0, len(jobs), K):
        J = jobs[b0:b0 + K]; Jp = J + [J[-1]] * (K - len(J))
        xyz = torch.stack([S[i]["parts"][:, :3] for i, _ in Jp]); q = torch.stack([S[i]["parts"][:, 3:7] for i, _ in Jp])
        A = torch.tensor([a for _, a in Jp])
        sim.set_state(xyz, q); before = sim.parts()
        p0, p1, ang = sim.pose(A); sim.sim.execute_action(p0, p1, ang); sim.sim.update_material_state(); after = sim.parts()
        for e, (i, _) in enumerate(J):
            g = rows[i]["goal"]; gm = emap[g]; m = torch.from_numpy(goal_mask(g)).to(dev).float(); dist = goal_dist(g).to(dev)
            ob = occ_for_scoring(before[e:e + 1, :, :3].to(dev)); oa = occ_for_scoring(after[e:e + 1, :, :3].to(dev))
            fin = lambda o: float((o * m).sum() / o.sum().clamp_min(1e-6))
            true[b0 + e] = (float(LM.lyap(oa, dist) - LM.lyap(ob, dist)), (fin(oa) - fin(ob)) / gm["mopt"])
        print(f"sim {b0 + len(J)}/{len(jobs)}", flush=True)
    for r in rows:
        for c in r["cps"]:
            for kind in ("lyap", "mass"):
                j = key[(r["i"], tuple(round(x, 7) for x in c[kind]["a"]))]
                c[kind]["true_dV"], c[kind]["true_dM"] = true[j]
    tmp = Path(str(out) + ".tmp"); tmp.write_text(json.dumps(rows)); os.replace(tmp, out)
    sim.env.destroy()
    print("wrote", out)


if __name__ == "__main__":
    main()
