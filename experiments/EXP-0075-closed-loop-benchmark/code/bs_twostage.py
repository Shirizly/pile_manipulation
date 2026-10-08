"""EXP-0075 two-stage model-switching planner test.  Stage 1: the cheapest model (vanilla64) scores a large pool (counts calibrated to T1 = 0.6 s of unit cost); Stage 2: the top candidates are re-scored by the ensemble (ens128) and
refined with GD for the remaining time (~0.8 s of GD steps), total <= 1.5 s per planning call (counts from results/timing_units*.json, not live clocks).
Three modes per task (10 tasks), all executed in the simulator, 4 pushes:
  greedy : H=1 re-planned from the TRUE state before every push (closed loop)
  open   : H=4 planned once from the start, executed open loop
  hybrid : the same H=4 plan; push 1 as planned, then before each of pushes 2..4 the planned action is RE-OPTIMISED on the real state (suffix of the plan kept fixed, candidates = noise around the planned action + fresh proposals,
           two-stage again).  Improvement of hybrid over open at push 2 (first divergence, identical real state) = (dv of hybrid push 2) - (dv of open push 2).
usage: bs_twostage.py [n_tasks]   -> results/budget_study/twostage_results.json + states_twostage.npz"""
import sys, os
from bs_lib import *
from bs_lib import _T2, _full
T1, TG = 0.6, 0.8                       # seconds of unit cost: stage 1 (vanilla64), stage 2 GD (ens128)
VC = 0.130 / 1000 / 4                   # vanilla64 cost per pushed rollout step (s)
ST1, ST4 = _T2["ens128 (zoom+vanilla)"]["gd_step_H1_n8_ms"] / 1000, _T2["ens128 (zoom+vanilla)"]["gd_step_H4_n8_ms"] / 1000


def gd_step_time(L):
    return ST1 + (L - 1) / 3 * (ST4 - ST1)


def two_stage(S0, goal, h, suffix=None, center=None, seed=0, k=8, lr=2e-3):
    """optimise h pushes (then the fixed suffix) from state S0; center (h,4): planned action(s) to perturb (hybrid mode). Returns dict."""
    torch.manual_seed(seed); mv, me = get_model("vanilla64"), get_model("ens128"); ov, oe = make_obj(mv, S0, goal), make_obj(me, S0, goal); L = h + (0 if suffix is None else len(suffix)); t0 = time.time()
    n1 = int(T1 / (VC * L)); steps = int(TG / gd_step_time(L)); first = sample_first(ov, max(n1, 64))
    if center is None:
        if h == 1: S = first[torch.randint(0, len(first), (n1,), device=DEV)][:, None]; C = ov.cost(_full(S, None, suffix)); i = C.argsort(); S, C = S[i], C[i]
        else: S, C = cem_pool(ov, first, h, n1, suffix=suffix)
    else:                                                    # hybrid refinement of one planned push: half noise around it, half fresh proposals
        c = center.to(DEV).float().reshape(1, 1, 4); noise = proj((c.reshape(1, 4) + 0.008 * torch.randn(n1 // 2, 4, device=DEV)).reshape(-1, 4)).reshape(-1, 1, 4)
        S = torch.cat([c, noise, first[torch.randint(0, len(first), (n1 - n1 // 2,), device=DEV)][:, None]]); C = ov.cost(_full(S, None, suffix)); i = C.argsort(); S, C = S[i], C[i]
    v_best = float(C[0]); cand = S[:k]
    if center is not None: cand = torch.cat([cand, center.to(DEV).float().reshape(1, 1, 4)])
    ce = oe.cost(_full(cand, None, suffix)); j = int(ce.argmin()); best, bc = cand[j], float(ce[j]); pre = bc
    sg, cg = gd_refine(oe, cand, max(steps, 1), lr, suffix=suffix)
    if cg < bc: best, bc = sg, cg
    return dict(seq=best.cpu(), ens_cost=bc, ens_cost_pre_gd=pre, stage1_vanilla_cost=v_best, n1=n1, gd_steps=steps, plan_time_s=time.time() - t0, L=L)


def main():
    nt = int(sys.argv[1]) if len(sys.argv) > 1 else 10; tasks = TASKS[:nt]; sim = Sim(32); n = nt * 3
    S0s = [START(s) for _, s in tasks]; p = sim.reset([S0s[i // 3] for i in range(n)]); goals = [tasks[i // 3][0] for i in range(n)] + [tasks[0][0]] * (32 - n)
    mode = lambda i: ("greedy", "open", "hybrid")[i % 3]; vals = [true_value(p, goals)]; parts = [p.numpy()]; log = []
    plans = [two_stage(S0s[t], tasks[t][0], 4, seed=100 + t) for t in range(nt)]; print("plans", [(round(x["ens_cost"], 3), round(x["stage1_vanilla_cost"], 3), x["n1"], x["gd_steps"], round(x["plan_time_s"], 1)) for x in plans][:3], flush=True)
    planned_all, legal_all = [], []
    for j in range(4):
        acts = torch.zeros(32, 4); entry = {}
        for i in range(n):
            t = i // 3; m = mode(i); pk = plans[t]["seq"]
            if m == "open" or (m == "hybrid" and j == 0): acts[i] = pk[j]
            elif m == "greedy": r = two_stage(p[i], goals[i], 1, seed=1000 + 10 * t + j); acts[i] = r["seq"][0]; entry[i] = r
            else: r = two_stage(p[i], goals[i], 1, suffix=pk[j + 1:].to(DEV) if j < 3 else None, center=pk[j], seed=2000 + 10 * t + j); acts[i] = r["seq"][0]; entry[i] = r
        acts[n:] = acts[0]; a, sh, ok, p = sim.push(acts); planned_all.append(acts.cpu().numpy()); legal_all.append(a.cpu().numpy()); vals.append(true_value(p, goals)); parts.append(p.numpy()); log.append({i: {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in e.items()} for i, e in entry.items()})
        print("push", j + 1, "done", flush=True)
    V = np.stack(vals, 1); out = []
    for t, (g, s) in enumerate(tasks):
        r = dict(goal=g, start=s, plan_pred=plans[t]["ens_cost"], plan_stage1_vanilla=plans[t]["stage1_vanilla_cost"], plan_pre_gd=plans[t]["ens_cost_pre_gd"], plan_time_s=plans[t]["plan_time_s"], plan_gd_steps=plans[t]["gd_steps"], plan_n1=plans[t]["n1"])
        for mi, m in enumerate(("greedy", "open", "hybrid")):
            i = 3 * t + mi; r[m] = dict(true_terminal=float(V[i, -1] - V[i, 0]), true_per_push=(V[i, 1:] - V[i, 0]).tolist(), step_dv=np.diff(V[i]).tolist())
        r["hybrid_minus_open_step2"] = r["hybrid"]["step_dv"][1] - r["open"]["step_dv"][1]; r["hybrid_minus_open_final"] = r["hybrid"]["true_terminal"] - r["open"]["true_terminal"]
        r["hybrid_refine"] = [{"ens_cost": log[j][3 * t + 2]["ens_cost"], "pre_gd": log[j][3 * t + 2]["ens_cost_pre_gd"], "plan_time_s": log[j][3 * t + 2]["plan_time_s"]} for j in range(1, 4)]; r["greedy_plan_time_s"] = [log[j][3 * t]["plan_time_s"] for j in range(4)]; out.append(r)
        print(g, s, {m: round(r[m]["true_terminal"], 3) for m in ("greedy", "open", "hybrid")}, "pred", round(plans[t]["ens_cost"], 3), "hybrid-open step2", round(r["hybrid_minus_open_step2"], 3), flush=True)
    json.dump(out, open(BS / "twostage_results.json", "w"), indent=1); np.savez_compressed(BS / "states_twostage.npz", parts=np.stack(parts, 1)[:n], planned=np.stack(planned_all, 1)[:n], legalised=np.stack(legal_all, 1)[:n], vals=V[:n]); sim.close()


if __name__ == "__main__":
    main()
