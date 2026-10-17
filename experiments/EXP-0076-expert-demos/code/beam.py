"""EXP-0076 expert / feasible episode runner: model-proposed, (optionally) simulator-verified beam search over TRUE Genesis states.

Per scenario a beam of B true states. At every tick each beam node is planned by the EXP-0075 model planner (wide_planner.plan_seq: pile-aware
bank + fetch / OT proposals, legal scoring, CEM [+ GD]); its k best DISTINCT legal first pushes (model order) are simulated from the node's true
state in parallel envs; the children are scored on the TRUE state and the B best (dedup'd) form the next beam. A scenario ends at the first tick
where a child reaches the success mass (>= 0.95; the fastest such child by physical time is kept) or at max_depth.
  B = 1, k = 1  -> the plain closed-loop model planner (feasible pipeline; the simulator is only 'reality').
  B = 1, k > 1  -> greedy with simulator verification of the model's top-k (expert).
  B > 1         -> beam search on true states (expert).
Many scenarios run at once (same cube count): env slots = sum over active scenarios of (beam size x k), new scenarios admitted as slots free.
Results: results/<tag>.json rewritten after every tick (atomic); every finished scenario is offered to the demo store (store.py).

usage: PYTHONPATH=. python -u beam.py --tag T --n 20 --ids ID ... | --strata ... --B 4 --k 8 --E 64 [--model ens2bal] [--H 1] [--gd 20]"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
import tasks as T
from simcore import Sim, REPO as _R   # noqa: F401  (sets sys.path for the EXP-0075 modules)
import wide_planner as wp
import store

DEV = "cuda"


def node_value(states, dist, mask):
    """True planner value (lyapunov - in-goal fraction; lower = better) and mass fraction for (N, n, >=3) states."""
    from simple_mpc.adapters import occ_for_scoring
    from simple_mpc.learned_mpc import lyap
    occ = occ_for_scoring(torch.as_tensor(states).float()[..., :3]); f = occ.reshape(len(occ), -1); m = T.mask_t(mask).to(occ.device).reshape(1, -1)
    mass = (f * m).sum(1) / f.sum(1)
    return (lyap(occ, dist.to(occ.device)) - mass).cpu().numpy(), mass.cpu().numpy()


def distinct(cands, k, thr=0.003):
    """Greedy: keep rows of cands (N, 4), in order, whose start AND end both differ by > thr from every kept row (max-norm over the 4 coords)."""
    keep = []
    for i in range(len(cands)):
        if len(keep) >= k:
            break
        if not keep or ((cands[keep] - cands[i]).abs().max(1).values > thr).all():
            keep.append(i)
    return cands[keep]


class Scenario:
    def __init__(self, sc, args, model):
        self.sc, self.id, self.n = sc, sc["id"], sc["n"]
        self.mask = T.goal_mask(sc["goal"]); self.mask_t = torch.from_numpy(self.mask.astype(np.float32))
        from Baselines.common.goals import dist_field_from_mask
        from simple_mpc.value_functions import uniform_layout
        self.dist = torch.from_numpy(dist_field_from_mask(self.mask.astype(bool))).float()
        self.slots = torch.from_numpy(uniform_layout(self.mask, self.n)).float()
        S0 = torch.tensor(sc["state0"], device="cpu").float(); v0, m0 = node_value(S0[None], self.dist, self.mask)
        self.beam = [dict(state=S0, actions=[], times=[], masses=[float(m0[0])], values=[float(v0[0])], states=[S0[:, :3].tolist()], quats=[S0[:, 3:7].tolist()],
                          pred=[], g=0.0, plan_s=[], prev0=sc.get("prev_end"))]
        self.depth, self.done, self.best, self.t0, self.n_sims, self.model = 0, False, None, time.time(), 0, model
        self.max_depth = args.max_depth or (12 if self.n <= 20 else 16)

    def record(self):
        b = self.best or min(self.beam, key=lambda z: z["values"][-1])
        return dict(id=self.id, n=self.n, init=self.sc["init"], kind=self.sc["kind"], goal=self.sc["goal"], solved=bool(self.best is not None),
                    pushes=len(b["actions"]), time_s=float(sum(b["times"])), final_mass=b["masses"][-1], actions=b["actions"], push_times=b["times"],
                    masses=b["masses"], values=b["values"], pred_cost=b["pred"], plan_s=b["plan_s"], phase=b.get("phase", []), acc_mass=b.get("acc", []), states=b["states"], quats=b["quats"], depth=self.depth,
                    wall_s=time.time() - self.t0, n_sims=self.n_sims, complete=self.done)


_MODELS = {}


def get_model(name):
    if name not in _MODELS:
        import closed_loop as CL
        _MODELS[name] = CL.MODELS[name]()
    return _MODELS[name]


def phase_cfg(args, mass):
    """--phases (JSON list): the first phase with mass < until_mass overrides args (model, H, pool, gd, n_pool, pop, iters); no phases -> args."""
    if not getattr(args, "phases", None):
        return args, 0
    import types
    for i, ph in enumerate(args.phases_list):
        if mass < ph.get("until_mass", 9.0):
            c = types.SimpleNamespace(**vars(args)); [setattr(c, kk.replace("-", "_"), v) for kk, v in ph.items() if kk != "until_mass"]
            return c, i
    return args, -1


def propose(scn, node, bank, args, k):
    """k distinct legal first pushes for one node, best model cost first. Returns (acts (m, 4) cuda, pred costs (m,))."""
    args, ph = phase_cfg(args, node["masses"][-1]); node["_phase"] = ph
    model = get_model(args.model)
    S = node["state"]; xy = S[:, :2].to(DEV)
    from model.retrieval.frame import yaw_from_quat
    legal = wp.make_legal_scorer(S[:, :2], yaw_from_quat(S[:, 3:7]))
    obj = wp.SeqObjective(model, S, scn.dist, scn.mask_t, 1.0, obj_mode="terminal" if args.H == 1 else "cumspill")
    kw = dict(extra=wp.fetch_candidates(xy, scn.mask_t), n_extra=getattr(args, "n_extra", 64), legal=legal, gd_steps=args.gd, topk=max(4 * k, 32), pool_mode=args.pool)
    if args.pool in ("ot", "mix"):
        kw["ot"] = wp.ot_candidates(xy, scn.mask_t, scn.slots); kw["cube_xy"] = xy
    out = wp.plan_seq(obj, bank, "cem", args.H, n_pool=args.n_pool, pop=args.pop, iters=args.iters, **kw)
    C = torch.cat([out["action"][None].to(DEV), out["top_seqs"][:, 0].to(DEV)]); c = torch.cat([torch.tensor([out["cost"]], device=DEV), out["top_costs"].to(DEV)])
    fin = torch.isfinite(c); C, c = C[fin], c[fin]; o = c.argsort(); C, c = C[o], c[o]
    if k > 1:                      # diversify with the model's best single pushes among the geometric proposals (fetch / OT), scored as executed
        geo = [kw["extra"]] + ([kw["ot"][0]] if "ot" in kw else [])
        G_ = torch.cat([g.to(DEV).float() for g in geo if len(g)]) if any(len(g) for g in geo) else torch.zeros(0, 4, device=DEV)
        if len(G_):
            a, _, ok = legal(wp.project_push(G_)[0]); a = a[ok]
            if len(a):
                cg = obj.cost(a[:, None, :]).to(DEV); o = cg.argsort(); a, cg = a[o], cg[o]
                if args.H == 1:    # same objective: merge in cost order
                    C = torch.cat([C, a]); c = torch.cat([c, cg]); o = c.argsort(); C, c = C[o], c[o]
                else:              # H>1 sequence costs are not comparable with single-push costs: planner rows first
                    C = torch.cat([C, a]); c = torch.cat([c, cg])
    if not len(C):                 # nothing legal from the planner: best legal bank rows by single-push model cost
        a, _, ok = legal(wp.project_push(bank.to(DEV).float())[0]); a = a[ok]
        c = obj.cost(a[:, None, :]).to(DEV); o = c.argsort(); C, c = a[o], c[o]
    mu_t = getattr(args, "time_mu", 0.0)
    if mu_t > 0:                   # time-aware re-ranking: among the top distinct candidates, order by cost + mu_t x physical push time from the plate pose
        D = distinct(C, max(16, k), args.distinct_mm / 1000.0); idx = [int(((C - d).abs().max(1).values).argmin()) for d in D]
        first = len(node["actions"]) == 0; prev = torch.zeros(2, device="cpu") if first else torch.tensor(node["actions"][-1][2:], device="cpu")
        tt = T.push_time_from(D.detach().cpu(), prev, first).float().to(DEV); sc_ = c[idx] + mu_t * tt; o = sc_.argsort()[:k]
        return D[o], c[idx][o], out
    D = distinct(C, k, args.distinct_mm / 1000.0); idx = [int(((C - d).abs().max(1).values).argmin()) for d in D]
    return D, c[idx], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--scenarios", default=str(REPO / "experiments/EXP-0076-expert-demos/results/scenarios.json"))
    ap.add_argument("--ids", nargs="*", default=None); ap.add_argument("--strata", nargs="*", default=None, help="init-kind pairs, e.g. scattered-disc")
    ap.add_argument("--all", action="store_true", help="every scenario of the file with this n");
    ap.add_argument("--first", type=int, default=0); ap.add_argument("--count", type=int, default=4, help="scenarios per stratum (from --first)")
    ap.add_argument("--B", type=int, default=4); ap.add_argument("--k", type=int, default=8); ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--model", default="ens2bal"); ap.add_argument("--H", type=int, default=1); ap.add_argument("--gd", type=int, default=20)
    ap.add_argument("--n-pool", type=int, default=256); ap.add_argument("--pop", type=int, default=256); ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--pool", default="fl"); ap.add_argument("--bank", type=int, default=4096)
    ap.add_argument("--mu", type=float, default=0.0, help="beam score = true value + mu x elapsed physical seconds")
    ap.add_argument("--success", type=float, default=0.95, help="mass needed to accept a child as solved (robust demos: > 0.95)")
    ap.add_argument("--pipeline-name", default=None, help="store pipeline directory override (e.g. expert_robust)")
    ap.add_argument("--recheck", action="store_true", help="re-simulate the top 2B children in other env slots; select on the mean value, accept success only if both sims >= 0.95")
    ap.add_argument("--n-extra", type=int, default=64, help="fetch proposals replacing bank rows in the initial pool")
    ap.add_argument("--time-mu", type=float, default=0.0, help="proposal re-ranking: model cost + time_mu x physical push time (value units per s)")
    ap.add_argument("--distinct-mm", type=float, default=3.0); ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--no-store", action="store_true")
    ap.add_argument("--phases", default=None, help='JSON list of phase overrides by current mass, e.g. [{"until_mass":0.8,"H":2,"pool":"ot"},{"H":1}]')
    ap.add_argument("--results-dir", default=str(REPO / "experiments/EXP-0076-expert-demos/results/runs"))
    a = ap.parse_args(); a.phases_list = json.loads(a.phases) if a.phases else None; T.SUCCESS = a.success
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    allsc = json.load(open(a.scenarios))["scenarios"]
    if a.all:
        todo = [s for s in allsc if s["n"] == a.n]
    elif a.ids:
        todo = [s for s in allsc if s["id"] in set(a.ids)]
    else:
        todo = []
        for st in a.strata:
            init, kind = st.split("-"); todo += [s for s in allsc if s["n"] == a.n and s["init"] == init and s["kind"] == kind][a.first:a.first + a.count]
    assert all(s["n"] == a.n for s in todo), "all scenarios of one run must share n"
    res_p = Path(a.results_dir) / f"{a.tag}.json"; res_p.parent.mkdir(parents=True, exist_ok=True)
    out = json.loads(res_p.read_text()) if res_p.exists() else dict(config=vars(a), scenarios={})
    todo = [s for s in todo if not out["scenarios"].get(s["id"], {}).get("complete")]
    print(f"{len(todo)} scenarios to run, B={a.B} k={a.k} E={a.E}", flush=True)
    method = dict(B=a.B, k=a.k, model=a.model, H=a.H, gd=a.gd, n_pool=a.n_pool, pop=a.pop, iters=a.iters, pool=a.pool, mu=a.mu, time_mu=a.time_mu, recheck=a.recheck, tag=a.tag, phases=a.phases_list,
                  pipeline=a.pipeline_name or ("feasible" if (a.B == 1 and a.k == 1) else "expert"), success=a.success)
    model = get_model(a.model)
    for ph in (a.phases_list or []):
        if "model" in ph:
            get_model(ph["model"])
    sim = Sim(a.n, a.E); per = a.B * a.k; S_max = max(1, a.E // per)
    queue, active = list(todo), []
    t_start = time.time(); tick = 0

    def save():
        tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(out)); os.replace(tmp, res_p)

    while queue or active:
        while queue and len(active) < S_max:
            active.append(Scenario(queue.pop(0), a, model))
        nodes = [(s, nd) for s in active for nd in s.beam]
        # banks for every node (one simulator call per E nodes)
        banks = []
        for i in range(0, len(nodes), a.E):
            banks += list(sim.bank(torch.stack([nd["state"] for _, nd in nodes[i:i + a.E]]), a.bank))
        kids = []                                     # (scenario, parent, act, pred)
        t_plan = time.time()
        for (s, nd), bank in zip(nodes, banks):
            k = 1 if a.k == 1 else max(1, min((a.E // max(1, len(active))) // len(s.beam), a.B * a.k // len(s.beam)))
            t_ = time.time(); acts, pc, _ = propose(s, nd, bank, a, k); nd["_plan_s"] = time.time() - t_
            kids += [(s, nd, acts[j].cpu(), float(pc[j])) for j in range(len(acts))]
        t_plan = time.time() - t_plan; t_sim = time.time(); child_states = []
        for i in range(0, len(kids), a.E):
            ch = kids[i:i + a.E]; child_states += list(sim.step_from(torch.stack([p["state"] for _, p, _, _ in ch]), torch.stack([x for _, _, x, _ in ch])))
        t_sim = time.time() - t_sim
        by = {}
        for (s, p, act, pc), cs in zip(kids, child_states):
            by.setdefault(s.id, []).append((p, act, pc, cs))
        rc = {}
        if a.recheck:                # re-simulate each scenario's top 2B children (by first-sim value) in shuffled env slots (different slot history)
            picks = []
            for s in active:
                ch = by.get(s.id, [])
                if ch:
                    V1, _ = node_value(torch.stack([c[3] for c in ch]), s.dist, s.mask); picks += [(s.id, int(j)) for j in np.argsort(V1)[: 2 * a.B]]
            picks = [picks[i] for i in np.random.permutation(len(picks))]; st2 = []
            for i in range(0, len(picks), a.E):
                cc = picks[i:i + a.E]
                st2 += list(sim.step_from(torch.stack([by[sid][j][0]["state"] for sid, j in cc]), torch.stack([by[sid][j][1] for sid, j in cc])))
            rc = {k: v for k, v in zip(picks, st2)}
        for s in list(active):
            ch = by.get(s.id, []); s.n_sims += len(ch); s.depth += 1
            if not ch:
                continue
            V, M = node_value(torch.stack([c[3] for c in ch]), s.dist, s.mask); new = []; ACC = M
            if a.recheck:
                idx = [j for j in range(len(ch)) if (s.id, j) in rc]; s.n_sims += len(idx)
                V2, M2 = node_value(torch.stack([rc[(s.id, j)] for j in idx]), s.dist, s.mask)
                ch = [ch[j] for j in idx]; V = (V[idx] + V2) / 2; ACC = np.minimum(M[idx], M2); M = M[idx]
            for (p, act, pc, cs), v, m, acc in zip(ch, V, M, ACC):
                first = len(p["actions"]) == 0; prev = torch.zeros(2, device="cpu") if first else torch.tensor(p["actions"][-1][2:], device="cpu")
                if first and p.get("prev0") is not None:      # continued episode (tail branch): transit from the recorded plate pose, lift charged
                    first, prev = False, torch.tensor(p["prev0"], device="cpu")
                dt = float(T.push_time_from(act[None], prev, first)[0])
                new.append(dict(state=cs, actions=p["actions"] + [act.tolist()], times=p["times"] + [dt], masses=p["masses"] + [float(m)], values=p["values"] + [float(v)],
                                states=p["states"] + [cs[:, :3].tolist()], quats=p["quats"] + [cs[:, 3:7].tolist()], pred=p["pred"] + [pc], g=p["g"] + dt,
                                plan_s=p["plan_s"] + [p.get("_plan_s", 0.0)], phase=p.get("phase", []) + [p.get("_phase", 0)], acc=p.get("acc", []) + [float(acc)]))
            sol = [z for z in new if z["masses"][-1] >= T.SUCCESS and z["acc"][-1] >= T.SUCCESS]
            if sol:
                s.best = min(sol, key=lambda z: z["g"]); s.done = True
            elif s.depth >= s.max_depth:
                s.beam = [min(new, key=lambda z: z["values"][-1])]; s.done = True
            else:
                new.sort(key=lambda z: z["values"][-1] + a.mu * z["g"]); keep = []
                for z in new:
                    if len(keep) >= a.B:
                        break
                    if all((z["state"][:, :2] - q["state"][:, :2]).norm(dim=1).mean() > 5e-4 for q in keep):
                        keep.append(z)
                s.beam = keep
            out["scenarios"][s.id] = dict(s.record(), method=method)
            if s.done:
                rec = out["scenarios"][s.id]
                if not a.no_store:
                    store.offer(rec, method)
                active.remove(s)
                print(f"  done {s.id}: solved={rec['solved']} pushes={rec['pushes']} time={rec['time_s']:.2f}s mass={rec['final_mass']:.3f} sims={rec['n_sims']} wall={rec['wall_s']:.0f}s", flush=True)
        tick += 1; save()
        print(f"tick {tick}: active {len(active)} queue {len(queue)} nodes {len(nodes)} kids {len(kids)} plan {t_plan:.1f}s sim {t_sim:.1f}s total {time.time() - t_start:.0f}s", flush=True)
    sim.destroy(); print("wrote", res_p)


if __name__ == "__main__":
    main()
