"""EXP-0070 RUN-0003+: closed-loop MPC that minimises expected total task time,
    J(a) = t(a | plate pose) + etc_g(V_soft(s) + dV_model(s, a)) - etc_g(V_soft(s)),
vs the lyapunov baseline (minimise dV_model), on the batched Genesis runner.

Cells (--cells, comma list):
  lyap      ModelObjective as in every earlier record (baseline).
  time      TimeObjective with the per-goal monotone map etc_g (--etc-map, results/etc_map_*.json).
  time_lin  same, etc_g replaced by its linear slope (isolates the t(a) penalty from the shape of etc).
  rate      greedy progress rate: minimise dV_model / (t_plan + t(a))  (a "do the cheapest useful push" baseline).
The planner, model, budget, candidate bank, legalisation and planner seed are identical across cells, so
paired episodes (same goal, start, seed) differ only in the objective. The objective is swapped by
replacing simple_mpc.learned_mpc.ModelObjective inside this process (experiment-local; the shared
module is not edited). The plate pose t(a) needs is tracked per env from the executed pushes.
Checkpoint: results/<tag>.json rewritten after every step (atomic).
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batched_closed_loop import goal_dist, goal_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring
from simple_mpc.genesis_oracle import GenesisOracleEnv
import simple_mpc.learned_mpc as LM
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, run_episodes_batched, lyap
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
from action_time import action_time, HOME

HERE = REPO / "experiments/EXP-0070-time-optimal-mpc"
K = 32
BaseObjective = LM.ModelObjective


def etc_torch(f, V, linear=False):
    kv = torch.tensor(f["knots_V"], device=V.device, dtype=V.dtype)
    ke = torch.tensor(f["knots_etc"], device=V.device, dtype=V.dtype)
    if linear:
        return f["slope_lin"] * V
    if "power" in f:                       # default planning map: smooth, strictly increasing
        return f["power"]["c"] * V.clamp_min(1e-6) ** f["power"]["p"]
    i = torch.searchsorted(kv, V.clamp(kv[0], kv[-1]).contiguous()).clamp(1, len(kv) - 1)
    x0, x1, y0, y1 = kv[i - 1], kv[i], ke[i - 1], ke[i]
    mid = y0 + (y1 - y0) * (V - x0) / (x1 - x0).clamp_min(1e-9)
    lo = ke[0] * V.clamp_min(0) / kv[0].clamp_min(1e-9)
    hi = ke[-1] + f["slope_hi"] * (V - kv[-1])
    return torch.where(V < kv[0], lo, torch.where(V > kv[-1], hi, mid))


class TimeObjective(BaseObjective):
    """Cost in SECONDS: action time + change in expected remaining time. `ctx` (per env, mutable)
    carries the plate pose, the goal's etc map, mode and the planning time charged per push."""

    def __init__(self, adapter, particles, dist, ctx=None, **kw):
        super().__init__(adapter, particles, dist, **kw)
        self.ctx, self.raw = ctx, False
        self.Vs = lyap(occ_for_scoring(particles[None].float()).to(self.ad.device), self.dist)[0]
        self.p_prev = torch.tensor(ctx["p_prev"], device=self.ad.device, dtype=torch.float32)
        if ctx["mode"] == "blend":           # v2 (RUN-0010): mass-gap readout, true soft in-goal mass of s
            occ_s = occ_for_scoring(particles[None].float()).to(self.ad.device)
            self.gm = ctx["map2"]; self.mask2 = torch.from_numpy(goal_mask(ctx["goal"])).to(self.ad.device).float()
            self.mrel_s = float((occ_s * self.mask2).sum() / occ_s.sum().clamp_min(1e-6)) / self.gm["mopt"]
        elif ctx["mode"] != "rate":
            self.etc0 = etc_torch(ctx["etc"], self.Vs[None], ctx["mode"] == "time_lin")[0]

    def __call__(self, act):
        dv = super().__call__(act)
        if self.raw:                       # time_rerank: the SEARCH runs on plain lyapunov dV
            return dv
        t = action_time(act.float(), self.p_prev, bool(self.ctx["first"]))
        if self.ctx["mode"] == "rate":
            return dv / (t + self.ctx["t_plan"])
        return t + etc_torch(self.ctx["etc"], self.Vs + dv, self.ctx["mode"] == "time_lin") - self.etc0


def make_objective(adapter, particles, dist, ctx=None, **kw):
    if ctx is None:
        return BaseObjective(adapter, particles, dist, **kw)
    return TimeObjective(adapter, particles, dist, ctx=ctx, **kw)


RERANK_K = 64
_orig_plan = LM.plan


def blend_select(obj, A, dv):
    """v2 selection (user's loop, RUN-0010): pool A (k, 4) from the lyap search; J = t + f(r') - f(r) on the mass-gap
    readout r = max(0, theta - mass_rel) with the calibrated model offset; pick argmin (1-lam) z(dV) + lam z(J)."""
    gm, key = obj.gm, obj.ctx["map_key"]
    m = obj.members[0]; occ_in = obj._occ_in[0]
    with torch.no_grad():
        pr = m.predict_step(occ_in.expand(len(A), -1, -1).contiguous(), A)
    fin = lambda o: (o * obj.mask2).reshape(len(o), -1).sum(1) / o.reshape(len(o), -1).sum(1).clamp_min(1e-6)
    dM = ((fin(pr) - fin(occ_in)) / gm["mopt"]).double().cpu().numpy()
    r0 = max(0.0, gm["theta"] - obj.mrel_s)
    cal = gm.get(key.split("|")[0] + "|offset")
    off = float(np.interp(r0, cal["r"], cal["bias"])) if cal else 0.0
    r1 = np.maximum(0.0, r0 - dM - off)
    f = gm[key]; fr = lambda r: f["c"] * np.clip(r, 0, None) ** f["p"]
    t = action_time(A.float(), obj.p_prev, bool(obj.ctx["first"])).double().cpu().numpy()
    J = (0.0 if obj.ctx.get("no_time") else t) + fr(r1) - fr(r0)      # blendnt<lam>: ablation without t(a)
    dv = np.asarray(dv, float); lam = obj.ctx["lam"]
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    i = int(((1 - lam) * z(dv) + lam * z(J)).argmin())
    return i, float(J[i]), float(J.min())


def meta_budget(obj):
    """Rule A (meta_replay.py): stop at the first checkpoint whose predicted net value
    f(r0) - f(r0 - r0 rho_k) - T_k - tau_k does not improve on the next; r0 = the state's true mass gap."""
    tb, CK = obj.ctx["meta"]["tables"], obj.ctx["meta"]["CK"]
    g = obj.ctx["goal"]; fam = "quadrant" if g.startswith("quad") else ("two_squares" if g == "two_squares" else "letter")
    t = tb.get(fam, tb["ALL"]); f = obj.gm[obj.ctx["map_key"]]
    r0 = max(0.0, obj.gm["theta"] - obj.mrel_s)
    if r0 <= 1e-6:
        return CK[0]
    F = lambda r: f["c"] * max(r, 0.0) ** f["p"]
    nh = [F(r0) - F(r0 - r0 * min(1.0, t["rho"][k])) - t["T"][k] - CK[k] for k in range(len(CK))]
    k = 0
    while k + 1 < len(CK) and nh[k + 1] > nh[k]:
        k += 1
    obj.ctx.setdefault("meta_log", []).append(CK[k])
    return CK[k]


def plan_rerank(objective, candidates, planner, budget_s, **kw):
    """`time_rerank` cell: search exactly as the lyap baseline (same planner, budget, seed), keep the
    RERANK_K best-predicted actions it evaluated, execute the one with the lowest
    J = t(a) + etc(V + dV) - etc(V). Added after RUN-0003's first 2 chunks showed CEM ON J collapsing
    into short cheap pushes (the smooth t(a) basin captures the elites). Re-ranking costs < 1 ms,
    outside the budget."""
    if isinstance(objective, TimeObjective) and objective.ctx["mode"] == "blend":
        if objective.ctx.get("meta"):
            budget_s = meta_budget(objective)
        objective.raw = True
        out = _orig_plan(objective, candidates, planner, budget_s, topk=RERANK_K, **kw)
        A = out.pop("topk_actions").to(objective.ad.device); dv = out.pop("topk_dv").cpu().numpy()
        i, Ji, Jmin = blend_select(objective, A, dv)
        out.update(action=A[i].cpu(), pred_dv=float(dv[i]))       # pred_dv stays the model lyap dV (the search value)
        return out
    if not (isinstance(objective, TimeObjective) and objective.ctx["mode"] == "time_rerank"):
        return _orig_plan(objective, candidates, planner, budget_s, **kw)
    objective.raw = True
    out = _orig_plan(objective, candidates, planner, budget_s, topk=RERANK_K, **kw)
    objective.raw = False
    A = out.pop("topk_actions").to(objective.ad.device); dv = out.pop("topk_dv")
    with torch.no_grad():
        J = objective(A)
    i = int(J.argmin())
    out.update(action=A[i].cpu(), pred_dv=float(J[i]), rerank_rank=i, rerank_dv=float(dv[i]), lyap_best_dv=float(dv.min()))
    return out


def load_etc(path):
    m = json.loads(Path(path).read_text()); out = {}
    for g, v in m.items():
        if "pass1" not in v:
            continue
        f = dict(v["pass2"] if v["n_completed"] < 0.5 * v["n_episodes"] else v["pass1"])
        kv, ke = np.array(f["knots_V"]), np.array(f["knots_etc"])
        f["slope_lin"] = float(np.polyfit(np.r_[0.0, kv], np.r_[0.0, ke], 1)[0])   # through the knots + origin
        f["which"] = "pass2" if v["n_completed"] < 0.5 * v["n_episodes"] else "pass1"
        out[g] = f
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--model", default="nfd_3ch_randlen")
    ap.add_argument("--planner", default="cem")
    ap.add_argument("--cells", default="lyap,time")
    ap.add_argument("--goals", nargs="*", default=["quadrant_0", "quadrant_3", "two_squares", "letter_O", "letter_T",
                                                    "letter_S", "letter_X"])
    ap.add_argument("--starts", type=int, nargs="*", default=list(range(40, 48)))
    ap.add_argument("--etc-map", default=str(HERE / "results/etc_map_v0.json"))
    ap.add_argument("--map2", default=str(HERE / "results/etc_map_m0.json"), help="etc_v2 map (blend<lam> cells)")
    ap.add_argument("--map-key", default="gap|td")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--n-cand", type=int, default=1024)
    ap.add_argument("--cem-pop", type=int, default=1024)
    ap.add_argument("--cem-elite-frac", type=float, default=0.25)
    ap.add_argument("--bank-per-env", type=int, default=16384)
    ap.add_argument("--planner-seed", type=int, default=0)
    a = ap.parse_args()
    LM.ModelObjective = make_objective            # run_episodes_batched / legal_fallback look it up here
    LM.plan = plan_rerank
    res_p = HERE / f"results/{a.tag}.json"
    etc = load_etc(a.etc_map)
    map2 = json.loads(Path(a.map2).read_text())
    meta_tab = json.loads((HERE / "results/meta_schedule.json").read_text())     # rule A tables (meta_replay.py)
    cells = a.cells.split(",")
    specs = [dict(model=a.model, planner=a.planner, goal=g, start=s, cell=c) for g in a.goals for s in a.starts for c in cells]   # cells interleaved: paired episodes share a chunk
    done = json.loads(res_p.read_text()) if res_p.exists() else {"config": vars(a), "etc_used": etc, "map2_used": map2, "episodes": []}
    done["episodes"] = [e for e in done["episodes"] if e.get("complete")]
    have = {(e["goal"], e["start"], e["cell"]) for e in done["episodes"]}
    todo = [s for s in specs if (s["goal"], s["start"], s["cell"]) not in have]
    rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
    starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in a.starts}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    adapter = make_occ_adapter(a.model, "cuda", "corner")
    dists = {g: goal_dist(g) for g in a.goals}
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    ctxs = []                                    # per env of the current chunk (None for lyap)

    def execute_batch(acts):
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float())
        z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
        sim.update_material_state()
        for k, c in enumerate(ctxs):
            if c is not None:
                c["p_prev"] = [float(acts[k, 2]), float(acts[k, 3])]; c["first"] = False
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def sample(m):
        s, e, _ = sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)
        return torch.cat([s[..., :2], e[..., :2]], -1).float().cpu()

    def legalize(acts, envs=None):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        st = sim._particle_state.detach().cpu().float()
        if envs is not None:
            st = st[envs]
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001,
                               box=(OCC_BOUNDS["x_min"] + LM.XY_MARGIN, OCC_BOUNDS["x_max"] - LM.XY_MARGIN))

    o = BaseObjective(adapter, starts[a.starts[0]], dists[a.goals[0]])
    LM.plan(o, torch.zeros(8, 4) + 0.01, "cem", 0.1)          # warm-up
    for c0 in range(0, len(todo), K):
        chunk = todo[c0:c0 + K]
        padded = chunk + [chunk[-1]] * (K - len(chunk))
        st = torch.stack([starts[s["start"]] for s in padded])
        sim.set_particle_state(st[:, :, :3].contiguous().to(gs.device), st[:, :, 3:7].contiguous().to(gs.device))
        sim.update_material_state()
        p0 = sim._particle_state[:, :, :7].detach().cpu().float().clone()
        ctxs.clear(); eps = []
        for s in padded:
            cell, _, bspec = s["cell"].partition("@")          # "<cell>@<budget s>" or "<cell>@meta" (RUN-0015)
            b_cell = float(bspec) if bspec and bspec != "meta" else a.budget
            s = dict(s, cell=cell)
            if s["cell"] == "lyap":
                ctx = None
            elif s["cell"].startswith("blend"):
                nt = s["cell"].startswith("blendnt")
                ctx = dict(mode="blend", lam=float(s["cell"][7 if nt else 5:]), no_time=nt, map2=map2["goals"][s["goal"]], map_key=a.map_key,
                           goal=s["goal"], etc=None, p_prev=list(HOME), first=True, t_plan=a.budget)
            else:
                ctx = dict(mode=s["cell"], etc=etc[s["goal"]], p_prev=list(HOME), first=True, t_plan=a.budget)
            if ctx is not None and bspec == "meta":
                ctx["meta"] = meta_tab
            ctxs.append(ctx)
            eps.append(dict(adapter=adapter, dist=dists[s["goal"]], planner=a.planner, budget_s=b_cell,
                            n_cand=a.n_cand, plan_kw=dict(cem_pop=a.cem_pop, cem_elite_frac=a.cem_elite_frac),
                            obj_kw={} if ctx is None else dict(ctx=ctx)))
        t0 = time.time()
        live = [dict(s, complete=False) for s in chunk]
        done["episodes"] += live

        def ckpt(t, recs):
            for e, r in zip(live, recs):
                e.update(r); e["wall_s"] = time.time() - t0
            tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, res_p)
            print(f"chunk {c0 // K}: step {t + 1}/{a.steps} ({time.time() - t0:.0f}s)", flush=True)
        recs = run_episodes_batched(eps, p0, execute_batch, sample, a.steps, n_cand=a.n_cand,
                                    bank_per_env=a.bank_per_env, on_step=ckpt, record_states=True,
                                    legalize=legalize, legalize_fallback=True, planner_seed=a.planner_seed)
        for e in live:
            e["complete"] = True
        ckpt(a.steps - 1, recs)
    env.destroy()
    print("wrote", res_p)


if __name__ == "__main__":
    main()
