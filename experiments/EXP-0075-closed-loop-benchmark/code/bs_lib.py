"""EXP-0075 budget study library: models, time-calibrated budgets, CEM / random pool / GD planners (H = 1 or 4), greedy open-loop, simulator runner that saves every transition.
Budgets are converted to EVALUATION COUNTS with the measured unit times (results/timing_units*.json) so parallel processes do not distort them: budget B seconds, model m, horizon H ->
 n_cem = f_cem * B / cost_fwd(m, H); gd_steps = f_gd * B / step_time(m, H, batch 8).  (f_cem = f_gd = 0.5 for 'cem_gd' / 'rand_gd'; f_cem = 1 for 'cem'.)"""
import json, sys, time, os
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code")); sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
import wide_planner as wp
from intensive import propose_from_occ, rand_later, proj, R74
DEV = wp.DEV; RES = REPO / "experiments/EXP-0075-closed-loop-benchmark/results"; BS = RES / "budget_study"; BS.mkdir(exist_ok=True)
TASKS = [("letter_T_w20", 40), ("letter_O_w20", 41), ("letter_S_w20", 42), ("letter_X_w20", 43), ("letter_T_w20", 44), ("letter_O_w20", 45), ("letter_S_w20", 46), ("letter_X_w20", 47), ("letter_O_w20", 40), ("letter_X_w20", 41)]
MODELS = {   # key -> (timing key, members)
    "ens128": ("ens128 (zoom+vanilla)", lambda: [wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")]),
    "zoom128": ("zoom128", lambda: [wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth")]),
    "vanilla128": ("vanilla128", lambda: [wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")]),
    "zoom64": ("zoom64 (f8)", lambda: [wp.Member("zoom", 64, R74 + "z64_f8_ms4/unet_best.pth")]),
    "vanilla64": ("vanilla64 (f4)", lambda: [wp.Member("world", 64, R74 + "w64_s0_ms4/unet_best.pth", feats=(4, 8, 16))]),
}
_T1, _T2 = json.load(open(RES / "timing_units.json")), json.load(open(RES / "timing_units2.json"))
_rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
START = lambda s: _rows.states[(_rows.slate_idx == s).nonzero()[0, 0]].float()
_cache = {}


def get_model(key):
    if key not in _cache:
        _cache[key] = wp.WideEns(MODELS[key][1](), balance=True, name=key)
    return _cache[key]


def unit(key, H):
    tk = MODELS[key][0]; return _T1[tk][f"fwd_H{H}_chunk256_ms_per_seq"] / 1000, _T2[tk][f"gd_step_H{H}_n8_ms"] / 1000


def counts(key, H, B, planner):
    c, g = unit(key, H)
    if planner == "cem":
        return dict(n_eval=max(40, int(B / c)), gd_steps=0)
    return dict(n_eval=max(40, int(0.5 * B / c)), gd_steps=max(1, int(0.5 * B / g)))


def make_obj(model, S0, goal):
    return wp.SeqObjective(model, S0, goal_dist(goal), torch.from_numpy(goal_mask(goal)), 1.0, chunk=256)


def sample_first(obj, n):
    return propose_from_occ(obj.model.occ0[0], max(n, 64))


def pool_seqs(first, n, H):
    f = first[torch.randint(0, len(first), (n,), device=DEV)][:, None]
    return torch.cat([f, rand_later(first, n, H)], 1) if H > 1 else f


def cem_pool(obj, first, H, n_eval, random_only=False, prefix=None, iters=4):
    """total n_eval evaluations: pool n0 + iters x pop n0 (n0 = n_eval/(iters+1)); random_only: n_eval random sequences, no refit. prefix (j,4): fixed pushes before the optimised H pushes (greedy open loop).
    Returns (seqs (M,H,4), costs (M,)) of everything sampled (sorted by cost, best first). costs are of the full prefix+sequence."""
    def cost(s):
        full = s if prefix is None else torch.cat([prefix[None].expand(len(s), -1, -1), s], 1); return obj.cost(full)
    if random_only:
        S = pool_seqs(first, n_eval, H); C = cost(S); i = C.argsort(); return S[i], C[i]
    n0 = max(8, n_eval // (iters + 1)); S = [pool_seqs(first, n0, H)]; C = [cost(S[0])]; n_el = max(2, n0 // 8); el = S[0][C[0].argsort()[:n_el]]
    for _ in range(iters):
        mean, std = el.mean(0), el.std(0).clamp_min(1e-3)
        s = proj((mean + std * torch.randn(n0, H, 4, device=DEV)).reshape(-1, 4)).reshape(n0, H, 4); c = cost(s); S.append(s); C.append(c); el = s[c.argsort()[:n_el]]
    S, C = torch.cat(S), torch.cat(C); i = C.argsort(); return S[i], C[i]


def gd_refine(obj, init, steps, lr, prefix=None):
    """Adam through the differentiable model on the (k,H,4) init (after prefix). Returns best (H,4), its cost."""
    x = init.clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=lr); H = init.shape[1]
    for _ in range(steps):
        opt.zero_grad(); xp = proj(x.reshape(-1, 4)).reshape(-1, H, 4); full = xp if prefix is None else torch.cat([prefix[None].expand(len(xp), -1, -1), xp], 1)
        pr = obj.model._predict(full[..., :2], full[..., 2:], grad=True)[-1]; obj._value(pr).sum().backward(); opt.step()
    with torch.no_grad():
        xp = proj(x.detach().reshape(-1, 4)).reshape(-1, H, 4); c = obj.cost(xp if prefix is None else torch.cat([prefix[None].expand(len(xp), -1, -1), xp], 1)); i = int(c.argmin())
    return xp[i], float(c[i])


def plan(key, S0, goal, H, B, planner, seed, lr=2e-3, gd_k=8):
    """planner: cem | cem_gd | rand_gd.  Returns dict(seq (H,4) list, cost, counts, plan_time_s, pred per-push values)."""
    torch.manual_seed(seed); model = get_model(key); obj = make_obj(model, S0, goal); cn = counts(key, H, B, planner); t0 = time.time()
    first = sample_first(obj, 4 * max(cn["n_eval"] // 5, 64) if planner != "rand_gd" else cn["n_eval"])
    S, C = cem_pool(obj, first, H, cn["n_eval"], random_only=(planner == "rand_gd")); best, bc = S[0], float(C[0])
    if cn["gd_steps"]:
        s, c = gd_refine(obj, S[:gd_k], cn["gd_steps"], lr)
        if c < bc: best, bc = s, c
    t = time.time() - t0
    with torch.no_grad():
        sq = best.to(DEV)[None]; pv = [float(obj._value(p)[0] - obj.v0) for p in model.predict(sq[..., :2], sq[..., 2:])]
    return dict(seq=best.cpu().tolist(), cost=bc, pred_per_push=pv, plan_time_s=t, **cn, B=B, H=H, planner=planner, model=key)


def greedy_open_loop(key, S0, goal, pool, steps_gd=150, k=24, lr=2e-3, seed=0, H=4):
    """sequential greedy, H=1 candidates (pool random pile-aware pushes of the CURRENT PREDICTED state's start-state bank), evaluated through the model on prefix + candidate, then GD (steps_gd x k best) on the last push."""
    torch.manual_seed(seed); model = get_model(key); obj = make_obj(model, S0, goal); t0 = time.time(); prefix = torch.zeros(0, 4, device=DEV); first0 = sample_first(obj, pool)
    for j in range(H):
        if j == 0: first = first0
        else:
            pr = model.predict(prefix[None, :, :2], prefix[None, :, 2:])[-1][0]; first = propose_from_occ(pr, pool)
            if len(first) < 32: first = first0
        cand = first[torch.randint(0, len(first), (pool,), device=DEV)][:, None]; full = lambda s: torch.cat([prefix[None].expand(len(s), -1, -1), s], 1)
        C = obj.cost(full(cand)); i = C.argsort(); S = cand[i]; best, bc = S[0], float(C[i][0])
        if steps_gd:
            s, c = gd_refine(obj, S[:k], steps_gd, lr, prefix=prefix)
            if c < bc: best, bc = s, c
        prefix = torch.cat([prefix, best.reshape(1, 4)], 0)
    sq = prefix.to(DEV)[None]
    with torch.no_grad(): pv = [float(obj._value(p)[0] - obj.v0) for p in model.predict(sq[..., :2], sq[..., 2:])]
    return dict(seq=prefix.cpu().tolist(), cost=pv[-1], pred_per_push=pv, plan_time_s=time.time() - t0, pool=pool, gd_steps=steps_gd, model=key, planner=f"greedy_pool{pool}", H=H)


class Sim:
    """one Genesis env batch (K envs); executes legalised pushes, records states / actions / values."""
    def __init__(self, K=32):
        from simple_mpc.genesis_oracle import GenesisOracleEnv
        from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
        from simple_mpc.oracle_mpc import load_oracle_config
        import genesis as gs
        self.gs, self.K = gs, K
        cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))); cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
        self.env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(self.env); self.sim = self.env._sim; self.sim._settle_steps = self.env._real_settle_steps; self.sim._clearance_ctrl_steps = self.env._real_clearance_steps

    def reset(self, states):                      # list of (20,7) cpu tensors (padded to K by repeating the first)
        st = torch.stack(states + [states[0]] * (self.K - len(states))); self.sim.set_particle_state(st[:, :, :3].to(self.gs.device), st[:, :, 3:7].to(self.gs.device)); self.sim.update_material_state(); return self.parts()

    def parts(self):
        return self.sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def legalize(self, acts):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        from simple_mpc.learned_mpc import XY_MARGIN
        st = self.sim._particle_state.detach().cpu().float()
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001, box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))

    def push(self, acts):                          # (K,4) planned -> legalised, executed. returns legalised acts, shift, ok, parts after
        from transforms.functional import action_to_pose
        a, shift, ok = self.legalize(acts); sx, sy, ex, ey, ang = action_to_pose(a.to(self.gs.device).float()); z = torch.full_like(sx, float(self.sim._operation_height))
        self.sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); self.sim.update_material_state(); return a.cpu(), shift.cpu().numpy(), ok.cpu().numpy(), self.parts()

    def close(self):
        self.env.destroy()


_GC = {}


def _goal(g):
    if g not in _GC: _GC[g] = (goal_dist(g), torch.from_numpy(goal_mask(g)).float().reshape(-1))
    return _GC[g]


def true_value(parts, goals):
    """(n,20,7+) particle states, goals list -> value (lyap - in-goal fraction) of the soft-truth occupancy"""
    from simple_mpc.adapters import occ_for_scoring
    from simple_mpc.learned_mpc import lyap
    occ = occ_for_scoring(parts[:, :, :3].float()).cpu(); f = occ.reshape(len(parts), -1); out = []
    for k, g in enumerate(goals):
        d, m = _goal(g); out.append(float(lyap(occ[k:k + 1], d)[0] - (f[k] * m).sum() / f[k].sum()))
    return np.array(out)


def replay(sim, items, tag):
    """items: list of dict(goal,start,seq (4,4),...) <= K. Executes open loop, saves results/budget_study/replay_<tag>.json and states/actions npz."""
    n = len(items); states = [START(it["start"]) for it in items]; goals = [it["goal"] for it in items] + [items[0]["goal"]] * (sim.K - n)
    p = sim.reset(states); vals = [true_value(p, goals)]; parts = [p.numpy()]; planned, legal, shifts, oks = [], [], [], []; seqs = torch.tensor([it["seq"] for it in items] + [items[0]["seq"]] * (sim.K - n)).float()
    for j in range(seqs.shape[1]):
        a, sh, ok, p = sim.push(seqs[:, j]); planned.append(seqs[:, j].cpu().numpy()); legal.append(a.cpu().numpy()); shifts.append(sh); oks.append(ok); vals.append(true_value(p, goals)); parts.append(p.numpy())
    vals = np.stack(vals, 1); out = []
    for k, it in enumerate(items):
        out.append(dict(**{x: it[x] for x in ("goal", "start", "model", "planner", "B", "H") if x in it}, pred_terminal=it.get("cost"), pred_per_push=it.get("pred_per_push"), true_terminal=float(vals[k, -1] - vals[k, 0]), true_per_push=(vals[k, 1:] - vals[k, 0]).tolist(),
                        shifted_pushes=int(sum(s[k] > 0 for s in shifts)), shift_mm=float(np.mean([s[k] for s in shifts]) * 1000), illegal=int(sum(not o[k] for o in oks)), plan_time_s=it.get("plan_time_s")))
    np.savez_compressed(BS / f"states_{tag}.npz", parts=np.stack(parts, 1)[:n], planned=np.stack(planned, 1)[:n], legalised=np.stack(legal, 1)[:n], shifts=np.stack(shifts, 1)[:n], vals=vals[:n], keys=np.array([f"{it['goal']}|{it['start']}|{it.get('model')}|{it.get('planner')}|{it.get('B')}" for it in items]))
    json.dump(out, open(BS / f"replay_{tag}.json.tmp", "w")); os.replace(BS / f"replay_{tag}.json.tmp", BS / f"replay_{tag}.json"); return out
