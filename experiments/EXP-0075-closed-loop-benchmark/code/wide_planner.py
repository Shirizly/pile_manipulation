"""EXP-0075 closed-loop benchmark: Sean-trained zoom / vanilla NFDs as the planner's model, horizon-H sequence planners, batched Genesis episodes.

Model side  (WideEns): members = zoom128 (variable-side push window cut from a 300x300 canvas of the CURRENT state) and/or vanilla128 (plain world raster), each unrolled-trained
(EXP-0074 *_ms4). predict(P0, P1) rolls a batch of N candidate push SEQUENCES (N, H, 2) through the members (own prediction fed back), averages the members' pasted 64x64 predictions at every step
and (optionally) applies the val-selected mass balance to every step's predicted change (EXP-0074 posthoc_fix). The expensive state encoding (canvas / raster) is done ONCE per decision and shared by
all candidates.
Planner side (plan_seq): `rank` = best of n sampled single pushes; `cem` = CEM over sequences (N, H, 4) with a pile-aware first push (from the Genesis bank), later pushes drawn from the same bank,
shift warm start from the previous decision, receding horizon (execute the first push), terminal objective = value of the predicted state after H pushes
(lyapunov distance - w * in-goal mass fraction, EXP-0052 success objective).
Runner (run_seq_episodes): K episodes side by side, one per simulator env; same legalisation / fallback / seeding conventions as learned_mpc.run_episodes_batched.
"""
import sys, time, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
import eval_wide as ew
from model.zoom_nfd.window_var import side_for, plates_v, extract_windows_v, paste_canvas_v, delta_to_world64_v
from model.zoom_nfd.world_res import raster_res, plates_res
from model.zoom_nfd.rollout import canvas_from_particles
from simple_mpc.adapters import occ_from_particles, occ_for_scoring
from simple_mpc.learned_mpc import project_push, L_MIN, L_MAX, lyap, legal_fallback

DEV = "cuda"
SIZES = lambda n: [[0.005] * 3] * n


def fix_delta_balance(d):
    """EXP-0074 posthoc_fix 'balance': rescale the positive and negative parts of the predicted change to equal mass (mass conserving)."""
    pos = d.clamp_min(0); neg = (-d).clamp_min(0)
    ps, ns = pos.sum((-1, -2), keepdim=True), neg.sum((-1, -2), keepdim=True); tg = (ps + ns) / 2
    return pos * tg / ps.clamp_min(1e-9) - neg * tg / ns.clamp_min(1e-9)


class Member:
    def __init__(self, kind, res, ckpt, feats=(8, 16, 32), C=300, thr=0.2):
        self.kind, self.res, self.C, self.thr = kind, res, C, thr
        self.net = ew.make_net(ckpt, feats)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def begin(self, S0):                       # S0 (n, 7) cpu
        n = S0.shape[0]
        if self.kind == "zoom":
            self.canvas0 = canvas_from_particles(S0[None].float().cpu(), SIZES(n), self.C, self.thr).float().to(DEV)
        else:
            self.st0 = raster_res(S0[None].float().cpu(), SIZES(n), self.res).to(DEV)

    @torch.no_grad()
    def rollout(self, P0, P1, occ0):           # P0, P1 (N, H, 2) on DEV; occ0 (1, 64, 64) -> list of H cumulative pasted predictions (N, 64, 64)
        N, H = P0.shape[:2]; cur = occ0.expand(N, -1, -1).clone(); out = []
        if self.kind == "zoom":
            canvas = self.canvas0.expand(N, -1, -1)
            for k in range(H):
                p0, p1 = P0[:, k], P1[:, k]; side = side_for((p1 - p0).norm(dim=-1))
                w = extract_windows_v(canvas, p0, p1, side, self.res)
                d = torch.sigmoid(self.net(torch.cat([w[:, None], plates_v(p0, p1, side, self.res)], 1))).squeeze(1) - w
                canvas = (canvas + paste_canvas_v(d, p0, p1, side, self.C)).clamp(0, 1)
                cur = (cur + delta_to_world64_v(d, p0, p1, side)).clamp(0, 1); out.append(cur)
        else:
            st = self.st0.expand(N, -1, -1)
            for k in range(H):
                p0, p1 = P0[:, k], P1[:, k]
                p = torch.sigmoid(self.net(torch.cat([st[:, None], plates_res(p0, p1, self.res)], 1))).squeeze(1)
                cur = (cur + ew.down_world(p - st, self.res)).clamp(0, 1); st = p; out.append(cur)
        return out


class WideEns:
    def __init__(self, members, balance=True, name="ens"):
        self.members, self.balance, self.name = members, balance, name

    def begin(self, S0):
        for m in self.members:
            m.begin(S0)
        self.occ0 = occ_from_particles(S0[None].float().cpu()).to(DEV)

    @torch.no_grad()
    def predict(self, P0, P1):
        outs = [m.rollout(P0, P1, self.occ0) for m in self.members]; H = P0.shape[1]
        raw = [torch.stack([o[k] for o in outs]).mean(0) for k in range(H)]
        if not self.balance:
            return raw
        prev_raw = prev = self.occ0.expand_as(raw[0]); res = []
        for r in raw:
            prev = (prev + fix_delta_balance(r - prev_raw)).clamp(0, 1); prev_raw = r; res.append(prev)
        return res


class SeqObjective:
    """cost(seqs (N, H, 4)) = value(predicted state after H pushes) - value(start); lower is better. value = lyapunov - w * in-goal mass fraction."""
    def __init__(self, model, S0, dist, mask, mass_weight=1.0, chunk=128, obj_mode="terminal"):
        self.model, self.chunk, self.obj_mode = model, chunk, obj_mode
        model.begin(S0); self.dist = dist.to(DEV).float(); self.mask = mask.to(DEV).float().reshape(1, -1); self.wm = float(mass_weight)
        self.v0 = self._value(model.occ0)
        self.ad = type("A", (), {"device": DEV})()

    def _value(self, occ):
        v = lyap(occ, self.dist); f = occ.reshape(occ.shape[0], -1)
        return v - self.wm * (f * self.mask).sum(1) / f.sum(1).clamp_min(1e-6)

    @torch.no_grad()
    def cost(self, seqs):
        out = []
        for i in range(0, len(seqs), self.chunk):
            s = seqs[i:i + self.chunk].to(DEV).float(); preds = self.model.predict(s[..., :2], s[..., 2:])
            if self.obj_mode == "prefixmin":      # best value anywhere along the predicted sequence: later (random) pushes cannot spoil a good prefix
                out.append(torch.stack([self._value(p) for p in preds]).min(0).values - self.v0)
            else:
                out.append(self._value(preds[-1]) - self.v0)
        return torch.cat(out)

    def __call__(self, act):                   # single-push cost, used by legal_fallback
        return self.cost(act[:, None, :].to(DEV).float())


@torch.no_grad()
def plan_seq(obj, bank, kind, H, push_len=None, n_pool=256, pop=256, iters=4, elite_frac=0.125, warm=None, rank_n=1280):
    """Returns dict(action (4,), seq (H,4), cost, n_evals, time_s). bank (M, 4) cpu: pile-aware candidates for the CURRENT state."""
    t0 = time.time(); lo, hi = (push_len, push_len) if push_len else (L_MIN, L_MAX)
    proj = lambda x: project_push(x, lo, hi)[0]
    b = proj(bank.to(DEV).float())
    if kind == "rank":
        cand = b[:rank_n]; c = obj(cand); i = int(c.argmin())
        return dict(action=cand[i].cpu(), seq=cand[i][None].cpu(), cost=float(c[i]), n_evals=len(cand), time_s=time.time() - t0)
    M = len(b); g = torch.Generator(device="cpu")
    first = b[:n_pool]
    later = b[torch.randint(0, M, (n_pool, max(H - 1, 0)), device="cpu").to(DEV)] if H > 1 else b[:0].reshape(n_pool, 0, 4)
    pool = torch.cat([first[:, None], later], 1)                                  # (n_pool, H, 4)
    if warm is not None and H > 1:                                                  # shift warm start: previous plan's steps 2..H, noisy copies, last step fresh
        w = warm.to(DEV).float()[: H - 1]; nw = min(16, n_pool // 4)
        noise = 0.005 * torch.randn(nw, w.shape[0], 4, device=DEV); pool[:nw, : w.shape[0]] = proj((w[None] + noise).reshape(-1, 4)).reshape(nw, -1, 4)
    pool = proj(pool.reshape(-1, 4)).reshape(n_pool, H, 4)
    cost = obj.cost(pool); n_evals = n_pool; n_el = max(2, int(elite_frac * pop))
    best_i = int(cost.argmin()); best, best_c = pool[best_i].clone(), float(cost[best_i])
    elite = pool[cost.argsort()[:n_el]]; mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
    for _ in range(iters):
        s = proj((mean + std * torch.randn(pop, H, 4, device=DEV)).reshape(-1, 4)).reshape(pop, H, 4)
        c = obj.cost(s); n_evals += pop; k = int(c.argmin())
        if float(c[k]) < best_c:
            best, best_c = s[k].clone(), float(c[k])
        elite = s[c.argsort()[:n_el]]; mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
    return dict(action=best[0].cpu(), seq=best.cpu(), cost=best_c, n_evals=n_evals, time_s=time.time() - t0)


def run_seq_episodes(episodes, particles0, execute_batch, sample_candidates_batch, n_steps, bank_per_env=4096, on_step=None, legalize=None, legalize_fallback=False, planner_seed=None):
    """episodes[k] = dict(model=WideEns, dist (H,W), mask (H,W), mass_weight, planner 'rank'|'cem', H, push_len, plan_kw={}). particles0 (K, n, 7) cpu."""
    K = len(episodes); parts = particles0.float(); recs = []
    for k, ep in enumerate(episodes):
        recs.append(dict(planner=ep["planner"], H=ep["H"], push_len=ep.get("push_len"), actions=[], pred_cost=[], n_evals=[], plan_time_s=[], states=[parts[k, :, :3].tolist()], quats=[parts[k, :, 3:7].tolist()]))
    warm = [None] * K

    def _seed(t, k):
        if planner_seed is not None:
            torch.manual_seed(int(planner_seed) * 1_000_003 + 100_003 * (t + 1) + k)

    for t in range(n_steps):
        _seed(t, -1); bank = sample_candidates_batch(bank_per_env); acts, objs = [], []
        for k, ep in enumerate(episodes):
            obj = SeqObjective(ep["model"], parts[k], ep["dist"], ep["mask"], ep.get("mass_weight", 1.0), obj_mode=ep.get("obj_mode", "terminal")); objs.append(obj); _seed(t, k)
            out = plan_seq(obj, bank[k], ep["planner"], ep["H"], push_len=ep.get("push_len"), warm=warm[k], **ep.get("plan_kw", {}))
            acts.append(out["action"]); warm[k] = out["seq"][1:] if ep["planner"] == "cem" and ep["H"] > 1 else None
            r = recs[k]; r["actions"].append(out["action"].tolist()); r["pred_cost"].append(out["cost"]); r["n_evals"].append(out["n_evals"]); r["plan_time_s"].append(out["time_s"])
        acts = torch.stack(acts)
        if legalize is not None:
            acts, shift, ok = legalize(acts); fb = torch.zeros(K, dtype=torch.long, device="cpu")
            if legalize_fallback:
                for k in (~ok).nonzero().flatten().tolist():
                    a2, s2, ok2 = legal_fallback(objs[k], bank[k], legalize, k, push_len=episodes[k].get("push_len"))
                    if ok2:
                        acts[k], shift[k], ok[k], fb[k] = a2, s2, True, 1; warm[k] = None
                    else:
                        raise RuntimeError(f"legalize_fallback: env {k} step {t}: no legal candidate")
            for k in range(K):
                recs[k].setdefault("legal_shift_m", []).append(float(shift[k])); recs[k].setdefault("legal_ok", []).append(bool(ok[k])); recs[k].setdefault("legal_fallback", []).append(int(fb[k]))
                recs[k]["actions"][-1] = acts[k].tolist()
                if shift[k] > 0:
                    warm[k] = None            # the executed push differs from the plan: the shifted plan no longer applies
        parts = execute_batch(acts).float()
        for k in range(K):
            recs[k]["states"].append(parts[k, :, :3].tolist()); recs[k]["quats"].append(parts[k, :, 3:7].tolist())
        if on_step is not None:
            on_step(t, recs)
    return recs
