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

    def _rollout(self, P0, P1, occ0):           # P0, P1 (N, H, 2) on DEV; occ0 (1, 64, 64) -> list of H cumulative pasted predictions (N, 64, 64)
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

    rollout = torch.no_grad()(_rollout)


class LFMember:
    """Switched-linear foresight (Suh & Tedrake; Baselines/LinearForesight) wide-fit operators as a planner model: hard length gate (the fitted model), own prediction fed back, on the 64x64 occupancy (EXP-0074 eval_lf_wide.LF).
    Moved here from bs_lib.py (2026-10-15, T0b) so closed_loop.MODELS and bs_lib share one copy; use inside WideEns(..., balance=False)."""
    kind = "lf"

    def __init__(self, ck="experiments/EXP-0074-wide-domain-zoom-nfd/runs/lf_wide/operators.pt"):
        from Baselines.LinearForesight.model import predict_switched, push_length_m
        from fit_linear_foresight import actions_to_pixels
        from simple_mpc.adapters import OCC_BOUNDS
        c = torch.load(ck, map_location="cpu", weights_only=False); self.res = c["res"]; self.edges = c["bin_edges"].to(DEV); self.ops = [o.to(DEV).float() for o in c["operators"]]
        self.lo = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]]); self.hi = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]]); self._ps, self._pl, self._ap = predict_switched, push_length_m, actions_to_pixels

    def begin(self, S0):
        pass

    def _rollout(self, P0, P1, occ0):
        N, H = P0.shape[:2]; cur = occ0.expand(N, -1, -1).clone(); out = []
        for k in range(H):
            act = torch.cat([P0[:, k], P1[:, k]], 1).float(); s, e = self._ap(act, self.lo, self.hi, tuple(cur.shape[-2:])); cur = self._ps(self.edges, self.ops, cur, s, e, self._pl(act), self.res, tuple(cur.shape[-2:])); out.append(cur)
        return out

    rollout = torch.no_grad()(_rollout)


class WideEns:
    def __init__(self, members, balance=True, name="ens"):
        self.members, self.balance, self.name = members, balance, name

    def begin(self, S0):
        for m in self.members:
            m.begin(S0)
        self.occ0 = occ_from_particles(S0[None].float().cpu()).to(DEV)

    def _predict(self, P0, P1, grad=False):
        outs = [(m._rollout if grad else m.rollout)(P0, P1, self.occ0) for m in self.members]; H = P0.shape[1]
        raw = [torch.stack([o[k] for o in outs]).mean(0) for k in range(H)]
        if not self.balance:
            return raw
        prev_raw = prev = self.occ0.expand_as(raw[0]); res = []
        for r in raw:
            prev = (prev + fix_delta_balance(r - prev_raw)).clamp(0, 1); prev_raw = r; res.append(prev)
        return res

    predict = torch.no_grad()(_predict)


class SeqObjective:
    """cost(seqs (N, H, 4)) = value(predicted state after H pushes) - value(start); lower is better. value = lyapunov - w * in-goal mass fraction."""
    def __init__(self, model, S0, dist, mask, mass_weight=1.0, chunk=128, obj_mode="terminal", spill_w=1.0, crowd=None):
        self.model, self.chunk, self.obj_mode, self.spill_w, self.crowd = model, chunk, obj_mode, float(spill_w), crowd   # crowd: captrans.CrowdTransport (objective 'captrans', 2026-10-11) or None
        self.S0 = S0; self._bind(); self.dist = dist.to(DEV).float(); self.mask = mask.to(DEV).float().reshape(1, -1); self.wm = float(mass_weight)
        self.v0 = self._value(model.occ0)
        self.ad = type("A", (), {"device": DEV})()

    def _bind(self):
        """The state encoding lives in the SHARED model object (model.begin); re-encode this objective's start state if another objective used
        the model since. Bug fixed 2026-10-11: before, an objective called after a later one had been built (run_seq_episodes' legal_fallback,
        after all envs were planned) scored its pushes on the LAST env's state encoding."""
        if getattr(self.model, "_bound_to", None) is not self:
            self.model.begin(self.S0); self.model._bound_to = self

    def _mass(self, occ):
        f = occ.reshape(occ.shape[0], -1)
        return (f * self.mask).sum(1) / f.sum(1).clamp_min(1e-6)

    def _value(self, occ):
        v = lyap(occ, self.dist); f = occ.reshape(occ.shape[0], -1)
        v = v - self.wm * (f * self.mask).sum(1) / f.sum(1).clamp_min(1e-6)
        return v + self.crowd(occ) if self.crowd is not None else v

    def _cost_body(self, s, grad=False):           # s (N, H, 4) on DEV -> (N,) cost under obj_mode; grad=True keeps the graph (GD refinement)
        preds = self.model._predict(s[..., :2], s[..., 2:], grad=grad)
        if self.obj_mode == "prefixmin":      # best value anywhere along the predicted sequence: later (random) pushes cannot spoil a good prefix
            return torch.stack([self._value(p) for p in preds]).min(0).values - self.v0
        if self.obj_mode in ("cumulative", "cumspill"):   # mean value over the H predicted states: early progress counts (2026-10-11 review item 3)
            c = torch.stack([self._value(p) for p in preds]).mean(0) - self.v0
            if self.obj_mode == "cumspill":   # + spill_w x every predicted drop of the in-goal mass fraction between consecutive states
                m = torch.stack([self._mass(self.model.occ0.expand_as(preds[0]))] + [self._mass(p) for p in preds])
                c = c + self.spill_w * (m[:-1] - m[1:]).clamp_min(0).sum(0)
            return c
        return self._value(preds[-1]) - self.v0

    @torch.no_grad()
    def cost(self, seqs):
        self._bind(); out = []
        for i in range(0, len(seqs), self.chunk):
            out.append(self._cost_body(seqs[i:i + self.chunk].to(DEV).float()))
        return torch.cat(out)

    def cost_grad(self, seqs):
        """Differentiable cost (same objective as `cost`, one chunk, no legalisation) for gradient refinement through the model."""
        self._bind(); return self._cost_body(seqs.to(DEV).float(), grad=True)

    def __call__(self, act):                   # single-push cost, used by legal_fallback
        return self.cost(act[:, None, :].to(DEV).float())

    @torch.no_grad()
    def crowd_parts(self, act=None):
        """(excess part, off-slot part) of the crowding-transport term at the current state (act None) or at the predicted state after the single push act (4,)."""
        self._bind(); occ = self.model.occ0 if act is None else self.model.predict(act[None, None, :2].to(DEV).float(), act[None, None, 2:].to(DEV).float())[-1]
        return tuple(float(x) for x in self.crowd.parts(occ))


CUBE_HALF, BLADE_HL, BLADE_HW = 0.0025, 0.02, 0.001


def inside_goal(xy, mask):
    """xy (n, 2) world -> bool (n,): cube centre's 64x64 pixel is in the goal mask (indexing mask[x_px, y_px], as analyse.py / churn)."""
    from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID
    lo = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]], device=xy.device); hi = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]], device=xy.device)
    uv = torch.round((xy - lo) / (hi - lo) * (OCC_GRID - 1)).long().clamp(0, OCC_GRID - 1)
    return mask.to(xy.device).bool()[uv[:, 0], uv[:, 1]]


def swept_hits(acts, pts, pad_perp=0.0, pad_along=0.0):
    """(N, 4) world pushes [sx, sy, ex, ey], (P, 2) world points -> (N, P) bool: point inside the push's swept rectangle -- the blade (half-length BLADE_HL,
    perpendicular to the push, half-thickness BLADE_HW) swept from start to end, i.e. scripts/probes/pool_common.py::swept_rectangle_corners' rectangle in
    world units, grown by pad_perp across and pad_along along the push (pad = CUBE_HALF: the cube footprint, approximated by its inscribed circle)."""
    acts, pts = acts.float(), pts.to(acts.device).float()
    s, d = acts[:, :2], acts[:, 2:] - acts[:, :2]; L = d.norm(dim=1, keepdim=True); u = d / L.clamp_min(1e-9); v = torch.stack([-u[:, 1], u[:, 0]], 1)
    r = pts[None] - s[:, None]; t = (r * u[:, None]).sum(-1); q = (r * v[:, None]).sum(-1).abs()
    return (t >= -(BLADE_HW + pad_along)) & (t <= L + BLADE_HW + pad_along) & (q <= BLADE_HL + pad_perp)


@torch.no_grad()
def bank_reach_frac(acts, xy, mask):
    """Fraction of candidate rows (N, 4) whose swept rectangle reaches (pad CUBE_HALF) at least one cube OUTSIDE the goal (xy (n, 2) world, mask (64, 64)).
    nan when no cube is outside. Per-push diagnostic (T0b, 2026-10-15): is the proposal bank even able to touch a stray?"""
    xy = xy.to(acts.device).float(); out = xy[~inside_goal(xy, mask)]
    if len(out) == 0 or len(acts) == 0:
        return float("nan")
    return float(swept_hits(acts, out, CUBE_HALF, CUBE_HALF).any(1).float().mean())


def make_spill_scorer(parts, mask):
    """Spill filter score (cell key spill_filter, T0b 2026-10-15) for candidates on ONE true state: parts (n, >=3) world particles, mask (64, 64).
    Filled goal pixels = goal pixels with occ_from_particles(state) > 0.5 (pixel centre lo + i/(G-1)(hi-lo), mask[x_px, y_px] as inside_goal).
    Returns acts (N, 4) -> (N,) the number of filled goal pixels whose centre lies in the push's swept rectangle (no pad), in CUBE EQUIVALENTS
    (/ (OCC_CUBE_SIZE / OCC_PITCH)^2 = 6.25 px per cube)."""
    from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID, OCC_CUBE_SIZE, OCC_PITCH
    occ = occ_from_particles(parts[None, :, :3].float().cpu())[0].to(DEV)
    filled = ((occ > 0.5) & mask.to(DEV).bool()).nonzero().float()
    lo = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]], device=DEV); hi = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]], device=DEV)
    pts = lo + filled / (OCC_GRID - 1) * (hi - lo); per = (OCC_CUBE_SIZE / OCC_PITCH) ** 2
    if len(pts) == 0:
        return lambda acts: torch.zeros(len(acts), device=DEV)
    return lambda acts: swept_hits(acts.to(DEV).float(), pts).float().sum(1) / per


@torch.no_grad()
def fetch_candidates(xy, mask, n_dirs=12, near=400, gaps=(0.004, 0.008), extras=(0.008, 0.015, 0.025), lo=None, hi=None):
    """Per-cube 'fetch' pushes (2026-10-11 bottleneck review, endgame_test.py): for every cube OUTSIDE the goal, a blade placed just behind it
    (gap behind the cube edge) pushing it toward one of n_dirs goal pixels spread over its `near` nearest goal pixels, long enough to carry it
    `extra` past its nearest goal pixel. xy (n, 2) world (DEV), mask (64, 64). Returns (M, 4) [sx, sy, ex, ey] (M = 0 when no cube is outside)."""
    from simple_mpc.adapters import OCC_BOUNDS, OCC_GRID
    dev = xy.device; m = mask.to(dev).bool()
    strays = xy[~inside_goal(xy, m)]
    if len(strays) == 0:
        return torch.zeros(0, 4, device=dev)
    lo_b = torch.tensor([OCC_BOUNDS["x_min"], OCC_BOUNDS["y_min"]], device=dev); hi_b = torch.tensor([OCC_BOUNDS["x_max"], OCC_BOUNDS["y_max"]], device=dev)
    mpts = lo_b + m.nonzero().float() / (OCC_GRID - 1) * (hi_b - lo_b)                     # (P, 2) goal pixel centres
    D = torch.cdist(strays, mpts); k = min(near, len(mpts)); dn, idx = D.topk(k, largest=False)   # sorted nearest first
    pick = torch.linspace(0, k - 1, min(n_dirs, k), device=dev).round().long()
    tg = mpts[idx[:, pick]]                                                                # (s, d, 2)
    u = tg - strays[:, None]; u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    reach = ((mpts[idx[:, 0]] - strays)[:, None] * u).sum(-1).clamp_min(0)                 # distance to the nearest goal pixel along u
    G = torch.tensor(gaps, device=dev); E = torch.tensor(extras, device=dev)
    nd, ng, ne = u.shape[1], len(G), len(E)
    uu = u[:, :, None, None, :]                                                            # (s, d, 1, 1, 2)
    s0 = strays[:, None, None, None, :] - uu * (CUBE_HALF + BLADE_HW + G)[None, None, :, None, None]   # (s, d, g, 1, 2)
    L = (reach[:, :, None, None] + G[None, None, :, None] + E[None, None, None, :]).clamp(lo or L_MIN, hi or L_MAX)   # (s, d, g, e)
    s0 = s0.expand(-1, nd, ng, ne, 2)
    e0 = s0 + uu * L[..., None]
    return torch.cat([s0, e0], -1).reshape(-1, 4)


def ot_candidates(xy, mask, slots, gaps=(0.004, 0.008), extras=(0.008, 0.015, 0.025), min_disp=0.005, lo=None, hi=None):
    """OT-seeded pushes (2026-10-11 OT-pool test, results/ot_pool/): Hungarian (Euclidean) assignment of the n cubes xy (n, 2) to the n goal slots `slots` (n, 2) world
    (default: value_functions.uniform_layout over the letter); for every cube displaced by more than min_disp, pushes along its displacement u = d/|d| with the blade
    CUBE_HALF + BLADE_HW + gap behind the cube centre and length |d| + gap + extra, clamped to [lo, hi] and projected. Cube-level (20 x 20), no Sinkhorn: ~1 ms.
    Returns (rows (M, 4) on xy.device, cube index (M,) long) -- M = 0 when every cube sits within min_disp of its slot."""
    from scipy.optimize import linear_sum_assignment
    dev = xy.device; P = xy.detach().cpu().double().numpy(); S = np.asarray(slots, dtype=float)
    r, c = linear_sum_assignment(np.linalg.norm(P[:, None] - S[None], axis=-1))
    d = torch.as_tensor(S[c] - P[r], dtype=torch.float32, device=dev); n = d.norm(dim=-1); keep = n > min_disp
    idx = torch.as_tensor(r, device=dev, dtype=torch.long)[keep]; d, n = d[keep], n[keep]
    if len(idx) == 0:
        return torch.zeros(0, 4, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    u = d / n[:, None]; G = torch.tensor(gaps, device=dev); E = torch.tensor(extras, device=dev); ng, ne = len(G), len(E)
    s0 = xy[idx].float()[:, None, None] - u[:, None, None] * (CUBE_HALF + BLADE_HW + G)[None, :, None, None]                 # (m, g, 1, 2)
    L = (n[:, None, None] + G[None, :, None] + E[None, None, :]).clamp(lo or L_MIN, hi or L_MAX)                               # (m, g, e)
    s0 = s0.expand(-1, ng, ne, 2); e0 = s0 + u[:, None, None] * L[..., None]
    rows = project_push(torch.cat([s0, e0], -1).reshape(-1, 4), lo or L_MIN, hi or L_MAX)[0]
    return rows, idx[:, None, None].expand(-1, ng, ne).reshape(-1)


def make_legal_scorer(xy, yaw):
    """Touchdown legaliser for candidates on ONE state (xy (n, 2), yaw (n,)): same constants and box as the driver's executed-push legaliser."""
    from Genesis.action_sampling import legalize_pushes_torch
    from simple_mpc.adapters import OCC_BOUNDS
    from simple_mpc.learned_mpc import XY_MARGIN
    box = (OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN); xy, yaw = xy[None].to(DEV).float(), yaw[None].to(DEV).float()
    return lambda acts: legalize_pushes_torch(acts.to(DEV).float(), xy, yaw, CUBE_HALF, BLADE_HL, BLADE_HW, box=box)


SRC = ("bank", "fetch", "ot")


def _nearest_cube(pts, cube_xy):
    return torch.cdist(pts[:, :2].float(), cube_xy.float()).argmin(1) if cube_xy is not None and len(cube_xy) else torch.full((len(pts),), -1, dtype=torch.long, device=pts.device)


def _ot_later(first, cid, otr, otc, union, cube_xy, H, n_pool, rint):
    """Later pushes (n_pool, H-1, 4) for the OT-structured pools: at each later step an `otr` row whose cube `otc` was not yet pushed in that sequence (cid = cube of the first push),
    a random `union` row when no such row exists (ot_pool test, 2026-10-11; shared by pool modes 'ot' and 'flot')."""
    used = cid[:, None]; lat = []; M = len(union)
    for _ in range(H - 1):
        allowed = (otc[None, :] != used[:, :, None]).all(1)                       # (n_pool, M_ot): OT rows of a cube not yet pushed in the sequence
        r = torch.rand(n_pool, len(otr), device="cpu").to(first.device); r[~allowed] = -1.0; pick = r.argmax(1); has = allowed.any(1)
        fb = union[rint(M, (n_pool,))]; rows = torch.where(has[:, None], otr[pick], fb); rc = torch.where(has, otc[pick], _nearest_cube(fb, cube_xy))
        lat.append(rows); used = torch.cat([used, rc[:, None]], 1)
    return torch.stack(lat, 1)


@torch.no_grad()
def plan_seq(obj, bank, kind, H, push_len=None, n_pool=256, pop=256, iters=4, elite_frac=0.125, warm=None, rank_n=1280, topk=0, trace=None, extra=None, n_extra=64, legal=None,
             pool_mode="fl", ot=None, cube_xy=None, gd_steps=0, gd_k=8, gd_lr=2e-3, spill=None, spill_thr=None, diag_xy=None, diag_mask=None):
    """Returns dict(action (4,), seq (H,4), cost, n_evals, time_s). bank (M, 4) cpu: pile-aware candidates for the CURRENT state.
    extra (E, 4): additional first-push proposals (fetch candidates); up to n_extra of them (random subset) REPLACE bank rows in the initial pool / rank set,
    so the evaluation count is unchanged. legal: callable acts (N, 4) -> (acts, shift, ok) -- if given, every scored FIRST push is legalised before it is
    scored (2026-10-11 review item 2) and pushes with no legal offset get cost +inf, so the planner scores the push that will be executed.
    pool_mode (2026-10-11 OT-pool test, results/ot_pool/): "fl" = the above (today's planner; later pushes of H>1 sequences = random rows of fetch u bank);
    "ot" = n_pool/2 OT rows (ot = (rows, cube index) from ot_candidates) + n_pool/4 fetch + n_pool/4 bank as first pushes, later pushes = OT rows of a cube
    not yet pushed in that sequence (cube of a bank / fetch row = nearest cube to its start, cube_xy (n, 2)); a union row when no such OT row exists;
    "mix" = n_pool/2 bank + n_pool/4 fetch + n_pool/4 OT, later pushes random union rows. Missing fetch / OT rows are filled with bank rows, so the pool size
    and the evaluation count are the same in every mode. gd_steps > 0: Adam (gd_lr) through the differentiable model on the gd_k best sequences after CEM
    (bs_lib.gd_refine), first pushes legalised and rescored, kept when better than the CEM best. Records the source of the initial pool's best row per source,
    the best-so-far trace (after the pool, after each CEM iteration, after GD), the stage that produced the chosen sequence and the source of the pool row
    nearest to its first push.
    spill (T0b, 2026-10-15; cell key spill_filter): callable acts (N, 4) -> (N,) filled-goal cube equivalents swept (make_spill_scorer); every scored FIRST push
    (after legalisation) with spill > spill_thr gets cost +inf, like legal. Guard: if no initial-pool row (rank: no rank row) passes, the filter is switched off
    for this decision and the pool is rescored (info spill_fallback = 1). info spill_reject_frac = fraction of initial-pool rows rejected.
    diag_xy (n, 2) + diag_mask (64, 64): record pool_reach_frac = bank_reach_frac of the initial pool's first pushes (before legalisation)."""
    t0 = time.time(); lo, hi = (push_len, push_len) if push_len else (L_MIN, L_MAX)
    INF = torch.tensor(float("inf"), device=DEV)

    def proj(x):
        return project_push(x, lo, hi)[0]

    def prep(seqs):                                    # (N, H, 4) -> projected (and first-push legalised) seqs, ok (N,)
        N_ = seqs.shape[0]; seqs = proj(seqs.reshape(-1, 4)).reshape(N_, -1, 4); ok = torch.ones(N_, dtype=torch.bool, device=DEV)
        if legal is not None:
            a, _, ok = legal(seqs[:, 0]); seqs = seqs.clone(); seqs[:, 0] = a.to(DEV)
        ok = ok.to(DEV)
        if spill_on[0]:
            ok = ok & (spill(seqs[:, 0]) <= spill_thr)
        return seqs, ok

    def score(seqs):
        seqs, ok = prep(seqs); c = obj.cost(seqs); return seqs, torch.where(ok, c, INF)

    def rint(n, shape):
        return torch.randint(0, n, shape, device="cpu").to(DEV)

    spill_on = [spill is not None and spill_thr is not None]

    def spill_guard(seqs):                             # True when the filter rejected every row: switch it off for this decision
        if not spill_on[0]:
            return False
        s1 = prep(seqs)[0][:, 0]; rej = spill(s1) > spill_thr; info["spill_reject_frac"] = float(rej.float().mean()); info["spill_fallback"] = int(bool(rej.all()))
        if rej.all():
            spill_on[0] = False
        return bool(rej.all())

    b = proj(bank.to(DEV).float()); info = {}; src = torch.zeros(n_pool, dtype=torch.long, device=DEV)
    if pool_mode in ("fl", "flot"):               # flot (captrans test, 2026-10-11): fl first pushes; later pushes = `ot` rows of a cube not yet pushed in the sequence
        if extra is not None and len(extra) and n_extra > 0:
            ex = proj(extra.to(DEV).float()); ex = ex[torch.randperm(len(ex), device="cpu")[:n_extra].to(DEV)]; info["n_extra"] = len(ex)
            b = torch.cat([ex, b]); src[:len(ex)] = 1
        if kind == "rank":
            if diag_xy is not None:
                info["pool_reach_frac"] = bank_reach_frac(b[:rank_n], diag_xy, diag_mask)
            spill_guard(b[:rank_n][:, None]); cand, c = score(b[:rank_n][:, None]); i = int(c.argmin())
            return dict(action=cand[i, 0].cpu(), seq=cand[i].cpu(), cost=float(c[i]), n_evals=len(cand), time_s=time.time() - t0, **info)
        M = len(b); first = b[:n_pool]
        later = b[rint(M, (n_pool, max(H - 1, 0)))] if H > 1 else b[:0].reshape(n_pool, 0, 4)
        if pool_mode == "flot" and H > 1 and ot is not None and len(ot[0]):
            otr, otc = proj(ot[0].to(DEV).float()), ot[1].to(DEV); info["n_ot_all"] = len(otr)
            later = _ot_later(first, _nearest_cube(first, cube_xy), otr, otc, b, cube_xy, H, n_pool, rint)
    else:
        assert kind == "cem" and pool_mode in ("ot", "mix")
        ex = proj(extra.to(DEV).float()) if extra is not None and len(extra) else b[:0]
        ex = ex[torch.randperm(len(ex), device="cpu")[: n_pool // 4].to(DEV)]
        otr, otc = (proj(ot[0].to(DEV).float()), ot[1].to(DEV)) if ot is not None and len(ot[0]) else (b[:0], torch.zeros(0, dtype=torch.long, device=DEV))
        sel = torch.randperm(len(otr), device="cpu")[: (n_pool // 2 if pool_mode == "ot" else n_pool // 4)].to(DEV); otp, otcp = otr[sel], otc[sel]
        nb = n_pool - len(ex) - len(otp); bk = b[:nb]
        first = torch.cat([otp, ex, bk]); src = torch.cat([torch.full((len(otp),), 2), torch.full((len(ex),), 1), torch.zeros(nb, dtype=torch.long)]).to(DEV)
        info["n_extra"], info["n_ot"], info["n_ot_all"] = len(ex), len(otp), len(otr)
        union = torch.cat([ex, otr, b]); M = len(union)
        if H <= 1:
            later = first[:0].reshape(n_pool, 0, 4)
        elif pool_mode == "ot" and len(otr):
            cid = _nearest_cube(first, cube_xy); cid[:len(otp)] = otcp
            later = _ot_later(first, cid, otr, otc, union, cube_xy, H, n_pool, rint)
        else:
            later = union[rint(M, (n_pool, H - 1))]
        b = union
    pool = torch.cat([first[:, None], later], 1)                                  # (n_pool, H, 4)
    if warm is not None and H > 1:                                                  # shift warm start: previous plan's steps 2..H, noisy copies, last step fresh
        w = warm.to(DEV).float()[: H - 1]; nw = min(16, n_pool // 4)
        noise = 0.005 * torch.randn(nw, w.shape[0], 4, device=DEV); pool[:nw, : w.shape[0]] = proj((w[None] + noise).reshape(-1, 4)).reshape(nw, -1, 4)
    if diag_xy is not None:
        info["pool_reach_frac"] = bank_reach_frac(pool[:, 0], diag_xy, diag_mask)
    spill_guard(pool)
    pool, cost = score(pool); n_evals = n_pool; n_el = max(2, int(elite_frac * pop))
    if "n_extra" in info and pool_mode in ("fl", "flot"):
        ne = info["n_extra"]; info["pool_best_extra"] = float(cost[:ne].min()); info["pool_best_bank"] = float(cost[ne:].min())
    for code, name in enumerate(SRC):
        if (src == code).any():
            info[f"pool_best_{name}"] = float(cost[src == code].min())
    allS, allC = [pool], [cost]
    best_i = int(cost.argmin()); best, best_c = pool[best_i].clone(), float(cost[best_i]); stage = "pool"; tr = [best_c]
    if trace is not None: trace.append((n_evals, best_c))
    elite = pool[cost.argsort()[:n_el]]; mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
    for it in range(iters):
        s, c = score(mean + std * torch.randn(pop, H, 4, device=DEV)); n_evals += pop; k = int(c.argmin()); allS.append(s); allC.append(c)
        if float(c[k]) < best_c:
            best, best_c = s[k].clone(), float(c[k]); stage = f"cem{it + 1}"
        elite = s[c.argsort()[:n_el]]; mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
        tr.append(best_c)
        if trace is not None: trace.append((n_evals, best_c))
    info["cem_best"] = best_c
    if gd_steps > 0:
        tg = time.time(); S_, C_ = torch.cat(allS), torch.cat(allC); idx = C_.argsort()[:gd_k]; idx = idx[torch.isfinite(C_[idx])]
        if len(idx):
            with torch.enable_grad():
                x = S_[idx].clone().requires_grad_(True); opt = torch.optim.Adam([x], lr=gd_lr)
                for _ in range(gd_steps):
                    opt.zero_grad(); obj.cost_grad(proj(x.reshape(-1, 4)).reshape(-1, H, 4)).sum().backward(); opt.step()
            s, c = score(x.detach()); k = int(c.argmin()); info["gd_best"] = float(c[k]); info["gd_best_unleg"] = float(obj.cost(proj(x.detach().reshape(-1, 4)).reshape(-1, H, 4)).min())
            if float(c[k]) < best_c:
                best, best_c = s[k].clone(), float(c[k]); stage = "gd"
        info["gd_time_s"] = time.time() - tg; info["gd_gain"] = info["cem_best"] - best_c; tr.append(best_c)
    info["trace"] = tr; info["best_stage"] = stage; info["chosen_src"] = SRC[int(src[int((first - best[0][None]).norm(dim=1).argmin())])]
    out = dict(action=best[0].cpu(), seq=best.cpu(), cost=best_c, n_evals=n_evals, time_s=time.time() - t0, **info)
    if topk > 0:
        S_, C_ = torch.cat(allS), torch.cat(allC); i = C_.argsort()[:topk]; out["top_seqs"], out["top_costs"] = S_[i].cpu(), C_[i].cpu()
    return out


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
            crowd = None
            if ep.get("objective", "cur") == "captrans":      # crowding-transport term (captrans.CrowdTransport) from the TRUE current state and the goal's capacity slots
                from captrans import CrowdTransport
                crowd = CrowdTransport(parts[k, :, :2].numpy(), ep["cap_slots"], ep["mask"].numpy(), None)
            if ep.get("objective", "cur") == "layout":        # layout-transport term (layout_term.LayoutTransport, 2026-10-11): Sinkhorn W1 of the predicted occupancy to the uniform_layout slots
                from layout_term import LayoutTransport
                crowd = LayoutTransport(ep["slots"].numpy(), ep["mask"].numpy(), lam=ep.get("layout_lam", 1.0)); crowd.n_free = 0
            obj = SeqObjective(ep["model"], parts[k], ep["dist"], ep["mask"], ep.get("mass_weight", 1.0), obj_mode=ep.get("obj_mode", "terminal"), spill_w=ep.get("spill_w", 1.0), crowd=crowd); objs.append(obj); _seed(t, k)
            kw = dict(ep.get("plan_kw", {}))
            if crowd is not None:
                recs[k].setdefault("crowd_now", []).append(obj.crowd_parts()); recs[k].setdefault("n_free_slots", []).append(crowd.n_free)
            if ep.get("captrans"):            # capacity-transport proposals REPLACE the fetch extras (same n_extra); H>1: distinct-cube suffixes from the same rows (pool_mode flot)
                from captrans import captrans_candidates
                t_ = time.time(); rows, cidx = captrans_candidates(parts[k, :, :2].to(DEV), ep["cap_slots"], lo=ep.get("push_len"), hi=ep.get("push_len"))
                kw["extra"] = rows; kw.setdefault("n_extra", ep.get("n_fetch", 64)); recs[k].setdefault("t_captrans_s", []).append(time.time() - t_)
                if ep["H"] > 1:
                    kw["ot"] = (rows, cidx); kw["cube_xy"] = parts[k, :, :2].to(DEV); kw.setdefault("pool_mode", "flot")
            elif ep.get("fetch"):
                t_ = time.time(); kw["extra"] = fetch_candidates(parts[k, :, :2].to(DEV), ep["mask"], lo=ep.get("push_len"), hi=ep.get("push_len")); kw.setdefault("n_extra", ep.get("n_fetch", 64))
                recs[k].setdefault("t_fetch_s", []).append(time.time() - t_)
            if ep.get("ot"):                  # OT-seeded proposals (ot_candidates) toward ep["slots"] (n, 2) world
                t_ = time.time(); kw["ot"] = ot_candidates(parts[k, :, :2].to(DEV), ep["mask"], ep["slots"], lo=ep.get("push_len"), hi=ep.get("push_len")); kw["cube_xy"] = parts[k, :, :2].to(DEV)
                recs[k].setdefault("t_ot_s", []).append(time.time() - t_)
            if ep.get("legal_score"):
                from model.retrieval.frame import yaw_from_quat
                kw["legal"] = make_legal_scorer(parts[k, :, :2], yaw_from_quat(parts[k, :, 3:7]))
            if ep.get("spill_filter") is not None:   # T0b: reject first pushes sweeping > spill_filter cube-equivalents of filled goal pixels
                kw["spill"] = make_spill_scorer(parts[k], ep["mask"]); kw["spill_thr"] = float(ep["spill_filter"])
            if ep.get("diag", True):                 # T0b per-push diagnostics: stray reach of the initial pool and of the whole Genesis bank
                kw["diag_xy"] = parts[k, :, :2].to(DEV); kw["diag_mask"] = ep["mask"]
                recs[k].setdefault("bank_reach_frac", []).append(bank_reach_frac(bank[k].to(DEV).float(), kw["diag_xy"], ep["mask"]))
            out = plan_seq(obj, bank[k], ep["planner"], ep["H"], push_len=ep.get("push_len"), warm=warm[k], **kw)
            acts.append(out["action"]); warm[k] = out["seq"][1:] if ep["planner"] == "cem" and ep["H"] > 1 else None
            r = recs[k]; r["actions"].append(out["action"].tolist()); r["pred_cost"].append(out["cost"]); r["n_evals"].append(out["n_evals"]); r["plan_time_s"].append(out["time_s"])
            for key in ("n_extra", "pool_best_extra", "pool_best_bank", "pool_best_fetch", "pool_best_ot", "n_ot", "n_ot_all", "trace", "cem_best", "gd_best", "gd_best_unleg", "gd_time_s", "gd_gain", "best_stage", "chosen_src", "pool_reach_frac", "spill_reject_frac", "spill_fallback"):
                if key in out:
                    r.setdefault(key, []).append(out[key])
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
        for k in range(K):                    # predicted single-push cost of the push that is actually EXECUTED (after legalisation / fallback)
            recs[k].setdefault("pred_cost_exec", []).append(float(objs[k](acts[k][None].to(DEV).float()).cpu()[0]))
            if objs[k].crowd is not None:
                recs[k].setdefault("crowd_exec", []).append(objs[k].crowd_parts(acts[k]))
        parts_prev = parts; parts = execute_batch(acts).float()
        for k in range(K):
            if episodes[k].get("diag", True):         # T0b: true value change of the executed push under the objective's value function, optimism, cubes entering / leaving
                from analyse import transition_diag
                dg = transition_diag(torch.stack([parts_prev[k, :, :3], parts[k, :, :3]]), episodes[k]["dist"], episodes[k]["mask"], episodes[k].get("mass_weight", 1.0))
                r = recs[k]; r.setdefault("true_dv_exec", []).append(float(dg["true_dv"][0])); r.setdefault("optimism_exec", []).append(r["pred_cost_exec"][-1] - float(dg["true_dv"][0]))
                r.setdefault("entering", []).append(int(dg["entering"][0])); r.setdefault("leaving", []).append(int(dg["leaving"][0]))
            recs[k]["states"].append(parts[k, :, :3].tolist()); recs[k]["quats"].append(parts[k, :, 3:7].tolist())
        if on_step is not None:
            on_step(t, recs)
    return recs
