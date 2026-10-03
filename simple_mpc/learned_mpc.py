"""simple_mpc/learned_mpc.py -- closed-loop MPC with a LEARNED model as the
objective, under a fixed wall-clock planning budget per decision.

Built for the benchmark question "do offline metrics predict closed-loop
performance?" (experiments/TODO.md G1b), so every convention matches the
offline benchmarks exactly:
  * model input: `adapters.occ_from_particles` (the 64x64 slate grid);
  * predicted dv: lyapunov of the model's predicted image minus lyapunov of
    its input image, per goal distance field (a COST, lower = better);
  * progress / ground truth: lyapunov of `adapters.occ_for_scoring` (the soft,
    mass-conserving truth scoring -- experiments/METRICS.md);
  * action legality: `project_push` (the EXP-0023 projection: endpoints 4 mm
    inside the +/-64 mm tray, push length clamped to 20-70 mm).

Planners (all spend at most `budget_s` of planning wall-clock per decision):
  rank  -- score the candidate pool with the model, take the best;
  gd    -- rank, then projected Adam from the top `n_restarts` candidates
           until the budget runs out, keep the best predicted (EXP-0027: one
           GD run is a noisy sample, so restarts matter);
  cem / mppi -- `sampling_optimizers` with the model as the cost, population
           seeded from the candidate pool, iterated until the budget runs out.

The runner is Genesis-agnostic about how an action is EXECUTED: it takes an
`execute(action (4,)) -> particles (n, >=3)` callable, so the same code runs
against `GenesisOracleEnv.step` or any other executor.
"""
from __future__ import annotations

import time
from typing import Callable

import torch

from simple_mpc.adapters import OCC_BOUNDS, occ_for_scoring, occ_from_particles

XY_MARGIN = 0.004
L_MIN, L_MAX = 0.020, 0.070

# The material physics of overnight_randlen (the dynamics models' training corpus)
# and Sean. Any simulator used to EVALUATE those models -- test-corpus collection,
# closed-loop execution, gradient-benchmark rollouts -- must run with these, or
# the models are scored off their training distribution (invariant
# `benchmark-physics-matches-training`). simple_mpc/config/config_oracle.yaml
# defaults to particle friction 0.25 / density 750 / box 0.3 / settle 100, and the
# binned collector to 0.3 / 1000 / 0.3: neither matches.
TRAINING_PHYSICS = dict(particle_friction=0.7, particle_density=450.0, box_friction=0.5,
                        settle_steps=3000)


def oracle_config_with_physics(cfg: dict, physics: dict = TRAINING_PHYSICS) -> dict:
    """Return `cfg` (a `load_oracle_config` dict) with `physics` written into the
    `dataset` keys GenesisEnv builds the sandbox from. Call BEFORE constructing
    GenesisOracleEnv, then `apply_physics(env)` after construction."""
    cfg["dataset"].update(physics)
    return cfg


def apply_physics(env, physics: dict = TRAINING_PHYSICS) -> None:
    """Push the material physics into a built simulator as the collectors do
    (`SandboxManipulation.set_material_properties`), so the value in force does
    not depend on which config path the build read."""
    env._sim.set_material_properties({
        "particle_friction": physics["particle_friction"],
        "particle_density": physics["particle_density"],
        "box_friction": physics["box_friction"],
        "sampled_particle_friction": None, "sampled_particle_density": None})


def project_push(act: torch.Tensor, l_min: float = L_MIN, l_max: float = L_MAX):
    """Clamp (B, 4) [sx, sy, ex, ey] pushes into the legal set: endpoints
    XY_MARGIN inside the tray, length in [L_MIN, L_MAX] along the current
    direction with the start held fixed. Returns (act, hit (B, 2) bool:
    [box clamped, length clamped]). Identical to EXP-0023's `project`."""
    lo = OCC_BOUNDS["x_min"] + XY_MARGIN
    hi = OCC_BOUNDS["x_max"] - XY_MARGIN
    clamped = act.clamp(lo, hi)
    hit_box = (clamped != act).any(dim=1)
    d = clamped[:, 2:4] - clamped[:, 0:2]
    L = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Lc = L.clamp(l_min, l_max)
    hit_len = (Lc != L).squeeze(-1)
    a = torch.cat([clamped[:, 0:2], clamped[:, 0:2] + d / L * Lc], dim=1)
    end = a[:, 2:4].clamp(lo, hi)
    hit_box = hit_box | (end != a[:, 2:4]).any(dim=1)
    return torch.cat([a[:, 0:2], end], dim=1), torch.stack([hit_box, hit_len], dim=1)


def lyap(occ: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """(B, H, W) images, (H, W) distance field -> (B,) mass-normalised lyapunov."""
    f = occ.reshape(occ.shape[0], -1)
    return (f * dist.reshape(1, -1)).sum(1) / f.sum(1).clamp_min(1e-6)


class ModelObjective:
    """Predicted dv of pushes from one state under one goal, via an
    `OccupancyGradientAdapter`'s `predict_step` (differentiable). `adapter` may
    be a LIST of adapters: the objective is then the mean of the members'
    predicted dv (an ensemble -- the mean of dv, not the dv of the mean image,
    because lyapunov is mass-normalised; EXP-0030/0035/0037 use the same rule)."""

    def __init__(self, adapter, particles: torch.Tensor, dist: torch.Tensor, chunk: int = 128,
                 mask: torch.Tensor | None = None, mass_weight: float = 0.0, signed_weight: float = 0.0,
                 value: str = "lyap", crowd_lam: float = 0.1):
        self.members = list(adapter) if isinstance(adapter, (list, tuple)) else [adapter]
        self.ad, self.chunk = self.members[0], chunk
        dev = self.ad.device
        self.occ0 = occ_from_particles(particles[None].float(), dev)
        self.dist = dist.to(dev)
        # capacity-aware value functions (EXP-0055, simple_mpc/value_functions.py); need `mask`.
        # "emd": sliced W1 to the uniform goal target; "emd_lin": lyapunov with the distance
        # field replaced by the OT dual potential from the CURRENT state; "crowd": lyapunov +
        # crowd_lam * over-capacity penalty ("crowd_floor": capacity >= one isolated cube).
        self.value, self.extra = value, None
        if value != "lyap":
            from simple_mpc import value_functions as vf
            tgt = vf.uniform_target(mask.cpu().numpy().astype(bool)).to(dev)
            if value == "emd":
                self.extra = vf.SlicedEMD(tgt, res=32, n_dirs=16, device=dev)
            elif value == "emd_lin":
                self.dist = vf.linearised_emd_field(self.occ0[0], tgt)
            elif value in ("crowd", "crowd_floor"):
                c = vf.CrowdingPenalty(mask.cpu().numpy().astype(bool), device=dev,
                                       floor_single=value == "crowd_floor")
                self.extra = lambda occ: float(crowd_lam) * c(occ)
            else:
                raise ValueError(f"unknown value {value!r}")
        # optional success-type terms (EXP-0052): cost -= w_m * in-goal mass fraction
        # + w_s * signed-mass fraction, both of the predicted image (differentiable)
        self.mask = mask.to(dev).float() if mask is not None else None
        self.wm, self.ws = float(mass_weight), float(signed_weight)
        self.v0 = self._value(self.occ0)
        # each member's INPUT representation (soft-occupancy NFD, EXP-0063, blurs the raster;
        # identity for every other model) and its value, so a member's predicted dv is measured
        # against its own representation of the start state (a no-op push predicts dv = 0)
        self._occ_in = [m.encode_state(self.occ0) if hasattr(m, "encode_state") else self.occ0
                        for m in self.members]
        self._v0_in = [self._value(o) for o in self._occ_in]

    def _value(self, occ):
        if self.value == "emd":
            return self.extra(occ)
        v = lyap(occ, self.dist)
        if self.value in ("crowd", "crowd_floor"):
            v = v + self.extra(occ)
        if self.mask is not None and (self.wm or self.ws):
            f = occ.reshape(occ.shape[0], -1); m = self.mask.reshape(1, -1)
            tot = f.sum(1).clamp_min(1e-6)
            inside = (f * m).sum(1) / tot
            v = v - self.wm * inside - self.ws * (2 * inside - 1)
        return v

    def __call__(self, act: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, act.shape[0], self.chunk):
            a = act[i:i + self.chunk]
            dvs = [self._value(m.predict_step(self._occ_in[j].expand(a.shape[0], -1, -1).contiguous(), a))
                   - self._v0_in[j] for j, m in enumerate(self.members)]
            outs.append(torch.stack(dvs).mean(0) if len(dvs) > 1 else dvs[0])
        return torch.cat(outs)


def plan(objective: ModelObjective, candidates: torch.Tensor, planner: str, budget_s: float,
         n_restarts: int = 8, lr: float = 1.5e-3, cem_elite_frac: float = 0.125,
         more_candidates=None, cem_pop: int | None = None, push_len: float | None = None) -> dict:
    """Choose one push. Returns {action (4,), pred_dv, n_evals, n_iters, time_s}.

    `more_candidates()` -> (K, 4): if given, the `rank` planner keeps drawing and
    scoring fresh candidate batches until the budget is spent (sample-and-rank at
    a matched budget); without it `rank` scores only `candidates` (the EXP-0032
    pilot's behaviour). `cem_pop` sets the CEM/MPPI population per iteration
    (default: the candidate-pool size, the EXP-0032/0039 behaviour); the pool
    still seeds the first elite set. `push_len` fixes the push length (e.g. 0.02 for the
    narrow 20 mm domain, EXP-0053): every candidate and iterate is projected to exactly
    that length (yaw stays perpendicular to the push by the action_to_pose convention)."""
    lo, hi = (push_len, push_len) if push_len else (L_MIN, L_MAX)
    proj = lambda x: project_push(x, lo, hi)
    t0 = time.time()
    dev = objective.ad.device
    cand = proj(candidates.to(dev).float())[0]
    with torch.no_grad():
        dv = objective(cand)
    n_evals, n_iters = len(cand), 0
    rank_s = time.time() - t0            # scoring the pool counts against the budget too
    last_iter, max_iter = 0.0, 0.0

    def time_left():
        # do not START an iteration the previous one says will not fit; the
        # first iteration may still overrun by its own duration (reported)
        return time.time() - t0 + last_iter < budget_s
    best_i = int(dv.argmin())
    best, best_dv = cand[best_i].clone(), float(dv[best_i])
    if planner == "rank":
        while more_candidates is not None and time_left():
            ti = time.time()
            extra = proj(more_candidates().to(dev).float())[0]
            with torch.no_grad():
                dv_x = objective(extra)
            n_evals += len(extra); n_iters += 1
            k = int(dv_x.argmin())
            if float(dv_x[k]) < best_dv:
                best, best_dv = extra[k].clone(), float(dv_x[k])
            last_iter = time.time() - ti; max_iter = max(max_iter, last_iter)
    elif planner == "gd":
        starts = cand[dv.argsort()[:n_restarts]].clone()
        p = starts.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        while time_left():
            ti = time.time()
            opt.zero_grad(set_to_none=True)
            objective(p).sum().backward()
            opt.step()
            with torch.no_grad():
                p.data.copy_(proj(p.data)[0])
                cur = objective(p.data)
                n_evals += len(cur); n_iters += 1
                k = int(cur.argmin())
                if float(cur[k]) < best_dv:
                    best, best_dv = p.data[k].clone(), float(cur[k])
            last_iter = time.time() - ti; max_iter = max(max_iter, last_iter)
    elif planner in ("cem", "mppi"):
        n_pop = int(cem_pop) if cem_pop else len(cand)
        n_elite = max(2, int(cem_elite_frac * n_pop))
        with torch.no_grad():
            elite = cand[dv.argsort()[:n_elite]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
            while time_left():
                ti = time.time()
                pop = proj(mean + std * torch.randn(n_pop, 4, device=dev))[0]
                cost = objective(pop)
                n_evals += n_pop; n_iters += 1
                k = int(cost.argmin())
                if float(cost[k]) < best_dv:
                    best, best_dv = pop[k].clone(), float(cost[k])
                if planner == "cem":
                    elite = pop[cost.argsort()[:n_elite]]
                    mean, std = elite.mean(0), elite.std(0).clamp_min(1e-3)
                else:  # mppi: exponentially weighted mean
                    w = torch.softmax(-(cost - cost.min()) / cost.std().clamp_min(1e-6), 0)
                    mean = (w[:, None] * pop).sum(0)
                    std = ((w[:, None] * (pop - mean) ** 2).sum(0)).sqrt().clamp_min(1e-3)
                last_iter = time.time() - ti; max_iter = max(max_iter, last_iter)
    else:
        raise ValueError(f"unknown planner {planner!r}")
    return dict(action=best.detach().cpu(), pred_dv=best_dv, n_evals=n_evals, n_iters=n_iters,
                time_s=time.time() - t0, max_iter_s=max_iter, rank_s=rank_s)


def run_episode(adapter, particles0: torch.Tensor, dist: torch.Tensor,
                execute: Callable[[torch.Tensor], torch.Tensor],
                sample_candidates: Callable[[torch.Tensor], torch.Tensor],
                planner: str, budget_s: float, n_steps: int,
                on_step: Callable[[dict], None] | None = None, rank_resample: bool = True,
                **plan_kw) -> dict:
    """One closed-loop episode. `execute(action) -> particles after the push`;
    `sample_candidates(particles) -> (K, 4) candidate pushes` for the current
    state. `on_step(record)` is called after every step (checkpoint hook).
    Returns {values (n_steps+1,), actions, pred_dv, true_dv, ...}; values are
    the TRUE lyapunov (soft scoring) of the state after each step."""
    dist = dist.float()
    parts = particles0.float()
    v = [float(lyap(occ_for_scoring(parts[None]), dist)[0])]
    rec = dict(planner=planner, budget_s=budget_s, values=v, actions=[], pred_dv=[], true_dv=[],
               n_evals=[], n_iters=[], plan_time_s=[])
    for t in range(n_steps):
        obj = ModelObjective(adapter, parts, dist)
        out = plan(obj, sample_candidates(parts), planner, budget_s,
                   more_candidates=(lambda: sample_candidates(parts)) if rank_resample else None, **plan_kw)
        parts = execute(out["action"]).float()
        v.append(float(lyap(occ_for_scoring(parts[None]), dist)[0]))
        rec["actions"].append(out["action"].tolist()); rec["pred_dv"].append(out["pred_dv"])
        rec["true_dv"].append(v[-1] - v[-2]); rec["n_evals"].append(out["n_evals"])
        rec["n_iters"].append(out["n_iters"]); rec["plan_time_s"].append(out["time_s"])
        if on_step is not None:
            on_step(rec)
    return rec


def run_episodes_batched(episodes: list, particles0: torch.Tensor,
                         execute_batch: Callable[[torch.Tensor], torch.Tensor],
                         sample_candidates_batch: Callable[[int], torch.Tensor],
                         n_steps: int, n_cand: int = 64, bank_per_env: int = 16384,
                         on_step: Callable[[int, list], None] | None = None,
                         record_states: bool = False) -> list:
    """K closed-loop episodes run side by side, one per simulator env (K = len(episodes)).

    `episodes[k]` = dict(adapter, dist (H, W), planner, budget_s, plan_kw={} optional,
    n_cand optional). `particles0` (K, n, >=3) are the start states.
    `execute_batch(actions (K, 4)) -> particles (K, n, >=3)` executes every env's own
    push at once. `sample_candidates_batch(m) -> (K, m, 4)` draws m candidates for
    every env's CURRENT state.

    Each step: one candidate bank of `bank_per_env` per env is drawn up front (outside
    the planning budget). Then every episode plans in turn, timed while the simulator
    is idle: its first `n_cand` bank rows are the pool, and rank's resampling walks the
    rest of the bank. Then all K chosen pushes execute together. This differs from
    `run_episode` in one way: there, rank's resampling called the sampler INSIDE the
    budget. Returns one record per episode, with the same keys as `run_episode`.
    `on_step(t, records)` is called after every step (checkpoint hook).
    `record_states=True` adds `states` (list of (n, 3) xyz lists, start + after every
    push) to each record, for success-type value functions and videos.
    """
    K = len(episodes)
    parts = particles0.float()
    recs = []
    for ep in episodes:
        recs.append(dict(planner=ep["planner"], budget_s=ep["budget_s"], values=[], actions=[], pred_dv=[],
                         true_dv=[], n_evals=[], n_iters=[], plan_time_s=[]))
    for k, ep in enumerate(episodes):
        recs[k]["values"].append(float(lyap(occ_for_scoring(parts[k:k + 1]), ep["dist"].float())[0]))
        if record_states:
            recs[k]["states"] = [parts[k, :, :3].tolist()]
    for t in range(n_steps):
        bank = sample_candidates_batch(bank_per_env)
        acts = []
        for k, ep in enumerate(episodes):
            n = ep.get("n_cand", n_cand)
            bk = bank[k]
            ptr = [n]

            def more(bk=bk, ptr=ptr, n=n):
                i = ptr[0] % len(bk); ptr[0] += n
                return bk[i:i + n]
            obj = ModelObjective(ep["adapter"], parts[k], ep["dist"].float(), **ep.get("obj_kw", {}))
            out = plan(obj, bk[:n], ep["planner"], ep["budget_s"],
                       more_candidates=more if ep["planner"] == "rank" else None, **ep.get("plan_kw", {}))
            acts.append(out["action"])
            r = recs[k]
            r["actions"].append(out["action"].tolist()); r["pred_dv"].append(out["pred_dv"])
            r["n_evals"].append(out["n_evals"]); r["n_iters"].append(out["n_iters"])
            r["plan_time_s"].append(out["time_s"])
        parts = execute_batch(torch.stack(acts)).float()
        for k, ep in enumerate(episodes):
            v = float(lyap(occ_for_scoring(parts[k:k + 1]), ep["dist"].float())[0])
            recs[k]["true_dv"].append(v - recs[k]["values"][-1]); recs[k]["values"].append(v)
            if record_states:
                recs[k]["states"].append(parts[k, :, :3].tolist())
        if on_step is not None:
            on_step(t, recs)
    return recs
