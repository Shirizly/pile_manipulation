"""EXP-0050: perfect-model (simulator-as-model) closed-loop MPC ceilings (DESIGN.md).

--mode greedy:    each push, simulate --n-cand candidates from every episode's true state
                  and keep the one with the lowest true lyapunov (its outcome is the next state).
--mode lookahead: K1 first pushes x K2 second pushes; execute the first push whose best
                  second outcome is lowest.
Simulation is batched ACROSS episodes: 32 envs, each set to its own (episode) state.
Results JSON rewritten after every push; a restart resumes from the last completed push.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import dist_field_from_mask, letter_goal_mask, quadrant_mask, two_squares_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import OCC_GRID, occ_for_scoring
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, lyap, oracle_config_with_physics, project_push
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose

HERE = REPO / "experiments/EXP-0050-oracle-ceiling"
K = 32  # default n_envs; overridden by --n-envs (EXP-0057)


def goal_mask(name):
    if name == "two_squares":
        return two_squares_mask(OCC_GRID, OCC_GRID)
    return quadrant_mask(OCC_GRID, OCC_GRID, int(name[-1])) if name.startswith("quadrant_") \
        else letter_goal_mask(name, OCC_GRID, OCC_GRID)  # EXP-0067: also letter_<L>_w<mm>


def goal_dist(name):
    return torch.from_numpy(dist_field_from_mask(goal_mask(name))).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["greedy", "lookahead", "cem"], default="greedy")
    ap.add_argument("--cem-pop", type=int, default=32)
    ap.add_argument("--cem-iters", type=int, default=3)
    ap.add_argument("--cem-elite", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--goals", nargs="*", default=["quadrant_0", "quadrant_3", "letter_O", "letter_T",
                                                   "letter_S", "letter_L", "letter_X", "letter_Z"])
    ap.add_argument("--starts", type=int, nargs="*", default=[40, 41])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--k1", type=int, default=16)
    ap.add_argument("--k2", type=int, default=8)
    ap.add_argument("--mass-weight", type=float, default=0.0,
                    help="selection cost = lyapunov - w x in-goal mass fraction (EXP-0052's objective)")
    ap.add_argument("--value", choices=["lyap", "crowd_floor"], default="lyap",
                    help="EXP-0057: planning objective; crowd_floor = lyapunov + 0.1 x CrowdingPenalty"
                         " (floor_single=True). Combine with --mass-weight for lyap-mass cells.")
    ap.add_argument("--n-envs", type=int, default=K, help="EXP-0057: batched sim envs (throughput vs memory)")
    ap.add_argument("--l-min", type=float, default=0.020, help="EXP-0057: push-length-range ablation")
    ap.add_argument("--l-max", type=float, default=0.070, help="EXP-0057: push-length-range ablation")
    ap.add_argument("--out-dir", default=None, help="EXP-0057: results dir override (default: EXP-0050's results/)")
    ap.add_argument("--seed-base", type=int, default=None,
                    help="seed pile-aware sampling and CEM refits per (goal, start, push, iteration) -> identical "
                         "random streams for a task at every ablation point (EXP-0057); default unseeded")
    ap.add_argument("--stop-solved", action="store_true",
                    help="cem mode: stop simulating an episode once in-goal mass >= 0.9 x the EXP-0046 optimum; "
                         "its state is carried forward (action None, 0 sims) so per-push arrays keep full length")
    ap.add_argument("--vstar-files", nargs="*", default=[],
                    help="EXP-0067: extra vstar.json files (repo-relative) whose per-goal optimum overrides/extends EXP-0046's")
    ap.add_argument("--legal", action="store_true",
                    help="G4 (2026-10-15), cem mode only: legalise every candidate (initial pool, every refit, fallback) with "
                         "Genesis/action_sampling.py::legalize_pushes_torch exactly as the EXP-0075 harness (oracle_legal.py); "
                         "un-legalisable candidates get cost inf and are never simulated or executed. Also records quaternions "
                         "(quats), per-decision fetch-like fraction of evaluations (reach_frac), entering/leaving, rel vs the "
                         "EXP-0067 vstar_width.json optimum, and legal shift / fallback counters. Off: behaviour unchanged.")
    a = ap.parse_args()
    if a.legal and a.mode != "cem":
        ap.error("--legal is implemented for --mode cem only")
    if a.legal and "experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json" not in a.vstar_files:
        a.vstar_files = list(a.vstar_files) + ["experiments/EXP-0067-letter-stroke-width-sweep/results/vstar_width.json"]
    n_envs = a.n_envs
    import zlib
    OPT = {}
    for vf in [REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json"] + [REPO / f for f in a.vstar_files]:
        OPT.update({g: v.get("mass_frac_best_placement", 1.0) for g, v in json.loads(Path(vf).read_text()).items()})

    def seed_of(e, t, it):
        return zlib.crc32(f"{e['goal']}|{e['start']}|{t}|{it}|{a.seed_base}".encode()) & 0x7FFFFFFF
    out_dir = Path(a.out_dir) if a.out_dir else (HERE / "results")
    res_p = out_dir / f"{a.tag}.json"; res_p.parent.mkdir(parents=True, exist_ok=True)
    rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = n_envs
    env = GenesisOracleEnv(cfg, n_envs=n_envs); apply_physics(env); sim = env._sim
    import genesis as gs
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    eps = [dict(goal=g, start=s) for g in a.goals for s in a.starts]
    D = {g: goal_dist(g) for g in a.goals}
    MK = {g: torch.from_numpy(goal_mask(g)).float() for g in a.goals}

    def inside(final, goal):
        f = occ_for_scoring(final[:, :, :3]).reshape(len(final), -1)
        return (f * MK[goal].reshape(1, -1)).sum(1) / f.sum(1).clamp_min(1e-6)

    def signed_inside(final, goal):
        f = occ_for_scoring(final[:, :, :3]).reshape(len(final), -1)
        m = MK[goal].reshape(1, -1); tot = f.sum(1).clamp_min(1e-6)
        return (2 * (f * m).sum(1) / tot - 1)

    CR = {}
    if a.value == "crowd_floor":
        from simple_mpc.value_functions import CrowdingPenalty
        CR = {g: CrowdingPenalty(MK[g].numpy().astype(bool), device="cpu", floor_single=True) for g in a.goals}

    if res_p.exists():
        out = json.loads(res_p.read_text())
        states = torch.tensor(out["state"])
    else:
        st0 = []
        for e in eps:
            st0.append(rows.states[(rows.slate_idx == e["start"]).nonzero()[0, 0]].float())
        # settle starts as the closed-loop runners do
        states = torch.stack(st0)
        out = {"config": vars(a), "episodes": [dict(e, cell=a.tag, values=[], actions=[]) for e in eps], "step": 0}
        for i in range(0, len(eps), n_envs):
            chunk = states[i:i + n_envs]; n = len(chunk)
            pad = torch.cat([chunk, chunk[:1].expand(n_envs - n, -1, -1)]) if n < n_envs else chunk
            sim.set_particle_state(pad[:, :, :3].to(gs.device), pad[:, :, 3:7].to(gs.device)); sim.update_material_state()
            states[i:i + n] = sim._particle_state[:n].detach().cpu().float().clone()
        for e, s in zip(out["episodes"], states):
            e["values"].append(float(lyap(occ_for_scoring(s[None, :, :3]), D[e["goal"]])[0]))
            e["mass_frac"] = [float(inside(s[None], e["goal"])[0])]
            e["signed_frac"] = [float(signed_inside(s[None], e["goal"])[0])]
            e["states"] = [s[:, :3].tolist()]
            if a.legal:
                e["quats"] = [s[:, 3:7].tolist()]
                e["rel"] = [e["mass_frac"][0] / OPT.get(e["goal"], 1.0)]

    def sample(state, n):
        """n pile-aware candidates from one state (all envs set to it)."""
        sim.set_particle_state(state[None, :, :3].to(gs.device), state[None, :, 3:7].to(gs.device))
        m = -(-n // n_envs)
        s, e, _ = sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)
        c = torch.cat([s[..., :2], e[..., :2]], -1).reshape(-1, 4).float().cpu()
        return project_push(c[:n], l_min=a.l_min, l_max=a.l_max)[0]

    def simulate(pairs):
        """pairs: list of (state (P,7), action (4,)) -> final states (len, P, 7)."""
        outs = []
        for i in range(0, len(pairs), n_envs):
            ch = pairs[i:i + n_envs]; n = len(ch)
            ch = ch + [ch[0]] * (n_envs - n)
            st = torch.stack([p[0] for p in ch]); ac = torch.stack([p[1] for p in ch])
            sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device))
            sx, sy, ex, ey, ang = action_to_pose(ac.to(gs.device))
            z = torch.full_like(sx, float(sim._operation_height))
            sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
            sim.update_material_state()
            outs.append(sim._particle_state[:n].detach().cpu().float().clone())
        return torch.cat(outs)

    def val(final, goal):
        occ = occ_for_scoring(final[:, :, :3])
        v = lyap(occ, D[goal])
        if a.value == "crowd_floor":
            v = v + 0.1 * CR[goal](occ)
        if a.mass_weight:
            v = v - a.mass_weight * inside(final, goal)
        return v

    if a.legal:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from oracle_legal import make_legaliser, masked_values, cem_refit, CUBE_HALF, BLADE_HL, BLADE_HW
        sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code"))
        res_p = res_p.resolve(); os.chdir(REPO)          # wide_planner inserts a cwd-relative EXP-0074 path
        import wide_planner as wp                        # imports Genesis (already loaded here)
        from analyse import transition_diag
        assert (wp.CUBE_HALF, wp.BLADE_HL, wp.BLADE_HW) == (CUBE_HALF, BLADE_HL, BLADE_HW), "legaliser constants drifted from EXP-0075"

    def legal_cem_step(t):
        """One decision for every live episode, every candidate legalised before it is simulated (cost inf if impossible)."""
        live = [i for i, e in enumerate(out["episodes"]) if not (a.stop_solved and e.get("solved_at"))]
        L = {i: make_legaliser(states[i]) for i in live}
        tl = 0.0

        def legalise(i, acts):
            nonlocal tl
            t1 = time.time(); a2, sh, ok = L[i](acts.float().cpu()); tl += time.time() - t1
            return a2.cpu(), sh.cpu(), ok.cpu()

        def seed(i, it):
            if a.seed_base is not None:
                torch.manual_seed(seed_of(out["episodes"][i], t, it)); np.random.seed(seed_of(out["episodes"][i], t, it))

        def fresh(i, it):
            seed(i, it)
            return legalise(i, sample(states[i], a.cem_pop))

        pop, okm, shm = {}, {}, {}
        for i in live:
            pop[i], shm[i], okm[i] = fresh(i, 0)
        best_v = {i: float("inf") for i in live}; best_a = {}; best_s = {}; best_sh = {}
        evals = {i: [] for i in live}; n_ill = {i: 0 for i in live}; n_imp = {i: 0 for i in live}; n_cand = {i: 0 for i in live}; n_resample = {i: 0 for i in live}

        def run_iter(it, ids):
            pairs, owner = [], []
            for i in ids:
                n_cand[i] += len(pop[i]); n_ill[i] += int(((shm[i] > 0) | ~okm[i]).sum()); n_imp[i] += int((~okm[i]).sum())
                for j in okm[i].nonzero().flatten().tolist():
                    pairs.append((states[i], pop[i][j])); owner.append((i, j))
            fin = simulate(pairs) if pairs else None
            vs = {}
            for i in ids:
                rows = [k for k, (o, _) in enumerate(owner) if o == i]
                v_legal = val(fin[rows], out["episodes"][i]["goal"]) if rows else torch.zeros(0, device="cpu")
                v = masked_values(v_legal, okm[i]); vs[i] = v
                evals[i].append(pop[i][okm[i]])
                if rows:
                    jj = int(v_legal.argmin())
                    if float(v_legal[jj]) < best_v[i]:
                        j = owner[rows[jj]][1]
                        best_v[i], best_a[i], best_s[i], best_sh[i] = float(v_legal[jj]), pop[i][j].clone(), fin[rows[jj]].clone(), float(shm[i][j])
            return vs

        for it in range(a.cem_iters if live else 0):
            vs = run_iter(it, live)
            if it == a.cem_iters - 1:
                break
            for i in live:
                e = out["episodes"][i]
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed_of(e, t, it + 1) if a.seed_base is not None else int(torch.randint(0, 2**31 - 1, (1,), device="cpu")))
                new = cem_refit(pop[i], vs[i], a.cem_elite, a.cem_pop, gen, a.l_min, a.l_max)
                if new is None:                      # no legal row at all: fresh pile-aware draw
                    n_resample[i] += 1; pop[i], shm[i], okm[i] = fresh(i, 1000 + it)
                else:
                    pop[i], shm[i], okm[i] = legalise(i, new)
        # fallback: an episode with no legal simulated candidate gets up to 3 fresh legalised pile-aware draws, never an illegal push
        for k in range(3):
            miss = [i for i in live if i not in best_a]
            if not miss:
                break
            for i in miss:
                n_resample[i] += 1; pop[i], shm[i], okm[i] = fresh(i, 2000 + k)
            run_iter(a.cem_iters + k, miss)
        for i, e in enumerate(out["episodes"]):
            if i not in live or i not in best_a:     # solved earlier, or no legal push found: carry the state forward, no push
                e["values"].append(e["values"][-1]); e["actions"].append(None)
                e.setdefault("obj_values", []).append(None)
                e.setdefault("sims_used", []).append(0 if i not in live else sum(len(x) for x in evals[i]))
                e.setdefault("legal_fail", []).append(i in live)
                for kk in ("reach_frac", "exec_reach", "illegal_frac", "unlegal_frac", "legal_shift_m", "n_resample", "entering", "leaving"):
                    e.setdefault(kk, []).append(None)
                continue
            prev = states[i].clone()
            states[i] = best_s[i]; e["values"].append(best_v[i]); e["actions"].append(best_a[i].tolist())
            e.setdefault("obj_values", []).append(best_v[i])
            ev = torch.cat(evals[i]) if evals[i] else torch.zeros(0, 4, device="cpu")
            e.setdefault("sims_used", []).append(len(ev))
            e.setdefault("legal_fail", []).append(False)
            m = MK[e["goal"]]
            e.setdefault("reach_frac", []).append(wp.bank_reach_frac(ev, prev[:, :2], m))           # fetch-like fraction of the decision's evaluations
            e.setdefault("exec_reach", []).append(wp.bank_reach_frac(best_a[i][None], prev[:, :2], m))
            e.setdefault("illegal_frac", []).append(n_ill[i] / max(n_cand[i], 1))                     # proposed candidates that were illegal as proposed (shifted or impossible)
            e.setdefault("unlegal_frac", []).append(n_imp[i] / max(n_cand[i], 1))                     # proposed candidates with no legal offset (cost inf, not simulated)
            e.setdefault("legal_shift_m", []).append(best_sh[i])
            e.setdefault("n_resample", []).append(n_resample[i])
            td = transition_diag(torch.stack([prev, states[i]]), D[e["goal"]], MK[e["goal"]])
            e.setdefault("entering", []).append(int(td["entering"][0])); e.setdefault("leaving", []).append(int(td["leaving"][0]))
        return tl

    for t in range(out["step"], a.steps):
        t0 = time.time(); t_legal_s = None
        if a.mode == "greedy":
            cands = [sample(states[i], a.n_cand) for i in range(len(eps))]
            pairs = [(states[i], c) for i in range(len(eps)) for c in cands[i]]
            fin = simulate(pairs).view(len(eps), a.n_cand, *states.shape[1:])
            for i, e in enumerate(out["episodes"]):
                v = val(fin[i], e["goal"]); j = int(v.argmin())
                states[i] = fin[i, j]; e["values"].append(float(v[j])); e["actions"].append(cands[i][j].tolist())
        elif a.mode == "cem" and a.legal:
            t_legal_s = legal_cem_step(t)
        elif a.mode == "cem":
            # simulator-as-model CEM: iteration 0 = pile-aware samples, then Gaussian refits to the
            # true-best elites (4-D action; project_push keeps it legal); keep the best ever simulated
            live = [i for i, e in enumerate(out["episodes"]) if not (a.stop_solved and e.get("solved_at"))]
            pop = {}
            for i in live:
                if a.seed_base is not None:
                    torch.manual_seed(seed_of(out["episodes"][i], t, 0)); np.random.seed(seed_of(out["episodes"][i], t, 0))
                pop[i] = sample(states[i], a.cem_pop)
            best_v = [float("inf")] * len(eps); best_a = [None] * len(eps); best_s = [None] * len(eps)
            for it in range(a.cem_iters if live else 0):
                fin = simulate([(states[i], c) for i in live for c in pop[i]]).view(len(live), a.cem_pop, *states.shape[1:])
                new = {}
                for li, i in enumerate(live):
                    e = out["episodes"][i]
                    v = val(fin[li], e["goal"]); j = int(v.argmin())
                    if float(v[j]) < best_v[i]:
                        best_v[i], best_a[i], best_s[i] = float(v[j]), pop[i][j].clone(), fin[li, j].clone()
                    el = pop[i][v.argsort()[:a.cem_elite]]
                    mu, sd = el.mean(0), el.std(0).clamp_min(1e-3)
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(seed_of(e, t, it + 1) if a.seed_base is not None else int(torch.randint(0, 2**31 - 1, (1,), device="cpu")))
                    new[i] = project_push(mu + sd * torch.randn(a.cem_pop, 4, device="cpu", generator=gen), l_min=a.l_min, l_max=a.l_max)[0]
                pop = new
            for i, e in enumerate(out["episodes"]):
                if i not in live:        # solved earlier: carry the state forward, no push
                    e["values"].append(e["values"][-1]); e["actions"].append(None)
                    e.setdefault("obj_values", []).append(None); e.setdefault("sims_used", []).append(0)
                    continue
                states[i] = best_s[i]; e["values"].append(best_v[i]); e["actions"].append(best_a[i].tolist())
                e.setdefault("obj_values", []).append(best_v[i])
                e.setdefault("sims_used", []).append(a.cem_pop * a.cem_iters)
        else:
            c1 = [sample(states[i], a.k1) for i in range(len(eps))]
            f1 = simulate([(states[i], c) for i in range(len(eps)) for c in c1[i]]).view(len(eps), a.k1, *states.shape[1:])
            c2 = [[sample(f1[i, j], a.k2) for j in range(a.k1)] for i in range(len(eps))]
            pairs = [(f1[i, j], c) for i in range(len(eps)) for j in range(a.k1) for c in c2[i][j]]
            f2 = simulate(pairs).view(len(eps), a.k1, a.k2, *states.shape[1:])
            for i, e in enumerate(out["episodes"]):
                v1 = val(f1[i], e["goal"])
                v2 = torch.stack([val(f2[i, j], e["goal"]).min() for j in range(a.k1)])
                j = int(v2.argmin())
                e.setdefault("greedy_first_value", []).append(float(v1.min()))
                states[i] = f1[i, j]; e["values"].append(float(v1[j])); e["actions"].append(c1[i][j].tolist())
        dt = time.time() - t0
        for i, e in enumerate(out["episodes"]):
            st = states[i:i + 1]
            if a.mass_weight or a.value != "lyap":   # values[] must stay pure lyapunov for comparability
                e["values"][-1] = float(lyap(occ_for_scoring(st[:, :, :3]), D[e["goal"]])[0])
            e.setdefault("mass_frac", []).append(float(inside(st, e["goal"])[0]))
            if a.stop_solved and not e.get("solved_at") and e["mass_frac"][-1] >= float(os.environ.get("ORACLE_SOLVED_THETA", 0.9)) * OPT.get(e["goal"], 1.0):
                e["solved_at"] = t + 1
            e.setdefault("signed_frac", []).append(float(signed_inside(st, e["goal"])[0]))
            e.setdefault("states", []).append(st[0, :, :3].tolist())
            e.setdefault("plan_time_s", []).append(dt / len(out["episodes"]))
            if a.legal:
                e.setdefault("quats", []).append(st[0, :, 3:7].tolist())
                e.setdefault("rel", []).append(e["mass_frac"][-1] / OPT.get(e["goal"], 1.0))
                e.setdefault("legal_time_s", []).append(t_legal_s / len(out["episodes"]))
        out["step"] = t + 1; out["state"] = states.tolist()
        tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(out)); os.replace(tmp, res_p)
        imp = np.mean([e["values"][0] - e["values"][-1] for e in out["episodes"]])
        msg = f"[{a.mode}] push {t + 1}/{a.steps}: mean improvement {imp:.3f} ({dt:.0f}s)"
        if a.legal:
            msg += f" rel {np.mean([e['rel'][-1] for e in out['episodes']]):.3f}"
        print(msg, flush=True)
    env.destroy()
    print("wrote", res_p)


if __name__ == "__main__":
    main()
