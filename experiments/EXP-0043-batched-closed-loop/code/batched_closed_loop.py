"""Batched closed-loop episodes (EXP-0043; reused by later benchmark runs).

Episodes = the cross product of --models x --planners x --goals x --starts (x --cells:
named plan-parameter overrides, JSON). They run in chunks of K=32, one episode per env.
Checkpoint: results/<tag>.json is rewritten after every step. Finished chunks are kept;
a half-finished chunk is dropped and rerun on restart.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import dist_field_from_mask, letter_mask, quadrant_mask, two_squares_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, run_episodes_batched
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose

HERE = REPO / "experiments/EXP-0043-batched-closed-loop"
OUT_DIRS = {}
K = 32


def goal_mask(name):
    if name == "two_squares":
        return two_squares_mask(OCC_GRID, OCC_GRID)
    return quadrant_mask(OCC_GRID, OCC_GRID, int(name[-1])) if name.startswith("quadrant_") \
        else letter_mask(name[-1], OCC_GRID, OCC_GRID)


def goal_dist(name):
    m = goal_mask(name)
    return torch.from_numpy(dist_field_from_mask(m)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", default=None, help="where <tag>.json goes (default: EXP-0043 results/)")
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--models", nargs="*", default=["nfd_3ch_randlen", "nfd_3ch_randlen_seed1", "linear_switched_soft",
                                                    "nfd_residual_worldframe_noaug_ep43"])
    ap.add_argument("--planners", nargs="*", default=["gd", "cem"])
    ap.add_argument("--goals", nargs="*", default=["quadrant_0", "letter_O", "letter_T", "letter_S"])
    ap.add_argument("--starts", type=int, nargs="*", default=[40, 41, 42, 43])
    ap.add_argument("--cells", default='{"default": {}}',
                    help='JSON {name: {budget_s, n_cand, n_restarts, lr, cem_pop, cem_elite_frac}} '
                         'or per planner {name: {"gd": {...}, "cem": {...}}}')
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=64)
    ap.add_argument("--bank-per-env", type=int, default=16384)
    ap.add_argument("--record-states", action="store_true", help="store particle xyz after every push")
    ap.add_argument("--starts-file", default=None,
                    help="narrow pools file (e.g. Genesis/data/narrow_l20_n20/test_pools_v2/pools_0.pt): --starts are then POOL "
                         "indices and each start is that pool's first row (DS-0016 has clump and scatter pools, field start_kind)")
    ap.add_argument("--legalize", action="store_true",
                    help="slide touchdown-illegal planned pushes back until the blade clears every cube "
                         "(Genesis.action_sampling.legalize_pushes; ISS-013, user rule 2026-10-03). Off = as before")
    a = ap.parse_args()
    res_p = (Path(a.results_dir) if a.results_dir else HERE / "results") / f"{a.tag}.json"; res_p.parent.mkdir(parents=True, exist_ok=True)
    cells = json.loads(a.cells)
    specs = [dict(model=m, planner=pl, goal=g, start=s, cell=c) for m in a.models for c in cells
             for pl in a.planners for g in a.goals for s in a.starts]
    done = json.loads(res_p.read_text()) if res_p.exists() else {"config": vars(a), "episodes": []}
    done["episodes"] = [e for e in done["episodes"] if e.get("complete")]
    have = {(e["model"], e["planner"], e["goal"], e["start"], e["cell"]) for e in done["episodes"]}
    todo = [s for s in specs if (s["model"], s["planner"], s["goal"], s["start"], s["cell"]) not in have]
    if a.starts_file:
        pf = torch.load(str(REPO / a.starts_file), map_location="cpu", weights_only=False)
        starts = {s: pf["states"][(pf["pool_idx"] == s).nonzero()[0, 0]].float() for s in a.starts}
    else:
        rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
        starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in a.starts}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    dev = "cuda"
    adapters, dists = {}, {g: goal_dist(g) for g in a.goals}
    for m in a.models:
        adapters[m] = make_occ_adapter(m, dev, "corner")
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps

    timing = {"sample_s": 0.0, "select_s": 0.0, "execute_s": 0.0}

    def execute_batch(acts):
        t_ex = time.time()
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float())
        z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang)
        sim.update_material_state()
        timing["execute_s"] += time.time() - t_ex
        return sim._particle_state[:, :, :3].detach().cpu().float().clone()

    env_specs = []        # per-env (sampler, n_cand, goal) of the current chunk, set below

    def sample(m):
        t_s = time.time()
        s, e, _ = sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)
        bank = torch.cat([s[..., :2], e[..., :2]], -1).float().cpu()
        # goal-aware candidate selection (EXP-0056, simple_mpc/goal_aware_sampling.py): the
        # planner's first n_cand rows are replaced by a selection from the pile-aware bank
        xy_all = sim._particle_state[:, :, :2].detach().cpu().float()
        timing["sample_s"] += time.time() - t_s; t_s = time.time()
        for k, (smp, n, g) in enumerate(env_specs):
            if smp == "pile":
                continue
            import simple_mpc.goal_aware_sampling as gas
            gen = torch.Generator().manual_seed(1000 * k + len(env_specs))
            msk, xy = goal_mask(g), xy_all[k]
            if smp == "misplaced":
                bank[k, :n] = gas.select_misplaced(bank[k], xy, msk, dists[g], n, gen)
            elif smp == "deposit":
                bank[k, :n] = gas.select_deposit(bank[k], xy, msk, dists[g], n, gen)
            elif smp == "ot_mix":
                h = n // 2
                prop = gas.ot_proposals(xy, msk, h, gen)
                if prop is not None:
                    bank[k, :h] = prop
            else:
                raise ValueError(f"unknown sampler {smp!r}")
        timing["select_s"] += time.time() - t_s
        return bank

    # warm-up (lazy imports) so the first episode's budget is not eaten
    from simple_mpc.learned_mpc import ModelObjective, plan
    o = ModelObjective(adapters[a.models[0]], starts[a.starts[0]], dists[a.goals[0]])
    plan(o, torch.zeros(8, 4) + 0.01, "gd", 0.2); plan(o, torch.zeros(8, 4) + 0.01, "cem", 0.1)
    for c0 in range(0, len(todo), K):
        chunk = todo[c0:c0 + K]
        padded = chunk + [chunk[-1]] * (K - len(chunk))
        st = torch.stack([starts[s["start"]] for s in padded])
        sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device))
        sim.update_material_state()
        p0 = sim._particle_state[:, :, :3].detach().cpu().float().clone()
        eps = []
        env_specs.clear()
        for s in padded:
            kw = cells[s["cell"]]
            kw = dict(kw.get(s["planner"], {}) if any(k in kw for k in ("rank", "gd", "cem", "mppi")) else kw)
            b = kw.pop("budget_s", a.budget); n = kw.pop("n_cand", a.n_cand)
            smp = kw.pop("sampler", "pile")
            env_specs.append((smp, n, s["goal"]))
            obj_kw = {k2: kw.pop(k2) for k2 in ("mass_weight", "signed_weight", "value", "crowd_lam") if k2 in kw}
            if obj_kw:
                obj_kw["mask"] = torch.from_numpy(goal_mask(s["goal"]))
            eps.append(dict(adapter=adapters[s["model"]], dist=dists[s["goal"]], planner=s["planner"],
                            budget_s=b, n_cand=n, plan_kw=kw, obj_kw=obj_kw))
        t0 = time.time()
        live = [dict(s, complete=False) for s in chunk]
        done["episodes"] += live

        def ckpt(t, recs):
            for e, r in zip(live, recs):
                e.update(r); e["wall_s"] = time.time() - t0
            tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, res_p)
            print(f"chunk {c0 // K}: step {t + 1}/{a.steps} ({time.time() - t0:.0f}s; cumulative "
                  + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()) + ")", flush=True)
        def legalize(acts):
            from Genesis.action_sampling import legalize_pushes
            from model.retrieval.frame import yaw_from_quat
            from simple_mpc.adapters import OCC_BOUNDS
            from simple_mpc.learned_mpc import XY_MARGIN
            st = sim._particle_state.detach().cpu().float()
            half = 0.02  # Genesis/configs/basic.yaml plate.size[0] / 2 (40 mm blade); 5 mm cubes below
            return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, half, 0.001,
                                   box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))
        recs = run_episodes_batched(eps, p0, execute_batch, sample, a.steps, n_cand=a.n_cand,
                                    bank_per_env=a.bank_per_env, on_step=ckpt,
                                    record_states=a.record_states,
                                    legalize=legalize if a.legalize else None)
        for e in live:
            e["complete"] = True
        ckpt(a.steps - 1, recs)
        for e in live:
            v = e["values"]
            print(f"  {e['model']:36s} {e['planner']:4s} {e['cell']:10s} {e['goal']:10s} {e['start']}: "
                  f"improve {v[0] - v[-1]:+.3f}; evals/step {np.mean(e['n_evals']):.0f}", flush=True)
    env.destroy()
    print("wrote", res_p)


if __name__ == "__main__":
    main()
