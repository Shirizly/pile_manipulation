"""EXP-0075 closed-loop driver (adapted from EXP-0043 batched_closed_loop.py): Sean-trained NFDs (EXP-0074) as the model, horizon-H CEM / rank planners (wide_planner.py).
Episodes = cells x goals x starts, run in chunks of K=32 (one per simulator env, all legal: --legalize --legalize-fallback, seeded planners). results/<tag>.json is rewritten after every step;
finished chunks are kept on restart. Steps are limited to <= 10 by the benchmark brief.
usage: PYTHONPATH=. python -u closed_loop.py --tag T --goals letter_O_w20 ... --starts 40..47 --cells '{"name": {"model": "ens2bal", "planner": "cem", "H": 2, "push_len": 0.02}}' --steps 10"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments/EXP-0075-closed-loop-benchmark/code"))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask, goal_dist
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
from simple_mpc.oracle_mpc import load_oracle_config
from transforms.functional import action_to_pose
import wide_planner as wp

K = 32
R74 = "experiments/EXP-0074-wide-domain-zoom-nfd/runs/"
MODELS = {
    "ens2bal": lambda: wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth"), wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True, name="ens2bal"),
    "zoom128bal": lambda: wp.WideEns([wp.Member("zoom", 128, R74 + "z128_f8_ms4/unet_best.pth")], balance=True, name="zoom128bal"),
    "vanilla128bal": lambda: wp.WideEns([wp.Member("world", 128, R74 + "w128_f8_ms4/unet_best.pth")], balance=True, name="vanilla128bal"),
    "vanilla64bal": lambda: wp.WideEns([wp.Member("world", 64, R74 + "w64_s0_ms4/unet_best.pth", feats=(4, 8, 16))], balance=True, name="vanilla64bal"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--results-dir", default=str(REPO / "experiments/EXP-0075-closed-loop-benchmark/results"))
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--goals", nargs="*", default=["letter_O_w20", "letter_T_w20", "letter_S_w20", "letter_X_w20"])
    ap.add_argument("--starts", type=int, nargs="*", default=list(range(40, 48)))
    ap.add_argument("--cells", required=True); ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--bank-per-env", type=int, default=4096); ap.add_argument("--planner-seed", type=int, default=0)
    a = ap.parse_args()
    assert a.steps <= 10, "benchmark brief: no task beyond 10 steps"
    res_p = Path(a.results_dir) / f"{a.tag}.json"; res_p.parent.mkdir(parents=True, exist_ok=True)
    cells = json.loads(a.cells)
    specs = [dict(cell=c, goal=g, start=s) for c in cells for g in a.goals for s in a.starts]
    done = json.loads(res_p.read_text()) if res_p.exists() else {"config": vars(a), "episodes": []}
    done["episodes"] = [e for e in done["episodes"] if e.get("complete")]
    have = {(e["cell"], e["goal"], e["start"]) for e in done["episodes"]}
    todo = [s for s in specs if (s["cell"], s["goal"], s["start"]) not in have]
    rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
    starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in a.starts}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K); apply_physics(env); sim = env._sim
    import genesis as gs
    models, dists, masks = {}, {g: goal_dist(g) for g in a.goals}, {g: torch.from_numpy(goal_mask(g)) for g in a.goals}
    for c in cells.values():
        if c["model"] not in models:
            models[c["model"]] = MODELS[c["model"]]()
    sim._settle_steps = env._real_settle_steps; sim._clearance_ctrl_steps = env._real_clearance_steps
    timing = {"sample_s": 0.0, "execute_s": 0.0}

    def execute_batch(acts):
        t_ex = time.time()
        sx, sy, ex, ey, ang = action_to_pose(acts.to(gs.device).float()); z = torch.full_like(sx, float(sim._operation_height))
        sim.execute_action(torch.stack([sx, sy, z], 1), torch.stack([ex, ey, z], 1), ang); sim.update_material_state()
        timing["execute_s"] += time.time() - t_ex
        return sim._particle_state[:, :, :7].detach().cpu().float().clone()

    def sample(m):
        t_s = time.time(); s, e, _ = sim.generate_action_samples(m, pile_aware=True, min_swath_particles=3)
        bank = torch.cat([s[..., :2], e[..., :2]], -1).float().cpu(); timing["sample_s"] += time.time() - t_s
        return bank

    def legalize(acts, envs=None):
        from Genesis.action_sampling import legalize_pushes
        from model.retrieval.frame import yaw_from_quat
        from simple_mpc.adapters import OCC_BOUNDS
        from simple_mpc.learned_mpc import XY_MARGIN
        st = sim._particle_state.detach().cpu().float()
        if envs is not None:
            st = st[envs]
        return legalize_pushes(acts, st[:, :, :2], yaw_from_quat(st[:, :, 3:7]), 0.0025, 0.02, 0.001, box=(OCC_BOUNDS["x_min"] + XY_MARGIN, OCC_BOUNDS["x_max"] - XY_MARGIN))

    for c0 in range(0, len(todo), K):
        chunk = todo[c0:c0 + K]; padded = chunk + [chunk[-1]] * (K - len(chunk))
        st = torch.stack([starts[s["start"]] for s in padded])
        sim.set_particle_state(st[:, :, :3].to(gs.device), st[:, :, 3:7].to(gs.device)); sim.update_material_state()
        p0 = sim._particle_state[:, :, :7].detach().cpu().float().clone()
        eps = []
        for s in padded:
            c = dict(cells[s["cell"]]); eps.append(dict(model=models[c.pop("model")], dist=dists[s["goal"]], mask=masks[s["goal"]], planner=c.pop("planner"), H=c.pop("H", 1),
                                                        push_len=c.pop("push_len", None), mass_weight=c.pop("mass_weight", 1.0), obj_mode=c.pop("obj_mode", "terminal"), plan_kw=c))
        t0 = time.time(); live = [dict(s, complete=False) for s in chunk]; done["episodes"] += live

        def ckpt(t, recs):
            for e, r in zip(live, recs):
                e.update(r); e["wall_s"] = time.time() - t0
            tmp = Path(str(res_p) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, res_p)
            print(f"chunk {c0 // K}: step {t + 1}/{a.steps} ({time.time() - t0:.0f}s; " + ", ".join(f"{k} {v:.0f}" for k, v in timing.items()) + ")", flush=True)
        recs = wp.run_seq_episodes(eps, p0, execute_batch, sample, a.steps, bank_per_env=a.bank_per_env, on_step=ckpt, legalize=legalize, legalize_fallback=True, planner_seed=a.planner_seed)
        for e in live:
            e["complete"] = True
        ckpt(a.steps - 1, recs)
    env.destroy(); print("wrote", res_p)


if __name__ == "__main__":
    main()
