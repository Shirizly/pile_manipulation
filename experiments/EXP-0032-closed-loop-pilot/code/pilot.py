"""EXP-0032: closed-loop MPC pilot with learned models (simple_mpc/learned_mpc.py).

Pilot, not a result: does the harness run end to end, how long does an
episode take, and how large is episode-to-episode variance (which sizes the
real study, TODO G1b)? Checkpoints after EVERY step to results/episodes.json.

Starts: DS-0005 start states (the offline benchmark states). Execution:
GenesisOracleEnv.step (full fidelity, 1 env, recording off). Candidates:
pile-aware samples from the simulator for the current state.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import dist_field_from_mask, letter_mask, quadrant_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import run_episode, oracle_config_with_physics, apply_physics
from simple_mpc.oracle_mpc import load_oracle_config

RES = REPO / "experiments/EXP-0032-closed-loop-pilot/results/episodes.json"


def goal_dist(name):
    m = quadrant_mask(OCC_GRID, OCC_GRID, int(name[-1])) if name.startswith("quadrant_") \
        else letter_mask(name[-1], OCC_GRID, OCC_GRID)
    return torch.from_numpy(dist_field_from_mask(m)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--models", nargs="*", default=["nfd_warped_randlen_flipaug", "linear_switched_hard", "nfd_3ch_finetuned"])
    ap.add_argument("--planners", nargs="*", default=["rank", "gd", "cem"])
    ap.add_argument("--goals", nargs="*", default=["letter_T", "quadrant_0"])
    ap.add_argument("--starts", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=64)
    a = ap.parse_args()
    RES.parent.mkdir(parents=True, exist_ok=True)
    rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
    starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in a.starts}
    cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = 1
    cfg = oracle_config_with_physics(cfg)      # the models' training physics (= DS-0006's)
    env = GenesisOracleEnv(cfg, n_envs=1)
    apply_physics(env)
    import genesis as gs
    sim = env._sim
    done = json.load(open(RES)) if RES.exists() else {"episodes": [], "config": vars(a)}
    have = {(e["model"], e["planner"], e["goal"], e["start"]) for e in done["episodes"] if e.get("complete")}

    def execute(action):
        env.step(action.numpy().astype(np.float32))
        return sim._particle_state[0].detach().cpu().float().clone()

    def sample_candidates(parts):
        s, e, _ = sim.generate_action_samples(a.n_cand, pile_aware=True, min_swath_particles=3)
        return torch.cat([s[0, :, :2], e[0, :, :2]], 1).float().cpu()

    for m in a.models:
        ad = make_occ_adapter(m, "cuda" if torch.cuda.is_available() else "cpu", "corner")
        for g in a.goals:
            D = goal_dist(g)
            for s0 in a.starts:
                for pl in a.planners:
                    key = (m, pl, g, s0)
                    if key in have:
                        continue
                    st = starts[s0]
                    env.restore_snapshot({"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)})
                    sim.update_material_state()
                    ep = dict(model=m, planner=pl, goal=g, start=s0, budget_s=a.budget, complete=False)
                    done["episodes"].append(ep)
                    t0 = time.time()

                    def ckpt(rec):
                        ep.update({k: v for k, v in rec.items()}); ep["wall_s"] = time.time() - t0
                        tmp = Path(str(RES) + ".tmp"); tmp.write_text(json.dumps(done, indent=1)); os.replace(tmp, RES)
                    rec = run_episode(ad, sim._particle_state[0].detach().cpu().float().clone(), D, execute,
                                      sample_candidates, pl, a.budget, a.steps, on_step=ckpt)
                    ep["complete"] = True; ckpt(rec)
                    v = rec["values"]
                    print(f"{m:28s} {pl:5s} {g:10s} start {s0}: V {v[0]:.3f} -> {v[-1]:.3f} "
                          f"(improve {v[0] - v[-1]:+.3f}); evals/step {np.mean(rec['n_evals']):.0f}; "
                          f"{time.time() - t0:.0f}s", flush=True)
        del ad; torch.cuda.empty_cache()
    env.destroy()


if __name__ == "__main__":
    main()
