"""EXP-0039: closed-loop MPC episodes for the model population (DESIGN.md).

Same harness as the EXP-0032 pilot (simple_mpc/learned_mpc.py, TRAINING_PHYSICS
executor, soft truth), with: the budget-matched rank planner (resamples candidates
until the budget is spent), ensemble arms (a list of adapters = mean predicted dv),
and resume: results/episodes.json is rewritten after EVERY step and completed
(model, planner, goal, start) cells are skipped on restart.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import dist_field_from_mask, letter_mask, quadrant_mask
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics, run_episode
from simple_mpc.oracle_mpc import load_oracle_config

RES = REPO / "experiments/EXP-0039-closed-loop-metric-validity/results/episodes.json"
ENSEMBLES = {"ensemble_nfd5": ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
                               "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43"]}
# Reduced by the user 2026-09-24 (DESIGN.md addendum 3): no ensemble, no warped NFDs;
# 4 models, ~3 h. The seed pair (nfd_3ch_randlen vs its seed1 retrain) is the
# closed-loop seed noise floor; the world-frame residual is the model whose offline
# ranking (6th) and gradient quality (2nd, EXP-0037) disagree most.
DEFAULT_MODELS = ["nfd_3ch_randlen", "nfd_3ch_randlen_seed1", "linear_switched_soft",
                  "nfd_residual_worldframe_noaug_ep43"]


def goal_dist(name):
    m = quadrant_mask(OCC_GRID, OCC_GRID, int(name[-1])) if name.startswith("quadrant_") \
        else letter_mask(name[-1], OCC_GRID, OCC_GRID)
    return torch.from_numpy(dist_field_from_mask(m)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--planners", nargs="*", default=["rank", "gd", "cem"])
    ap.add_argument("--goals", nargs="*", default=["quadrant_0", "letter_O", "letter_T", "letter_S"])
    ap.add_argument("--starts", type=int, nargs="*", default=[40, 41, 42, 43])
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--n-cand", type=int, default=64)
    a = ap.parse_args()
    RES.parent.mkdir(parents=True, exist_ok=True)
    rows = BinnedSlateCorpus.load(str(REPO / a.corpus)).step(0)
    starts = {s: rows.states[(rows.slate_idx == s).nonzero()[0, 0]].float() for s in a.starts}
    cfg = oracle_config_with_physics(load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml")))
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = 1
    env = GenesisOracleEnv(cfg, n_envs=1); apply_physics(env)
    import genesis as gs
    sim = env._sim
    done = json.load(open(RES)) if RES.exists() else {"episodes": [], "config": vars(a)}
    done["episodes"] = [e for e in done["episodes"] if e.get("complete")]      # drop half-finished episodes
    have = {(e["model"], e["planner"], e["goal"], e["start"]) for e in done["episodes"]}

    def execute(action):
        env.step(action.numpy().astype(np.float32))
        return sim._particle_state[0].detach().cpu().float().clone()

    def sample_candidates(parts):
        s, e, _ = sim.generate_action_samples(a.n_cand, pile_aware=True, min_swath_particles=3)
        return torch.cat([s[0, :, :2], e[0, :, :2]], 1).float().cpu()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for m in a.models:
        members = ENSEMBLES.get(m, [m])
        ad = [make_occ_adapter(x, dev, "corner") for x in members]
        ad = ad if len(ad) > 1 else ad[0]
        for g in a.goals:
            D = goal_dist(g)
            for s0 in a.starts:
                for pl in a.planners:
                    if (m, pl, g, s0) in have:
                        continue
                    st = starts[s0]
                    env.restore_snapshot({"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)})
                    sim.update_material_state()
                    ep = dict(model=m, planner=pl, goal=g, start=s0, budget_s=a.budget, members=members, complete=False)
                    done["episodes"].append(ep); t0 = time.time()

                    def ckpt(rec):
                        ep.update(rec); ep["wall_s"] = time.time() - t0
                        tmp = Path(str(RES) + ".tmp"); tmp.write_text(json.dumps(done)); os.replace(tmp, RES)
                    rec = run_episode(ad, sim._particle_state[0].detach().cpu().float().clone(), D, execute,
                                      sample_candidates, pl, a.budget, a.steps, on_step=ckpt)
                    ep["complete"] = True; ckpt(rec)
                    v = rec["values"]
                    print(f"{m:36s} {pl:5s} {g:10s} start {s0}: improve {v[0] - v[-1]:+.3f}; evals/step "
                          f"{np.mean(rec['n_evals']):.0f}; {time.time() - t0:.0f}s", flush=True)
        del ad; torch.cuda.empty_cache()
    env.destroy()


if __name__ == "__main__":
    main()
