"""
EXP-0024 pilot: does repeating the SAME action from the SAME state produce
materially different dv (physical stochasticity), and if so, how does that
within-action variance compare to the between-action variance that slateN
actually ranks on?

Design: build one GenesisOracleEnv with n_envs = N_ACTIONS * N_REPEATS.
For each of N_STATES settled states: snapshot it, build an act_seqs tensor
of shape (n_envs, 1, 4) holding N_ACTIONS distinct actions, each replicated
N_REPEATS times, and roll all of them out in ONE rollout_candidates() call
(all envs share the frozen snapshot, so this is exactly "same state, same
action, repeated" x N_REPEATS interleaved with "same state, different
action" x N_ACTIONS). Compute dv = lyapunov(occ_after, d) - lyapunov(occ0, d)
for every one of the n_envs outcomes, goal = corner.

Reports per state: within-action variance (mean of per-action-group variance
of dv) vs between-action variance (variance of the N_ACTIONS group means),
and their ratio. Also checks whether repeats are bit-identical (deterministic
sim) or genuinely differ.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import load_oracle_config
from model.eulerian_wrapper import EulerianModelWrapper
from control_utility_test import lyapunov, lyapunov_weights

N_ACTIONS = 3
N_REPEATS = 5
N_STATES = 3
GRID_RES = (64, 64)

FULL_FIDELITY = "--full-fidelity" in sys.argv
if FULL_FIDELITY:
    N_STATES = 1  # full fidelity is ~3x slower per rollout; one state is enough
                  # to check whether determinism survives at real settle depth

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = load_oracle_config()
    cfg['dataset']['headless'] = True
    # keep it small & fast -- this is a pilot, not a production collection
    cfg['dataset']['num_objects'] = 20
    cfg['dataset']['settle_steps'] = 60
    cfg['dataset']['reset_warmup_steps'] = 10
    cfg.setdefault('mpc', {})['rollout_settle_steps'] = 20

    n_envs = N_ACTIONS * N_REPEATS
    t0 = time.time()
    env = GenesisOracleEnv(cfg, n_envs=n_envs)
    print(f"[build] {time.time()-t0:.1f}s", flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_bounds = EulerianModelWrapper.default_bounds(cfg, convention='genesis')
    d_field = lyapunov_weights(GRID_RES, "corner", device)
    footprint_r = env.default_footprint_radius_voxels(grid_bounds, GRID_RES)

    # Three distinct candidate actions, chosen to sweep across the pile from
    # different directions -- action_to_pose takes [sx,sy,ex,ey] (world m,
    # centred at 0), yaw derived from travel direction.
    w = float(cfg['dataset']['wkspc_w'])
    actions_world = torch.tensor([
        [-0.6 * w, 0.0,      0.6 * w, 0.0],       # sweep +x, through the middle
        [0.0,     -0.6 * w,  0.0,     0.6 * w],   # sweep +y, through the middle
        [-0.5 * w, -0.5 * w, 0.3 * w, 0.3 * w],   # diagonal sweep
    ], dtype=torch.float32)

    results = []
    for state_idx in range(N_STATES):
        t_state = time.time()
        env.reset()
        snapshot = env.snapshot_particles()
        pos0 = snapshot['pos']  # (1, P, 3)
        occ0 = env.particles_to_occ(pos0, grid_bounds, GRID_RES, footprint_r)
        v0 = lyapunov(occ0, d_field)  # (1,)
        v0_scalar = float(v0[0])

        # (n_envs, 1, 4): repeat each action N_REPEATS times, contiguous blocks
        act_seqs = actions_world.repeat_interleave(N_REPEATS, dim=0).unsqueeze(1)
        assert act_seqs.shape == (n_envs, 1, 4)

        t_roll = time.time()
        terminal_pos = env.rollout_candidates(
            act_seqs, snapshot, use_rollout_fidelity=not FULL_FIDELITY, record=False)
        roll_s = time.time() - t_roll

        occ1 = env.particles_to_occ(terminal_pos, grid_bounds, GRID_RES, footprint_r)
        v1 = lyapunov(occ1, d_field)  # (n_envs,)
        dv = (v1 - v0_scalar).detach().cpu().numpy()  # (n_envs,)

        dv_by_action = dv.reshape(N_ACTIONS, N_REPEATS)

        # also grab raw terminal COM per env, as a model-free cross-check
        com = terminal_pos.mean(dim=1).detach().cpu().numpy()  # (n_envs, 3)
        com_by_action = com.reshape(N_ACTIONS, N_REPEATS, 3)

        within_var_per_action = dv_by_action.var(axis=1, ddof=1)  # (N_ACTIONS,)
        within_var_mean = float(within_var_per_action.mean())
        action_means = dv_by_action.mean(axis=1)  # (N_ACTIONS,)
        between_var = float(action_means.var(ddof=1))
        ratio = within_var_mean / between_var if between_var > 1e-12 else float('inf')

        max_repeat_com_spread_mm = float(
            np.linalg.norm(com_by_action - com_by_action.mean(axis=1, keepdims=True),
                            axis=-1).max() * 1000.0
        )

        rec = dict(
            state_idx=state_idx,
            v0=v0_scalar,
            dv_by_action=dv_by_action.tolist(),
            action_means=action_means.tolist(),
            within_var_per_action=within_var_per_action.tolist(),
            within_var_mean=within_var_mean,
            between_var=between_var,
            ratio_within_over_between=ratio,
            max_repeat_com_spread_mm=max_repeat_com_spread_mm,
            rollout_s=roll_s,
            state_total_s=time.time() - t_state,
        )
        results.append(rec)
        print(json.dumps(rec, indent=2), flush=True)

    env.destroy()

    suffix = "_fullfidelity" if FULL_FIDELITY else ""
    out_path = os.path.join(OUT_DIR, f"results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(dict(N_ACTIONS=N_ACTIONS, N_REPEATS=N_REPEATS, N_STATES=N_STATES,
                        results=results, total_s=time.time() - t0), f, indent=2)
    print(f"[done] total {time.time()-t0:.1f}s, wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
