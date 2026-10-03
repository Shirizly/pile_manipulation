"""Ground-truth action-outcome collection, batched across N simulated envs.

Same job and same on-disk contract as collect_true_action_results.py: for each
of N initial states, the identical particle state is replayed n_actions times,
one independently-sampled push per replay.  The difference is that the replays
run concurrently -- FlexEnvMulti lays out `n_envs` identical copies of the task
env on a grid inside one Flex solver, so a batch of `n_envs` actions costs one
push instead of `n_envs` pushes.

Output layout, manifest schema, file names and the action sequence itself are
byte-for-byte the same as the serial collector (actions are drawn with the same
per-(state, action) seed), so the two datasets are interchangeable and a run
started with one can be finished with the other.

Config: the serial `dataset` + `slates` sections, plus an optional `parallel`
section:

    parallel:
      n_envs: 9            # rounded up to a square-ish grid
      tile_pitch_scale: 2  # tile spacing in units of global_scale
      tile_seed: 12345     # scene RNG seed; identical for every tile

Usage:
    PYFLEX_HEADLESS_OVERRIDE=0 xvfb-run -a -s "-screen 0 1280x1024x24" \
        python collect_true_action_results_parallel.py config/data_gen/slates_parallel.yaml
"""

import argparse
import collections
import json
import os
import time

import cv2
import numpy as np

from env.flex_env_multi import FlexEnvMulti
from utils import load_yaml, set_seed

from collect_true_action_results import (
    load_manifest_done,
    save_obs,
    state_initialized,
)
from collect_transitions import resolve_count_group


def action_for(seed, state_idx, action_idx, env):
    """Sample action (state_idx, action_idx), unbinned.

    Seeded exactly as in the serial collector so both produce the same action
    for the same indices -- that is what makes the two datasets comparable and
    lets a run switch between them.
    """
    np.random.seed(seed * 1_000_003 + state_idx * 10_000 + action_idx)
    action, _ = env.sample_action(1)
    return action[0, 0]


def plan_binned_actions(env, seed, state_idx, n_actions, edges_frac=None,
                        edges_abs=None, sampler='uniform', max_rounds=400):
    """Actions for one state, grouped into push-length bins, ordered by bin.

    Every env in a batch runs for the *longest* push in that batch (shorter
    ones hold their final pose), so mixing a 1-unit nudge with a 14-unit sweep
    wastes most of the batch. Grouping actions of similar length means a batch
    costs about what its own pushes cost.

    Bins are fractions of X, the longest push the workspace allows: with
    endpoints drawn uniformly from [lo, hi]^2, that is the diagonal,
    sqrt(2) * (hi - lo). Actions shorter than the first edge are not sampled at
    all, so `edges_frac[0] > 0` deliberately excludes near-zero pushes.

    Sampling is rejection-based against the same uniform action distribution
    env.sample_action() uses, so within a bin the actions are distributed
    exactly as unbinned sampling would give -- the bins only change how many
    are drawn from each length range (equal counts), not their shape. Long
    pushes are rare (the top bin holds ~0.14% of uniform samples), so this
    deliberately oversamples them relative to uniform sampling.

    Deterministic in (seed, state_idx): a resumed run regenerates the same plan.
    """
    lo = -env.wkspc_w + env.action_margin
    hi = env.wkspc_w - env.action_margin
    # Absolute edges are preferred: bin boundaries must be the SAME push
    # lengths across every dataset, because a separate model is trained per
    # bin. Fraction-of-X edges silently move whenever action_margin changes,
    # since X = sqrt(2) * (hi - lo).
    if edges_abs is not None:
        edges = np.asarray(edges_abs, dtype=float)
    else:
        edges = np.asarray(edges_frac, dtype=float) * np.sqrt(2.0) * (hi - lo)
    n_bins = len(edges) - 1

    # Spread any remainder over the low bins rather than dropping actions.
    per_bin = [n_actions // n_bins] * n_bins
    for i in range(n_actions - sum(per_bin)):
        per_bin[i] += 1

    rng = np.random.RandomState((seed * 1_000_003 + state_idx) % (2 ** 32))

    def propose(k):
        if sampler == 'obj_biased':
            return env.sample_action_obj_biased_local(k, env_idx=0, rng=rng)
        return lo + (hi - lo) * rng.rand(k, 4)

    actions, bin_ids, lengths, shortfall = [], [], [], {}
    for b in range(n_bins):
        want = per_bin[b]
        b_lo, b_hi = edges[b], edges[b + 1]
        got, rounds = [], 0
        while len(got) < want and rounds < max_rounds:
            cand = propose(4096)
            d = np.linalg.norm(cand[:, 2:] - cand[:, :2], axis=1)
            keep = cand[(d >= b_lo) & (d < b_hi)]
            got.extend(keep[:want - len(got)])
            rounds += 1
        if len(got) < want:
            # A bin can be genuinely unreachable: with obj_biased starts, a
            # compact pile in the middle of the workspace simply cannot produce
            # a push long enough for the top bins. Take what exists and let the
            # caller log the shortfall rather than spinning forever.
            shortfall[b] = want - len(got)
        if not got:
            continue
        got = np.array(got)
        actions.append(got)
        bin_ids.extend([b] * len(got))
        lengths.extend(np.linalg.norm(got[:, 2:] - got[:, :2], axis=1))

    if not actions:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0), shortfall
    return (np.concatenate(actions, axis=0), np.array(bin_ids),
            np.array(lengths), shortfall)


def gen_slates_parallel(config):
    ds = config['dataset']
    slates_cfg = config['slates']
    n_states = slates_cfg['n_states']
    n_actions = slates_cfg['n_actions']
    seed = slates_cfg['seed']
    global_scale = ds['global_scale']
    bins_cfg = slates_cfg.get('bins') or {}
    edges_frac = bins_cfg.get('edges_frac')
    edges_abs = bins_cfg.get('edges_abs')
    # 'uniform' (both endpoints uniform) or 'obj_biased' (start drawn near the
    # pile). Bin edges are unchanged either way, so datasets collected with
    # different samplers stay comparable bin for bin.
    sampler = slates_cfg.get('action_sampler', 'uniform')
    # Cycle the pile generator across states so the dataset covers both compact
    # localized blobs and broad spreads. `init_pos` is read per reset(), so
    # setting it on the env before reset() is all that is needed.
    init_pos_mix = slates_cfg.get('init_pos_mix')
    # Alternative to init_pos_mix: target an exact carrot-count range per
    # state, split into len(count_groups) equal contiguous blocks of states
    # (see collect_transitions.py's resolve_count_group, reused here so both
    # collectors assign identical groups/targets for the same state_idx).
    count_groups = slates_cfg.get('count_groups')

    out_dir = ds['folder']
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, 'manifest.jsonl')

    done_actions = load_manifest_done(manifest_path)
    manifest_f = open(manifest_path, 'a')

    def log_record(rec):
        rec['timestamp'] = time.time()
        manifest_f.write(json.dumps(rec) + '\n')
        manifest_f.flush()
        os.fsync(manifest_f.fileno())

    env = FlexEnvMulti(config)
    n_envs = env.n_envs

    run_config_path = os.path.join(out_dir, 'run_config.json')
    if not os.path.exists(run_config_path):
        with open(run_config_path, 'w') as f:
            json.dump({**config,
                       'parallel_resolved': {'n_envs': n_envs,
                                             'grid': [env.grid_nx, env.grid_nz],
                                             'tile_pitch': env.tile_pitch,
                                             'tile_seed': env.tile_seed}},
                      f, indent=2)

    try:
        for state_idx in range(n_states):
            state_dir = os.path.join(out_dir, '%d' % state_idx)
            os.makedirs(state_dir, exist_ok=True)
            init_pos_path = os.path.join(state_dir, 'initial_particles.npy')
            init_state_path = os.path.join(state_dir, 'initial_state.npz')

            pending = [a for a in range(n_actions)
                       if (state_idx, a) not in done_actions]
            already_init = state_initialized(manifest_path, state_idx)

            if already_init and not pending:
                print('state %d already complete, skipping' % state_idx)
                continue

            # reset() must run even when resuming: it is what calls
            # pyflex.set_scene(), which allocates the particle/rigid buffers.
            # Without it, set/get on those buffers segfaults with no traceback
            # in a fresh process. (Same trap as the serial collector's resume.)
            # The tile seed is what distinguishes one initial state from the
            # next: all tiles share it (that is what makes them identical
            # copies), so it must vary per state or every state would be the
            # same pile.
            group_idx = None
            target_num_carrots = None
            if count_groups:
                group_idx, target_num_carrots = resolve_count_group(
                    count_groups, n_states, state_idx, seed)
                env.init_pos = 'count_target'
                env.target_num_carrots = target_num_carrots
            elif init_pos_mix:
                env.init_pos = init_pos_mix[state_idx % len(init_pos_mix)]
            set_seed(seed * 1_000_003 + state_idx)
            env.reset(tile_seed=seed * 1_000_003 + state_idx)

            if already_init and os.path.exists(init_state_path):
                saved = np.load(init_state_path)
                init_state = {k: saved[k] for k in
                              ('positions', 'velocities',
                               'rigid_rotations', 'rigid_translations')}
                print('state %d: resuming (%d/%d actions left)'
                      % (state_idx, len(pending), n_actions))
            else:
                # reset() already stamped tile 0's settled state onto every
                # tile, so tile 0 is the canonical initial state.
                init_state = env.get_env_state(0)
                init_state['velocities'][:] = 0.0
                np.savez(init_state_path, **init_state)
                np.save(init_pos_path, init_state['positions'].reshape(-1))
                obs0 = env.render_env(0, step_first=True)
                save_obs(state_dir, 'initial', obs0, global_scale)
                log_record({
                    'type': 'state_init',
                    'state_idx': state_idx,
                    'n_particles': int(init_state['positions'].shape[0]),
                    'positions_path': os.path.relpath(init_pos_path, out_dir),
                    'color_path': os.path.relpath(
                        os.path.join(state_dir, 'initial_color.png'), out_dir),
                    'init_pos': env.init_pos,
                    'count_group': group_idx,
                    'target_num_carrots': target_num_carrots,
                })
                print('state %d: initialized (%s, %d particles/env, %d envs)'
                      % (state_idx, env.init_pos,
                         init_state['positions'].shape[0], n_envs))

            # Build this state's action plan once. With bins, it is ordered by
            # push length, and batches are cut so they never straddle a bin --
            # that is the whole point, since a batch costs its longest push.
            if edges_frac is not None or edges_abs is not None:
                plan_actions, plan_bins, plan_lengths, shortfall = plan_binned_actions(
                    env, seed, state_idx, n_actions, edges_frac=edges_frac,
                    edges_abs=edges_abs, sampler=sampler)
                if shortfall:
                    print('state %d: could not fill bins %s (unreachable push '
                          'lengths for this pile under %s sampling)'
                          % (state_idx, dict(shortfall), sampler))
                    log_record({'type': 'bin_shortfall', 'state_idx': state_idx,
                                'shortfall': {str(k): int(v) for k, v in shortfall.items()},
                                'sampler': sampler})
            else:
                plan_actions = np.stack([action_for(seed, state_idx, a, env)
                                         for a in range(n_actions)])
                plan_bins = np.full(n_actions, -1)
                plan_lengths = np.linalg.norm(
                    plan_actions[:, 2:] - plan_actions[:, :2], axis=1)

            # A bin shortfall makes the plan shorter than n_actions; never
            # index past what was actually planned.
            pending = [a for a in pending if a < len(plan_actions)]

            # Work bin by bin through a queue: an aborted batch puts its
            # survivors back on the queue instead of recording them.
            for b in sorted(set(plan_bins.tolist())):
                queue = [a for a in pending if plan_bins[a] == b]
                attempts = collections.Counter()
                max_attempts = 4
                while queue:
                    batch = queue[:n_envs]
                    queue = queue[n_envs:]
                    actions = plan_actions[batch]
                    if len(batch) < n_envs:
                        # Short final batch: the grid simulates every tile
                        # whether or not we use it, so pad with repeats and
                        # discard their results rather than leaving tiles idle.
                        pad = np.tile(actions[-1], (n_envs - len(batch), 1))
                        actions = np.concatenate([actions, pad], axis=0)

                    env.set_all_envs_state(init_state)
                    t0 = time.time()
                    obs_list = env.step(actions)
                    dt = time.time() - t0

                    if getattr(env, 'last_aborted', False):
                        # A blow-up stopped the batch early. Only the envs that
                        # actually blew up get recorded (as invalid, permanently
                        # -- the action is deterministic and would blow up
                        # again). The rest were mid-push and never settled, so
                        # their state is not the outcome of their action; they
                        # go back on the queue to run in a clean batch. Each
                        # abort removes at least one action, so this terminates.
                        # Slots at or beyond len(batch) are padding: duplicates
                        # of a real action run only to keep the grid busy. They
                        # must never decide a real action's fate, and a batch
                        # where ONLY padding exploded still leaves the real
                        # actions inconclusive -- they get retried, not dropped.
                        blew = [slot for slot in env.last_exploded_envs
                                if slot < len(batch)]
                        retry = [batch[slot] for slot in range(len(batch))
                                 if slot not in blew]
                        for slot in blew:
                            action_idx = batch[slot]
                            log_record({
                                'type': 'action',
                                'state_idx': state_idx,
                                'action_idx': action_idx,
                                'action': actions[slot].tolist(),
                                'valid': False,
                                'env_slot': slot,
                                'bin': int(plan_bins[action_idx]),
                                'push_length': float(plan_lengths[action_idx]),
                                'failure': 'solver_explosion',
                            })
                            done_actions.add((state_idx, action_idx))
                        # Bound the retries instead of dropping anything: an
                        # action that keeps landing in aborted batches is
                        # recorded as invalid with its own reason, so every
                        # planned action ends up in the manifest exactly once.
                        requeue = []
                        for action_idx in retry:
                            attempts[action_idx] += 1
                            if attempts[action_idx] >= max_attempts:
                                log_record({
                                    'type': 'action',
                                    'state_idx': state_idx,
                                    'action_idx': action_idx,
                                    'action': plan_actions[action_idx].tolist(),
                                    'valid': False,
                                    'bin': int(plan_bins[action_idx]),
                                    'push_length': float(plan_lengths[action_idx]),
                                    'failure': 'repeated_batch_abort',
                                })
                                done_actions.add((state_idx, action_idx))
                            else:
                                requeue.append(action_idx)
                        queue = requeue + queue
                        print('state %d bin %d batch %s: exploded in slots %s, '
                              '%d requeued (%.1fs)'
                              % (state_idx, b, batch, blew, len(requeue), dt))
                        continue

                    n_valid = 0
                    for slot, action_idx in enumerate(batch):
                        obs_after = obs_list[slot]
                        valid = obs_after is not None
                        n_valid += int(valid)
                        record = {
                            'type': 'action',
                            'state_idx': state_idx,
                            'action_idx': action_idx,
                            'action': actions[slot].tolist(),
                            'valid': valid,
                            'env_slot': slot,
                            'bin': int(plan_bins[action_idx]),
                            'push_length': float(plan_lengths[action_idx]),
                        }
                        if valid:
                            after_pos = env.get_env_positions(slot).copy()
                            tag = '%d_after' % action_idx
                            after_pos_path = os.path.join(
                                state_dir, '%s_particles.npy' % tag)
                            np.save(after_pos_path, after_pos)
                            save_obs(state_dir, tag, obs_after, global_scale)
                            record['after_positions_path'] = os.path.relpath(
                                after_pos_path, out_dir)
                            record['after_color_path'] = os.path.relpath(
                                os.path.join(state_dir, '%s_color.png' % tag),
                                out_dir)
                        log_record(record)
                        done_actions.add((state_idx, action_idx))

                    print('state %d bin %d actions %s: %d/%d ok in %.1fs '
                          '(%.2fs per action, len %.1f-%.1f)'
                          % (state_idx, b, batch, n_valid, len(batch), dt,
                             dt / len(batch), plan_lengths[batch].min(),
                             plan_lengths[batch].max()))

                    if n_valid == 0:
                        # Every tile invalid without an abort: rebuild the scene
                        # before continuing. Deterministic, so the rebuilt tiles
                        # match the saved initial state.
                        print('state %d: all envs invalid, rebuilding scene' % state_idx)
                        set_seed(seed * 1_000_003 + state_idx)
                        env.reset(tile_seed=seed * 1_000_003 + state_idx)
    finally:
        manifest_f.close()
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default='config/data_gen/slates_parallel.yaml',
                        help='YAML config with `dataset` + `slates` (+ optional '
                             '`parallel`) sections')
    args = parser.parse_args()
    gen_slates_parallel(load_yaml(args.config))
