"""Ground-truth (state, action) -> next-state TRAJECTORY collection.

Different job from the `slates` collectors (`collect_true_action_results*.py`):
those replay N independent actions from the *same* fixed initial state (good
for ranking many actions against one state). This script instead samples
`transitions.n_states` independent initial states and, for each, applies
`transitions.n_steps` actions *in sequence* -- each action's outcome becomes
the state the next action is sampled from and applied to. That gives many more
genuinely distinct VISITED states for the same action budget, at the cost of
not being able to compare actions against a shared fixed state anymore.

Runs single-env (`FlexEnvMulti(config, n_envs=1)`), deliberately not batched:
a trajectory's own steps are inherently sequential (step k needs step k-1's
settled outcome), so there is nothing to batch *within* one trajectory. Across
trajectories, carrots' initial-state sampler (`rand_blob`/`rand_spread`) draws
a continuously-random particle count per state (confirmed empirically: 5
different counts from 2000 draws, before even counting each carrot's own
particle count also varying with its random scale), which conflicts with
FlexEnvMulti's batched-tiling requirement that every tile have the exact same
particle count (one shared buffer, sliced at a fixed stride -- see
`_env_slice` in env/flex_env_multi.py). Coffee's bean count is fixed by
config, so it wouldn't hit this, but for one simple, low-risk code path that
works for both materials, throughput instead comes from OS-process sharding:
run this script several times concurrently with different (--shard, --n-shards)
values, each an independent process/GPU context handling a disjoint slice of
states -- see --merge below to combine their output.

Still uses FlexEnvMulti (not plain FlexEnv) even at n_envs=1, specifically to
reuse its already-validated per-env state machinery (get_env_state /
set_all_envs_state / sample_action_obj_biased_local) and, for coffee, to get
yx_CoffeeGrid's per-state bean-position jitter (plain FlexEnv's obj='coffee'
branch uses the non-tiled, fully-deterministic yx_Coffee scene with no
randomness at all -- confirmed empirically, two different seeds produced
byte-identical positions). With n_envs=1 there is exactly one tile, so the
"every tile must be identical" requirement is trivially satisfied regardless
of material.

Config: `dataset` section as in the slates configs, plus:

    transitions:
      n_states: 2000
      n_steps: 10
      seed: 0
      init_pos_mix: ['rand_blob', 'rand_spread']   # optional, carrots only
      action_sampler: 'obj_biased'                 # or 'uniform'
      max_retries: 20       # resample attempts for one step before giving up
      bins:
        edges_abs: [...]   # optional, informational only -- see below

Each state's trajectory is all-or-nothing-per-run: on resume, a state with
all `n_steps` transitions already in its shard's manifest is skipped; any
other state (absent, or left partially done by a crash) is redone from
scratch. `--merge` combines all `manifest_shard*.jsonl` files in a folder into
one `manifest.jsonl`, keeping the LAST record for any duplicate (state_idx,
step_idx) -- which is what a resume-then-redo leaves behind.

A state's `env.reset()` (not its action steps -- those already retry via
`max_retries`) can rarely segfault the whole process natively: confirmed
cause is a degenerate convex mesh from `CreateRandomConvexMesh` in carrots'
scene generation (`PyFleX/bindings/helpers.h`), which occasional large
`num_carrots` draws hit by chance. That cannot be caught from Python, so
`reset_crash_attempts_shard*.json` records an attempt BEFORE every reset()
call; after `transitions.max_reset_attempts` (default 2) crashes on the same
state, it's logged `state_failed` and permanently skipped rather than
re-crashing forever. Run shards under `run_transitions_shard.sh`, which
auto-restarts a crashed process -- this ledger is what keeps that loop from
spinning on one bad state.

Usage:
    PYFLEX_HEADLESS_OVERRIDE=0 xvfb-run -a -s "-screen 0 1280x1024x24" \
        python -u collect_transitions.py config/data_gen/transitions_carrots.yaml \
        --shard 0 --n-shards 4
    python collect_transitions.py config/data_gen/transitions_carrots.yaml --merge
"""

import argparse
import collections
import glob
import json
import os
import time

import numpy as np

from env.flex_env_multi import FlexEnvMulti
from utils import load_yaml, set_seed

from collect_true_action_results import save_obs


def shard_manifest_path(out_dir, shard_idx):
    return os.path.join(out_dir, 'manifest_shard%d.jsonl' % shard_idx)


def crash_ledger_path(out_dir, shard_idx):
    return os.path.join(out_dir, 'reset_crash_attempts_shard%d.json' % shard_idx)


def load_crash_counts(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        try:
            return {int(k): v for k, v in json.load(f).items()}
        except (json.JSONDecodeError, ValueError):
            return {}


def record_reset_attempt(path, counts, state_idx):
    """Persist an incremented attempt count BEFORE the risky reset() call.

    env.reset() can segfault natively (confirmed: a rare degenerate convex
    mesh in CreateRandomConvexMesh, see PyFleX/bindings/helpers.h) -- a crash
    there kills the whole process with no Python-level exception to catch, so
    the only way to bound retries across process restarts is a count written
    to disk *before* the call that might never return.
    """
    counts[state_idx] = counts.get(state_idx, 0) + 1
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump({str(k): v for k, v in counts.items()}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    return counts[state_idx]


def load_shard_progress(manifest_path, n_steps):
    """Return {state_idx: is_complete} for every state_idx mentioned in this
    shard's manifest. Complete = all of step_idx 0..n_steps-1 present as valid
    'transition' records, or a terminal 'trajectory_failed' record exists --
    either way, re-running would be wasted work, so these are skipped."""
    valid_steps = collections.defaultdict(set)
    failed = set()
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('type') == 'transition' and rec.get('valid'):
                valid_steps[rec['state_idx']].add(rec['step_idx'])
            elif rec.get('type') in ('trajectory_failed', 'state_failed'):
                failed.add(rec['state_idx'])
    done = {}
    for state_idx, steps in valid_steps.items():
        done[state_idx] = len(steps) >= n_steps
    for state_idx in failed:
        done[state_idx] = True
    return done


def sample_step_action(env, sampler, rng_seed):
    np.random.seed(rng_seed)
    if sampler == 'obj_biased':
        return env.sample_action_obj_biased_local(1, env_idx=0)[0]
    action, _ = env.sample_action(1)
    return action[0, 0]


def resolve_count_group(count_groups, n_states, state_idx, seed):
    """Map a state_idx to (group_idx, target_num_carrots) for the
    `transitions.count_groups` config (a list of [lo, hi] object-count
    ranges, states split into len(count_groups) equal contiguous blocks).
    target_num_carrots is drawn uniformly from the group's range, seeded
    deterministically by (seed, state_idx) so a resumed/redone state draws
    the identical target."""
    n_groups = len(count_groups)
    states_per_group = n_states // n_groups
    group_idx = min(state_idx // states_per_group, n_groups - 1)
    lo, hi = count_groups[group_idx]
    rng = np.random.RandomState((seed * 1_000_003 + state_idx) % (2 ** 32))
    target = int(rng.randint(lo, hi + 1))
    return group_idx, target


def bin_of(push_length, edges_abs):
    if edges_abs is None:
        return None
    edges = np.asarray(edges_abs, dtype=float)
    # Last edge is inclusive (a push exactly at the longest configured length
    # should still land in the top bin, not fall off the end).
    for b in range(len(edges) - 1):
        if edges[b] <= push_length < edges[b + 1] or (
                b == len(edges) - 2 and push_length >= edges[b + 1]):
            return b
    return 0 if push_length < edges[0] else len(edges) - 2


def collect_shard(config, shard_idx, n_shards):
    ds = config['dataset']
    tr_cfg = config['transitions']
    n_states = tr_cfg['n_states']
    n_steps = tr_cfg['n_steps']
    seed = tr_cfg['seed']
    sampler = tr_cfg.get('action_sampler', 'obj_biased')
    max_retries = int(tr_cfg.get('max_retries', 20))
    init_pos_mix = tr_cfg.get('init_pos_mix')
    count_groups = tr_cfg.get('count_groups')
    edges_abs = (tr_cfg.get('bins') or {}).get('edges_abs')
    global_scale = ds['global_scale']

    out_dir = ds['folder']
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = shard_manifest_path(out_dir, shard_idx)
    done = load_shard_progress(manifest_path, n_steps)

    shard_states = list(range(shard_idx, n_states, n_shards))
    pending_states = [s for s in shard_states if not done.get(s)]
    if not pending_states:
        # Constructing FlexEnvMulti calls pyflex.init(); if no state ever
        # calls reset() (which is what allocates the scene/particle buffers
        # via pyflex.set_scene()), pyflex.clean() on a never-reset context
        # segfaults -- same root cause as the documented "must call reset()
        # before touching buffers" resume hazard. So when a shard's work is
        # already entirely done, skip touching pyflex at all.
        print('shard %d/%d: all %d assigned states already complete'
              % (shard_idx, n_shards, len(shard_states)))
        return

    manifest_f = open(manifest_path, 'a')

    def log_record(rec):
        rec['timestamp'] = time.time()
        manifest_f.write(json.dumps(rec) + '\n')
        manifest_f.flush()
        os.fsync(manifest_f.fileno())

    run_config_path = os.path.join(out_dir, 'run_config.json')
    if not os.path.exists(run_config_path):
        with open(run_config_path, 'w') as f:
            json.dump(config, f, indent=2)

    ledger_path = crash_ledger_path(out_dir, shard_idx)
    crash_counts = load_crash_counts(ledger_path)
    max_reset_attempts = int(tr_cfg.get('max_reset_attempts', 2))

    env = FlexEnvMulti(config, n_envs=1)

    try:
        for state_idx in pending_states:
            if crash_counts.get(state_idx, 0) >= max_reset_attempts:
                log_record({
                    'type': 'state_failed',
                    'state_idx': state_idx,
                    'reason': 'repeated_reset_crash',
                    'attempts': crash_counts[state_idx],
                })
                print('state %d: skipping, reset() already crashed %d time(s)'
                      % (state_idx, crash_counts[state_idx]))
                continue

            state_dir = os.path.join(out_dir, '%d' % state_idx)
            os.makedirs(state_dir, exist_ok=True)

            group_idx = None
            target_num_carrots = None
            if count_groups:
                group_idx, target_num_carrots = resolve_count_group(
                    count_groups, n_states, state_idx, seed)
                env.init_pos = 'count_target'
                env.target_num_carrots = target_num_carrots
            elif init_pos_mix:
                env.init_pos = init_pos_mix[state_idx % len(init_pos_mix)]
            state_seed = seed * 1_000_003 + state_idx
            set_seed(state_seed)
            # Written BEFORE the call: reset() can segfault natively (see
            # record_reset_attempt's docstring), in which case this process
            # dies here with no chance to log anything else -- the next
            # restart reads this count back and, past max_reset_attempts,
            # skips the state instead of crashing on it forever.
            record_reset_attempt(ledger_path, crash_counts, state_idx)
            env.reset(tile_seed=state_seed)

            init_state = env.get_env_state(0)
            init_pos = init_state['positions'].copy()
            np.save(os.path.join(state_dir, 'initial_particles.npy'),
                    init_pos.reshape(-1))
            np.savez(os.path.join(state_dir, 'initial_state.npz'), **init_state)
            obs0 = env.render_env(0, step_first=True)
            save_obs(state_dir, 'initial', obs0, global_scale)
            log_record({
                'type': 'state_init',
                'state_idx': state_idx,
                'n_particles': int(init_pos.shape[0]),
                'positions_path': os.path.relpath(
                    os.path.join(state_dir, 'initial_particles.npy'), out_dir),
                'color_path': os.path.relpath(
                    os.path.join(state_dir, 'initial_color.png'), out_dir),
                'init_pos': getattr(env, 'init_pos', None),
                'count_group': group_idx,
                'target_num_carrots': target_num_carrots,
            })
            print('state %d: initialized (%d particles)'
                  % (state_idx, init_pos.shape[0]))

            cur_state = init_state
            for step_idx in range(n_steps):
                success = False
                action = None
                obs_list = None
                for attempt in range(max_retries):
                    env.set_all_envs_state(cur_state)
                    action = sample_step_action(
                        env, sampler,
                        state_seed * 100 + step_idx * 1000 + attempt)
                    obs_list = env.step(action.reshape(1, 4))
                    if getattr(env, 'last_aborted', False):
                        continue
                    success = True
                    break

                if not success:
                    log_record({
                        'type': 'trajectory_failed',
                        'state_idx': state_idx,
                        'step_idx': step_idx,
                        'reason': 'max_retries_exceeded',
                        'retries': max_retries,
                    })
                    print('state %d step %d: gave up after %d retries, '
                          'ending trajectory early' % (state_idx, step_idx, max_retries))
                    break

                after_pos = env.get_env_positions(0).copy()
                tag = '%d_after' % step_idx
                after_pos_path = os.path.join(state_dir, '%s_particles.npy' % tag)
                np.save(after_pos_path, after_pos)
                save_obs(state_dir, tag, obs_list[0], global_scale)

                push_length = float(np.linalg.norm(action[2:] - action[:2]))
                log_record({
                    'type': 'transition',
                    'state_idx': state_idx,
                    'step_idx': step_idx,
                    'action': action.tolist(),
                    'valid': True,
                    'push_length': push_length,
                    'bin': bin_of(push_length, edges_abs),
                    'retries': attempt,
                    'after_positions_path': os.path.relpath(after_pos_path, out_dir),
                    'after_color_path': os.path.relpath(
                        os.path.join(state_dir, '%s_color.png' % tag), out_dir),
                })
                print('state %d step %d: ok (retries=%d, len=%.2f)'
                      % (state_idx, step_idx, attempt, push_length))

                cur_state = env.get_env_state(0)
    finally:
        manifest_f.close()
        env.close()


def merge_manifests(out_dir):
    """Combine every manifest_shard*.jsonl into one manifest.jsonl, keeping
    the LAST record for a duplicate (state_idx, step_idx) or (state_idx,
    'state_init') -- the only way a duplicate arises is a crash-then-redo,
    and the later record is the one that actually produced the kept files."""
    shard_paths = sorted(glob.glob(os.path.join(out_dir, 'manifest_shard*.jsonl')))
    if not shard_paths:
        print('no manifest_shard*.jsonl files found in %s' % out_dir)
        return

    kept = {}
    order = []
    for path in shard_paths:
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec['type'] == 'state_init':
                    key = ('state_init', rec['state_idx'])
                elif rec['type'] == 'transition':
                    key = ('transition', rec['state_idx'], rec['step_idx'])
                else:
                    key = ('other', rec['state_idx'], rec.get('step_idx'), rec.get('timestamp'))
                if key not in kept:
                    order.append(key)
                kept[key] = rec

    out_path = os.path.join(out_dir, 'manifest.jsonl')
    with open(out_path, 'w') as f:
        for key in order:
            f.write(json.dumps(kept[key]) + '\n')
    print('merged %d shard manifest(s), %d records -> %s'
          % (len(shard_paths), len(order), out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='YAML config with `dataset` + `transitions` sections')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--merge', action='store_true',
                         help='Merge manifest_shard*.jsonl into manifest.jsonl and exit')
    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.merge:
        merge_manifests(config['dataset']['folder'])
    else:
        collect_shard(config, args.shard, args.n_shards)
