"""Particle-transition dataset for the object-count-grouped GNN study.

Reads `true_action_transitions_carrots_grouped` (see
`config/data_gen/transitions_carrots_grouped.yaml`): a sequential-rollout
dataset (collect_transitions.py) whose 2000 states are split into 4 object-
count groups (10-30 / 50-70 / 100-150 / 400-500 carrots). Produces the EXACT
`__getitem__` contract `dataset.dataset_gnn_dyn.ParticleDataset` does --
`(states, states_delta, attrs, particle_num, particle_den, color_imgs)`, same
shapes/semantics -- so `train/train_gnn_dyn_grouped.py` (a copy of
`train/train_gnn_dyn.py` with only this import swapped) runs the IDENTICAL
training loop unmodified, per the project decision to use the baseline
training process rather than the unified `training/trainer.py` for this
study (results should not be confounded by a different training procedure).

Differences from the original ParticleDataset, each necessary because our
data was never collected through a simulated depth camera (unlike the
original pipeline's `*_depth.png` + `depth2fgpcd` + FPS-from-depth sampling
step), not an attempt to change the training contract:

  - No depth-image round trip. We have ground-truth particle positions
    directly, so "which particles to track" is chosen by farthest-point
    sampling (FPS) directly on the true particle cloud of the window's first
    frame, by INDEX (not by nearest-neighbour matching a depth-derived point
    cloud to the nearest real particle, which is what the original did to
    bridge "sampled from an image" to "a real tracked particle"). The same
    indices are then reused across the window's remaining frames, exactly as
    the original keeps one fixed index set once chosen.
  - Node budget: `min(30, state's target object count)`, not the original's
    particle_den-driven FPS radius (itself driven by a RANDOM per-sample
    density in [15, 6500] -- a domain-randomisation knob for real, noisy,
    depth-sensed data that has no equivalent here, since our ground truth is
    exact). 30 is this study's fixed node budget (see conversation: the
    O(N^2) adjacency in `PropNetDiffDenModel.predict_one_step` cannot scale
    to the ~400-500-object group's actual particle count -- tens of
    thousands -- so every state is compressed to at most 30 representative
    points; FPS spreads them across the whole pile so the compression, not
    the graph SIZE, is what should carry the object-count signal).
  - `particle_den` is a FIXED constant (`PARTICLE_DEN_CONST`), not a random
    draw -- we have no analogous sensing-noise quantity to randomise over.
  - No camera-frame round trip (`opengl2cam`/`opencv_T_world`): positions are
    used directly in PyFleX world coordinates, reordered to (x, z, y) so the
    push-plane projection math below (identical to
    `transforms.functional.build_action_delta`, reused verbatim) operates on
    the correct horizontal plane -- our action's two push coordinates ARE
    (x, z) in world units (not a camera-relative frame the original frame's
    points had to be converted into).

Train/valid split is PER-STATE (never per-window, which would leak a
state's own future frames across the split) and stratified: a fixed
`train_valid_ratio` fraction of states is held out for 'valid' INDEPENDENTLY
within each object-count group's contiguous block, so both splits have every
group represented in the same proportion -- a global split could otherwise
starve a small group's validation set by chance.
"""

from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from transforms.functional import build_action_delta

PARTICLE_DEN_CONST = 1000.0  # see module docstring: fixed, not sensing-noise-randomised
GLOBAL_SCALE = 24.0  # must match the collecting config's dataset.global_scale
# Positions below are divided by GLOBAL_SCALE (see _frame_path usage in
# __getitem__) to match dataset_gnn_dyn.ParticleDataset's own final
# normalisation step -- model/config defaults (PropModuleDiffDen's
# adj_thresh=0.08, this module's PLATE_HALF_WIDTH) are tuned to that
# normalised range, not raw PyFleX world units; skipping this would silently
# make adj_thresh ~24x too small relative to actual particle spacing.
PLATE_HALF_WIDTH = 1.2 / GLOBAL_SCALE  # 0.05 * global_scale, normalised; see SKILL.md

# EXP-0064 port (2026-10-03): the collector stores actions as (x, -z) of the
# particle files (invariant `flex-action-frame-neg-z`; re-measured on DS-0021 /
# DS-0022 by code/frame_check.py: cos(moved displacement, push dir) 0.99 under
# z = -a1 vs ~0 under the literal reading). The study AS RUN used the literal
# reading (sign +1), i.e. a z-mirrored push. `train.action_z_sign` selects it;
# the default +1 reproduces the as-run model (MODEL-0011), -1 is the corrected
# frame (MODEL-0012).
def action_z_sign(config: dict) -> float:
    return float(config.get('train', {}).get('action_z_sign', 1.0))


# EXP-0064 issues.md I-3: `train.action_encoding` = 'tube' (default, as run: build_action_delta's Gaussian
# tube x full push vector) or 'orig' (the source baseline ParticleDataset's encoding, dataset_gnn_dyn.py
# l.150-215: distance-to-push-end along the push x push direction x hard length gate x soft width gate,
# pusher half-width 0.8/24, decay 0.01 -- same normalised units as here, since both divide by 24).
ORIG_PUSHER_W = 0.8 / GLOBAL_SCALE


def orig_action_delta(s_cur: torch.Tensor, p_start: torch.Tensor, p_stop: torch.Tensor) -> torch.Tensor:
    """(n, 3) nodes (push plane in cols 0, 1), (3,) start/stop -> (n, 3) per-node action displacement."""
    d = (p_stop - p_start)[:2]
    L = d.norm().clamp_min(1e-9)
    d = d / L
    o = torch.stack([-d[1], d[0]])
    diff = s_cur[:, :2] - p_start[None, :2]
    proj, proj_o = diff @ d, diff @ o
    lmask = ((proj < L) & (proj > 0)).float()
    wmask = torch.exp(-torch.maximum((-ORIG_PUSHER_W - proj_o).clamp_min(0), (proj_o - ORIG_PUSHER_W).clamp_min(0)) / 0.01)
    to_end = (p_stop[None, :2] - s_cur[:, :2]) @ d
    out = torch.zeros_like(s_cur)
    out[:, :2] = (to_end * lmask * wmask)[:, None] * d[None]
    return out


def encode_action(kind: str, s_cur, p_start, p_stop):
    if kind == 'orig':
        return orig_action_delta(s_cur, p_start, p_stop)
    return build_action_delta(s_cur, p_start, p_stop, sigma_m=PLATE_HALF_WIDTH)


def _fps_indices(points: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sampling, returning the chosen INDICES into `points`
    (not copies of the points themselves), so the same indices can be reused
    to track the same physical particles across a window's later frames."""
    n = points.shape[0]
    k = min(k, n)
    chosen = [int(np.random.randint(n))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(points - points[nxt], axis=1))
    return np.array(chosen, dtype=np.int64)


def _load_merged_manifest(root: str) -> list[dict]:
    merged_path = os.path.join(root, 'manifest.jsonl')
    if not os.path.exists(merged_path):
        raise FileNotFoundError(
            f'{merged_path} not found -- run collect_transitions.py --merge first.')
    with open(merged_path, 'r') as f:
        return [json.loads(l) for l in f if l.strip()]


def _frame_path(state_dir: str, frame_idx: int) -> str:
    """frame_idx 0 is the pre-push state; frame_idx k>0 is after step (k-1)."""
    if frame_idx == 0:
        return os.path.join(state_dir, 'initial_particles.npy')
    return os.path.join(state_dir, '%d_after_particles.npy' % (frame_idx - 1))


def _color_path(state_dir: str, frame_idx: int) -> str:
    if frame_idx == 0:
        return os.path.join(state_dir, 'initial_color.png')
    return os.path.join(state_dir, '%d_after_color.png' % (frame_idx - 1))


class GroupedParticleDataset(Dataset):
    """
    Parameters
    ----------
    data_root : str -- e.g. 'data/true_action_transitions_carrots_grouped'
    config    : dict -- the gnn_dyn-style train config; reads
                config['train']['n_history'], ['n_rollout'],
                ['train_valid_ratio'], and
                config['train']['particle']['node_budget'] (default 30).
    phase     : 'train' | 'valid'
    """

    def __init__(self, data_root: str, config: dict, phase: str):
        assert phase in ('train', 'valid')
        self.data_root = data_root
        self.n_his = int(config['train']['n_history'])
        self.n_roll = int(config['train']['n_rollout'])
        self.window_len = self.n_his + self.n_roll
        self.node_budget = int(config['train']['particle'].get('node_budget', 30))
        self.z_sign = action_z_sign(config)
        self.encoding = str(config['train'].get('action_encoding', 'tube'))
        self.drop_escaped = bool(config['train'].get('drop_escaped', False))  # issues.md I-2

        records = _load_merged_manifest(data_root)
        state_init = {r['state_idx']: r for r in records if r['type'] == 'state_init'}
        valid_steps: dict[int, set] = {}
        for r in records:
            if r['type'] == 'transition' and r['valid']:
                valid_steps.setdefault(r['state_idx'], set()).add(r['step_idx'])
        actions: dict[tuple[int, int], list] = {}
        for r in records:
            if r['type'] == 'transition' and r['valid']:
                actions[(r['state_idx'], r['step_idx'])] = r['action']

        # Only states with every step complete -- a short-circuited trajectory
        # (trajectory_failed / state_failed) has no reliable full window.
        n_steps_total = 10
        complete_states = sorted(
            s for s, steps in valid_steps.items() if len(steps) >= n_steps_total)

        # Stratified per-group split: group states by their recorded
        # count_group, split each group's own state list independently.
        by_group: dict[int, list[int]] = {}
        for s in complete_states:
            g = state_init[s].get('count_group')
            by_group.setdefault(g, []).append(s)

        ratio = float(config['train'].get('train_valid_ratio', 0.9))
        self.states: list[int] = []
        for g, states in by_group.items():
            states = sorted(states)
            n_train = int(round(len(states) * ratio))
            self.states.extend(states[:n_train] if phase == 'train' else states[n_train:])
        self.states.sort()
        self.n_dropped_escaped = 0
        if self.drop_escaped:  # drop states whose window frames contain any |x| or |z| > 10 (DS-0019 escape_abs)
            keep = []
            for st in self.states:
                d = os.path.join(data_root, str(st))
                esc = any(np.abs(np.load(_frame_path(d, f)).reshape(-1, 4)[:, [0, 2]]).max() > 10.0
                          for f in range(self.window_len))
                if not esc:
                    keep.append(st)
            self.n_dropped_escaped = len(self.states) - len(keep)
            self.states = keep

        self.state_init = state_init
        self.actions = actions

        # One window per state for now: starting at frame 0, length
        # n_his+n_roll (<=11 frames available: initial + 10 after-steps).
        if self.window_len > n_steps_total + 1:
            raise ValueError(
                'n_history+n_rollout=%d exceeds the %d frames a complete '
                'trajectory has.' % (self.window_len, n_steps_total + 1))
        self.windows = [(s, 0) for s in self.states]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        state_idx, start = self.windows[idx]
        state_dir = os.path.join(self.data_root, str(state_idx))
        target_count = self.state_init[state_idx].get('target_num_carrots')
        node_budget = min(self.node_budget, int(target_count)) if target_count else self.node_budget

        # Choose tracked particle indices via FPS on the window's first frame.
        first = np.load(_frame_path(state_dir, start)).reshape(-1, 4)[:, :3] / GLOBAL_SCALE
        idxs = _fps_indices(first, node_budget)
        particle_num = idxs.shape[0]

        states = np.zeros((self.window_len, particle_num, 3), dtype=np.float32)
        states_delta = np.zeros((self.window_len - 1, particle_num, 3), dtype=np.float32)
        color_imgs = np.zeros((self.window_len, 720, 720, 3), dtype=np.uint8)

        for i in range(start, start + self.window_len):
            pos = np.load(_frame_path(state_dir, i)).reshape(-1, 4)[idxs, :3] / GLOBAL_SCALE
            # (x, y, z) -> (x, z, y): puts the push plane in columns [0, 1],
            # matching build_action_delta's [:2] convention. See module
            # docstring -- no camera transform, this is a pure axis swap.
            pos_remap = pos[:, [0, 2, 1]]
            states[i - start] = pos_remap

            if i < start + self.window_len - 1:
                action = self.actions[(state_idx, i)]  # [x0, z0, x1, z1], world units
                a = np.array(action, dtype=np.float32) / GLOBAL_SCALE
                a[[1, 3]] *= self.z_sign  # see action_z_sign (EXP-0064 port)
                p_start = np.array([a[0], a[1], 0.0], dtype=np.float32)
                p_stop = np.array([a[2], a[3], 0.0], dtype=np.float32)
                s_cur_t = torch.from_numpy(pos_remap.astype(np.float32))
                delta = encode_action(self.encoding, s_cur_t, torch.from_numpy(p_start), torch.from_numpy(p_stop))
                states_delta[i - start] = delta.numpy()

            img = cv2.imread(_color_path(state_dir, i))
            if img is not None:
                color_imgs[i - start] = img[:, :, ::-1]

        attrs = np.ones((self.window_len, particle_num), dtype=np.float32)

        return (
            torch.from_numpy(states),
            torch.from_numpy(states_delta),
            torch.from_numpy(attrs),
            particle_num,
            PARTICLE_DEN_CONST,
            color_imgs,
        )
