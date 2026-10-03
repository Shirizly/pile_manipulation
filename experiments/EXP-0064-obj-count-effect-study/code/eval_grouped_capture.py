"""Evaluate a GNN dynamics model (trained by train/train_gnn_dyn_grouped.py)
on the object-count-grouped TEST set (slates format: a pool of independent
candidate actions from the same fixed initial state per test state --
config/data_gen/slates_objbiased_carrots_grouped.yaml).

Two metrics, both defined in METRICS.md and re-stated here for this
particle/raw-coordinate setting (no occupancy image involved):

accuracy
    1 - rms(pred - true) / rms(persistence - true), pooled over all
    (state, action) samples -- persistence = predicting "nothing moved"
    (the model's own input state). Reported overall and per object-count
    group.

capture (METRICS.md's slateN, generalised to an arbitrary value function --
see METRICS.md's "slateN generalised to an arbitrary VALUE function" section)
    Value function here is Lyapunov-style distance-to-goal-centroid (project
    decision, see conversation): for one test state, a goal point g is drawn
    uniformly in the workspace, SEEDED BY THAT STATE'S INDEX (reproducible,
    one fixed goal per state regardless of which model is being scored).
    cost(particles) = mean distance from every (subsampled, tracked)
    particle to g -- lower is better, dv < 0 is an improving push, exactly
    METRICS.md's "lyapunov... COST (lower is better)" convention.
    For the pool of K candidate actions from that state:
        capture = (mean_c(true_dv) - true_dv[argmin_c(pred_dv)])
                  / (mean_c(true_dv) - min_c(true_dv))
    0 = model's pick no better than the pool average picked at random; 1 =
    model picked the oracle's (true-best) action. Reported per state, then
    mean (+/- sem) overall and per group.

Both metrics use the SAME node subsampling as training (FPS to
min(30, state's target object count) tracked particle indices, chosen once
per state from its initial frame) and the SAME position normalisation
(divide by GLOBAL_SCALE) as dataset/dataset_grouped_particles.py, so the
model sees distributionally the same input it was trained on.

Usage:
    python eval_grouped_capture.py <checkpoint.pth> [--config config/train/gnn_dyn_grouped.yaml]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from dataset.dataset_grouped_particles import (
    GLOBAL_SCALE,
    PARTICLE_DEN_CONST,
    PLATE_HALF_WIDTH,
    _fps_indices,
    _load_merged_manifest,
)
from model.gnn_dyn import PropNetDiffDenModel
from transforms.functional import build_action_delta
from utils import load_yaml

TEST_DATA_ROOT = 'data/true_action_slates_objbiased_carrots_grouped'
COUNT_GROUPS = [[10, 30], [50, 70], [100, 150], [400, 500]]
GROUP_LABELS = ['10-30', '50-70', '100-150', '400-500']


def load_model(config: dict, ckpt_path: str) -> PropNetDiffDenModel:
    use_gpu = torch.cuda.is_available()
    model = PropNetDiffDenModel(config, use_gpu)
    state = torch.load(ckpt_path, map_location='cuda' if use_gpu else 'cpu')
    model.load_state_dict(state)
    if use_gpu:
        model = model.cuda()
    model.eval()
    return model


def goal_point_for_state(state_idx: int, wkspc_w: float = 5.0) -> np.ndarray:
    """Seeded per test state (not per model/call) -- see module docstring."""
    rng = np.random.RandomState(state_idx)
    return rng.uniform(-wkspc_w, wkspc_w, size=2).astype(np.float32) / GLOBAL_SCALE


def cost_to_goal(points_xz: np.ndarray, goal_xz: np.ndarray) -> float:
    return float(np.linalg.norm(points_xz - goal_xz[None, :], axis=1).mean())


def evaluate(config: dict, ckpt_path: str, max_states_per_group: int | None = None):
    model = load_model(config, ckpt_path)
    device = next(model.parameters()).device

    records = _load_merged_manifest(TEST_DATA_ROOT)
    state_init = {r['state_idx']: r for r in records if r['type'] == 'state_init'}
    actions_by_state: dict[int, list[dict]] = {}
    for r in records:
        if r['type'] == 'action' and r['valid']:
            actions_by_state.setdefault(r['state_idx'], []).append(r)

    per_group_pred_sqerr = [0.0] * len(COUNT_GROUPS)
    per_group_copy_sqerr = [0.0] * len(COUNT_GROUPS)
    per_group_dof = [0] * len(COUNT_GROUPS)
    per_group_capture = [[] for _ in COUNT_GROUPS]

    overall_pred_sqerr = 0.0
    overall_copy_sqerr = 0.0
    overall_dof = 0
    overall_capture = []

    per_state_results = []

    n_done_per_group = [0] * len(COUNT_GROUPS)

    for state_idx in sorted(actions_by_state.keys()):
        rec0 = state_init.get(state_idx)
        if rec0 is None:
            continue
        group_idx = rec0.get('count_group')
        if group_idx is None:
            continue
        if max_states_per_group is not None and n_done_per_group[group_idx] >= max_states_per_group:
            continue
        n_done_per_group[group_idx] += 1

        target_count = rec0.get('target_num_carrots')
        node_budget = min(30, int(target_count)) if target_count else 30

        state_dir = os.path.join(TEST_DATA_ROOT, str(state_idx))
        init_pos = np.load(os.path.join(state_dir, 'initial_particles.npy')).reshape(-1, 4)[:, :3] / GLOBAL_SCALE
        idxs = _fps_indices(init_pos, node_budget)
        s0 = init_pos[idxs][:, [0, 2, 1]]  # (x, z, y) remap, see dataset module
        n = s0.shape[0]

        goal = goal_point_for_state(state_idx)
        cost0 = cost_to_goal(s0[:, :2], goal)

        acts = actions_by_state[state_idx]
        B = len(acts)
        s_cur = torch.from_numpy(np.tile(s0[None], (B, 1, 1))).float().to(device)
        a_cur = torch.ones((B, n), dtype=torch.float32, device=device)
        particle_dens = torch.full((B,), PARTICLE_DEN_CONST, dtype=torch.float32, device=device)

        s_delta = torch.zeros((B, n, 3), dtype=torch.float32)
        true_next = np.zeros((B, n, 3), dtype=np.float32)
        for bi, rec in enumerate(acts):
            a = np.array(rec['action'], dtype=np.float32) / GLOBAL_SCALE
            p_start = torch.tensor([a[0], a[1], 0.0])
            p_stop = torch.tensor([a[2], a[3], 0.0])
            s_delta[bi] = build_action_delta(
                torch.from_numpy(s0.astype(np.float32)), p_start, p_stop,
                sigma_m=PLATE_HALF_WIDTH)
            after = np.load(os.path.join(TEST_DATA_ROOT, rec['after_positions_path'])).reshape(-1, 4)[:, :3] / GLOBAL_SCALE
            true_next[bi] = after[idxs][:, [0, 2, 1]]
        s_delta = s_delta.to(device)

        with torch.no_grad():
            pred = model.predict_one_step(a_cur, s_cur, s_delta, particle_dens)
        pred_np = pred.cpu().numpy()

        pred_sqerr = ((pred_np - true_next) ** 2).sum()
        copy_sqerr = ((s0[None] - true_next) ** 2).sum()
        dof = B * n * 3

        per_group_pred_sqerr[group_idx] += pred_sqerr
        per_group_copy_sqerr[group_idx] += copy_sqerr
        per_group_dof[group_idx] += dof
        overall_pred_sqerr += pred_sqerr
        overall_copy_sqerr += copy_sqerr
        overall_dof += dof

        true_cost = np.array([cost_to_goal(true_next[bi, :, :2], goal) for bi in range(B)])
        pred_cost = np.array([cost_to_goal(pred_np[bi, :, :2], goal) for bi in range(B)])
        true_dv = true_cost - cost0
        pred_dv = pred_cost - cost0

        denom = true_dv.mean() - true_dv.min()
        if denom > 1e-9:
            chosen = int(np.argmin(pred_dv))
            capture = (true_dv.mean() - true_dv[chosen]) / denom
            per_group_capture[group_idx].append(capture)
            overall_capture.append(capture)

        per_state_results.append({
            'state_idx': state_idx, 'group': group_idx, 'n_particles_tracked': n,
            'n_actions': B, 'capture': float(capture) if denom > 1e-9 else None,
        })

    def rms(sqerr, dof):
        return float(np.sqrt(sqerr / max(1, dof)))

    def accuracy(pred_sqerr, copy_sqerr, dof):
        r_pred = rms(pred_sqerr, dof)
        r_copy = rms(copy_sqerr, dof)
        return 1.0 - r_pred / r_copy if r_copy > 0 else float('nan')

    results = {
        'overall': {
            'accuracy': accuracy(overall_pred_sqerr, overall_copy_sqerr, overall_dof),
            'capture_mean': float(np.mean(overall_capture)) if overall_capture else None,
            'capture_sem': float(np.std(overall_capture, ddof=1) / np.sqrt(len(overall_capture))) if len(overall_capture) > 1 else None,
            'n_states': len(overall_capture),
        },
        'by_group': {},
        'per_state': per_state_results,
    }
    for g, label in enumerate(GROUP_LABELS):
        caps = per_group_capture[g]
        results['by_group'][label] = {
            'range': COUNT_GROUPS[g],
            'accuracy': accuracy(per_group_pred_sqerr[g], per_group_copy_sqerr[g], per_group_dof[g]),
            'capture_mean': float(np.mean(caps)) if caps else None,
            'capture_sem': float(np.std(caps, ddof=1) / np.sqrt(len(caps))) if len(caps) > 1 else None,
            'n_states': len(caps),
        }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', help='Path to a trained net_*.pth checkpoint')
    parser.add_argument('--config', default='config/train/gnn_dyn_grouped.yaml')
    parser.add_argument('--out', default=None, help='Write results JSON here')
    parser.add_argument('--max-states-per-group', type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    results = evaluate(config, args.checkpoint, args.max_states_per_group)

    print(json.dumps({'overall': results['overall'], 'by_group': results['by_group']}, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print('wrote', args.out)
