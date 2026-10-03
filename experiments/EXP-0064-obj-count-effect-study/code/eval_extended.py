"""EXP-0064 RUN-0006 -- extended, eval-only re-scoring of the object-count
GNN study on DS-0022 (test slates), run from THIS repo.

What `eval_grouped_capture.py` (RUN-0004, as run in the source repo) did not
do, and this adds -- per experiment-log standards:

  * action frame: models are fed the action with an explicit z sign
    (+1 = the as-run, z-MIRRORED reading of MODEL-0011; -1 = corrected,
    invariant `flex-action-frame-neg-z`); truth never depends on it.
  * baselines: `persistence` (predict nothing moved; its argmin over an
    all-zero dv is arbitrary, so its capture is ~random by construction) and
    `field` (pred = s0 + the corrected `build_action_delta` field, i.e. the
    GNN's own action input used as a dynamics model -- an analytic
    do-something baseline the GNN must beat).
  * FPS seeding: the as-run eval used an unseeded random FPS start; here
    `np.random.seed(1000 * rep + state_idx)`, R reps -> node-sampling noise.
  * escaped rows: rows marked valid whose after-state has any particle with
    |x| or |z| > 10 (DS-0019's `escape_abs`) are flagged, so the summary can
    drop them (the as-run eval kept them).
  * goals: the as-run single point goal per state (rng(state_idx)), plus
    G_EXTRA more point goals per state (rng(10_000 + 100 * state_idx + g)).
  * truth on ALL particles, not only the <= 30 tracked nodes: the model's
    node displacements are carried to every particle by nearest node (xz),
    and both point-goal cost and occupancy are computed on all particles.
  * project-standard goal masks: random_quadrant (seed = state_idx) / ring_O
    / T x lyapunov / mass_in_region / signed_mass on a 64x64 binary disk
    raster, +-7.2 FleX units, grid row = x, col = -z (DS-0019's frame and
    grid). A LOCAL approximation of eval_report's FleX path (binary disk
    raster for both truth and prediction), not eval_report itself.

Writes one npz per state into --out (atomic: .tmp then os.replace) plus a
manifest.json rewritten after every state; resumes by skipping states whose
npz exists. Summaries are computed by summarize_extended.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_grouped_particles import (GLOBAL_SCALE, PARTICLE_DEN_CONST,  # noqa: E402
                                       PLATE_HALF_WIDTH, _fps_indices,
                                       _load_merged_manifest, encode_action)
from Baselines.GNN.model.gnn_dyn import PropNetDiffDenModel  # noqa: E402
from Baselines.common import goals as G  # noqa: E402
from control_utility_test import lyapunov  # noqa: E402
from transforms.functional import build_action_delta  # noqa: E402

TEST_ROOT = 'datasets/DS-0022-flex-carrots-countgroups-test-slates/data'
ESCAPE_ABS = 10.0
GRID_HALF, GRID_RES = 7.2, 64
WKSPC_W = 5.0


def load_gnn(cfg_path, ckpt, device):
    cfg = yaml.safe_load(open(cfg_path))
    m = PropNetDiffDenModel(cfg, device.type == 'cuda')
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


def raster(xz_raw: np.ndarray) -> np.ndarray:
    """(B, N, 2) raw FleX (x, z) -> (B, 64, 64) binary, row = x, col = -z,
    radius-1 disk (centre + 4-neighbours) footprint."""
    B = xz_raw.shape[0]
    px = GRID_HALF * 2 / GRID_RES
    r = np.floor((xz_raw[..., 0] + GRID_HALF) / px).astype(np.int64)
    c = np.floor((-xz_raw[..., 1] + GRID_HALF) / px).astype(np.int64)
    out = np.zeros((B, GRID_RES + 2, GRID_RES + 2), dtype=bool)
    ok = (r >= 0) & (r < GRID_RES) & (c >= 0) & (c < GRID_RES)
    b = np.broadcast_to(np.arange(B)[:, None], r.shape)
    for dr, dc in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        out[b[ok], r[ok] + 1 + dr, c[ok] + 1 + dc] = True
    return out[:, 1:-1, 1:-1].astype(np.float32)


def mask_goals(state_idx):
    H = W = GRID_RES
    q, _ = G.random_quadrant_mask(H, W, state_idx)
    return {'random_quadrant': q, 'ring_O': G.letter_mask('O', H, W), 'T': G.letter_mask('T', H, W)}


def mask_values(occ: np.ndarray, masks: dict) -> np.ndarray:
    """-> (9, B): goal-major x (lyapunov, mass_in_region, signed_mass)."""
    o = torch.from_numpy(occ)
    rows = []
    for name in ('random_quadrant', 'ring_O', 'T'):
        m = masks[name]
        d = torch.from_numpy(G.dist_field_from_mask(m))
        mt = torch.from_numpy(m)
        rows += [lyapunov(o, d).numpy(), G.mass_in_region(o, mt).numpy(), G.signed_mass_in_region(o, mt).numpy()]
    return np.stack(rows)


def point_goals(state_idx, n_extra):
    g0 = np.random.RandomState(state_idx).uniform(-WKSPC_W, WKSPC_W, size=2)  # as-run goal (raw units)
    gs = [g0] + [np.random.RandomState(10_000 + 100 * state_idx + g).uniform(-WKSPC_W, WKSPC_W, size=2)
                 for g in range(n_extra)]
    return np.array(gs, dtype=np.float32)  # (G, 2) raw (x, z)


def mean_dist(xz: np.ndarray, goals_xz: np.ndarray) -> np.ndarray:
    """xz (B, N, 2), goals (G, 2) -> (G, B) mean distance."""
    return np.stack([np.linalg.norm(xz - g[None, None], axis=-1).mean(-1) for g in goals_xz])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--models', default='gnn_asrun:+1:weights/MODEL-0011-gnn-flex-particles-countgroups-mirrored-action/checkpoint.pth')
    ap.add_argument('--config', default='experiments/EXP-0064-obj-count-effect-study/code/configs/gnn_dyn_grouped.yaml')
    ap.add_argument('--fps-reps', type=int, default=3)
    ap.add_argument('--n-extra-goals', type=int, default=16)
    ap.add_argument('--states', default=None, help='comma list (smoke)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnns = {}
    for spec in args.models.split(','):
        parts = spec.split(':')  # name:z_sign:ckpt[:encoding tube|orig]
        name, sign, ckpt = parts[:3]
        enc = parts[3] if len(parts) > 3 else 'tube'
        gnns[name] = (float(sign), enc, load_gnn(args.config, ckpt, device))
    model_names = list(gnns) + ['persistence', 'field']

    recs = _load_merged_manifest(TEST_ROOT)
    st_init = {r['state_idx']: r for r in recs if r['type'] == 'state_init'}
    by_state = {}
    for r in recs:
        if r['type'] == 'action' and r['valid']:
            by_state.setdefault(r['state_idx'], []).append(r)
    states = sorted(by_state) if args.states is None else [int(s) for s in args.states.split(',')]
    man_path = os.path.join(args.out, 'manifest.json')
    manifest = {'models': args.models, 'model_names': model_names, 'fps_reps': args.fps_reps,
                'n_extra_goals': args.n_extra_goals, 'test_root': TEST_ROOT, 'done': []}
    if os.path.exists(man_path):
        manifest['done'] = json.load(open(man_path))['done']

    for s in states:
        fn = os.path.join(args.out, f'state_{s:04d}.npz')
        if os.path.exists(fn):
            continue
        t0 = time.time()
        rec0 = st_init[s]
        acts = by_state[s]
        B = len(acts)
        p0 = np.load(os.path.join(TEST_ROOT, rec0['positions_path'])).reshape(-1, 4)[:, :3]
        after = np.stack([np.load(os.path.join(TEST_ROOT, r['after_positions_path'])).reshape(-1, 4)[:, :3] for r in acts])
        escaped = (np.abs(after[..., [0, 2]]) > ESCAPE_ABS).any(-1).any(-1)
        A = np.array([r['action'] for r in acts], dtype=np.float32)
        tc = int(rec0.get('target_num_carrots') or 30)
        nb = min(30, tc)
        pgoals = point_goals(s, args.n_extra_goals)
        masks = mask_goals(s)
        out = {'state_idx': s, 'group': rec0['count_group'], 'target_num_carrots': tc,
               'n_particles': p0.shape[0], 'action_idx': np.array([r['action_idx'] for r in acts]),
               'push_length': np.array([r['push_length'] for r in acts]), 'escaped': escaped,
               'point_goals': pgoals}
        # truth on all particles (independent of FPS rep)
        out['true_full_point'] = mean_dist(after[..., [0, 2]], pgoals)            # (G, B)
        out['true_full_mask'] = mask_values(raster(after[..., [0, 2]]), masks)     # (9, B)
        out['v0_full_point'] = mean_dist(p0[None, :, [0, 2]], pgoals)[:, 0]
        for rep in range(args.fps_reps):
            np.random.seed(1000 * rep + s)
            idx = _fps_indices(p0 / GLOBAL_SCALE, nb)
            n = idx.shape[0]
            s0 = (p0[idx] / GLOBAL_SCALE)[:, [0, 2, 1]].astype(np.float32)   # (n, 3) normalised (x, z, y)
            true_nodes = (after[:, idx] / GLOBAL_SCALE)[..., [0, 2, 1]]       # (B, n, 3)
            out[f'r{rep}_true_nodes_point'] = mean_dist(true_nodes[..., :2] * GLOBAL_SCALE, pgoals)
            moved = np.linalg.norm((true_nodes - s0[None])[..., :2], axis=-1) * GLOBAL_SCALE > 0.1  # (B, n)
            out[f'r{rep}_moved_nodes'] = moved.sum(1)
            # nearest node (xz) of every particle, for carrying node motion to all particles
            dd = ((p0[:, None, [0, 2]] / GLOBAL_SCALE - s0[None, :, :2]) ** 2).sum(-1)
            nn = dd.argmin(1)

            def deltas(sign, enc='tube'):
                sd = torch.zeros((B, n, 3))
                for bi in range(B):
                    a = A[bi] / GLOBAL_SCALE
                    ps = torch.tensor([a[0], sign * a[1], 0.0])
                    pe = torch.tensor([a[2], sign * a[3], 0.0])
                    sd[bi] = encode_action(enc, torch.from_numpy(s0), ps, pe)
                return sd

            preds = {}
            for name, (sign, enc, m) in gnns.items():
                sd = deltas(sign, enc).to(device)
                with torch.no_grad():
                    pr = m.predict_one_step(torch.ones((B, n), device=device),
                                            torch.from_numpy(np.tile(s0[None], (B, 1, 1))).to(device),
                                            sd, torch.full((B,), PARTICLE_DEN_CONST, device=device))
                preds[name] = pr.cpu().numpy()
            preds['persistence'] = np.tile(s0[None], (B, 1, 1))
            preds['field'] = s0[None] + deltas(-1.0).numpy()
            preds['field_orig'] = s0[None] + deltas(-1.0, 'orig').numpy()
            for name, pr in preds.items():
                err = pr - true_nodes
                out[f'r{rep}_{name}_sqerr'] = (err ** 2).sum((1, 2))                     # (B,)
                out[f'r{rep}_{name}_sqerr_moved'] = ((err ** 2).sum(-1) * moved).sum(1)
                out[f'r{rep}_{name}_pred_nodes_point'] = mean_dist(pr[..., :2] * GLOBAL_SCALE, pgoals)
                disp_raw = (pr - s0[None])[..., :2] * GLOBAL_SCALE                         # (B, n, 2)
                carried = p0[None, :, [0, 2]] + disp_raw[:, nn]                              # (B, N, 2)
                out[f'r{rep}_{name}_pred_full_point'] = mean_dist(carried, pgoals)
                out[f'r{rep}_{name}_pred_full_mask'] = mask_values(raster(carried), masks)
            out[f'r{rep}_persist_sqerr'] = ((s0[None] - true_nodes) ** 2).sum((1, 2))
            out[f'r{rep}_persist_sqerr_moved'] = (((s0[None] - true_nodes) ** 2).sum(-1) * moved).sum(1)
            out[f'r{rep}_n_nodes'] = n
        tmp = fn + '.tmp.npz'
        np.savez_compressed(tmp, **out)
        os.replace(tmp, fn)
        manifest['done'].append(s)
        with open(man_path + '.tmp', 'w') as f:
            json.dump(manifest, f)
        os.replace(man_path + '.tmp', man_path)
        print(f'state {s} group {rec0["count_group"]} B={B} esc={int(escaped.sum())} N={p0.shape[0]} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
