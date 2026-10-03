"""EXP-0064 RUN-0007 -- action-frame and escaped-row audit of DS-0021 / DS-0022.

For a seeded sample of valid rows: (a) mean cosine between the mean displacement
of moved (> 0.1), non-escaped particles and the push direction, under the
literal (x, z) and the (x, -z) reading of the stored action; (b) the share of
in-path particles (inside the plate swath, half-width 1.0) that move, both
readings; (c) cosine between true node displacement and the GNN's action input
(GroupedParticleDataset.states_delta) for action_z_sign +1 (as run) and -1.
Writes results/frame_check.json.
"""
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_grouped_particles import GroupedParticleDataset  # noqa: E402

E = 'experiments/EXP-0064-obj-count-effect-study'
ROOTS = {'DS-0021': ('datasets/DS-0021-flex-carrots-countgroups-train/data', 'transition'),
         'DS-0022': ('datasets/DS-0022-flex-carrots-countgroups-test-slates/data', 'action')}


def rows_of(root, kind):
    recs = [json.loads(l) for l in open(os.path.join(root, 'manifest.jsonl'))]
    return [r for r in recs if r['type'] == kind and r['valid']]


def before_path(root, r, kind):
    if kind == 'transition' and r['step_idx'] > 0:
        return os.path.join(root, str(r['state_idx']), '%d_after_particles.npy' % (r['step_idx'] - 1))
    return os.path.join(root, str(r['state_idx']), 'initial_particles.npy')


out = {}
for name, (root, kind) in ROOTS.items():
    rows = rows_of(root, kind)
    rng = np.random.default_rng(1)
    cos = {1: [], -1: []}; inpath = {1: [], -1: []}
    for i in rng.choice(len(rows), 300, replace=False):
        r = rows[i]
        p0 = np.load(before_path(root, r, kind)).reshape(-1, 4)[:, [0, 2]]
        p1 = np.load(os.path.join(root, r['after_positions_path'])).reshape(-1, 4)[:, [0, 2]]
        d = p1 - p0
        a = np.array(r['action'])
        moved = np.linalg.norm(d, axis=1) > 0.1
        ok = moved & (np.abs(p1).max(1) < 10)
        for sg in (1, -1):
            st = np.array([a[0], sg * a[1]]); en = np.array([a[2], sg * a[3]])
            v = en - st; L = np.linalg.norm(v); u = v / L; nrm = np.array([-u[1], u[0]])
            if ok.sum() >= 3:
                md = d[ok].mean(0); cos[sg].append(float(md @ u / np.linalg.norm(md)))
            q = p0 - st; t = q @ u; w = np.abs(q @ nrm)
            ip = (t > 0) & (t < L) & (w < 1.0)
            if ip.sum():
                inpath[sg].append(float(moved[ip].mean()))
    out[name] = {'n_rows_sampled': 300,
                 'cos_disp_push_literal': float(np.mean(cos[1])), 'cos_disp_push_negz': float(np.mean(cos[-1])),
                 'inpath_moved_literal': float(np.mean(inpath[1])), 'inpath_moved_negz': float(np.mean(inpath[-1])),
                 'n_inpath_rows_literal': len(inpath[1]), 'n_inpath_rows_negz': len(inpath[-1])}

cfg = yaml.safe_load(open(os.path.join(E, 'code/configs/gnn_dyn_grouped.yaml')))
cfg['train']['data_root'] = ROOTS['DS-0021'][0]
for sg in (1.0, -1.0):
    cfg['train']['action_z_sign'] = sg
    np.random.seed(0)
    ds = GroupedParticleDataset(cfg['train']['data_root'], cfg, 'train')
    cs = []
    for i in range(0, len(ds), 60):
        st, sd, *_ = ds[i]
        tr = (st[1] - st[0])[:, :2].numpy(); dl = sd[0][:, :2].numpy()
        cs.append(float((tr * dl).sum() / (np.linalg.norm(tr) * np.linalg.norm(dl) + 1e-12)))
    out[f'gnn_input_cos_node_disp_vs_s_delta_sign{int(sg):+d}'] = float(np.mean(cs))
os.makedirs(os.path.join(E, 'results'), exist_ok=True)
json.dump(out, open(os.path.join(E, 'results/frame_check.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
