"""EXP-0065 RUN-0005: the EXP-0039/0044 closed-loop model set re-scored offline on DS-0006 pools,
all candidates vs LEGAL-only (exact SAT), over all 160 states and over the closed-loop start states
40-47 only (lyapunov / mass_in_region, 30 goals, soft truth -- EXP-0030 cache convention). Adds
optimism at the pick (true dv - predicted dv for the chosen push, cost sense) and top-1 regret.
Missing predictions are computed with EXP-0030's own path (make_occ_adapter + predict_step).
Writes results/closed_loop_models_legal_rescore.json."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments/EXP-0029-state-vs-pool-split/code')
sys.path.insert(0, 'experiments/EXP-0059-retrieval-transition-model/code')
from split_test import capture, goal_tensors, values, VFS, save_atomic
from audit_tool_placement import _row_illegal
from Baselines.common.goals import higher_is_better_for
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from simple_mpc.adapters import make_occ_adapter, occ_from_particles

CORPUS = 'Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys'
A30 = 'experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001'
OUT = 'experiments/EXP-0065-legality-audit-ds0001-ds0006'
SRC = {'nfd_3ch_randlen': f'{A30}/pred_nfd_3ch_randlen.pt',
       'nfd_residual_worldframe_noaug_ep43': f'{A30}/pred_nfd_residual_worldframe_noaug_ep43.pt',
       'nfd_3ch_randlen_seed1': 'experiments/EXP-0036-seed-noise-floor/artifacts/RUN-0002/pred_DS-0006_nfd_3ch_randlen_seed1.pt',
       'linear_switched_soft': f'{OUT}/artifacts/pred_linear_switched_soft.pt'}
rows = BinnedSlateCorpus.load(CORPUS).step(0)
slate = rows.slate_idx.long().numpy(); S = slate.max() + 1
dvt = torch.load(f'{A30}/truth.pt', weights_only=False)['dv'].numpy()
acts = torch.cat([rows.p_starts[:, :2], rows.p_stops[:, :2]], 1).float()
import os; os.makedirs(f'{OUT}/artifacts', exist_ok=True)
if not os.path.exists(SRC['linear_switched_soft']):
    dev = torch.device('cpu'); t0 = time.time()
    masks, dists = goal_tensors(dev); ad = make_occ_adapter('linear_switched_soft', dev, 'corner')
    out = torch.zeros(len(slate), dvt.shape[1], 2)
    for s in range(S):
        ix = torch.from_numpy(np.nonzero(slate == s)[0])
        occ0 = occ_from_particles(rows.states[ix[0]].float()[None], dev); v0 = values(occ0, masks, dists)
        with torch.no_grad():
            pred = ad.predict_step(occ0.expand(len(ix), -1, -1).contiguous(), acts[ix])
            out[ix] = values(pred.float(), masks, dists) - v0
    save_atomic({'dv': out, 'model': 'linear_switched_soft'}, SRC['linear_switched_soft'])
    print(f'pred linear_switched_soft {time.time() - t0:.0f}s', flush=True)
dvp = {m: torch.load(p, weights_only=False)['dv'].numpy() for m, p in SRC.items()}
d = torch.load(f'{CORPUS}/step0.pt', map_location='cpu', weights_only=False)
illegal, _ = _row_illegal(d['states'].float(), d['p_starts'].float()[:, :2], d['angles'].float(), tol=0.0)
res = {}
for vi, vf in enumerate(VFS):
    hib = higher_is_better_for(vf); sg = 1 if hib else -1
    for scope, states in (('all160', range(S)), ('cl_starts_40_47', range(40, 48))):
        for pool in ('all', 'legal'):
            tab = {}
            for m in dvp:
                caps, opt, reg = [], [], []
                for s in states:
                    ix = np.flatnonzero(slate == s)
                    if pool == 'legal':
                        ix = ix[~illegal[ix]]
                    vp, vt = dvp[m][ix, :, vi], dvt[ix, :, vi]
                    caps.append(np.nanmean(capture(vp, vt, hib)))
                    pick = (sg * vp).argmax(0); g = np.arange(vt.shape[1])
                    opt.append(np.mean(sg * (vp[pick, g] - vt[pick, g])))   # >0: predicted more improvement than realised
                    reg.append(np.mean((sg * vt).max(0) - sg * vt[pick, g]))
                tab[m] = dict(slateN=float(np.mean(caps)), optimism=float(np.mean(opt)), top1_regret=float(np.mean(reg)), n_states=len(caps))
            res[f'{vf}|{scope}|{pool}'] = tab
save_atomic(res, f'{OUT}/results/closed_loop_models_legal_rescore.json')
for k, t in res.items():
    print(k, ' '.join(f"{m.replace('nfd_','').replace('_randlen','')[:18]}={v['slateN']:.3f}/opt{v['optimism']:+.4f}" for m, v in sorted(t.items(), key=lambda x: -x[1]['slateN'])))
