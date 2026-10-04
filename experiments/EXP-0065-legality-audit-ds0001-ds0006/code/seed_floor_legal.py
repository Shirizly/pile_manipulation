"""EXP-0065 RUN-0006: EXP-0036's NFD training-seed floor on DS-0006, all candidates vs legal-only (lyapunov /
mass_in_region, 30 goals, soft truth). Seed 0 = EXP-0030's nfd_3ch_randlen cache; seeds 1-3 = EXP-0036 RUN-0002.
Writes results/seed_floor_legal.json."""
import json, sys
import numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments/EXP-0029-state-vs-pool-split/code'); sys.path.insert(0, 'experiments/EXP-0059-retrieval-transition-model/code')
from split_test import capture, VFS
from audit_tool_placement import _row_illegal
from Baselines.common.goals import higher_is_better_for
C = 'Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/step0.pt'
d = torch.load(C, map_location='cpu', weights_only=False)
illegal, _ = _row_illegal(d['states'].float(), d['p_starts'].float()[:, :2], d['angles'].float(), tol=0.0)
slate = d['slate_idx'].long().numpy()
dvt = torch.load('experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001/truth.pt', weights_only=False)['dv'].numpy()
P = {'seed0': 'experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001/pred_nfd_3ch_randlen.pt'}
for k in (1, 2, 3):
    P[f'seed{k}'] = f'experiments/EXP-0036-seed-noise-floor/artifacts/RUN-0002/pred_DS-0006_nfd_3ch_randlen_seed{k}.pt'
dvp = {k: torch.load(v, weights_only=False)['dv'].numpy() for k, v in P.items()}
out = {}
for vi, vf in enumerate(VFS):
    hib = higher_is_better_for(vf)
    for pool in ('all', 'legal'):
        per = {}
        for k in dvp:
            caps = []
            for s in range(slate.max() + 1):
                ix = np.flatnonzero(slate == s)
                if pool == 'legal':
                    ix = ix[~illegal[ix]]
                caps.append(np.nanmean(capture(dvp[k][ix, :, vi], dvt[ix, :, vi], hib)))
            per[k] = float(np.mean(caps))
        v = np.array(list(per.values()))
        out[f'{vf}|{pool}'] = dict(per_seed=per, mean=float(v.mean()), sd=float(v.std(ddof=1)), range=float(v.max() - v.min()))
        print(vf, pool, {k: round(x, 3) for k, x in per.items()}, 'sd', round(v.std(ddof=1), 4))
json.dump(out, open('experiments/EXP-0065-legality-audit-ds0001-ds0006/results/seed_floor_legal.json', 'w'), indent=1)
