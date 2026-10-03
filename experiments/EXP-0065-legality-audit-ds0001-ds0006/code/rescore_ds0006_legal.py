"""EXP-0065 RUN-0002: re-score EXP-0030's cached DS-0006 predictions (8 models + mean-dv NFD ensemble,
30 goals x {lyapunov, mass_in_region}, soft truth) on LEGAL-only candidates vs the full 128-push pool,
with a size-matched random-subset control (slateN depends on pool size, C-031).
Reads experiments/EXP-0030-*/artifacts/RUN-0001/{truth,pred_*}.pt; writes results/rescore_ds0006_legal.json."""
import glob, json, os, sys
import numpy as np, torch
from scipy.stats import kendalltau
sys.path.insert(0, 'experiments/EXP-0059-retrieval-transition-model/code')
sys.path.insert(0, 'experiments/EXP-0029-state-vs-pool-split/code')
from audit_tool_placement import _row_illegal
from split_test import capture, VFS
from Baselines.common.goals import higher_is_better_for

ART = 'experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001'
d = torch.load('Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/step0.pt', map_location='cpu', weights_only=False)
illegal, _ = _row_illegal(d['states'].float(), d['p_starts'].float()[:, :2], d['angles'].float(), tol=0.0)
slate = d['slate_idx'].long().numpy()
dvt = torch.load(f'{ART}/truth.pt', weights_only=False)['dv'].numpy()
preds = {}
for f in sorted(glob.glob(f'{ART}/pred_*.pt')):
    x = torch.load(f, weights_only=False); preds[x['model']] = x['dv'].numpy()
nfd = [m for m in preds if m.startswith('nfd')]
preds['ensemble_nfd'] = np.mean([preds[m] for m in nfd], 0)
models = list(preds)
S = slate.max() + 1
rng = np.random.default_rng(0)
out = {'n_rows': int(len(slate)), 'frac_illegal': float(illegal.mean()), 'models': models, 'by_vf': {}}
for vi, vf in enumerate(VFS):
    hib = higher_is_better_for(vf)
    res = {m: {'all': [], 'legal': [], 'size_matched': []} for m in models}
    best_illegal, dv_ill, dv_leg = [], [], []
    for s in range(S):
        ix = np.flatnonzero(slate == s); leg = ix[~illegal[ix]]
        if len(leg) < 8:
            continue
        vt = dvt[ix, :, vi]
        sgn = 1 if hib else -1
        best_illegal.append(float(np.mean(illegal[ix][(sgn * vt).argmax(0)])))
        dv_ill.append(float(np.mean(sgn * dvt[ix[illegal[ix]], :, vi]))) if illegal[ix].any() else None
        dv_leg.append(float(np.mean(sgn * dvt[leg, :, vi])))
        subs = [rng.choice(ix, len(leg), replace=False) for _ in range(20)]
        for m in models:
            res[m]['all'].append(np.nanmean(capture(preds[m][ix, :, vi], vt, hib)))
            res[m]['legal'].append(np.nanmean(capture(preds[m][leg, :, vi], dvt[leg, :, vi], hib)))
            res[m]['size_matched'].append(np.nanmean([np.nanmean(capture(preds[m][q, :, vi], dvt[q, :, vi], hib)) for q in subs]))
    tab = {m: {k: float(np.mean(v)) for k, v in r.items()} for m, r in res.items()}
    for m in models:
        dlt = np.array(res[m]['legal']) - np.array(res[m]['size_matched'])
        bs = np.random.default_rng(1).choice(dlt, (2000, len(dlt))).mean(1)
        tab[m]['legal_minus_size_matched'] = [float(dlt.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    a = [tab[m]['all'] for m in models]; l = [tab[m]['legal'] for m in models]; z = [tab[m]['size_matched'] for m in models]
    out['by_vf'][vf] = {'per_model': tab, 'kendall_all_vs_legal': float(kendalltau(a, l)[0]),
                        'kendall_sizematched_vs_legal': float(kendalltau(z, l)[0]),
                        'order_all': [models[i] for i in np.argsort(a)[::-1]], 'order_legal': [models[i] for i in np.argsort(l)[::-1]],
                        'share_of_true_best_that_is_illegal': float(np.mean(best_illegal)),
                        'mean_improvement_illegal_vs_legal_candidates': [float(np.mean(dv_ill)), float(np.mean(dv_leg))],
                        'n_states': len(best_illegal)}
os.makedirs('experiments/EXP-0065-legality-audit-ds0001-ds0006/results', exist_ok=True)
json.dump(out, open('experiments/EXP-0065-legality-audit-ds0001-ds0006/results/rescore_ds0006_legal.json', 'w'), indent=1)
for vf, r in out['by_vf'].items():
    print(vf, 'tau all-vs-legal', round(r['kendall_all_vs_legal'], 3), 'best-is-illegal', round(r['share_of_true_best_that_is_illegal'], 3), 'impr ill/leg', np.round(r['mean_improvement_illegal_vs_legal_candidates'], 5))
    for m in models:
        t = r['per_model'][m]; print(f"  {m:40s} all {t['all']:.3f} legal {t['legal']:.3f} sizematched {t['size_matched']:.3f} legal-sm {t['legal_minus_size_matched'][0]:+.3f} [{t['legal_minus_size_matched'][1]:+.3f},{t['legal_minus_size_matched'][2]:+.3f}]")
