"""Appends T5-T8 (models vs ceiling, metric correlations, ceilings in each metric) to results/ceiling/tables.md (run after ceiling_report.py)."""
import json, numpy as np
OUT = 'experiments/EXP-0072-zoom-window-nfd/results/ceiling/'
cs = json.load(open(OUT + 'ceiling_scores.json')); mv = json.load(open(OUT + 'metrics_vs_slate.json')); L = []; w = L.append
lv = [0, 0.01, 0.05, 0.25, 0.5, 1.0]
def curve(fr): return [np.mean([cs[t]['onestep']['all'][fr] for t in cs if not t.startswith('_') and float(t.split('mm')[0]) == l]) for l in lv]
def equiv(fr, val):
    c = curve(fr)
    if val >= c[0]: return 0.0
    for i in range(len(lv) - 1):
        if c[i] >= val >= c[i + 1]: return float(np.interp(val, [c[i + 1], c[i]], [lv[i + 1], lv[i]]))
    return float('nan')
rows = [("nfd64", "w64_native", 0.559, 0.710), ("world128 (300ep)", "window64", 0.807, 0.728), ("world128 (300ep)", "paste_world128", 0.591, 0.728),
        ("zoom64 ft300", "window64", 0.846, 0.765), ("zoom64 ft300", "paste_zoom64", 0.622, 0.765), ("zoom128", "window128", 0.859, 0.809), ("zoom128", "paste_zoom128", 0.640, 0.809)]
tl = cs['_true_label_cap']
w("## T5 models vs ceiling (accuracy_1 on DS-0016 chains, same frame; ceiling = mean of two 0 mm re-sims; slateN in pasted frame; window slateN not given)")
w("| model | frame | model acc | ceiling@0mm | true-label cap | model/ceiling | equivalent xy noise (mm, interpolated) | model slateN | slateN ceiling@0mm same frame | true-label slateN cap |\n|---|---|---|---|---|---|---|---|---|---|")
for m, f, a, s in rows:
    c0 = np.mean([cs['0mm_r0']['onestep']['all'][f], cs['0mm_r1']['onestep']['all'][f]])
    sf = f if f in ('w64_native', 'paste_zoom64', 'paste_zoom128') else None
    sc = np.mean([cs['0mm_r0']['pools_slateN'][sf], cs['0mm_r1']['pools_slateN'][sf]]) if sf else float('nan'); st = tl['pools_slateN'].get(sf, float('nan')) if sf else float('nan')
    w(f"| {m} | {f} | {a:.3f} | {c0:.3f} | {tl['onestep'][f]:.3f} | {a / c0:.2f} | {equiv(f, a):.2f} | {s:.3f} | {sc:.3f} | {st:.3f} |")
w("\n## T6 metric vs slateN across models (model-level Kendall tau / Spearman rho; boot = tau mean+-sd over 200 pool-bootstraps; per-pool = mean over models of within-model Spearman across the 32 pools; xm = pool-demeaned across-model Spearman)")
for s in ('core7', 'all15'):
    w(f"\n**{s}** (n={7 if s == 'core7' else 15} models)\n\n| metric | pools tau | rho | boot tau | chains tau | within-model per-pool rho | within-pool across-model rho |\n|---|---|---|---|---|---|---|")
    C = mv['correlations_model_level'][s]; P = mv['correlations_per_pool'][s]
    for k, v in C.items():
        if k.startswith('pools'):
            n = k[6:]; w(f"| {n} | {v['kendall']:+.2f} | {v['spearman']:+.2f} | {v['boot_tau_mean']:+.2f}+-{v['boot_tau_sd']:.2f} | {C['chains:' + n]['kendall']:+.2f} | {P[n]['mean_within_model_spearman_over_pools']:+.2f} | {P[n]['within_pool_across_models_spearman']:+.2f} |")
ks = list(mv['models']['ft300']['metrics_pools'].keys()); T = ('0mm_r0', '0.25mm_r0', '0.5mm_r0', '1mm_r0')
w("\n## T7 ceiling in every metric (perfect-physics re-sim, pool rows; columns frame@level)")
w("| metric | model range (15 models) | " + " | ".join(f"{fr[:5]}@{t.split('mm')[0]}" for t in T for fr in ('w64_native', 'paste_zoom64')) + " |\n|" + "---|" * 10)
for k in ks:
    v = [mv['models'][m]['metrics_pools'][k] for m in mv['models']]
    w(f"| {k} | {min(v):+.2f}..{max(v):+.2f} | " + " | ".join(f"{mv['ceiling'][t][fr]['pools'][k]:+.2f}" for t in T for fr in ('w64_native', 'paste_zoom64')) + " |")
w("\n## T8 models (pool rows, pasted world-64 frame): slateN and metrics\n| model | slateN | " + " | ".join(ks) + " |\n|" + "---|" * (2 + len(ks)))
for m, v in mv['models'].items(): w(f"| {m} | {v['slateN']:.3f} | " + " | ".join(f"{v['metrics_pools'][k]:+.2f}" for k in ks) + " |")
open(OUT + 'tables.md', 'a').write("\n\n" + "\n".join(L)); print("\n".join(L))
