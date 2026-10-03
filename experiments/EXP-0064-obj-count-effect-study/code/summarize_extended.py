"""Summarise eval_extended.py outputs (EXP-0064 RUN-0006) into results/.

capture (slateN form, METRICS.md): cost -> (mean(t) - t_chosen) / (mean(t) - min(t));
value -> (t_chosen - mean(t)) / (max(t) - mean(t)). Ties in the model's own
prediction at its best are broken by EXPECTATION (t_chosen = mean of t over
the tied-at-best set), so `persistence` (all-zero dv) scores exactly 0 instead
of the as-run eval's arbitrary first-index pick. Pools whose denominator is
< 1e-12 are skipped (counted).

Usage: python code/summarize_extended.py --dirs <npz dir> [<npz dir> ...] --out results/
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

GROUPS = ['10-30', '50-70', '100-150', '400-500']
MASK_CELLS = [(g, v) for g in ('random_quadrant', 'ring_O', 'T') for v in ('lyapunov', 'mass_in_region', 'signed_mass')]
HIB = {'lyapunov': False, 'mass_in_region': True, 'signed_mass': True}


def capture(pred, true, hib, idx=None):
    if idx is not None:
        pred, true = pred[idx], true[idx]
    if hib:
        pred, true = -pred, -true
    best = pred.min()
    tied = np.abs(pred - best) <= 1e-12 * max(1.0, abs(best))
    tc = true[tied].mean()
    den = true.mean() - true.min()
    if den < 1e-12:
        return np.nan
    return (true.mean() - tc) / den


def regret(pred, true, hib):
    if hib:
        pred, true = -pred, -true
    best = pred.min()
    tied = np.abs(pred - best) <= 1e-12 * max(1.0, abs(best))
    return true[tied].mean() - true.min(), true.mean() - true.min()


def boot(x, n=2000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return [float('nan')] * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def boot_diff(a, b, n=2000, seed=0):
    """Unpaired difference of group means (different states), mean(a) - mean(b)."""
    a = np.asarray([v for v in a if np.isfinite(v)]); b = np.asarray([v for v in b if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    d = rng.choice(a, (n, len(a))).mean(1) - rng.choice(b, (n, len(b))).mean(1)
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def load(dirs):
    states = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, 'state_*.npz'))):
            z = dict(np.load(f))
            s = int(z['state_idx'])
            if s in states:
                states[s].update({k: v for k, v in z.items() if k not in states[s]})
            else:
                states[s] = z
    return states


def model_names(states):
    z = next(iter(states.values()))
    return sorted({k[3:-len('_sqerr')] for k in z if k.startswith('r0_') and k.endswith('_sqerr') and not k.startswith('r0_persist_')})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k-eq-draws', type=int, default=50)
    args = ap.parse_args()
    S = load(args.dirs)
    models = model_names(S)
    reps = sorted({int(k[1:k.index('_')]) for k in next(iter(S.values())) if k.startswith('r') and k[1].isdigit()})
    clean_sizes = [int((~z['escaped']).sum()) for z in S.values()]
    K_EQ = int(min(clean_sizes))
    rng = np.random.default_rng(0)

    res = {'n_states': len(S), 'models': models, 'fps_reps': len(reps), 'K_eq': K_EQ, 'by_group': {}, 'per_state': {}}
    per = {m: {} for m in models}   # m -> metric -> list per state (with group)
    groups_of = {}
    for s, z in sorted(S.items()):
        g = int(z['group']); groups_of[s] = g
        esc = z['escaped'].astype(bool); keep = ~esc
        clean_idx = np.flatnonzero(keep)
        eq_draws = [rng.choice(clean_idx, K_EQ, replace=False) for _ in range(args.k_eq_draws)]
        for m in models:
            row = {}
            for r in reps:
                # accuracy pieces (summed across reps later)
                for tag, mask in (('all', np.ones_like(keep)), ('clean', keep)):
                    row.setdefault(f'sq_{tag}', 0.0); row.setdefault(f'sqp_{tag}', 0.0)
                    row.setdefault(f'sqm_{tag}', 0.0); row.setdefault(f'sqpm_{tag}', 0.0)
                    row[f'sq_{tag}'] += z[f'r{r}_{m}_sqerr'][mask].sum(); row[f'sqp_{tag}'] += z[f'r{r}_persist_sqerr'][mask].sum()
                    row[f'sqm_{tag}'] += z[f'r{r}_{m}_sqerr_moved'][mask].sum(); row[f'sqpm_{tag}'] += z[f'r{r}_persist_sqerr_moved'][mask].sum()
            # point-goal captures, averaged over FPS reps
            def over_reps(fn):
                return float(np.nanmean([fn(r) for r in reps]))
            pn = lambda r: z[f'r{r}_{m}_pred_nodes_point']
            tn = lambda r: z[f'r{r}_true_nodes_point']
            pf = lambda r: z[f'r{r}_{m}_pred_full_point']
            tf = z['true_full_point']
            row['cap_asrun_def'] = over_reps(lambda r: capture(pn(r)[0], tn(r)[0], False))                     # goal 0, nodes truth, escaped kept
            row['cap_nodes_g0_clean'] = over_reps(lambda r: capture(pn(r)[0], tn(r)[0], False, clean_idx))
            row['cap_nodes_17g_clean'] = over_reps(lambda r: np.nanmean([capture(pn(r)[g], tn(r)[g], False, clean_idx) for g in range(tf.shape[0])]))
            row['cap_full_17g_clean'] = over_reps(lambda r: np.nanmean([capture(pf(r)[g], tf[g], False, clean_idx) for g in range(tf.shape[0])]))
            row['cap_full_17g_Keq'] = over_reps(lambda r: np.nanmean([[capture(pf(r)[g], tf[g], False, d) for g in range(tf.shape[0])] for d in eq_draws[:10]]))
            rg = [regret(pf(r)[g][clean_idx], tf[g][clean_idx], False) for r in reps for g in range(tf.shape[0])]
            row['regret_full_17g'] = float(np.mean([a for a, _ in rg])); row['spread_full_17g'] = float(np.mean([b for _, b in rg]))
            # mask-goal slateN (full truth), per value function, goal-averaged
            pm = lambda r: z[f'r{r}_{m}_pred_full_mask']
            tm = z['true_full_mask']
            for vf in ('lyapunov', 'mass_in_region', 'signed_mass'):
                cells = [i for i, (_, v) in enumerate(MASK_CELLS) if v == vf]
                row[f'slateN_{vf}'] = over_reps(lambda r: np.nanmean([capture(pm(r)[i], tm[i], HIB[vf], clean_idx) for i in cells]))
                row[f'slateN_{vf}_Keq'] = over_reps(lambda r: np.nanmean([[capture(pm(r)[i], tm[i], HIB[vf], d) for i in cells] for d in eq_draws[:10]]))
            row['slateN_mean3'] = float(np.mean([row[f'slateN_{v}'] for v in HIB]))
            row['slateN_mean3_Keq'] = float(np.mean([row[f'slateN_{v}_Keq'] for v in HIB]))
            per[m][s] = row
        # pool facts (model-independent)
        res['per_state'][s] = {'group': g, 'n_valid': int(len(esc)), 'n_escaped': int(esc.sum()), 'n_particles': int(z['n_particles']),
                               'target_num_carrots': int(z['target_num_carrots']),
                               'frac_zero_node_motion': float(np.mean([(z[f'r{r}_moved_nodes'][keep] == 0).mean() for r in reps])),
                               'n_nodes': int(z['r0_n_nodes'])}

    metrics = [k for k in next(iter(per[models[0]].values())) if not k.startswith('sq')]
    for gi, gl in enumerate(GROUPS + ['overall']):
        sel = [s for s in S if gi == 4 or groups_of[s] == gi]
        if not sel:
            continue
        G = {'n_states': len(sel)}
        ps = [res['per_state'][s] for s in sel]
        G['valid_per_state'] = float(np.mean([p['n_valid'] for p in ps]))
        G['escaped_per_state'] = float(np.mean([p['n_escaped'] for p in ps]))
        G['frac_pool_zero_node_motion'] = float(np.mean([p['frac_zero_node_motion'] for p in ps]))
        G['n_particles_median'] = float(np.median([p['n_particles'] for p in ps]))
        for m in models:
            M = {}
            for tag in ('all', 'clean'):
                sq = sum(per[m][s][f'sq_{tag}'] for s in sel); sqp = sum(per[m][s][f'sqp_{tag}'] for s in sel)
                sqm = sum(per[m][s][f'sqm_{tag}'] for s in sel); sqpm = sum(per[m][s][f'sqpm_{tag}'] for s in sel)
                M[f'accuracy_nodes_{tag}'] = float(1 - np.sqrt(sq / sqp))
                M[f'accuracy_moved_nodes_{tag}'] = float(1 - np.sqrt(sqm / sqpm))
            for k in metrics:
                M[k] = boot([per[m][s][k] for s in sel])
            G[m] = M
        res['by_group'][gl] = G
    # group contrasts: 400-500 minus 10-30, per model, key metrics
    if not (any(g == 3 for g in groups_of.values()) and any(g == 0 for g in groups_of.values())):
        res['contrast_400-500_minus_10-30'] = {m: {} for m in models}
    else:
      res['contrast_400-500_minus_10-30'] = {m: {k: boot_diff([per[m][s][k] for s in S if groups_of[s] == 3],
                                                            [per[m][s][k] for s in S if groups_of[s] == 0])
                                               for k in ('cap_asrun_def', 'cap_full_17g_clean', 'cap_full_17g_Keq', 'slateN_mean3', 'slateN_mean3_Keq', 'regret_full_17g', 'spread_full_17g')}
                                           for m in models}
    # paired model - field baseline per group (same states)
    res['paired_vs_field'] = {}
    for m in models:
        if m == 'field':
            continue
        res['paired_vs_field'][m] = {}
        for gi, gl in enumerate(GROUPS + ['overall']):
            sel = [s for s in S if gi == 4 or groups_of[s] == gi]
            if not sel:
                continue
            res['paired_vs_field'][m][gl] = {k: boot([per[m][s][k] - per['field'][s][k] for s in sel])
                                             for k in ('cap_full_17g_clean', 'slateN_mean3')}
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, 'extended_summary.json.tmp')
    json.dump(res, open(tmp, 'w'), indent=1, default=float)
    os.replace(tmp, os.path.join(args.out, 'extended_summary.json'))

    # markdown table
    L = [f'# EXP-0064 extended re-scoring (RUN-0006) -- generated by code/summarize_extended.py\n',
         f'States {len(S)}, FPS reps {len(reps)}, K_eq {K_EQ} (smallest clean pool). Cells: mean [95% state-bootstrap CI].\n']
    def f(x):
        return f'{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]' if isinstance(x, list) else f'{x:.3f}'
    L.append('\n## Pool facts\n\n| group | valid/state | escaped-in-valid/state | pool share with no tracked-node motion | median particles |\n|---|---|---|---|---|')
    for gl in [g for g in GROUPS + ['overall'] if g in res['by_group']]:
        G = res['by_group'][gl]
        L.append(f"| {gl} | {G['valid_per_state']:.1f} | {G['escaped_per_state']:.2f} | {G['frac_pool_zero_node_motion']:.3f} | {G['n_particles_median']:.0f} |")
    cols = [('accuracy_nodes_all', 'acc nodes (as-run def)'), ('accuracy_moved_nodes_clean', 'acc moved nodes, clean'),
            ('cap_asrun_def', 'capture as-run def'), ('cap_full_17g_clean', 'capture 17 pt goals, all-particle truth'),
            ('cap_full_17g_Keq', f'same at K={K_EQ}'), ('slateN_mean3', 'slateN mask goals (mean 3 vf)'),
            ('slateN_mean3_Keq', f'same at K={K_EQ}'), ('regret_full_17g', 'regret (FleX units)'), ('spread_full_17g', 'mean-best spread')]
    for m in models:
        L.append(f'\n## {m}\n\n| group | ' + ' | '.join(c[1] for c in cols) + ' |\n|' + '---|' * (len(cols) + 1))
        for gl in [g for g in GROUPS + ['overall'] if g in res['by_group']]:
            M = res['by_group'][gl][m]
            L.append(f'| {gl} | ' + ' | '.join(f(M[c[0]]) for c in cols) + ' |')
    L.append('\n## 400-500 minus 10-30 (unpaired, state bootstrap)\n\n| model | ' + ' | '.join(res['contrast_400-500_minus_10-30'][models[0]].keys()) + ' |\n|' + '---|' * 8)
    for m in models:
        L.append(f'| {m} | ' + ' | '.join(f(v) for v in res['contrast_400-500_minus_10-30'][m].values()) + ' |')
    L.append('\n## model minus `field` baseline (paired per state)\n\n| model | group | capture 17 pt goals | slateN mask mean3 |\n|---|---|---|---|')
    for m, d in res['paired_vs_field'].items():
        for gl, v in d.items():
            L.append(f"| {m} | {gl} | {f(v['cap_full_17g_clean'])} | {f(v['slateN_mean3'])} |")
    open(os.path.join(args.out, 'extended_summary.md'), 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
