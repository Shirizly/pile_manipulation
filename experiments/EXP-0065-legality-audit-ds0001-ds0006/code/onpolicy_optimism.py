"""EXP-0065 RUN-0004: on executed closed-loop pushes, is the model's predicted dv more optimistic on the
illegal-touchdown pushes than on legal ones? (illegal = 'sure' bound of audit_closed_loop_actions.py).
dv sign: the records' planning objective is a COST (lyapunov, or lyapunov - w*mass), so optimism =
true_dv - pred_dv (> 0: the model predicted more improvement than happened). Writes results/onpolicy_optimism.json."""
import glob, json, sys
import numpy as np
sys.path.insert(0, 'experiments/EXP-0065-legality-audit-ds0001-ds0006/code')
from audit_closed_loop_actions import flags  # noqa  (module runs its audit on import? guarded below)

out = {}
for f in sorted(glob.glob('experiments/EXP-00[5]*/results/*.json')):
    try:
        J = json.load(open(f))
    except Exception:
        continue
    eps = J.get('episodes') if isinstance(J, dict) else None
    if not isinstance(eps, list) or not eps or not isinstance(eps[0], dict) or not eps[0].get('states') or not eps[0].get('pred_dv'):
        continue
    by = {}
    for e in eps:
        if not e.get('states') or not e.get('actions') or not e.get('pred_dv') or not e.get('true_dv'):
            continue
        for t, a in enumerate(e['actions']):
            if a is None or t >= len(e['states']) or e['states'][t] is None or e['pred_dv'][t] is None or e['true_dv'][t] is None:
                continue
            sure, _ = flags(e['states'][t], a)
            by.setdefault(f"{e.get('model','?')}|{e.get('planner','?')}", []).append((bool(sure), float(e['pred_dv'][t]), float(e['true_dv'][t])))
    res = {}
    for k, v in by.items():
        v = np.array(v)
        ill, pr, tr = v[:, 0].astype(bool), v[:, 1], v[:, 2]
        def st(m):
            return dict(n=int(m.sum()), pred=float(pr[m].mean()), true=float(tr[m].mean()), optimism=float((tr[m] - pr[m]).mean())) if m.sum() else None
        res[k] = dict(illegal=st(ill), legal=st(~ill))
    if res:
        out[f] = res
json.dump(out, open('experiments/EXP-0065-legality-audit-ds0001-ds0006/results/onpolicy_optimism.json', 'w'), indent=1)
for f, r in out.items():
    print(f.split('/')[1][:8], f.split('/')[-1])
    for k, v in r.items():
        fmt = lambda d: 'n/a' if d is None else f"n={d['n']} pred {d['pred']:+.4f} true {d['true']:+.4f} opt {d['optimism']:+.4f}"
        print(f"   {k:45s} ILLEGAL {fmt(v['illegal'])} | LEGAL {fmt(v['legal'])}")
