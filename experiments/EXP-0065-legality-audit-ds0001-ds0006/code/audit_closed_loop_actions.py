"""EXP-0065 RUN-0003: touchdown legality of EXECUTED closed-loop pushes (ISS-013 item b).
Episodes recorded with record_states store cube centres (no yaw) + executed actions [x0, y0, x1, y1] (m).
Blade = 40 x 2 mm rectangle centred at the start, long axis perpendicular to the push. Cube yaw unknown, so
legality is bounded: illegal_sure = blade rect within 2.5 mm (inscribed radius) of a cube centre,
illegal_possible = within 3.54 mm (circumscribed). Calibrated on DS-0006 candidates, where the exact SAT
answer (with yaw) is known. Writes results/audit_closed_loop_actions.json."""
import glob, json
import numpy as np, torch

HL, HW = 0.020, 0.001
R_IN, R_OUT = 0.0025, 0.0025 * np.sqrt(2)


def rect_dist(c, p, ang):
    """distance from points c (n,2) to a rectangle centred p, long axis at angle ang."""
    u = np.array([np.cos(ang), np.sin(ang)]); v = np.array([-u[1], u[0]])
    q = c - p
    du = np.maximum(np.abs(q @ u) - HL, 0); dv = np.maximum(np.abs(q @ v) - HW, 0)
    return np.hypot(du, dv)


def flags(cubes, act):
    p = np.array(act[:2]); d = np.array(act[2:4]) - p
    ang = np.arctan2(d[1], d[0]) + np.pi / 2
    m = rect_dist(np.asarray(cubes)[:, :2], p, ang).min()
    return m < R_IN, m < R_OUT


def main():
    out = {}
    # calibration on DS-0006 (known exact answer)
    d = torch.load('Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys/step0.pt', map_location='cpu', weights_only=False)
    st = d['states'][:, :, :2].numpy(); ps = d['p_starts'][:, :2].numpy(); pe = d['p_stops'][:, :2].numpy()
    fl = np.array([flags(st[i], np.r_[ps[i], pe[i]]) for i in range(0, len(st), 4)])
    out['calibration_DS-0006_every4th'] = dict(n=len(fl), sure=float(fl[:, 0].mean()), possible=float(fl[:, 1].mean()),
                                                exact_SAT_reference=0.543)
    for f in sorted(glob.glob('experiments/EXP-00[3-5]*/results/*.json')):
        try:
            J = json.load(open(f))
        except Exception:
            continue
        eps = J.get('episodes') if isinstance(J, dict) else None
        if not isinstance(eps, list) or not eps or not isinstance(eps[0], dict) or not eps[0].get('states') or not eps[0].get('actions'):
            continue
        rows = []
        for e in eps:
            if not e.get('states') or not e.get('actions'):
                continue
            for t, a in enumerate(e['actions']):
                if a is not None and t < len(e['states']) and e['states'][t] is not None and len(a) == 4:
                    s, p = flags(e['states'][t], a)
                    rows.append((e.get('model', '?'), e.get('planner', '?'), t, s, p))
        if not rows:
            continue
        arr = np.array([(r[3], r[4]) for r in rows], dtype=float)
        by = {}
        for m, pl, t, s, p in rows:
            by.setdefault(f'{m}|{pl}', []).append((s, p))
        out[f] = dict(n_pushes=len(rows), n_episodes=len(eps), illegal_sure=float(arr[:, 0].mean()), illegal_possible=float(arr[:, 1].mean()),
                      first_push_sure=float(np.mean([r[3] for r in rows if r[2] == 0])),
                      by_model_planner={k: [float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])), len(v)] for k, v in by.items()})
    json.dump(out, open('experiments/EXP-0065-legality-audit-ds0001-ds0006/results/audit_closed_loop_actions.json', 'w'), indent=1)
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != 'by_model_planner'})


if __name__ == '__main__':
    main()
