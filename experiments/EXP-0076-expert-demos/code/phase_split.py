"""Phase-resolved comparison of recorded runs: bulk = start -> first state with mass >= 0.8; tail = that state -> first state >= 0.95.
Per arm on the 36-scenario ablation set: pushes / physical time per phase (only episodes that reach the phase end), reach rates, and paired
differences vs the ens baseline pair (f20_all, F0_seed1) on scenarios where both reach the phase end."""
import json, numpy as np, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import tasks as T
ARMS = {"ens (f20_all)": "f20_all", "ens seed1 (F0)": "F0_seed1", "v64 4x budget (F1)": "F1_v64big", "ens H2 OT (F2)": "F2_ensH2ot",
        "ens H2->H1 (F3)": "F3_phaseH2H1", "ens 2x budget (F4)": "F4_ensbig", "v64 H2 OT (F5)": "F5_v64H2ot", "time_mu .02 (F6)": "F6_tmu02",
        "time_mu .05 (F7)": "F7_tmu05", "tail rule (F8)": "F8_tailrule"}
R = {k: json.load(open(HERE.parent / f"results/runs/{v}.json"))["scenarios"] for k, v in ARMS.items()}
ids = sorted(R["v64 4x budget (F1)"])


def phases(r):
    m = np.array(r["masses"]); t = np.concatenate([[0], np.cumsum(r["push_times"])])
    i8 = np.nonzero(m >= 0.8)[0]; i95 = np.nonzero(m >= 0.95)[0]
    b = (int(i8[0]), float(t[i8[0]])) if len(i8) else None
    tl = (int(i95[0] - i8[0]), float(t[i95[0]] - t[i8[0]])) if len(i8) and len(i95) else None
    return b, tl, (float(m[i8[0]]) if len(i8) else None)


P = {k: {i: phases(R[k][i]) for i in ids} for k in R}
print(f"{'arm':22s} | bulk reached | bulk pushes / s | tail reached (of bulk) | tail pushes / s | entry mass")
for k in R:
    b = [P[k][i][0] for i in ids if P[k][i][0]]; t = [P[k][i][1] for i in ids if P[k][i][1]]; e = [P[k][i][2] for i in ids if P[k][i][2]]
    print(f"{k:22s} | {len(b):2d}/36 | {np.mean([x[0] for x in b]):.2f} / {np.mean([x[1] for x in b]):.2f} | {len(t):2d}/{len(b):2d} | "
          f"{np.mean([x[0] for x in t]):.2f} / {np.mean([x[1] for x in t]):.2f} | {np.mean(e):.3f}")


def paired(a, b, ph):
    s = [i for i in ids if P[a][i][ph] and P[b][i][ph]]
    d = np.array([P[b][i][ph][1] - P[a][i][ph][1] for i in s]); rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    return len(s), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


print("\npaired time difference vs ens (f20_all), per phase, on scenarios where both reach the phase end (s, 95 % bootstrap CI):")
for k in R:
    if k == "ens (f20_all)": continue
    for ph, nm in ((0, "bulk"), (1, "tail")):
        n, m, lo, hi = paired("ens (f20_all)", k, ph); print(f"  {k:22s} {nm}: n {n:2d}  {m:+.2f} [{lo:+.2f}, {hi:+.2f}]")
