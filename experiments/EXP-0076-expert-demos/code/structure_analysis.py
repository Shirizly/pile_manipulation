"""EXP-0076 structure analysis: which structural features of push sequences distinguish fast from slow / failed episodes.
Reads results/runs/*.json (f20_all = main feasible baseline; F*, p1A/B/C pooled where labelled), writes results/structure_analysis.md.
CPU only, Genesis-free.  usage: PYTHONPATH=. python -u experiments/EXP-0076-expert-demos/code/structure_analysis.py
Sections: per-push features, phase structure, solved vs unsolved, fast vs slow, physical time + dynamics-ignoring reorder bound,
pooled-config checks.  The 'Candidate rules' text at the bottom (RULES) was written by hand after reading the tables."""
import json, glob, os, sys, itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from action_time import episode_action_times, trap_time
import torch
RUNS = HERE.parent / "results" / "runs"; OUT = HERE.parent / "results" / "structure_analysis.md"
LO, PITCH, G = T.LO, T.PITCH, T.G


_C = {}


def load(tag):
    if tag not in _C: _C[tag] = json.load(open(RUNS / f"{tag}.json"))["scenarios"]
    return _C[tag]


def pix(x):
    return np.clip(np.rint((np.asarray(x) - LO) / PITCH).astype(int), 0, G - 1)


def inmask(mask, xy):
    xy = np.asarray(xy); return mask[pix(xy[..., 0]), pix(xy[..., 1])] > 0


def analyse(rec, tag):
    """-> (episode dict, [push dicts])."""
    mask = T.goal_mask(rec["goal"]); A = np.asarray(rec["actions"], float).reshape(-1, 4); S = np.asarray(rec["states"], float)
    M = np.asarray(rec["masses"], float); N = len(A)
    ep = dict(tag=tag, id=rec["id"], init=rec["init"], kind=rec["kind"], stratum=f"{rec['init']}-{rec['kind']}", solved=bool(rec["solved"]), pushes=N,
              time=float(sum(rec["push_times"])), rho=rec["goal"]["rho"], mass0=M[0], n=rec["n"])
    in0 = inmask(mask, S[0, :, :2]); ep["n_out0"] = int((~in0).sum())
    dt = distance_transform_edt(mask == 0) * PITCH * 1000                   # mm from goal (0 inside)
    d_out = dt[pix(S[0, :, 0]), pix(S[0, :, 1])]
    ep["mean_dout0"] = float(d_out[~in0].mean()) if (~in0).any() else 0.0
    ep["sum_dout0"] = float(d_out.sum())
    if N == 0:
        return ep, []
    tp = episode_action_times(A.tolist())
    prev = np.vstack([[0, 0], A[:-1, 2:4]]); pushes = []
    for i in range(N):
        d = np.linalg.norm(S[i + 1, :, :2] - S[i, :, :2], axis=1); mv = d > 0.002
        a, b = inmask(mask, S[i, :, :2]), inmask(mask, S[i + 1, :, :2])
        s_in, e_in = bool(inmask(mask, A[i, 0:2])), bool(inmask(mask, A[i, 2:4]))
        pushes.append(dict(tag=tag, id=rec["id"], idx=i + 1, last=(i == N - 1), solved=ep["solved"], init=ep["init"], kind=ep["kind"], stratum=ep["stratum"],
                           length=float(np.linalg.norm(A[i, 2:4] - A[i, 0:2])) * 1000, transit_d=float(np.linalg.norm(A[i, 0:2] - prev[i])) * 1000,
                           t=tp["total"][i], t_transit=tp["transit"][i], t_sweep=tp["sweep"][i], t_fixed=tp["up"][i] + tp["down"][i],
                           mass_before=float(M[i]), gain=float(M[i + 1] - M[i]), moved=int(mv.sum()), moved_in=int((mv & a).sum()),
                           enter=int((~a & b).sum()), leave=int((a & ~b).sum()), start_in=s_in, end_in=e_in,
                           cls=("in>in" if s_in and e_in else "in>out" if s_in else "out>in" if e_in else "out>out"),
                           n_in_before=int(a.sum()), n_in_after=int(b.sum())))
    ep["n_in_final"] = pushes[-1]["n_in_after"]
    P = pushes
    ep.update(mean_len=np.mean([p["length"] for p in P]), mean_transit=np.mean([p["transit_d"] for p in P]), t_transit=sum(p["t_transit"] for p in P),
              t_sweep=sum(p["t_sweep"] for p in P), t_fixed=sum(p["t_fixed"] for p in P), frac_start_in=np.mean([p["start_in"] for p in P]),
              leave_tot=sum(p["leave"] for p in P), enter_tot=sum(p["enter"] for p in P), moved_mean=np.mean([p["moved"] for p in P]),
              first_gain=P[0]["gain"], neg_pushes=sum(p["gain"] < 0 for p in P), tiny_pushes=sum(p["gain"] < 0.02 for p in P),
              mean_gain=(M[-1] - M[0]) / N, final_mass=M[-1])
    # dynamics-ignoring reorder bound: Held-Karp over the same pushes, minimise sum of transit times (start from HOME)
    ep["t_transit_min"], ep["t_transit_greedy"] = reorder_bound(A)
    return ep, pushes


def reorder_bound(A):
    N = len(A); st, en = torch.tensor(A[:, :2]), torch.tensor(A[:, 2:])
    C = trap_time((en[:, None, :] - st[None, :, :]).norm(dim=2)).numpy()          # C[i, j]: finish i then start j
    h = trap_time((st - torch.zeros(1, 2, dtype=st.dtype)).norm(dim=1)).numpy()   # from HOME
    # greedy nearest-start
    left, cur, tot = set(range(N)), None, 0.0
    while left:
        j = min(left, key=lambda j: h[j] if cur is None else C[cur, j]); tot += h[j] if cur is None else C[cur, j]; left.remove(j); cur = j
    if N > 13:
        return float("nan"), tot
    INF = 1e9; dp = np.full((1 << N, N), INF)
    for j in range(N):
        dp[1 << j, j] = h[j]
    for m in range(1, 1 << N):
        row = dp[m]
        if (row >= INF).all(): continue
        cand = (row[:, None] + C).min(0)                       # best cost to reach k from any j in m
        for k in range(N):
            if (m >> k) & 1: continue
            if cand[k] < dp[m | (1 << k), k]: dp[m | (1 << k), k] = cand[k]
    return float(dp[(1 << N) - 1].min()), tot


# ------------------------------------------------------------------ small helpers
def f(x, nd=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def mean(xs, k):
    xs = [x[k] for x in xs]; return float(np.mean(xs)) if len(xs) else float("nan")


def boot_ci(x, nb=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 3: return (float("nan"), float("nan"))
    r = np.random.default_rng(seed); m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(nb)]; return tuple(np.percentile(m, [2.5, 97.5]))


def sp(x, y):
    if len(x) < 5: return "n<5"
    r, p = spearmanr(x, y); return f"{r:+.2f} (p={p:.3f})"


def main():
    L = []; w = L.append
    ids = {}
    EP, PU = {}, {}
    tags = ["f20_all"] + [t for t in ("F0_seed1", "F1_v64big", "F2_ensH2ot", "F3_phaseH2H1", "F4_ensbig", "F5_v64H2ot", "p1A_feas", "p1B_greedy16", "p1C_beam4x8")
                           if (RUNS / f"{t}.json").exists()] + [Path(p).stem for p in sorted(glob.glob(str(RUNS / "E20_*.json")))]
    for t in tags:
        EP[t], PU[t] = [], []
        for rec in load(t).values():
            if rec["n"] != 20 or "states" not in rec or len(rec["states"]) != len(rec["actions"]) + 1: continue
            e, p = analyse(rec, t); EP[t].append(e); PU[t] += p
    E, P = EP["f20_all"], PU["f20_all"]
    ES = [e for e in E if e["solved"]]; EU = [e for e in E if not e["solved"]]
    w("# EXP-0076 structure analysis (what distinguishes fast from slow / failed solutions)\n")
    w("Generated by `code/structure_analysis.py` (CPU only). Main data: `f20_all` = feasible planner (ens2bal, H=1, CEM+GD20, B=1,k=1), 270 n20 scenarios, one seed; "
      "episodes stop at the first push reaching in-goal mass >= 0.95 (solved) or after 12 pushes (unsolved). Physical time = EXP-0070 action_time "
      "(lift 0.14 s + transit + descent 0.14 s + sweep; first push from HOME without lift). 'Cube in goal' = cube centre's pixel inside the mask. "
      "'Moved' = centre displaced > 2 mm. Gain = change in in-goal MASS fraction (soft scoring raster). One seed, one config: all numbers are descriptive, "
      "CIs are bootstrap over episodes and ignore planner randomness.\n")
    w("## 0. Data\n")
    w(table(["run", "role", "episodes", "solved", "mean pushes (solved)", "mean time s (solved)"],
            [[t, ("main feasible" if t == "f20_all" else "expert beam B4xk8" if t.startswith("p1C") else "greedy-verify k16" if t.startswith("p1B") else "feasible ablation/pilot"),
              len(EP[t]), sum(e["solved"] for e in EP[t]), f(mean([e for e in EP[t] if e["solved"]], "pushes")), f(mean([e for e in EP[t] if e["solved"]], "time"))] for t in tags]))
    w(f"Per-push records in f20_all: {len(P)} pushes ({len(ES)} solved episodes, {len(EU)} unsolved episodes of 12 pushes each). Unsolved episodes are censored at 12 pushes. "
      "No E20_* expert runs existed when this was run (the pooled expert evidence is only p1C, 18 scenarios).\n")

    # ---- 1 per-push
    w("## 1. Per-push features (f20_all)\n")
    def feat_row(name, ps):
        return [name, len(ps), f(mean(ps, "length"), 1), f(mean(ps, "transit_d"), 1), f(mean(ps, "t"), 2), f(mean(ps, "t_transit"), 2), f(mean(ps, "t_sweep"), 2),
                f(mean(ps, "t_fixed"), 2), f(mean(ps, "gain"), 3), f(mean(ps, "moved"), 1), f(mean(ps, "enter"), 2), f(mean(ps, "leave"), 2), f(np.mean([p["start_in"] for p in ps]) if ps else float("nan"), 2)]
    H1 = ["group", "pushes", "length mm", "transit d mm", "time s", "transit s", "sweep s", "lift+descent s", "mass gain", "cubes moved", "enter", "leave", "starts in goal"]
    rows = [feat_row("all", P), feat_row("solved episodes", [p for p in P if p["solved"]]), feat_row("unsolved episodes", [p for p in P if not p["solved"]])]
    for init in T.INITS: rows.append(feat_row(init, [p for p in P if p["init"] == init]))
    for k in ("disc", "square", "letter"): rows.append(feat_row(k, [p for p in P if p["kind"] == k]))
    w(table(H1, rows))
    w("Push class by (start in goal, end in goal) (all pushes of f20_all):\n")
    rows = []
    for c in ("out>out", "out>in", "in>in", "in>out"):
        ps = [p for p in P if p["cls"] == c]; g = np.array([p["gain"] for p in ps])
        rows.append([c, len(ps), f(len(ps) / len(P), 2), f(g.mean(), 3) if len(g) else "-", f(np.median(g), 3) if len(g) else "-", f(np.mean(g < 0.02), 2) if len(g) else "-",
                     f(mean(ps, "length"), 1) if ps else "-", f(mean(ps, "t"), 2) if ps else "-", f(g.sum() / sum(p["t"] for p in ps), 3) if ps else "-"])
    w(table(["class", "n", "share", "mean gain", "median gain", "frac gain<0.02", "length mm", "time s", "gain / s"], rows))
    w("Correlation of per-push gain with features (Spearman, all pushes): " + "; ".join(
        f"{k}: {sp([p[k] for p in P], [p['gain'] for p in P])}" for k in ("length", "transit_d", "moved", "mass_before", "idx")) + "\n")

    # ---- 2 phases
    w("## 2. Phase structure (f20_all)\n")
    def phase_rows(keyname, groups):
        rows = []
        for name, sel in groups:
            ps = [p for p in P if sel(p)]
            if not ps: continue
            g = np.array([p["gain"] for p in ps])
            rows.append([name, len(ps), f(g.mean(), 3), f(np.median(g), 3), f(np.mean(g < 0.02), 2), f(np.mean(g <= 0), 2), f(mean(ps, "t"), 2), f(g.sum() / sum(p["t"] for p in ps), 3),
                         f(mean(ps, "leave"), 2), f(mean(ps, "enter"), 2), f(mean(ps, "moved"), 1)])
        return table([keyname, "pushes", "mean gain", "median gain", "frac gain<0.02", "frac gain<=0", "time s", "gain/s", "leave", "enter", "moved"], rows)
    w("By push index (all episodes; later indices are dominated by unsolved episodes):\n")
    w(phase_rows("push #", [(str(i), lambda p, i=i: p["idx"] == i) for i in range(1, 9)] + [("9-12", lambda p: p["idx"] >= 9)]))
    w("By current mass before the push (bins in the brief; episodes end at 0.95):\n")
    bins = [(0, .5), (.5, .8), (.8, .9), (.9, .95)]
    w(phase_rows("mass before", [(f"{a}-{b}", lambda p, a=a, b=b: a <= p["mass_before"] < b) for a, b in bins]))
    w("Same, solved episodes only / unsolved episodes only:\n")
    w(phase_rows("solved: mass before", [(f"{a}-{b}", lambda p, a=a, b=b: p["solved"] and a <= p["mass_before"] < b) for a, b in bins]))
    w(phase_rows("unsolved: mass before", [(f"{a}-{b}", lambda p, a=a, b=b: (not p["solved"]) and a <= p["mass_before"] < b) for a, b in bins]))
    # time spent in each bin per episode
    rows = []
    for nm, EE in (("solved", ES), ("unsolved", EU)):
        tot = sum(e["time"] for e in EE); r = [nm, len(EE), f(tot / len(EE), 1)]
        for a, b in bins:
            r.append(f(sum(p["t"] for p in P if (p["solved"] == (nm == "solved")) and a <= p["mass_before"] < b) / tot, 2))
        rows.append(r)
    w("Share of episode physical time spent in each mass bin:\n")
    w(table(["group", "episodes", "mean time s"] + [f"{a}-{b}" for a, b in bins], rows))
    # last pushes of solved
    sol = [p for p in P if p["solved"]]
    last = [p for p in sol if p["last"]]; lastbut = [p for p in sol if not p["last"]]
    w(f"Solved episodes: the finishing push gains {mean(last, 'gain'):.3f} on average (it must cross the 0.95 line); the pushes before it gain {mean(lastbut, 'gain'):.3f}. "
      f"Pushes with gain < 0.02 take {100*sum(p['t'] for p in P if p['gain']<0.02)/sum(p['t'] for p in P):.0f}% of all physical time in f20_all "
      f"({100*sum(p['t'] for p in sol if p['gain']<0.02)/sum(p['t'] for p in sol):.0f}% in solved episodes, {100*sum(p['t'] for p in P if (not p['solved']) and p['gain']<0.02)/sum(p['t'] for p in P if not p['solved']):.0f}% in unsolved ones).\n")
    # unsolved: stall
    w("Unsolved episodes: mass after push k (mean), showing the stall:\n")
    rows = []
    for nm, EE in (("solved", ES), ("unsolved", EU)):
        r = [nm]
        for i in (0, 2, 4, 6, 8, 10, 12):
            ms = [load("f20_all")[e["id"]]["masses"] for e in EE]; vals = [m[i] for m in ms if len(m) > i]
            r.append(f(np.mean(vals), 3) if vals else "-")
        rows.append(r)
    w(table(["group", "m0", "m2", "m4", "m6", "m8", "m10", "m12"], rows))

    # ---- 3 solved vs unsolved
    w("## 3. Solved vs unsolved (f20_all)\n")
    rows = []
    for s in sorted({e["stratum"] for e in E}):
        a = [e for e in ES if e["stratum"] == s]; b = [e for e in EU if e["stratum"] == s]
        rows.append([s, len(a) + len(b), f"{len(a)}/{len(a)+len(b)}", f(mean(a, "rho")), f(mean(b, "rho")), f(mean(a, "mass0")), f(mean(b, "mass0")), f(mean(a, "mean_dout0"), 1), f(mean(b, "mean_dout0"), 1),
                     f(mean(a, "n_out0"), 1), f(mean(b, "n_out0"), 1)])
    w(table(["stratum", "n", "solved", "rho solved", "rho unsol.", "mass0 solved", "mass0 unsol.", "mean dist of outside cubes mm (solved)", "(unsol.)", "cubes outside goal at start (solved)", "(unsol.)"], rows))
    w("Solve rate by rho bin x init (solved/total):\n")
    rb = [(0.15, .25), (.25, .35), (.35, .45), (.45, .61)]; rows = []
    for a, b in rb:
        r = [f"{a}-{b}"]
        for init in T.INITS:
            EE = [e for e in E if e["init"] == init and a <= e["rho"] < b]; r.append(f"{sum(e['solved'] for e in EE)}/{len(EE)}")
        EE = [e for e in E if a <= e["rho"] < b]; r.append(f"{sum(e['solved'] for e in EE)}/{len(EE)}"); rows.append(r)
    w(table(["rho"] + list(T.INITS) + ["all"], rows))
    # within-stratum rank correlations with solved
    def strat_resid(EE, key):
        """rank within stratum (demeaned) so pooled correlations are within-stratum."""
        out = {}
        for s in {e["stratum"] for e in EE}:
            idx = [i for i, e in enumerate(EE) if e["stratum"] == s]
            v = np.array([EE[i][key] for i in idx], float); rk = v.argsort().argsort().astype(float) if len(v) else v
            from scipy.stats import rankdata
            rk = rankdata(v); rk = (rk - rk.mean()) / max(1, len(v))
            for i, r in zip(idx, rk): out[i] = r
        return np.array([out[i] for i in range(len(EE))])
    def wcorr(EE, key, yk):
        x = strat_resid(EE, key); y = strat_resid(EE, yk)
        if len(EE) < 8 or x.std() == 0 or y.std() == 0: return "n/a"
        r, p = spearmanr(x, y); return f"{r:+.2f} (p={p:.3f})"
    w("Within-stratum Spearman correlation (ranks demeaned per init x kind stratum, 270 episodes) with the solved flag:\n")
    for e in E: e["solved_i"] = float(e["solved"])
    feats = ["rho", "mass0", "n_out0", "mean_dout0", "sum_dout0"]
    w(table(["feature"] + ["corr with solved"], [[k, wcorr(E, k, "solved_i")] for k in feats]))
    w("Process features of unsolved vs solved episodes (episode level; for unsolved these cover 12 pushes, so counts are not like-for-like — use the per-push means):\n")
    rows = []
    for nm, EE in (("solved", ES), ("unsolved", EU)):
        rows.append([nm, len(EE), f(mean(EE, "mean_len"), 1), f(mean(EE, "mean_transit"), 1), f(mean(EE, "moved_mean"), 1), f(mean(EE, "frac_start_in"), 2),
                     f(np.mean([e["leave_tot"] / e["pushes"] for e in EE]), 2), f(np.mean([e["enter_tot"] / e["pushes"] for e in EE]), 2), f(np.mean([e["neg_pushes"] / e["pushes"] for e in EE]), 2),
                     f(np.mean([e["tiny_pushes"] / e["pushes"] for e in EE]), 2), f(np.mean([e["time"] / e["pushes"] for e in EE]), 2)])
    w(table(["group", "n", "mean len mm", "mean transit mm", "cubes moved/push", "frac pushes starting in goal", "leave/push", "enter/push", "frac gain<0", "frac gain<0.02", "time per push s"], rows))
    w("Unsolved episodes: final in-goal cube count vs n (cube-centre count) and final mass:\n")
    w(f"mean final mass {mean(EU,'final_mass'):.3f} (median {np.median([e['final_mass'] for e in EU]):.3f}); share of unsolved with final mass >= 0.90: "
      f"{np.mean([e['final_mass']>=0.9 for e in EU]):.2f}; >=0.80: {np.mean([e['final_mass']>=0.8 for e in EU]):.2f}. Mean centre-in-goal cubes at end: {mean(EU,'n_in_final'):.1f} of 20 "
      f"(solved: {mean(ES,'n_in_final'):.1f}).\n")

    # ---- 4 fast vs slow
    w("## 4. Fast vs slow within solved episodes (f20_all)\n")
    w("Fast = pushes <= stratum median, slow = pushes > stratum median (init x kind stratum medians; solved episodes only, so conditioned on success -> survivor bias: slow-but-solved episodes exclude the failures).\n")
    rows = []; fast, slow = [], []
    for s in sorted({e["stratum"] for e in ES}):
        a = [e for e in ES if e["stratum"] == s]; med = np.median([e["pushes"] for e in a]); fa = [e for e in a if e["pushes"] <= med]; sl = [e for e in a if e["pushes"] > med]
        fast += fa; slow += sl
        rows.append([s, len(a), f(med, 1), len(fa), len(sl), f(mean(fa, "time"), 1), f(mean(sl, "time"), 1), f(mean(fa, "rho")), f(mean(sl, "rho")), f(mean(fa, "mass0")), f(mean(sl, "mass0")),
                     f(mean(fa, "n_out0"), 1), f(mean(sl, "n_out0"), 1)])
    w(table(["stratum", "solved n", "median pushes", "n fast", "n slow", "time fast s", "time slow s", "rho fast", "rho slow", "mass0 fast", "mass0 slow", "n_out0 fast", "n_out0 slow"], rows))
    rows = []
    for k, nm, nd in (("pushes", "pushes", 1), ("time", "time s", 1), ("mean_len", "mean push length mm", 1), ("mean_transit", "mean transit mm", 1), ("moved_mean", "cubes moved / push", 1),
                      ("frac_start_in", "frac pushes starting in goal", 2), ("first_gain", "first-push gain", 3), ("mean_gain", "mean gain / push", 3), ("neg_pushes", "pushes with gain<0", 2), ("tiny_pushes", "pushes with gain<0.02", 2),
                      ("leave_tot", "cubes leaving goal (episode)", 2), ("rho", "rho", 3), ("mass0", "mass0", 3), ("n_out0", "cubes outside goal at start", 1), ("mean_dout0", "mean dist of outside cubes mm", 1)):
        a, b = [e[k] for e in fast], [e[k] for e in slow]
        rows.append([nm, f(np.mean(a), nd + 1 if nd < 3 else nd), f(np.mean(b), nd + 1 if nd < 3 else nd), wcorr(ES, k, "pushes"), wcorr(ES, k, "time")])
    w(table(["feature", f"fast (n={len(fast)})", f"slow (n={len(slow)})", "within-stratum Spearman vs pushes", "vs time s"], rows))
    w("Time per push vs number of pushes (solved, f20_all) — is a many-push episode slower per push?\n")
    rows = []
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 12)):
        a = [e for e in ES if lo <= e["pushes"] <= hi]
        rows.append([f"{lo}-{hi}", len(a), f(mean(a, "time"), 1), f(np.mean([e["time"] / e["pushes"] for e in a]), 2), f(mean(a, "mean_len"), 1), f(mean(a, "mean_transit"), 1),
                     f(np.mean([e["t_transit"] / e["time"] for e in a]), 2), f(np.mean([e["t_fixed"] / e["time"] for e in a]), 2)])
    w(table(["pushes", "n", "mean time s", "time / push s", "mean length mm", "mean transit mm", "transit share of time", "lift+descent share"], rows))
    w(f"Corr(pushes, time) over solved episodes: Spearman {sp([e['pushes'] for e in ES],[e['time'] for e in ES])}; "
      f"slope of time on pushes by OLS: {np.polyfit([e['pushes'] for e in ES],[e['time'] for e in ES],1)[0]:.2f} s/push (intercept {np.polyfit([e['pushes'] for e in ES],[e['time'] for e in ES],1)[1]:.2f}).\n")

    # ---- 4b decomposition of slow episodes
    w("Decomposition of the extra pushes of slow solved episodes (f20_all; per-episode counts by push class and by gain):\n")
    def cnt(EE, fn):
        ids = {e["id"] for e in EE}; return sum(1 for p in P if p["id"] in ids and fn(p)) / len(EE)
    rows = []
    for nm, fn in (("all pushes", lambda p: True), ("out>in (deliveries)", lambda p: p["cls"] == "out>in"), ("out>out", lambda p: p["cls"] == "out>out"),
                   ("gain >= 0.02", lambda p: p["gain"] >= 0.02), ("gain < 0.02", lambda p: p["gain"] < 0.02), ("gain >= 0.05", lambda p: p["gain"] >= 0.05),
                   ("finishing push", lambda p: p["last"])):
        a, b = cnt(fast, fn), cnt(slow, fn); rows.append([nm, f(a, 2), f(b, 2), f(b - a, 2)])
    w(table(["pushes per episode", "fast", "slow", "slow - fast"], rows))
    w("Out>out pushes in solved episodes by mass bin (do they pay off later? share of such pushes with gain<0.02 and mean gain):\n")
    rows = []
    for a, b in bins:
        ps = [p for p in P if p["solved"] and p["cls"] == "out>out" and a <= p["mass_before"] < b]; qs = [p for p in P if p["solved"] and p["cls"] == "out>in" and a <= p["mass_before"] < b]
        rows.append([f"{a}-{b}", len(ps), f(mean(ps, "gain"), 3), f(np.mean([p["gain"] < 0.02 for p in ps]), 2) if ps else "-", len(qs), f(mean(qs, "gain"), 3)])
    w(table(["mass before", "out>out n", "mean gain", "frac gain<0.02", "out>in n", "mean gain"], rows))

    # ---- 5 time
    w("## 5. Physical time: transit share and reorder bound (f20_all)\n")
    def tshare(EE, nm):
        tt = sum(e["time"] for e in EE); return [nm, len(EE), f(tt / len(EE), 1), f(sum(e["t_transit"] for e in EE) / tt, 3), f(sum(e["t_sweep"] for e in EE) / tt, 3), f(sum(e["t_fixed"] for e in EE) / tt, 3)]
    rows = [tshare(ES, "solved"), tshare(EU, "unsolved (12 pushes)"), tshare(E, "all")]
    for init in T.INITS: rows.append(tshare([e for e in ES if e["init"] == init], f"solved {init}"))
    for k in ("disc", "square", "letter"): rows.append(tshare([e for e in ES if e["kind"] == k], f"solved {k}"))
    w(table(["group", "n", "mean time s", "transit share", "sweep share", "lift+descent share"], rows))
    w("**Dynamics-ignoring bound**: keep the same pushes (same start/end, same set), re-order them to minimise total transit time (exact Held-Karp over the 4-coordinate pushes; "
      "start from HOME). This ignores that pushes depend on each other (a push's effect depends on the pile state left by the previous ones), so it is an UPPER bound on what "
      "any reordering could save, not an achievable number. 'Greedy' = nearest-start-next.\n")
    rows = []
    for nm, EE in (("solved", ES), ("unsolved", EU)) + tuple((f"solved {i}", [e for e in ES if e["init"] == i]) for i in T.INITS):
        EE = [e for e in EE if not np.isnan(e["t_transit_min"])]
        sav = np.array([e["t_transit"] - e["t_transit_min"] for e in EE]); gr = np.array([e["t_transit"] - e["t_transit_greedy"] for e in EE]); tt = np.array([e["time"] for e in EE])
        ci = boot_ci(sav)
        rows.append([nm, len(EE), f(tt.mean(), 2), f(np.mean([e["t_transit"] for e in EE]), 2), f(np.mean([e["t_transit_min"] for e in EE]), 2), f(sav.mean(), 2) + f" [{ci[0]:.2f},{ci[1]:.2f}]",
                     f(100 * sav.sum() / tt.sum(), 1) + "%", f(gr.mean(), 2), f(np.mean(sav <= 1e-9), 2)])
    w(table(["group", "n", "mean time s", "transit s", "min transit s (any order)", "max saving s [95% CI]", "saving / time", "greedy saving s", "frac already optimal order"], rows))
    # saving vs pushes
    rows = []
    for lo, hi in ((2, 3), (4, 5), (6, 7), (8, 9), (10, 12)):
        EE = [e for e in ES if lo <= e["pushes"] <= hi and not np.isnan(e["t_transit_min"])]
        if EE: rows.append([f"{lo}-{hi}", len(EE), f(np.mean([e["time"] for e in EE]), 1), f(np.mean([e["t_transit"] - e["t_transit_min"] for e in EE]), 2), f(100 * sum(e["t_transit"] - e["t_transit_min"] for e in EE) / sum(e["time"] for e in EE), 1) + "%"])
    w("Reorder-bound saving by episode length (solved):\n"); w(table(["pushes", "n", "mean time s", "max saving s", "saving / time"], rows))

    # ---- 6 pooled
    w("## 6. Pooled-config checks (labelled)\n")
    w("Within-scenario fixed effects: for scenarios solved by >= 2 of the pooled runs {f20_all, F0..F5, p1A, p1B, p1C} (different planner configs / seeds on the same scenario), "
      "does the solution with more pushes take longer? Pooled across configs, so planners differ in more than push count.\n")
    byid = {}
    for t in tags:
        for e in EP[t]:
            if e["solved"]: byid.setdefault(e["id"], []).append(e)
    pairs = []; dxs, dys = [], []
    for sid, es in byid.items():
        if len(es) < 2: continue
        px = np.array([e["pushes"] for e in es], float); ty = np.array([e["time"] for e in es], float)
        dxs += list(px - px.mean()); dys += list(ty - ty.mean())
    dxs, dys = np.array(dxs), np.array(dys)
    nsc = sum(len(es) >= 2 for es in byid.values())
    if nsc >= 3:
        b = (dxs * dys).sum() / (dxs ** 2).sum(); r = spearmanr(dxs, dys)[0]
        w(f"{nsc} scenarios with >= 2 solved runs, {len(dxs)} solutions: within-scenario slope of time on pushes = {b:.2f} s/push, Spearman of demeaned values {r:+.2f}.\n")
    # fast/slow across configs same scenario: best vs worst
    rows = []
    gaps = []
    for sid, es in byid.items():
        if len(es) < 2: continue
        bt = min(es, key=lambda e: e["time"]); wt = max(es, key=lambda e: e["time"])
        gaps.append((wt["time"] - bt["time"], wt["pushes"] - bt["pushes"], wt["t_transit"] - bt["t_transit"], np.mean([e["time"] for e in es])))
    if gaps:
        g = np.array(gaps)
        w(f"Across runs on the same scenario (n={len(g)}): slowest solution minus fastest = {g[:,0].mean():.2f} s on average (mean time {g[:,3].mean():.2f} s), of which "
          f"{g[:,1].mean():+.2f} pushes more and {g[:,2].mean():+.2f} s more transit; fastest solution had fewer pushes than the slowest in {np.mean(g[:,1]>0):.2f} of scenarios, "
          f"equal pushes in {np.mean(g[:,1]==0):.2f}, MORE pushes in {np.mean(g[:,1]<0):.2f}.\n")
    # expert vs feasible
    if "p1C_beam4x8" in EP and "p1A_feas" in EP:
        w("Expert beam (p1C, B=4 x k=8, true-state lookahead) vs feasible (p1A, same planner, no lookahead) on the 18 pilot scenarios (n small):\n")
        rows = []
        for t in ("p1A_feas", "p1B_greedy16", "p1C_beam4x8"):
            pp = PU[t]; ee = [e for e in EP[t] if e["solved"]]
            rows.append([t, len(EP[t]), len(ee), f(mean(ee, "pushes"), 2), f(mean(ee, "time"), 2), f(mean(pp, "length"), 1), f(mean(pp, "transit_d"), 1), f(mean(pp, "t_transit"), 2), f(mean(pp, "gain"), 3),
                         f(mean(pp, "moved"), 1), f(mean(pp, "leave"), 2), f(np.mean([p["start_in"] for p in pp]), 2), f(np.mean([p["gain"] < 0.02 for p in pp]), 2)])
        w(table(["run", "episodes", "solved", "pushes", "time s", "length mm", "transit mm", "transit s/push", "gain/push", "moved", "leave/push", "starts in goal", "frac gain<0.02"], rows))
        # same scenarios solved by both
        a = {e["id"]: e for e in EP["p1A_feas"] if e["solved"]}; c = {e["id"]: e for e in EP["p1C_beam4x8"] if e["solved"]}
        both = sorted(set(a) & set(c))
        if both:
            d = np.array([[a[i]["pushes"], c[i]["pushes"], a[i]["time"], c[i]["time"], a[i]["t_transit"], c[i]["t_transit"]] for i in both])
            w(f"On the {len(both)} scenarios solved by both: beam - feasible = {np.mean(d[:,1]-d[:,0]):+.2f} pushes, {np.mean(d[:,3]-d[:,2]):+.2f} s total, {np.mean(d[:,5]-d[:,4]):+.2f} s transit.\n")
    # pooled per-push (all ablation configs) check of main class table
    allP = [p for t in tags if t != "f20_all" and not t.startswith("p1C") for p in PU[t]]
    if allP:
        w(f"Cross-check of the push-class table on the pooled ablation/pilot runs (excluding f20_all and the beam), {len(allP)} pushes:\n")
        rows = []
        for c in ("out>out", "out>in", "in>in", "in>out"):
            ps = [p for p in allP if p["cls"] == c]; g = np.array([p["gain"] for p in ps])
            rows.append([c, len(ps), f(len(ps) / len(allP), 2), f(g.mean(), 3), f(np.mean(g < 0.02), 2)])
        w(table(["class", "n", "share", "mean gain", "frac gain<0.02"], rows))
    w(RULES)
    OUT.write_text("\n".join(L)); print("\n".join(L))


RULES = """
## 7. Candidate rules (empirical, descriptive; f20_all unless labelled; one planner config, one seed, n20 only)

1. **The 0.8-0.95 tail eats the time.** In solved episodes 40% of physical time (0.23 + 0.17) is spent at mass 0.8-0.95, i.e. for the last 0.15 of mass.
   Mean gain per push by mass bin 0.250 / 0.099 / 0.037 / 0.015 (gain/s 0.194 / 0.081 / 0.035 / 0.016); pushes cost nearly the same
   (1.29 s -> 0.94 s). Rule: a pipeline should treat the tail as a separate, straggler-targeting phase (tail pushes should be chosen
   by the explicit 'cubes still outside' set), not as 'more of the same'. Caveat: the 0.95 threshold is on the soft raster, so some tail gain is the
   raster's blur, not a real straggler; the bin structure is partly mechanical (gain is bounded by 0.95 - mass).
2. **Stall signature separates failures from slow successes.** Pushes with gain < 0.02: 9% of pushes in solved episodes vs 50% in unsolved; they use 13% vs
   46% of episode time. Unsolved episodes are near misses (final mass mean 0.88, 64% >= 0.90, 95% >= 0.80; 18.4 of 20 cube centres in the goal at the end).
   Rule: two consecutive pushes with gain < 0.02 (or any gain <= 0 after mass 0.8) predict failure within the budget; branch/re-plan (expert) or
   drop the episode as a demonstration. Caveat: unsolved episodes are censored at 12 pushes; no held-out test of the stall detector was made.
3. **Slow = extra pushes that end outside the goal, not extra deliveries.** Slow solved episodes (pushes > stratum median) have 8.15 vs 4.66 pushes; of the
   +3.49 extra pushes, +2.57 are out>out (start and end of the blade outside the goal mask) and +1.58 have gain < 0.02, only +0.89 are out>in deliveries.
   Push class out>in: mean gain 0.144, 0.121 gain/s; out>out: 0.040, 0.037 gain/s (52% below 0.02). Pooled ablation runs reproduce this (0.140 vs 0.044).
   Rule: prefer candidates whose sweep ends inside the goal (or drags a straggler across the boundary); penalise tail pushes that end outside.
   Caveat: classification uses the blade start/end point, not cubes; early out>out pushes are useful gathering (gain 0.29 below mass 0.5, n = 22).
4. **Difficulty is set before the first push (rho, how far cubes are), not by the opening.** Solve rate by rho: 85% (rho < 0.25), 70%, 76%, 41% (rho >= 0.45, 22/54).
   Within init x kind stratum, Spearman with solved: rho -0.39, sum of outside-cube distance -0.23, mean outside distance -0.20, mass0 +0.17;
   with pushes (solved): rho +0.57, outside distance +0.45, mass0 -0.37. First-push gain is identical in fast and slow episodes (0.349 vs 0.352).
   Rule: predict a push budget (and whether to bother) from rho, number of cubes outside and their mean distance to the goal; piled starts with a
   large goal (rho > 0.5, mass0 ~ 0.02) are the failing corner (piled disc/square failures have mass0 0.02 and rho 0.53-0.56, 3-4 failures each of 30).
   Caveat: rho is confounded with goal kind (letters have rho 0.15-0.25) and with init; the solved/unsolved split uses one planner.
5. **More pushes = slower, essentially linearly; nothing here contradicts it.** OLS 1.25 s per extra push across solved episodes (Spearman 0.98); within
   the same scenario across 47 scenarios / 236 solutions of mixed configs the slope is 0.99 s/push, and the fastest solution never had MORE pushes than
   the slowest (0/47; fewer in 83%, equal in 17%). The nuance: time per push is not constant (0.93 s at 1-2 pushes, 1.21-1.24 s at 5-8) because
   late pushes are shorter, yet a near-useless late push still costs ~0.8-0.9 s (mass 0.9-0.95 bin: 0.94 s mean) since lift + descent (0.27 s) and
   transit (0.5 s) are paid whatever the sweep. Rule: minimise push count first; do not expect cheap 'small cleanup' pushes. Caveat: configs pooled in
   the within-scenario check differ in planner, not only in push count.
6. **Only a third of time is pushing.** Solved: transit 45%, sweep 33%, lift+descent 22% (stable 43-46% transit across 1-12 pushes). Mean transit
   distance (54 mm) exceeds mean sweep length (39 mm). Rule: transit is the largest single component; start-point proximity to the previous
   end matters about as much as the sweep.
7. **Reordering bound (dynamics-ignoring, UPPER bound, not achievable as such).** Optimal re-ordering of the same pushes saves 0.64 s [0.55, 0.72] =
   9.0% of solved time (unsolved: 1.40 s, 10.9%; 2-3 pushes 2.7%, >= 8 pushes ~10%); nearest-start greedy gets about half (0.32 s); only 15% of episodes
   are already optimally ordered. Realistic gain after respecting push dependencies is smaller; a transit term in the planner objective (not reordering)
   is the usable form. Caveat: pushes are pile-state dependent; a re-ordered sequence would give different cube outcomes.
8. **Lookahead on true states shortens episodes and removes stall pushes (labelled: 18 pilot scenarios, one seed).** Beam B4xk8 vs same-planner feasible
   on the 14 jointly solved: -0.79 pushes, -0.86 s total, -0.41 s transit; beam gain/push 0.128 vs 0.103, share of pushes with gain < 0.02 12% vs 25%,
   cubes leaving 0.07 vs 0.12 per push, 18/18 vs 14/18 solved. Caveat: tiny sample, the beam also solves the harder scenarios, so the overall mean pushes
   (5.78 vs 5.71) looks equal while the matched comparison does not.
9. **Cubes leaving the goal is a minor, localised cost.** 0.09 cubes leave per push, in>out pushes are 1% of pushes, in>in 2% (3% of pushes start in the goal).
   But slow solved episodes lose 1.08 cubes to it vs 0.20 in fast ones (Spearman with pushes +0.40). Rule: a mild leave-penalty is enough; the main
   waste is out>out pushes (rule 3). Caveat: the cube-centre in/out test uses the pixel of the cube centre; 'leave' may include boundary jitter.
10. **Fast episodes move more cubes per push** (7.97 vs 6.15; within-stratum Spearman with pushes -0.38) with slightly longer pushes (40.5 vs 38.7 mm,
   -0.19). Rule: among candidate pushes with similar predicted gain, prefer the one that sweeps more cubes. Caveat: moved-count is highly collinear with mass gain
   (Spearman 0.75 per push), so this is not independent evidence for a separate mechanism.
"""
if __name__ == "__main__":
    main()
