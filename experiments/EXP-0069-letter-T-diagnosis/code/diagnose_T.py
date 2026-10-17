"""EXP-0069: zero-simulation diagnosis of why letter_T behaves differently in legal closed loop.
Reads EXP-0066 per-episode cells (results/*.json, 32 scatter starts x 4 letters, 16 pushes, states recorded)
and EXP-0039 RUN-0002 cem/rank episodes (8 letters x 16 starts), recomputes in-goal mass per push from the
recorded particle states with the scoring splat (occ_for_scoring + mass_in_region, the EXP-0066 formula),
and tests H-a (headroom), H-b (geometry / legality / push direction), H-c (capacity), H-d (model-specific),
H-e (mask artefact). Writes results/diagnosis.json, results/diagnosis.md, results/figures/*.png.
CPU only. Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -u experiments/EXP-0069-letter-T-diagnosis/code/diagnose_T.py
"""
import json, sys, math
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common.goals import letter_mask, mass_in_region  # noqa: E402
from simple_mpc.adapters import occ_for_scoring, OCC_GRID, OCC_BOUNDS  # noqa: E402

HERE = REPO / "experiments/EXP-0069-letter-T-diagnosis"; R = HERE / "results"; FIG = R / "figures"
R66 = REPO / "experiments/EXP-0066-closed-loop-power-study/results"
R39 = REPO / "experiments/EXP-0039-closed-loop-metric-validity/results/run0002"
VS = json.loads((REPO / "experiments/EXP-0046-goal-ceiling/results/vstar.json").read_text())
K0, K1 = 4, 16
LETTERS4 = ["letter_O", "letter_T", "letter_S", "letter_X"]
LETTERS8 = ["letter_O", "letter_T", "letter_S", "letter_L", "letter_X", "letter_Z", "letter_C", "letter_H"]
CELLS66 = {"nfd": ["nfd_s0_ps0", "nfd_s1_ps0", "nfd_s2_ps0"], "lin": ["lin_orig_ps0", "lin_b1_ps0", "lin_b2_ps0"],
           "broad": ["broad_s0_ps0", "broad_s1_ps0"]}
PX_MM = (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) / OCC_GRID * 1000.0
XY_MARGIN = 0.004
rng = np.random.default_rng(0)


def mask(g):
    return letter_mask(g[-1], OCC_GRID, OCC_GRID)


def mask_stats(g):
    m = mask(g); px = int(m.sum()); rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    ii, jj = np.indices(m.shape); cx = float((ii * m).sum() / px); cy = float((jj * m).sum() / px)
    edt = distance_transform_edt(m)
    # distance of mask pixels to the tray edge (px); the legal push box is XY_MARGIN inside the tray
    edge = np.minimum(np.minimum(ii, OCC_GRID - 1 - ii), np.minimum(jj, OCC_GRID - 1 - jj))
    return dict(mask_px=px, bbox_x_px=[int(rows.min()), int(rows.max())], bbox_y_px=[int(cols.min()), int(cols.max())],
                bbox_mm=dict(x=[(rows.min() - 31.5) * PX_MM, (rows.max() - 31.5) * PX_MM], y=[(cols.min() - 31.5) * PX_MM, (cols.max() - 31.5) * PX_MM]),
                centroid_mm=[(cx - 31.5) * PX_MM, (cy - 31.5) * PX_MM], stroke_width_mm_max=float(2 * edt.max() * PX_MM),
                min_edge_dist_px=int(edge[m].min()), frac_px_within_6px_of_edge=float((edge[m] <= 6).mean()),
                sym_flip_x_iou=float((m & m[::-1]).sum() / (m | m[::-1]).sum()), sym_flip_y_iou=float((m & m[:, ::-1]).sum() / (m | m[:, ::-1]).sum()),
                capacity=VS[g]["capacity"], mass_frac_best_placement=VS[g]["mass_frac_best_placement"], vstar=VS[g]["vstar_map_value"],
                n_inside_vstar=VS[g]["n_inside_vstar"])


# letter_T sub-regions: crossbar = mask columns (world y) where the mask is >= 10 px wide in x; stem = the rest
def t_parts():
    m = mask("letter_T"); width = m.sum(0); bar_cols = width >= 10
    bar = m & bar_cols[None, :]; stem = m & ~bar_cols[None, :]
    return bar, stem


def load_episodes(f):
    d = json.loads(f.read_text()); out = []
    for e in d["episodes"]:
        if not e.get("complete") or not e.get("states"):
            continue
        S = torch.tensor(e["states"]).float(); occ = occ_for_scoring(S); tot = occ.reshape(len(S), -1).sum(1)
        m = torch.from_numpy(mask(e["goal"]))
        mf = (mass_in_region(occ, m) / tot).numpy(); rel = mf / VS[e["goal"]]["mass_frac_best_placement"]
        rec = dict(model=e["model"], planner=e["planner"], goal=e["goal"], start=e["start"], rel=rel.tolist(), mf=mf.tolist(),
                   score=float(rel[K0:K1 + 1].mean()), k0=float(rel[0]), k4=float(rel[K0]), k16=float(rel[K1]),
                   values=e["values"], pred_dv=e["pred_dv"], true_dv=e["true_dv"], actions=e["actions"],
                   legal_ok=e.get("legal_ok", []), legal_fallback=e.get("legal_fallback", []), legal_shift_m=e.get("legal_shift_m", []),
                   occ0=occ[0].numpy(), occ16=occ[K1].numpy(), states=S.numpy())
        if e["goal"] == "letter_T":
            bar, stem = t_parts()
            rec["mf_bar"] = (mass_in_region(occ, torch.from_numpy(bar)) / tot).numpy().tolist()
            rec["mf_stem"] = (mass_in_region(occ, torch.from_numpy(stem)) / tot).numpy().tolist()
        out.append(rec)
    return out


def q(x):
    x = np.asarray(x, float); bs = rng.choice(x, (4000, len(x))).mean(1)
    return dict(n=int(len(x)), mean=float(x.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0)


def push_geom(acts):
    """(N,4) [sx,sy,ex,ey] m -> angle deg (atan2(dy,dx)), start near-wall flag, end near-wall flag, length mm."""
    a = np.asarray(acts, float).reshape(-1, 4); d = a[:, 2:] - a[:, :2]
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])); L = np.linalg.norm(d, axis=1) * 1000
    lo = OCC_BOUNDS["x_min"] + XY_MARGIN + 1e-6; hi = OCC_BOUNDS["x_max"] - XY_MARGIN - 1e-6
    s_wall = ((a[:, :2] <= lo) | (a[:, :2] >= hi)).any(1); e_wall = ((a[:, 2:] <= lo) | (a[:, 2:] >= hi)).any(1)
    return ang, s_wall, e_wall, L, a


def axis_alignment(ang):
    """fraction of pushes within 22.5 deg of an axis (0/90/180/270), and of each axis"""
    a = np.mod(ang, 360); dx = np.minimum(np.abs(np.mod(a, 180)), 180 - np.abs(np.mod(a, 180)))  # distance to x-axis
    dy = np.abs(np.mod(a, 180) - 90)
    return dict(along_x=float((dx <= 22.5).mean()), along_y=float((dy <= 22.5).mean()), diagonal=float(((dx > 22.5) & (dy > 22.5)).mean()))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = dict(mask_stats={g: mask_stats(g) for g in LETTERS8})
    # ---------------- EXP-0066 cells ----------------
    cells = {}
    for fam, tags in CELLS66.items():
        for t in tags:
            f = R66 / f"{t}.json"
            if f.exists():
                cells[t] = load_episodes(f)
    fam_of = {t: fam for fam, tags in CELLS66.items() for t in tags}
    res["cells_loaded"] = sorted(cells)
    # per-letter per-cell table
    tbl = {}
    for t, eps in cells.items():
        for g in LETTERS4:
            E = [e for e in eps if e["goal"] == g]
            if not E:
                continue
            ang, sw, ew, L, _ = push_geom([a for e in E for a in e["actions"]])
            n_steps = sum(len(e["actions"]) for e in E)
            tbl[(t, g)] = dict(score=q([e["score"] for e in E]), k0=float(np.mean([e["k0"] for e in E])), k4=float(np.mean([e["k4"] for e in E])),
                               k16=float(np.mean([e["k16"] for e in E])), mf_k16=float(np.mean([e["mf"][K1] for e in E])),
                               gain_4_16=float(np.mean([e["k16"] - e["k4"] for e in E])),
                               fallback_rate=float(sum(int(x) for e in E for x in e["legal_fallback"]) / n_steps),
                               shift_rate=float(sum(1 for e in E for s in e["legal_shift_m"] if s > 0) / n_steps),
                               mean_shift_mm=float(np.mean([s for e in E for s in e["legal_shift_m"]]) * 1000),
                               illegal_executed=int(sum(1 for e in E for o in e["legal_ok"] if not o)),
                               start_on_wall=float(sw.mean()), end_on_wall=float(ew.mean()), push_len_mm=float(L.mean()),
                               axis=axis_alignment(ang),
                               pred_true_corr=float(np.corrcoef([p for e in E for p in e["pred_dv"]], [p for e in E for p in e["true_dv"]])[0, 1]),
                               pred_true_mae=float(np.mean(np.abs(np.array([p for e in E for p in e["pred_dv"]]) - np.array([p for e in E for p in e["true_dv"]])))),
                               optimism=float(np.mean(np.array([p for e in E for p in e["true_dv"]]) - np.array([p for e in E for p in e["pred_dv"]]))),
                               mean_true_dv=float(np.mean([p for e in E for p in e["true_dv"]])),
                               frac_true_dv_neg=float(np.mean([p < 0 for e in E for p in e["true_dv"]])))
    res["per_cell_goal"] = {f"{t}|{g}": v for (t, g), v in tbl.items()}
    # family means per letter (ps0 cells)
    fam_tbl = {}
    for fam in CELLS66:
        tags = [t for t in CELLS66[fam] if t in cells]
        for g in LETTERS4:
            rows = [tbl[(t, g)] for t in tags if (t, g) in tbl]
            if rows:
                fam_tbl[f"{fam}|{g}"] = {k: float(np.mean([r[k] for r in rows])) for k in ("k0", "k4", "k16", "mf_k16", "gain_4_16", "fallback_rate", "shift_rate", "start_on_wall", "end_on_wall", "pred_true_corr", "pred_true_mae", "optimism", "mean_true_dv", "frac_true_dv_neg")}
                fam_tbl[f"{fam}|{g}"]["score"] = float(np.mean([r["score"]["mean"] for r in rows]))
                fam_tbl[f"{fam}|{g}"]["axis"] = {a: float(np.mean([r["axis"][a] for r in rows])) for a in ("along_x", "along_y", "diagonal")}
    res["per_family_goal"] = fam_tbl
    # headroom (H-a): optimum - k0 (relative units: 1 - k0), and the fraction of headroom closed by k=16
    res["headroom"] = {g: dict(k0_rel=fam_tbl[f"nfd|{g}"]["k0"], headroom_rel=1 - fam_tbl[f"nfd|{g}"]["k0"],
                              closed_by_k16={fam: (fam_tbl[f"{fam}|{g}"]["k16"] - fam_tbl[f"{fam}|{g}"]["k0"]) / (1 - fam_tbl[f"{fam}|{g}"]["k0"]) for fam in CELLS66 if f"{fam}|{g}" in fam_tbl})
                       for g in LETTERS4}
    # paired per-start gap (NFD mean - lin mean) per letter: distribution, not just the mean
    nfd = [t for t in CELLS66["nfd"] if t in cells]; lin = [t for t in CELLS66["lin"] if t in cells]
    gaps = {}
    for g in LETTERS4:
        starts = sorted({e["start"] for e in cells[nfd[0]] if e["goal"] == g})
        per_start = []
        for s in starts:
            xn = np.mean([e["score"] for t in nfd for e in cells[t] if e["goal"] == g and e["start"] == s])
            xl = np.mean([e["score"] for t in lin for e in cells[t] if e["goal"] == g and e["start"] == s])
            per_start.append(dict(start=s, nfd=float(xn), lin=float(xl), gap=float(xn - xl)))
        gp = np.array([p["gap"] for p in per_start])
        gaps[g] = dict(per_start=per_start, mean=float(gp.mean()), sd=float(gp.std(ddof=1)), frac_pos=float((gp > 0).mean()),
                       corr_gap_vs_k0=float(np.corrcoef(gp, [np.mean([e["k0"] for e in cells[nfd[0]] if e["goal"] == g and e["start"] == p["start"]]) for p in per_start])[0, 1]))
    res["paired_gap_per_start"] = gaps
    # per-push trajectory of in-goal mass per letter per family (for the figure and the json)
    traj = {}
    for fam in CELLS66:
        tags = [t for t in CELLS66[fam] if t in cells]
        for g in LETTERS4:
            M = np.array([e["rel"] for t in tags for e in cells[t] if e["goal"] == g])
            traj[f"{fam}|{g}"] = dict(mean=M.mean(0).tolist(), sd=M.std(0).tolist(), n=int(len(M)))
    res["trajectory_rel"] = traj
    # letter_T crossbar vs stem: where does the mass go, per family
    bar, stem = t_parts(); barfrac_opt = None
    tparts = {}
    for fam in CELLS66:
        tags = [t for t in CELLS66[fam] if t in cells]
        E = [e for t in tags for e in cells[t] if e["goal"] == "letter_T"]
        B = np.array([e["mf_bar"] for e in E]); S = np.array([e["mf_stem"] for e in E])
        tparts[fam] = dict(bar_mean=B.mean(0).tolist(), stem_mean=S.mean(0).tolist(), bar_k16=float(B[:, K1].mean()), stem_k16=float(S[:, K1].mean()))
    # capacity split of the optimum placement: cubes inside bar vs stem at V*
    cen = np.array(VS["letter_T"]["vstar_centres_m"]); uv = ((cen[:, :2] - OCC_BOUNDS["x_min"]) / (OCC_BOUNDS["x_max"] - OCC_BOUNDS["x_min"]) * (OCC_GRID - 1)).round().astype(int)
    in_bar = bar[uv[:, 0], uv[:, 1]].sum(); in_stem = stem[uv[:, 0], uv[:, 1]].sum()
    tparts["vstar_cubes_in_bar"] = int(in_bar); tparts["vstar_cubes_in_stem"] = int(in_stem); tparts["bar_px"] = int(bar.sum()); tparts["stem_px"] = int(stem.sum())
    res["letter_T_parts"] = tparts
    # mean occupancy maps per letter per family at k=0 and k=16; occupancy inside/outside the mask by band of distance to the mask
    maps = {}
    for fam in CELLS66:
        tags = [t for t in CELLS66[fam] if t in cells]
        for g in LETTERS4:
            E = [e for t in tags for e in cells[t] if e["goal"] == g]
            maps[(fam, g)] = dict(k0=np.mean([e["occ0"] for e in E], 0), k16=np.mean([e["occ16"] for e in E], 0))
    # distance-to-mask profile of the final mass (how far the misplaced mass sits), per letter per family
    prof = {}
    for (fam, g), mm in maps.items():
        m = mask(g); d = distance_transform_edt(~m); occ = mm["k16"]; tot = occ.sum()
        bands = [(0, 0.5), (0.5, 2.5), (2.5, 5.5), (5.5, 10.5), (10.5, 99)]
        prof[f"{fam}|{g}"] = {f"{a}-{b}px": float(occ[(d >= a) & (d < b)].sum() / tot) for a, b in bands}
        prof[f"{fam}|{g}"]["mean_dist_px"] = float((occ * d).sum() / tot)
    res["final_mass_distance_profile"] = prof
    # ---------------- EXP-0039 RUN-0002 (8 letters, cem + rank) ----------------
    r39 = {}
    for arm in ("cem", "rank"):
        files = sorted(R39.glob(f"hm_{arm}_*.json"))
        eps = {f.stem.replace(f"hm_{arm}_", ""): load_episodes(f) for f in files}
        models = sorted(eps)
        per = {}
        for g in LETTERS8:
            row = {}
            for mdl in models:
                E = [e for e in eps[mdl] if e["goal"] == g]
                if E:
                    row[mdl] = dict(score=float(np.mean([e["score"] for e in E])), k0=float(np.mean([e["k0"] for e in E])), k16=float(np.mean([e["k16"] for e in E])),
                                    illegal=int(sum(1 for e in E for o in e["legal_ok"] if not o)),
                                    pred_true_corr=float(np.corrcoef([p for e in E for p in e["pred_dv"]], [p for e in E for p in e["true_dv"]])[0, 1]))
            per[g] = row
        # within-goal spread of model scores and the pooled-order agreement (Kendall) recomputed
        pooled = {mdl: np.mean([e["score"] for e in eps[mdl]]) for mdl in models}
        order = sorted(models, key=lambda m: -pooled[m])
        def kendall(a, b):
            s = 0; n = 0
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    s += np.sign((a[i] - a[j]) * (b[i] - b[j])); n += 1
            return float(s / n)
        agree = {g: kendall([pooled[m] for m in models], [per[g][m]["score"] for m in models]) for g in LETTERS8 if all(m in per[g] for m in models)}
        spread = {g: float(np.std([per[g][m]["score"] for m in models], ddof=1)) for g in LETTERS8 if all(m in per[g] for m in models)}
        # noise floor for the per-goal order: within-letter SE of a model's 16-episode mean
        se = {g: float(np.mean([np.std([e["score"] for e in eps[m] if e["goal"] == g], ddof=1) / 4 for m in models])) for g in LETTERS8}
        # linear vs NFD gap per letter (linear_narrow_l20_v2_res64 vs nfd_3ch_narrow_l20_v2 seeds 0/1/2 mean)
        nf = [m for m in models if m in ("nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2")]
        gapT = {}
        for g in LETTERS8:
            starts = sorted({e["start"] for e in eps["linear_narrow_l20_v2_res64"] if e["goal"] == g})
            dd = [np.mean([e["score"] for m in nf for e in eps[m] if e["goal"] == g and e["start"] == s]) - np.mean([e["score"] for e in eps["linear_narrow_l20_v2_res64"] if e["goal"] == g and e["start"] == s]) for s in starts]
            gapT[g] = q(dd)
        r39[arm] = dict(models=models, pooled_order=order, per_goal=per, tau_vs_pooled=agree, model_spread_sd=spread, within_goal_se=se, nfd3_minus_linear_res64=gapT)
    res["exp0039_run0002"] = r39
    # ---------------- figures ----------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # F1: in-goal mass (rel) vs push index per letter, families
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    col = {"nfd": "#1f77b4", "lin": "#d62728", "broad": "#2ca02c"}
    for ax, g in zip(axs, LETTERS4):
        for fam in CELLS66:
            tr = traj.get(f"{fam}|{g}")
            if tr:
                mu = np.array(tr["mean"]); sd = np.array(tr["sd"]) / np.sqrt(tr["n"])
                ax.plot(range(17), mu, color=col[fam], label=f"{fam} (n={tr['n']})"); ax.fill_between(range(17), mu - 2 * sd, mu + 2 * sd, color=col[fam], alpha=0.2)
        ax.axvspan(K0, K1, color="k", alpha=0.05); ax.set_title(f"{g}: cap {VS[g]['capacity']}, opt {VS[g]['mass_frac_best_placement']:.3f}"); ax.set_xlabel("push k"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("in-goal mass / optimum"); axs[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "f1_ingoal_vs_push.png", dpi=130); plt.close(fig)
    # F2: mean occupancy heat maps at k=16 per letter x family with mask outline, plus k=0
    fams = list(CELLS66); fig, axs = plt.subplots(len(fams) + 1, 4, figsize=(13, 3.2 * (len(fams) + 1)))
    for j, g in enumerate(LETTERS4):
        m = mask(g)
        for i, fam in enumerate(["k0"] + fams):
            ax = axs[i, j]; mm = maps[(fams[0], g)]["k0"] if fam == "k0" else maps[(fam, g)]["k16"]
            ax.imshow(mm.T, origin="upper", cmap="magma", vmin=0, vmax=max(0.02, float(np.percentile(mm, 99.5))))  # x right, y down
            ax.contour(m.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
            ax.set_title(f"{g} {'start (k=0)' if fam == 'k0' else fam + ' k=16'}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("mean scoring occupancy (world x right, y down); cyan = goal mask", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "f2_occupancy_maps.png", dpi=120); plt.close(fig)
    # F3: push direction roses per letter per family (EXP-0066 ps0 cells)
    fig, axs = plt.subplots(len(fams), 4, figsize=(13, 3.2 * len(fams)), subplot_kw=dict(projection="polar"))
    for j, g in enumerate(LETTERS4):
        for i, fam in enumerate(fams):
            tags = [t for t in CELLS66[fam] if t in cells]
            ang, *_ = push_geom([a for t in tags for e in cells[t] if e["goal"] == g for a in e["actions"]])
            h, edges = np.histogram(np.radians(ang), bins=16, range=(-np.pi, np.pi)); ax = axs[i, j]
            ax.bar((edges[:-1] + edges[1:]) / 2, h / h.sum(), width=2 * np.pi / 16, color=col[fam], alpha=0.8)
            ax.set_title(f"{g} {fam}", fontsize=9); ax.set_yticklabels([]); ax.set_theta_zero_location("E"); ax.set_theta_direction(-1)
    fig.suptitle("push direction (atan2(dy,dx), world x right, y down)", fontsize=10); fig.tight_layout(); fig.savefig(FIG / "f3_push_directions.png", dpi=120); plt.close(fig)
    # F4: letter_T crossbar vs stem mass vs push, per family; and the per-start paired gap per letter
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for fam in fams:
        axs[0].plot(range(17), tparts[fam]["bar_mean"], color=col[fam], label=f"{fam} crossbar"); axs[0].plot(range(17), tparts[fam]["stem_mean"], color=col[fam], ls="--", label=f"{fam} stem")
    axs[0].set_title(f"letter_T: mass fraction in crossbar ({tparts['bar_px']} px, {tparts['vstar_cubes_in_bar']} cubes at V*) vs stem ({tparts['stem_px']} px, {tparts['vstar_cubes_in_stem']})", fontsize=8); axs[0].set_xlabel("push k"); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.3)
    for j, g in enumerate(LETTERS4):
        gp = [p["gap"] for p in gaps[g]["per_start"]]; axs[1].scatter(np.full(len(gp), j) + rng.uniform(-0.15, 0.15, len(gp)), gp, s=12, alpha=0.7)
        axs[1].plot([j - 0.3, j + 0.3], [gaps[g]["mean"]] * 2, color="k")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].set_xticks(range(4)); axs[1].set_xticklabels(LETTERS4); axs[1].set_ylabel("NFD - linear, per start"); axs[1].set_title("paired per-start family gap (EXP-0066)", fontsize=9); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "f4_T_parts_and_gaps.png", dpi=130); plt.close(fig)
    # ---------------- write ----------------
    tmp = R / "diagnosis.json.tmp"; tmp.write_text(json.dumps(res, indent=1, default=float)); tmp.replace(R / "diagnosis.json")
    L = ["# EXP-0069 diagnosis (generated by code/diagnose_T.py)\n", f"EXP-0066 cells loaded: {', '.join(res['cells_loaded'])}\n",
         "## Mask geometry (64 px grid, 2.03 mm/px; world mm, tray +-64 mm, legal push box +-60 mm)",
         "| goal | px | capacity | opt mass frac | V* | bbox x mm | bbox y mm | centroid mm | stroke max mm | min edge dist px | frac px <=6 px of edge | flip-x IoU | flip-y IoU |", "|---" * 13 + "|"]
    for g in LETTERS8:
        s = res["mask_stats"][g]
        L.append(f"| {g} | {s['mask_px']} | {s['capacity']} | {s['mass_frac_best_placement']:.3f} | {s['vstar']:.4f} | [{s['bbox_mm']['x'][0]:.0f}, {s['bbox_mm']['x'][1]:.0f}] | [{s['bbox_mm']['y'][0]:.0f}, {s['bbox_mm']['y'][1]:.0f}] | ({s['centroid_mm'][0]:.1f}, {s['centroid_mm'][1]:.1f}) | {s['stroke_width_mm_max']:.1f} | {s['min_edge_dist_px']} | {s['frac_px_within_6px_of_edge']:.2f} | {s['sym_flip_x_iou']:.2f} | {s['sym_flip_y_iou']:.2f} |")
    L += ["\n## EXP-0066 per family x letter (ps0 cells; 3 NFD seeds, 3 linear instances, 2 broad seeds; 32 starts each)",
          "| family | goal | score | k0 | k4 | k16 | gain 4->16 | headroom closed by k16 | fallback rate | shift rate | start on wall | end on wall | along x | along y | diag | pred-true corr | MAE | optimism | mean true dv |", "|---" * 19 + "|"]
    for fam in CELLS66:
        for g in LETTERS4:
            v = fam_tbl.get(f"{fam}|{g}")
            if v:
                L.append(f"| {fam} | {g} | {v['score']:.3f} | {v['k0']:.3f} | {v['k4']:.3f} | {v['k16']:.3f} | {v['gain_4_16']:+.3f} | {res['headroom'][g]['closed_by_k16'][fam]:.2f} | {v['fallback_rate']:.3f} | {v['shift_rate']:.2f} | {v['start_on_wall']:.2f} | {v['end_on_wall']:.2f} | {v['axis']['along_x']:.2f} | {v['axis']['along_y']:.2f} | {v['axis']['diagonal']:.2f} | {v['pred_true_corr']:+.2f} | {v['pred_true_mae']:.4f} | {v['optimism']:+.4f} | {v['mean_true_dv']:+.4f} |")
    L += ["\n## EXP-0066 per cell x letter (score mean [CI])", "| cell | " + " | ".join(LETTERS4) + " |", "|---" * 5 + "|"]
    for t in cells:
        L.append(f"| {t} | " + " | ".join(f"{tbl[(t, g)]['score']['mean']:.3f} [{tbl[(t, g)]['score']['ci'][0]:.3f}, {tbl[(t, g)]['score']['ci'][1]:.3f}]" for g in LETTERS4) + " |")
    L += ["\n## Paired per-start gap NFD(3-seed mean) - linear(3-instance mean)", "| goal | mean | sd over starts | frac starts > 0 | corr(gap, k0) |", "|---|---|---|---|---|"]
    for g in LETTERS4:
        v = gaps[g]; L.append(f"| {g} | {v['mean']:+.3f} | {v['sd']:.3f} | {v['frac_pos']:.2f} | {v['corr_gap_vs_k0']:+.2f} |")
    L += ["\n## Final (k=16) mass by distance to the mask (fraction of total mass)", "| family | goal | on mask | 0.5-2.5 px | 2.5-5.5 px | 5.5-10.5 px | >10.5 px | mean dist px |", "|---" * 8 + "|"]
    for k, v in prof.items():
        fam, g = k.split("|"); L.append(f"| {fam} | {g} | " + " | ".join(f"{v[b]:.3f}" for b in ("0-0.5px", "0.5-2.5px", "2.5-5.5px", "5.5-10.5px", "10.5-99px")) + f" | {v['mean_dist_px']:.2f} |")
    L += [f"\n## letter_T crossbar vs stem (crossbar {tparts['bar_px']} px holds {tparts['vstar_cubes_in_bar']} of 20 cubes at V*; stem {tparts['stem_px']} px holds {tparts['vstar_cubes_in_stem']})",
          "| family | bar k0 | bar k16 | stem k0 | stem k16 |", "|---|---|---|---|---|"]
    for fam in fams:
        v = tparts[fam]; L.append(f"| {fam} | {v['bar_mean'][0]:.3f} | {v['bar_k16']:.3f} | {v['stem_mean'][0]:.3f} | {v['stem_k16']:.3f} |")
    for arm in ("cem", "rank"):
        a = r39[arm]
        L += [f"\n## EXP-0039 RUN-0002 {arm}: per-letter model scores (recomputed), tau vs pooled order, model spread, within-goal SE", "| goal | tau | sd over models | SE of a model mean | NFD3 - linear res64 [CI] | " + " | ".join(a["models"]) + " |", "|---" * (5 + len(a["models"])) + "|"]
        for g in LETTERS8:
            gg = a["nfd3_minus_linear_res64"][g]
            L.append(f"| {g} | {a['tau_vs_pooled'].get(g, float('nan')):+.2f} | {a['model_spread_sd'].get(g, float('nan')):.3f} | {a['within_goal_se'][g]:.3f} | {gg['mean']:+.3f} [{gg['ci'][0]:+.3f}, {gg['ci'][1]:+.3f}] | " + " | ".join(f"{a['per_goal'][g][m]['score']:.3f}" for m in a["models"]) + " |")
    (R / "diagnosis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
