"""EXP-0069 follow-up: discriminate capacity (H-c) from the linear model's junction-blob policy (H-d) using the
8-letter EXP-0039 RUN-0002 cem episodes (16 starts each), plus letter_T spatial checks on EXP-0066.
Writes results/mechanism.json / .md and figures/f5_blob_8letters.png. CPU only.
Usage: CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -u experiments/EXP-0069-letter-T-diagnosis/code/mechanism_8letters.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0069-letter-T-diagnosis/code"))
from diagnose_T import mask, load_episodes, LETTERS8, LETTERS4, VS, R39, R66, CELLS66, K1, R, FIG, PX_MM, t_parts  # noqa: E402

NFD3 = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2"]
MODELS = {"linear_res64": ["linear_narrow_l20_v2_res64"], "linear_switched_soft": ["linear_switched_soft"], "nfd_narrow3": NFD3, "nfd_randlen": ["nfd_3ch_randlen"]}


def blob(mean_map, m, r=3.5):
    """peak of the mean k=16 map; fraction of mass within r px of it; whether the peak is on the mask; local EDT there"""
    pk = np.unravel_index(np.argmax(mean_map), mean_map.shape); ii, jj = np.indices(mean_map.shape)
    near = (ii - pk[0]) ** 2 + (jj - pk[1]) ** 2 <= r * r
    edt = distance_transform_edt(m)
    return dict(peak_px=[int(pk[0]), int(pk[1])], peak_mm=[(pk[0] - 31.5) * PX_MM, (pk[1] - 31.5) * PX_MM], peak_density=float(mean_map.max() / mean_map.sum()),
                frac_mass_within_r=float(mean_map[near].sum() / mean_map.sum()), peak_on_mask=bool(m[pk]), mask_frac_in_disk=float(m[near].mean()),
                edt_at_peak_mm=float(edt[pk] * 2 * PX_MM), top5pct_mass=float(np.sort(mean_map.ravel())[::-1][: int(0.05 * mean_map.size)].sum() / mean_map.sum()))


def main():
    out = {}
    # letter geometry relevant to transport: distance field statistics over the tray
    geo = {}
    for g in LETTERS8:
        m = mask(g); d = distance_transform_edt(~m); edt = distance_transform_edt(m)
        geo[g] = dict(mean_dist_px=float(d.mean()), frac_tray_gt10px=float((d > 10).mean()), frac_tray_gt15px=float((d > 15).mean()), max_dist_px=float(d.max()),
                      junction_edt_max_mm=float(edt.max() * 2 * PX_MM), mask_px=int(m.sum()), capacity=VS[g]["capacity"], opt=VS[g]["mass_frac_best_placement"])
    out["geometry"] = geo
    # EXP-0039 cem: per letter per model group, final-map blob statistics and score
    eps = {f.stem.replace("hm_cem_", ""): load_episodes(f) for f in sorted(R39.glob("hm_cem_*.json"))}
    per = {}
    for g in LETTERS8:
        m = mask(g); row = {}
        for grp, mdls in MODELS.items():
            E = [e for md in mdls for e in eps[md] if e["goal"] == g]
            mm = np.mean([e["occ16"] for e in E], 0)
            row[grp] = dict(score=float(np.mean([e["score"] for e in E])), n=len(E), **blob(mm, m))
            # per-episode concentration (not of the mean map): fraction of each episode's final mass in its own densest 5 % pixels, and the
            # mean pairwise distance between cubes at k=16 (mm) -- a blob policy gives a small spread
            sp = []
            for e in E:
                S = e["states"][K1][:, :2] * 1000; sp.append(float(np.linalg.norm(S[:, None] - S[None], axis=-1).mean()))
            row[grp]["cube_spread_mm_k16"] = float(np.mean(sp))
            S0 = [float(np.linalg.norm((e["states"][0][:, :2] * 1000)[:, None] - (e["states"][0][:, :2] * 1000)[None], axis=-1).mean()) for e in E]
            row[grp]["cube_spread_mm_k0"] = float(np.mean(S0))
        row["gap_nfd3_minus_linear_res64"] = row["nfd_narrow3"]["score"] - row["linear_res64"]["score"]
        per[g] = row
    out["exp0039_cem_per_letter"] = per
    # correlations of the 8-letter gap with candidate predictors (descriptive, n = 8)
    gap = np.array([per[g]["gap_nfd3_minus_linear_res64"] for g in LETTERS8])
    preds = dict(capacity=[geo[g]["capacity"] for g in LETTERS8], mask_px=[geo[g]["mask_px"] for g in LETTERS8], mean_dist_px=[geo[g]["mean_dist_px"] for g in LETTERS8],
                 junction_edt_max_mm=[geo[g]["junction_edt_max_mm"] for g in LETTERS8],
                 linear_blob_frac=[per[g]["linear_res64"]["frac_mass_within_r"] for g in LETTERS8], linear_score=[per[g]["linear_res64"]["score"] for g in LETTERS8],
                 nfd_score=[per[g]["nfd_narrow3"]["score"] for g in LETTERS8], linear_spread_k16=[per[g]["linear_res64"]["cube_spread_mm_k16"] for g in LETTERS8],
                 nfd_spread_k16=[per[g]["nfd_narrow3"]["cube_spread_mm_k16"] for g in LETTERS8])
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    out["gap_correlations_n8"] = {k: dict(pearson=float(np.corrcoef(gap, v)[0, 1]), spearman=spearman(gap, v)) for k, v in preds.items()}
    # letter_T spatial check on EXP-0066: where the off-mask mass sits at k=16 (top band above the crossbar vs lower quadrants beside the stem)
    bar, stem = t_parts(); m = mask("letter_T"); ii, jj = np.indices(m.shape)
    bar_cols = np.where(bar.any(0))[0]; y_bar_lo, y_bar_hi = bar_cols.min(), bar_cols.max()
    stem_rows = np.where(stem.any(1))[0]; x_stem_lo, x_stem_hi = stem_rows.min(), stem_rows.max()
    regions = dict(above_bar=(jj < y_bar_lo) & ~m, bar_band_off_mask=(jj >= y_bar_lo) & (jj <= y_bar_hi) & ~m,
                   lower_quadrants=(jj > y_bar_hi) & ((ii < x_stem_lo) | (ii > x_stem_hi)), below_stem=(jj > np.where(stem.any(0))[0].max()) & ~m, on_mask=m)
    tsp = {}
    for fam in CELLS66:
        E = []
        for t in CELLS66[fam]:
            f = R66 / f"{t}.json"
            if f.exists():
                E += [e for e in load_episodes(f) if e["goal"] == "letter_T"]
        mm16 = np.mean([e["occ16"] for e in E], 0); mm0 = np.mean([e["occ0"] for e in E], 0)
        tsp[fam] = {k: dict(k0=float(mm0[rg].sum() / mm0.sum()), k16=float(mm16[rg].sum() / mm16.sum()), tray_frac=float(rg.mean())) for k, rg in regions.items()}
    out["letter_T_regions_exp0066"] = tsp
    tmp = R / "mechanism.json.tmp"; tmp.write_text(json.dumps(out, indent=1, default=float)); tmp.replace(R / "mechanism.json")
    # figure: 8-letter linear vs NFD mean final maps
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 8, figsize=(20, 5.4))
    for j, g in enumerate(LETTERS8):
        m = mask(g)
        for i, grp in enumerate(["linear_res64", "nfd_narrow3"]):
            E = [e for md in MODELS[grp] for e in eps[md] if e["goal"] == g]; mm = np.mean([e["occ16"] for e in E], 0)
            ax = axs[i, j]; ax.imshow(mm.T, origin="upper", cmap="magma", vmin=0, vmax=max(0.02, float(np.percentile(mm, 99.5)))); ax.contour(m.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)
            ax.set_title(f"{g[-1]} {grp}\nscore {per[g][grp]['score']:.3f}, blob {per[g][grp]['frac_mass_within_r']:.2f}", fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("EXP-0039 RUN-0002 cem, mean scoring occupancy at k=16 (x right, y down); blob = mass within 3.5 px of the map peak", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "f5_blob_8letters.png", dpi=120); plt.close(fig)
    L = ["# EXP-0069 mechanism (generated by code/mechanism_8letters.py)\n", "## Letter geometry (distance-to-mask field over the 64x64 tray)",
         "| goal | capacity | mask px | opt | mean dist px | frac tray > 10 px | frac tray > 15 px | max dist px | thickest stroke mm |", "|---" * 9 + "|"]
    for g in LETTERS8:
        v = geo[g]; L.append(f"| {g} | {v['capacity']} | {v['mask_px']} | {v['opt']:.3f} | {v['mean_dist_px']:.2f} | {v['frac_tray_gt10px']:.2f} | {v['frac_tray_gt15px']:.2f} | {v['max_dist_px']:.1f} | {v['junction_edt_max_mm']:.1f} |")
    L += ["\n## EXP-0039 RUN-0002 cem, 8 letters x 16 starts: score, final-map blob (mass within 3.5 px of the mean-map peak), cube spread at k=16 (mm)",
          "| goal | gap NFD3-lin | " + " | ".join(f"{grp} score / blob / peak on mask / spread" for grp in MODELS) + " |", "|---" * (2 + len(MODELS)) + "|"]
    for g in LETTERS8:
        L.append(f"| {g} | {per[g]['gap_nfd3_minus_linear_res64']:+.3f} | " + " | ".join(f"{per[g][grp]['score']:.3f} / {per[g][grp]['frac_mass_within_r']:.2f} / {'Y' if per[g][grp]['peak_on_mask'] else 'n'} / {per[g][grp]['cube_spread_mm_k16']:.0f}" for grp in MODELS) + " |")
    L += ["\n## Correlation of the 8-letter gap (NFD3 - linear res64) with candidate predictors (n = 8, descriptive)", "| predictor | pearson | spearman |", "|---|---|---|"]
    for k, v in out["gap_correlations_n8"].items():
        L.append(f"| {k} | {v['pearson']:+.2f} | {v['spearman']:+.2f} |")
    L += ["\n## letter_T (EXP-0066): where the mass sits at k=0 / k=16, fraction of total mass (tray fraction of the region in brackets)",
          "| family | " + " | ".join(regions) + " |", "|---" * (1 + len(regions)) + "|"]
    for fam, v in tsp.items():
        L.append(f"| {fam} | " + " | ".join(f"{v[k]['k0']:.3f} -> {v[k]['k16']:.3f} [{v[k]['tray_frac']:.2f}]" for k in regions) + " |")
    (R / "mechanism.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
