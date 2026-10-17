"""EXP-0071 analysis: paired (goal x start) comparisons of the gated ensemble / equal blend vs the RUN-0002 member cells.
Score = EXP-0039 RUN-0002 episode score (in-goal mass / optimum, mean over pushes 4..16), reused verbatim."""
import sys, json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO/"experiments/EXP-0039-closed-loop-metric-validity/code"))
from analyse_run0002 import episode_scores
HERE = REPO/"experiments/EXP-0071-gated-ensemble"; R1 = REPO/"experiments/EXP-0039-closed-loop-metric-validity/results/run0002"; R2 = HERE/"results/run0001"
rng = np.random.default_rng(0)
NFD = ["nfd_3ch_narrow_l20_v2", "nfd_3ch_narrow_l20_v2_seed1", "nfd_3ch_narrow_l20_v2_seed2"]
def boot(d, n=5000):
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]; return float(d.mean()), np.percentile(b, [2.5, 97.5]).round(4).tolist()
out = {}; L = ["# EXP-0071 closed-loop results (paired over (goal, start) episodes)\n"]
for arm in ("rank", "cem"):
    cells = {m: episode_scores(R1/f"hm_{arm}_{m}.json") for m in NFD + ["linear_narrow_l20_v2_res64"]}
    for m in ("gated_ens_narrow_v2", "nfd3_meanblend_narrow_v2"):
        f = R2/f"hm_{arm}_{m}.json"
        if f.exists(): cells[m] = episode_scores(f)
    if "gated_ens_narrow_v2" not in cells: continue
    keys = sorted(set.intersection(*[set(c) for c in cells.values()])); X = {m: np.array([c[k]["score"] for k in keys]) for m, c in cells.items()}
    L.append(f"\n## {arm} arm: {len(keys)} common episodes\n\n| cell | mean | k8 | k16 |\n|---|---|---|---|")
    for m, c in cells.items(): L.append(f"| {m} | {X[m].mean():.3f} | {np.mean([c[k]['k8'] for k in keys]):.3f} | {np.mean([c[k]['k16'] for k in keys]):.3f} |")
    nfd_mean = np.mean([X[m] for m in NFD], 0); best = max(NFD, key=lambda m: X[m].mean())
    comps = {"gate - NFD3 mean (H1)": X["gated_ens_narrow_v2"] - nfd_mean, f"gate - best NFD seed ({best}; selected post hoc)": X["gated_ens_narrow_v2"] - X[best],
             "gate - linear res64": X["gated_ens_narrow_v2"] - X["linear_narrow_l20_v2_res64"]}
    if "nfd3_meanblend_narrow_v2" in X:
        comps["gate - equal blend (H2)"] = X["gated_ens_narrow_v2"] - X["nfd3_meanblend_narrow_v2"]; comps["equal blend - NFD3 mean"] = X["nfd3_meanblend_narrow_v2"] - nfd_mean
    L.append("\n| comparison | diff [95% paired-bootstrap CI] |\n|---|---|"); out[arm] = {}
    for n, d in comps.items():
        m, ci = boot(d); out[arm][n] = dict(diff=m, ci=ci); L.append(f"| {n} | {m:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
    for g in sorted({k[0] for k in keys}):
        ix = [i for i, k in enumerate(keys) if k[0] == g]; L.append(f"- per goal {g}: gate - NFD3 mean {np.mean((X['gated_ens_narrow_v2']-nfd_mean)[ix]):+.3f}")
    # selection note: also report seed-to-seed spread of the NFD cells for scale
    L.append(f"- NFD seed cell means {[round(float(X[m].mean()),3) for m in NFD]}")
(HERE/"results").mkdir(exist_ok=True); (HERE/"results/analysis.json").write_text(json.dumps(out, indent=1)); (HERE/"results/analysis.md").write_text("\n".join(L)); print("\n".join(L))
