"""EXP-0038 (metric-validity study, part 1): a table of candidate OFFLINE metrics per
model on DS-0006 (160 states x 128 pushes, 30 goals, soft truth, lyapunov), from
EXP-0030's saved predictions -- to be correlated with closed-loop performance (G1b).

Per (state, goal) pool, then averaged (per-state means kept for resampling):
  slateN        capture of the model's argmin pick
  top1_regret   true dv(pick) - true best dv in pool (>= 0; lower = better)
  spearman      rank correlation of predicted vs true dv over the pool
  pearson       linear correlation of predicted vs true dv
  optimism      predicted dv(pick) - true dv(pick) (< 0 = over-promises)
  top8_capture  mean capture of the model's 8 best-predicted pushes (a sampler's view)
Plus accuracy (randlen_test, EXP-0028 soft report) where the model is in the harness.
"""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats
REPO = Path(__file__).resolve().parents[3]
ART = REPO / "experiments/EXP-0030-state-superiority-ds0005/artifacts/RUN-0001"
sys.path.insert(0, str(REPO))
from Genesis.binned_slate_dataset import BinnedSlateCorpus
MODELS = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug", "nfd_warped_randlen_flipaug_epoch30",
          "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43", "nfd_3ch_finetuned", "linear_switched_hard"]
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
sl = rows.slate_idx.numpy(); S = sl.max() + 1
dvt = torch.load(ART / "truth.pt", weights_only=False)["dv"].numpy()[..., 0]          # lyapunov (N, G)
pred = {m: torch.load(ART / f"pred_{m}.pt", weights_only=False)["dv"].numpy()[..., 0] for m in MODELS}
pred["ensemble_nfd"] = np.mean([pred[m] for m in MODELS if m.startswith("nfd")], 0)   # 7 'nfd*' models (EXP-0030/0035)
ENS5 = ["nfd_3ch_randlen", "nfd_warped_randlen", "nfd_warped_randlen_flipaug",
        "nfd_residual_warped_flipaug_randlen", "nfd_residual_worldframe_noaug_ep43"]   # EXP-0037 / EXP-0039 members
pred["ensemble_nfd5"] = np.mean([pred[m] for m in ENS5], 0)
acc_rep = json.load(open(REPO / "experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-many/report.json"))["randlen_test"]
ALIAS = {"nfd_3ch_randlen": "nfd_randlen"}
table = {}
for m, P in pred.items():
    per_state = {k: [] for k in ("slateN", "top1_regret", "spearman", "pearson", "optimism", "top8_capture")}
    for s in range(S):
        ix = np.nonzero(sl == s)[0]
        vals = {k: [] for k in per_state}
        for g in range(dvt.shape[1]):
            t, p = dvt[ix, g], P[ix, g]
            mean, best = t.mean(), t.min()
            den = mean - best
            if den < 1e-9: continue
            k = int(p.argmin()); top8 = np.argsort(p)[:8]
            vals["slateN"].append((mean - t[k]) / den)
            vals["top1_regret"].append(t[k] - best)
            vals["spearman"].append(stats.spearmanr(p, t).statistic)
            vals["pearson"].append(np.corrcoef(p, t)[0, 1])
            vals["optimism"].append(p[k] - t[k])
            vals["top8_capture"].append(np.mean((mean - t[top8]) / den))
        for key in per_state: per_state[key].append(float(np.nanmean(vals[key])))
    table[m] = {k: float(np.mean(v)) for k, v in per_state.items()} | {"per_state": per_state}
    a = acc_rep.get(ALIAS.get(m, m), {}).get("accuracy")
    table[m]["accuracy_randlen_test"] = a
    print(f"{m:40s} " + " ".join(f"{k} {table[m][k]:+.3f}" for k in ("slateN", "top1_regret", "spearman", "pearson", "optimism", "top8_capture"))
          + (f" acc {a:.3f}" if a is not None else ""))
names = list(table)
keys = ["slateN", "top1_regret", "spearman", "pearson", "optimism", "top8_capture"]
print("\nKendall tau between offline metrics across models (how redundant are they?):")
for i, a in enumerate(keys):
    print("   " + a.ljust(13) + " ".join(f"{stats.kendalltau([table[n][a] for n in names], [table[n][b] for n in names]).statistic:+.2f}" for b in keys))
(REPO / "experiments/EXP-0038-offline-metric-table/results").mkdir(exist_ok=True)
json.dump(table, open(REPO / "experiments/EXP-0038-offline-metric-table/results/metric_table.json", "w"))
