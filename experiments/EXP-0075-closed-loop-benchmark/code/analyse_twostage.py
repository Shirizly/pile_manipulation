"""summary of results/budget_study/twostage_results.json -> results/twostage_summary.md"""
import json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"; d = json.load(open(R / "budget_study" / "twostage_results.json")); rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (10000, len(x))).mean(1); return f"{x.mean():+.3f} [{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
g, o, h = (np.array([r[m]["true_terminal"] for r in d]) for m in ("greedy", "open", "hybrid")); pred = np.array([r["plan_pred"] for r in d])
out = [f"# Two-stage model-switching planner (vanilla64 pool -> ens128 GD), {len(d)} tasks, 4 pushes, true value change in the simulator (lower = better)\n",
       f"Stage 1: vanilla64, ~{int(np.mean([r['plan_n1'] for r in d]))} H=4 sequences (0.6 s unit cost); stage 2: ens128 re-score of top 8 + GD ({int(np.mean([r['plan_gd_steps'] for r in d]))} steps, 8 sequences); measured wall time per planning call (exclusive GPU, after warm-up) ~1.3-1.6 s.\n",
       "| mode | true value | vs open (paired) |", "|---|---|---|", f"| greedy H=1 closed loop (re-plan each push) | {ci(g)} | {ci(g - o)} |", f"| H=4 open loop (plan once) | {ci(o)} | - |", f"| hybrid: H=4 once, each later push re-optimised on the real state | {ci(h)} | {ci(h - o)} |",
       f"\nhybrid - greedy: {ci(h - g)}", f"\nPlan optimism (predicted - true, open loop): {ci(pred - o)}  (predicted {pred.mean():+.3f})",
       f"Stage-1 vanilla64 best own-cost -> ens128 cost of the same candidates -> after GD: {np.mean([r['plan_stage1_vanilla'] for r in d]):+.3f} -> {np.mean([r['plan_pre_gd'] for r in d]):+.3f} -> {pred.mean():+.3f}",
       "\n## Divergence from the plan (hybrid vs open, from the SAME real state)\n", f"* push 2 (first divergence): improvement (hybrid dv - open dv; negative = refinement better): {ci([r['hybrid_minus_open_step2'] for r in d])}; better in {sum(r['hybrid_minus_open_step2'] < 0 for r in d)}/{len(d)} tasks"]
for j in (2, 3):
    x = [r["hybrid"]["step_dv"][j] - r["open"]["step_dv"][j] for r in d]; out.append(f"* push {j + 1} dv difference (states already diverged): {ci(x)}")
out.append(f"* terminal: {ci([r['hybrid_minus_open_final'] for r in d])}")
out.append("\nMean per-push dv: " + "; ".join(f"{m}: " + ", ".join(f"{np.mean([r[m]['step_dv'][j] for r in d]):+.3f}" for j in range(4)) for m in ("greedy", "open", "hybrid")))
out.append("\nPer task (greedy / open / hybrid / plan prediction): " + "; ".join(f"{r['goal'][7]}/{r['start']}: {r['greedy']['true_terminal']:+.2f}/{r['open']['true_terminal']:+.2f}/{r['hybrid']['true_terminal']:+.2f}/{r['plan_pred']:+.2f}" for r in d))
(R / "twostage_summary.md").write_text("\n".join(out)); print("\n".join(out))
