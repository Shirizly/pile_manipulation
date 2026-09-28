"""EXP-0043: batched replicate vs EXP-0039 (sequential), per (model, planner), paired
over the 16 (goal, start) episodes. -> results/compare_exp0039.json"""
import json
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
A = json.loads((REPO / "experiments/EXP-0039-closed-loop-metric-validity/results/episodes.json").read_text())["episodes"]
B = json.loads((REPO / "experiments/EXP-0043-batched-closed-loop/results/replicate_exp0039.json").read_text())["episodes"]
key = lambda e: (e["model"], e["planner"], e["goal"], e["start"])
imp = lambda e: e["values"][0] - e["values"][-1]
a = {key(e): imp(e) for e in A if e["complete"]}; b = {key(e): imp(e) for e in B if e["complete"]}
out = {}
rng = np.random.default_rng(0)
for m in sorted({k[0] for k in b}):
    for pl in ("gd", "cem"):
        ks = sorted(k for k in b if k[0] == m and k[1] == pl)
        x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks]); d = y - x
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        out[f"{m}/{pl}"] = dict(n=len(ks), seq=float(x.mean()), batched=float(y.mean()), diff=float(d.mean()),
                                ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                r=float(np.corrcoef(x, y)[0, 1]), repeat_sd=float(d.std(ddof=1) / np.sqrt(2)),
                                between_episode_sd=float(np.concatenate([x, y]).std(ddof=1)),
                                evals_seq=float(np.mean([np.mean(e["n_evals"]) for e in A if key(e) in ks])),
                                evals_batched=float(np.mean([np.mean(e["n_evals"]) for e in B if key(e) in ks])))
        r = out[f"{m}/{pl}"]
        print(f"{m:36s} {pl:4s} seq {r['seq']:.3f} batched {r['batched']:.3f} diff {r['diff']:+.3f} "
              f"[{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  r {r['r']:+.2f}  repeat sd {r['repeat_sd']:.3f} "
              f"(episode sd {r['between_episode_sd']:.3f})  evals {r['evals_seq']:.0f}->{r['evals_batched']:.0f}")
# model ordering in the batched run, per planner, with paired diffs vs worldframe
for pl in ("gd", "cem"):
    ms = sorted({k[0] for k in b})
    means = {m: np.mean([b[k] for k in b if k[0] == m and k[1] == pl]) for m in ms}
    print(pl, "batched order:", sorted(ms, key=lambda m: -means[m]))
Path(REPO / "experiments/EXP-0043-batched-closed-loop/results/compare_exp0039.json").write_text(json.dumps(out, indent=1))
