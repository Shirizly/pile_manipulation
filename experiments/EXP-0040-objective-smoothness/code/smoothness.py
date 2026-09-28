"""EXP-0040 (exploratory T0): is a model's PREDICTED objective smoother along
action-space lines for the arms that optimise well (EXP-0037)? Per (state, arm):
16 random line segments through the arm's top pool pushes (+/-8 mm on each
endpoint coordinate along a random unit direction, 41 points), predicted
lyapunov dv per goal; roughness = mean |2nd difference| / (profile range), and the
number of interior local minima. Correlated across arms with EXP-0037 gain_capture.
Checkpoint: results/smoothness.json after every arm."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0027-many-goal-benchmarks/code"))
import stage1_grad_many as s27
from stage1_grad_many import lyap_rows
from simple_mpc.adapters import make_occ_adapter, occ_from_particles
from simple_mpc.learned_mpc import project_push
OUT = REPO / "experiments/EXP-0040-objective-smoothness/results/smoothness.json"
s1 = torch.load(REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/artifacts/RUN-0001/stage1.pt", weights_only=False)
res37 = json.load(open(REPO / "experiments/EXP-0037-gradient-benchmark-ds0006/results/analysis.json"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
s27.GOALS = s1["goals"]; D = s27.goal_fields(dev); G = len(s1["goals"])
STATES, NL, NP, AMP = list(range(10)), 16, 41, 0.008
rng = np.random.default_rng(0)
lines = {s: [(rng.normal(size=4) / 1.0) for _ in range(NL)] for s in STATES}
out = json.load(open(OUT)) if OUT.exists() else {}
for arm in s1["arms"]:
    if arm in out: continue
    members = s1["config"]["nfd_members"] if arm == "ensemble_nfd" else [arm]
    ads = [make_occ_adapter(m, dev, "corner") for m in members]
    rough, nmin = [], []
    for s in STATES:
        occ0 = occ_from_particles(s1["states0"][s][None], dev)
        v0 = lyap_rows(occ0.expand(G, -1, -1), D)
        base = s1["arms"][arm]["a_rank"][s]                     # (G, 4) the arm's own top pick per goal
        for li, d in enumerate(lines[s]):
            dvec = torch.tensor(d / np.linalg.norm(d), dtype=torch.float32)
            t = torch.linspace(-AMP, AMP, NP)
            for g in range(G):
                A = project_push(base[g][None] + t[:, None] * dvec[None])[0].to(dev)
                with torch.no_grad():
                    occ = occ0.expand(NP, -1, -1).contiguous()
                    dv = torch.stack([lyap_rows(ad.predict_step(occ, A), D[g].expand(NP, -1, -1)) for ad in ads]).mean(0) - v0[g]
                p = dv.cpu().numpy(); rng_ = p.max() - p.min()
                if rng_ < 1e-9: continue
                rough.append(np.abs(np.diff(p, 2)).mean() / rng_)
                nmin.append(int(((p[1:-1] < p[:-2]) & (p[1:-1] < p[2:])).sum()))
    out[arm] = dict(roughness=float(np.mean(rough)), local_minima=float(np.mean(nmin)),
                    gain_capture=res37["gain_capture"]["means"][arm], grad_capture=res37["grad_capture"]["means"][arm],
                    rank_capture=res37["rank_capture"]["means"][arm])
    tmp = Path(str(OUT) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, OUT)
    print(f"{arm:40s} roughness {out[arm]['roughness']:.4f}  local minima/line {out[arm]['local_minima']:.2f}  "
          f"gain {out[arm]['gain_capture']:+.3f}  grad {out[arm]['grad_capture']:.3f}", flush=True)
    del ads; torch.cuda.empty_cache()
arms = list(out)
for k in ("roughness", "local_minima"):
    for tgt in ("gain_capture", "grad_capture", "rank_capture"):
        r = stats.spearmanr([out[a][k] for a in arms], [out[a][tgt] for a in arms])
        print(f"Spearman({k}, {tgt}) over {len(arms)} arms: {r.statistic:+.2f} (p {r.pvalue:.2f})")
