"""EXP-0034 (G2b): is the accuracy/slateN disagreement within the NFD family
driven by pixel-level cube-orientation artifacts? Accuracy (swept-region,
ratio of means vs persistence, `fit_linear_foresight.metrics`) is recomputed on
randlen_test after blurring prediction, truth and previous frame with the SAME
Gaussian (sigma in px), which removes 2x2-px orientation detail while keeping
where mass is. Then Kendall tau vs lyapunov slateN (EXP-0028 soft, 30 goals).
Non-GNN models only (the GNN accuracy path resamples through nodes).
Checkpoint: results/blur_accuracy.json rewritten after every model.
"""
import json, os, sys
from pathlib import Path
import torch, torch.nn.functional as F
from scipy import stats
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from Baselines.common import eval_report as er
from Baselines.common.eval_baseline import _predictor_batch
from fit_linear_foresight import actions_to_pixels, metrics, swept_region_mask

OUT = REPO / "experiments/EXP-0034-orientation-artifacts-accuracy/results/blur_accuracy.json"
SIGMAS = [0.0, 1.0, 2.0]
def blur(x, s):
    if s == 0: return x
    r = int(3 * s + 0.5); g = torch.exp(-torch.arange(-r, r + 1).float() ** 2 / (2 * s * s)); g = g / g.sum()
    x = x[:, None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    return x[:, 0]
cell = er._load_cell(er.CORPORA["randlen_test"], tag="randlen_test")
H, W = cell.occ0.shape[-2:]
s_px, e_px = actions_to_pixels(cell.actions, cell.workspace_min, cell.workspace_max, (H, W))
plate_px = 0.04 / 0.128 * W
region = swept_region_mask(s_px, e_px, (H, W), 0.5 * plate_px + 2.0, 0.5 * plate_px)
truth, prev = cell.occ1.float(), cell.occ0.float()
rep = json.load(open(REPO / "experiments/EXP-0028-soft-truth-scoring/artifacts/RUN-0001-soft-many/report.json"))["randlen_test"]
res = json.load(open(OUT)) if OUT.exists() else {"sigmas": SIGMAS, "models": {}}
models = [m for m in rep if m not in ("persistence", "random") and not er.MODELS[m]["is_gnn"]]
for m in models:
    if m in res["models"]: continue
    pred = er._load_predictor(er.MODELS[m]).predict_occ(_predictor_batch(cell) if hasattr(cell, "states") else cell).float()
    res["models"][m] = {"slateN": rep[m]["capture"]["averaged_over_goals"]["lyapunov"],
                        **{f"acc_s{s}": float(metrics(blur(pred, s), blur(truth, s), blur(prev, s), region=region)["accuracy"]) for s in SIGMAS}}
    tmp = Path(str(OUT) + ".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, OUT)
    print(m, res["models"][m], flush=True)
for label, pick in (("all non-GNN", lambda m: True), ("NFD family", lambda m: m.startswith("nfd"))):
    ms = [m for m in res["models"] if pick(m)]
    for s in SIGMAS:
        t = stats.kendalltau([res["models"][m][f"acc_s{s}"] for m in ms], [res["models"][m]["slateN"] for m in ms])
        res.setdefault("agreement", {})[f"{label}|sigma{s}"] = dict(tau=float(t.statistic), p=float(t.pvalue), n=len(ms))
        print(f"{label:12s} sigma {s}: Kendall tau accuracy vs slateN {t.statistic:+.2f} (p {t.pvalue:.2f}, n={len(ms)})")
tmp = Path(str(OUT) + ".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, OUT)
