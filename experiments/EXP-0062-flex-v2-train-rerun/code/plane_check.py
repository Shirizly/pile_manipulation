"""EXP-0062 RUN-0003 -- is the GNN's constant-plane height y = 0.24 still right on DS-0020 v2?

v2 / DS-0019 have no depth, so: a PARTICLE-based visible-surface estimator (per image pixel, the max
particle-centre height of the particles projecting within 1 particle radius) is CALIBRATED on DS-0020
v1, which has true depth PNGs (median true fg surface height vs median estimator), then applied to v2
and DS-0019. Also reports the consequence of a plane error for the GNN: a node at true height h placed
at plane p moves in table XY by factor (18 - p)/(18 - h) -> |dXY| = r * |h - p| / (18 - h).
    python -u experiments/EXP-0062-flex-v2-train-rerun/code/plane_check.py
"""
import json, os, sys
from pathlib import Path
import cv2, numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO)); os.chdir(REPO)
from FlexData.image_mask import Camera
cam = Camera.load(); R_P = 0.125
D20 = Path("datasets/DS-0020-training-data-flex-N864"); D19 = Path("datasets/DS-0019-slates-flex-pile-varN")

def est(parts, color):
    fg = (color != 255).any(2)
    uv = cam.project(parts[:, :3])
    rad = cam.f * R_P / (cam.cam_h - parts[:, 1])
    top = np.full(fg.shape, -np.inf, np.float32)
    order = np.argsort(parts[:, 1])                      # paint low->high: max wins
    for i in order:
        u, v, r = uv[i, 0], uv[i, 1], rad[i]
        u0, u1 = int(max(0, u - r)), int(min(719, u + r)); v0, v1 = int(max(0, v - r)), int(min(719, v + r))
        top[v0:v1 + 1, u0:u1 + 1] = parts[i, 1]
    sel = fg & np.isfinite(top)
    return top[sel], fg

rng = np.random.default_rng(0); out = {}
# v1 calibration
t_ids = rng.choice(2000, 40, replace=False)
dep, es = [], []
for t in t_ids:
    k = int(rng.integers(0, 11))
    d = D20 / "old_data" / str(t)
    color = cv2.imread(str(d / f"{k}_color.png")); depth = cv2.imread(str(d / f"{k}_depth.png"), cv2.IMREAD_UNCHANGED) / 1000.0
    p = np.load(d / f"{k}_particles.npy").reshape(-1, 4)
    e, fg = est(p, color); es.append(np.median(e)); dep.append(np.median(cam.cam_h - depth[fg & (depth > 0)]))
out["v1"] = dict(n=len(t_ids), true_depth_surface_median=float(np.median(dep)), estimator_median=float(np.median(es)))
off = out["v1"]["true_depth_surface_median"] - out["v1"]["estimator_median"]
sp = json.load(open(D20 / "splits.json"))["splits"]["train"]
v2 = []
for t in rng.choice(sp, 60, replace=False):
    d = D20 / "data/true_action_transitions_carrots" / str(t); k = int(rng.integers(0, 10))
    e, _ = est(np.load(d / f"{k}_after_particles.npy").reshape(-1, 4), cv2.imread(str(d / f"{k}_after_color.png")))
    v2.append(np.median(e))
paths = json.load(open(D19 / "cache/image_paths.json"))["states"]
s19 = []
raw19 = Path(json.load(open(D19 / "config.yaml")) if False else D19)
for s in sorted(paths, key=int)[::2]:
    cpath = D19 / paths[s]["initial_color"]
    ppath = cpath.parent / "initial_particles.npy"
    if not ppath.exists():
        continue
    e, _ = est(np.load(ppath).reshape(-1, 4), cv2.imread(str(cpath))); s19.append(np.median(e))
out["v2"] = dict(n=len(v2), estimator_median=float(np.median(v2)), calibrated=float(np.median(v2) + off),
                 p10_p90=[float(x + off) for x in np.percentile(v2, [10, 90])])
out["ds0019"] = dict(n=len(s19), estimator_median=float(np.median(s19)) if s19 else None,
                     calibrated=float(np.median(s19) + off) if s19 else None)
out["calibration_offset"] = off
for k in ("v2", "ds0019"):
    h = out[k]["calibrated"]
    if h is not None:
        out[k]["xy_shift_at_r5_if_plane_0.24"] = float(5 * abs(h - 0.24) / (18 - h))
out["px_per_unit"] = 64 / 14.4
print(json.dumps(out, indent=1))
Path("experiments/EXP-0062-flex-v2-train-rerun/results").mkdir(exist_ok=True)
json.dump(out, open("experiments/EXP-0062-flex-v2-train-rerun/results/gnn_plane_check.json", "w"), indent=1)
