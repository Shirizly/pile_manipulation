"""Fit/verify the cam_idx-0 pinhole (f, cx, cy) by maximising IoU between the
colour mask and projected particles (discs of the particle radius in px), on
states from BOTH corpora; writes cam_params.json beside this file.
python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/image_mask/fit_camera.py"""
import json, sys, time
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from image_mask import Camera, segment, CAM_JSON
REPO = HERE.parents[3]
D20 = REPO / "datasets/DS-0020-training-data-flex-N864/old_data"
D19 = next((REPO / "datasets").glob("DS-0019-*")) / "data"

def states():
    rng = np.random.default_rng(0)
    out = [(f"DS-0020/{t}/{k}", D20 / f"{t}/{k}_color.png", D20 / f"{t}/{k}_particles.npy")
           for t, k in zip(rng.integers(0, 2000, 8), rng.integers(0, 11, 8))]
    for s in rng.choice(100, 8, replace=False):
        out.append((f"DS-0019/{s}/init", D19 / f"{s}/initial_color.png", D19 / f"{s}/initial_particles.npy"))
    return out

def render(uv, r):
    img = np.zeros((720, 720), np.uint8)
    for u, v in np.round(uv).astype(int):
        if 0 <= u < 720 and 0 <= v < 720:
            cv2.circle(img, (u, v), r, 1, -1)
    return img > 0

def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)

S = []
for name, c, p in states():
    m = segment(cv2.imread(str(c)))
    P = np.load(p).reshape(-1, 4)[:, :3]
    P = P[np.isfinite(P).all(1) & (np.abs(P).max(1) < 10)]
    S.append((name, m, P))

R_PX = 6   # particle_r 0.125 * f / ~18
def loss(th, subset=S):
    cam = Camera(f=th[0], cx=th[1], cy=th[2])
    return -np.mean([iou(render(cam.project(P), R_PX), m) for _, m, P in subset])

t = time.time()
nominal = Camera()
x0 = [nominal.f, nominal.cx, nominal.cy]
res = minimize(loss, x0, method="Nelder-Mead",
               options=dict(initial_simplex=[x0, [x0[0] + 15, x0[1], x0[2]], [x0[0], x0[1] + 3, x0[2]],
                                             [x0[0], x0[1], x0[2] + 3]], xatol=0.05, fatol=1e-5, maxiter=150))
fit = Camera(f=res.x[0], cx=res.x[1], cy=res.x[2])
per = [(n, iou(render(nominal.project(P), R_PX), m), iou(render(fit.project(P), R_PX), m)) for n, m, P in S]
print(f"fit in {time.time()-t:.0f}s: f={res.x[0]:.2f} cx={res.x[1]:.2f} cy={res.x[2]:.2f}")
for n, a, b in per:
    print(f"  {n:18s} IoU nominal {a:.4f} fitted {b:.4f}")
fov_fit = 2 * np.degrees(np.arctan(360 / res.x[0]))
out = dict(f=nominal.f, cx=nominal.cx, cy=nominal.cy, cam_h=18.0, width=720, height=720, plane_y=0.0,
           model="u = cx + f*x/(cam_h - y), v = cy + f*z/(cam_h - y); FleX world, y up; camera at (0,18,0) pitch -90",
           source="derived: PyFleX default vertical fov 45 deg -> f = 360/tan(22.5 deg); cx=cy=359.5 (pixel centres); "
                  "cam pose from dyn-res-pile-manip env/flex_env.py:242-251 (cam_idx 0, global_scale 24). Verified by fit below.",
           fit=dict(f=float(res.x[0]), cx=float(res.x[1]), cy=float(res.x[2]), fov_deg=float(fov_fit),
                    objective=f"mean IoU(colour mask, particle discs r={R_PX}px) over {len(S)} states (8 DS-0020, 8 DS-0019)",
                    mean_iou_nominal=float(np.mean([a for _, a, _ in per])),
                    mean_iou_fit=float(np.mean([b for _, _, b in per])),
                    per_state={n: [float(a), float(b)] for n, a, b in per}))
tmp = CAM_JSON.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=1)); tmp.replace(CAM_JSON)
print("wrote", CAM_JSON)
