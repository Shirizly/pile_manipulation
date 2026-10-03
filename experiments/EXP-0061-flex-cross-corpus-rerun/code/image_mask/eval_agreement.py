"""Image-mask occupancy vs particle raster, ~200 states per corpus.
python -u experiments/EXP-0061-flex-cross-corpus-rerun/code/image_mask/eval_agreement.py
Writes figures/image_mask/agreement.jsonl (one record per state, appended as it goes)
and agreement_grids.npz (frac / raster / mass grids, for the figure script)."""
import json, sys, time
from pathlib import Path
import cv2, numpy as np, torch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from image_mask import Camera, GridSpec, segment, mask_to_occupancy
from FlexData.dataset import rasterize_disk, flex_xz_to_table
from transforms.functional import splat_particles_mass

OUT = REPO / "experiments/EXP-0061-flex-cross-corpus-rerun/figures/image_mask"
D20 = REPO / "datasets/DS-0020-training-data-flex-N864/old_data"
D19 = next((REPO / "datasets").glob("DS-0019-*")) / "data"
G = GridSpec(); CAM = Camera.load()
THR = [0.0, 0.05, 0.1, 0.25, 0.5]
PLANES = [0.0, 0.15, 0.3]


def sample_states(n=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for t, k in zip(rng.choice(2000, n, replace=False), rng.integers(0, 11, n)):
        out.append(dict(ds="DS-0020", id=f"{t}/{k}", color=D20 / f"{t}/{k}_color.png",
                        depth=D20 / f"{t}/{k}_depth.png", parts=D20 / f"{t}/{k}_particles.npy", meta={}))
    man = [json.loads(l) for l in open(D19.parent / "manifest.jsonl")]
    init = {r["state_idx"]: r for r in man if r["type"] == "state_init"}
    acts = {}
    for r in man:
        if r["type"] == "action" and r["valid"]:
            acts.setdefault(r["state_idx"], []).append(r)
    for s in sorted(init):     # every initial state (100) + one random after-state per slate (100)
        meta = dict(init_pos=init[s].get("init_pos"), n_particles=init[s]["n_particles"])
        out.append(dict(ds="DS-0019", id=f"{s}/init", color=D19 / init[s]["color_path"], depth=None,
                        parts=D19 / init[s]["positions_path"], meta=meta))
        a = acts[s][rng.integers(len(acts[s]))]
        out.append(dict(ds="DS-0019", id=f"{s}/{a['action_idx']}", color=D19 / a["after_color_path"], depth=None,
                        parts=D19 / a["after_positions_path"], meta=meta))
    return out


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def hidden_fraction(P, cell=0.125, layer=0.25):
    """Fraction of particles more than `layer` (one diameter) below the highest
    particle in their 0.125-unit XZ column: material a top-down view cannot see."""
    ij = np.floor(P[:, [0, 2]] / cell).astype(np.int64)
    ij -= ij.min(0)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    top = np.full(key.max() + 1, -np.inf); np.maximum.at(top, key, P[:, 1])
    return float((P[:, 1] < top[key] - layer).mean())


def main():
    states = sample_states()
    OUT.mkdir(parents=True, exist_ok=True)
    jl = OUT / "agreement.jsonl"; jl.write_text("")
    grids = {k: [] for k in ("frac", "frac_depth", "raster", "mass", "key")}
    mask_to_occupancy(np.zeros((720, 720), bool), CAM, G)       # warm LUT
    for n, s in enumerate(states):
        t0 = time.perf_counter(); color = cv2.imread(str(s["color"])); t1 = time.perf_counter()
        m = segment(color); frac = mask_to_occupancy(m, CAM, G); t2 = time.perf_counter()
        fr_planes = {h: mask_to_occupancy(m, Camera(**{**CAM.__dict__, "plane_y": h}), G) for h in PLANES[1:]}
        fd = None
        if s["depth"] is not None:
            d = cv2.imread(str(s["depth"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            fd = mask_to_occupancy(m, CAM, G, depth=d)
        P = np.load(s["parts"]).reshape(-1, 4)[:, :3]
        P = P[np.isfinite(P).all(1) & (np.abs(P).max(1) < 10)]
        px = flex_xz_to_table(P[:, [0, 2]]) * G.to_pxl + G.ctr
        ras = rasterize_disk(px, 64, 64, 1.0) > 0
        mass = splat_particles_mass(torch.from_numpy(px[None]).float(), (64, 64))[0].numpy()
        rec = dict(ds=s["ds"], id=s["id"], **s["meta"], n_particles_loaded=len(P),
                   y_max=float(P[:, 1].max()), hidden_frac=hidden_fraction(P),
                   raster_area=int(ras.sum()), mask_area_frac=float(frac.sum()),
                   t_read_ms=1e3 * (t1 - t0), t_seg_warp_ms=1e3 * (t2 - t1),
                   l1_frac=float(np.abs(frac - ras).mean()),
                   tv_mass=float(0.5 * np.abs(mass / mass.sum() - frac / frac.sum()).sum()),
                   tv_raster_vs_mass=float(0.5 * np.abs(mass / mass.sum() - ras / ras.sum()).sum()))
        for t in THR:
            b = frac > t
            rec[f"iou_t{t}"] = iou(b, ras); rec[f"area_ratio_t{t}"] = float(b.sum() / max(ras.sum(), 1))
            rec[f"l1_bin_t{t}"] = float(np.abs(b.astype(float) - ras).mean())
        for h, f in fr_planes.items():
            rec[f"iou_plane{h}_t0.1"] = iou(f > 0.1, ras)
        if fd is not None:
            rec["iou_depth_t0.1"] = iou(fd > 0.1, ras)
            rec["l1_depth_vs_plane"] = float(np.abs(fd - frac).mean())
        with open(jl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        grids["frac"].append(frac.astype(np.float16)); grids["raster"].append(ras)
        grids["mass"].append(mass.astype(np.float32)); grids["key"].append(f"{s['ds']}/{s['id']}")
        grids["frac_depth"].append((fd if fd is not None else np.full((64, 64), np.nan)).astype(np.float16))
        if n % 50 == 0:
            print(n, rec["ds"], rec["id"], f"iou0.1={rec['iou_t0.1']:.3f}", flush=True)
    tmp = OUT / "agreement_grids.tmp.npz"
    np.savez_compressed(tmp, **{k: np.array(v) for k, v in grids.items()}); tmp.replace(OUT / "agreement_grids.npz")
    print("done", len(states))


if __name__ == "__main__":
    main()
