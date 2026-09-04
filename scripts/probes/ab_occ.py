"""Same fit, same metric, two occupancy sources. Isolates whether the report's
'nothing beats persistence' on scattered cubes is a property of the physics or
of the rasteriser the dataset registry bakes in."""
import argparse, torch
from dmdc_baseline import load_transition_arrays, split_by_episode
from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (canonicalise, fit_operator, metrics,
                                  swept_region_mask, actions_to_pixels)
from transforms.functional import (blend_push_prediction, from_push_frame,
                                   push_frame_validity_mask)

ap = argparse.ArgumentParser()
ap.add_argument("--cfg", default="configs/dataset/genesis_foresight_L040.yaml")
ap.add_argument("--glob", default="Genesis/data/foresight/L040/**/*_data.pt")
ap.add_argument("--cube-size", type=float, default=0.005)
ap.add_argument("--res", type=int, default=64)
ap.add_argument("--crop", type=float, default=0.5)
ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--min-push-mm", type=float, default=39.0)
ap.add_argument("--ridge", type=float, default=1.0)
a = ap.parse_args()
H = W = 64

def gblur(x, sig):
    if sig <= 0: return x
    k = int(2 * round(3 * sig) + 1)
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sig * sig)); g = g / g.sum()
    x = torch.nn.functional.conv2d(x.unsqueeze(1), g.view(1,1,1,-1), padding=(0,k//2))
    return torch.nn.functional.conv2d(x, g.view(1,1,-1,1), padding=(k//2,0)).squeeze(1)

def evaluate(name, o0, o1, act, ep):
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique()
    g = torch.Generator().manual_seed(0)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps)//4)].tolist())
    te = torch.tensor([int(e) in val for e in ep]); tr = ~te
    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], a.res, a.crop).reshape(int(tr.sum()), -1).T
    A = fit_operator(Y0, Y1, a.ridge, toward_identity=True)
    bmd = (Y1 - Y0).mean(1, keepdim=True)
    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5*plate + 2.0, 0.5*plate)
    base = metrics(ote, o1te, ote, region=region)["rms"]
    Y0te = canonicalise(ote, ste, ete, a.res, a.crop).reshape(int(te.sum()), -1).T
    msk = push_frame_validity_mask(ste, ete, (H, W), (a.res, a.res), a.crop)
    def world(Yp):
        back = from_push_frame(Yp.T.reshape(-1, a.res, a.res), ste, ete, (H,W), a.crop)
        return blend_push_prediction(back, ote, msk).clamp_min(0.0)
    pct = lambda P: 100 * metrics(P, o1te, ote, region=region)["rms"] / base
    print(f"{name:34s} occ_mean={float(o0.mean()):.4f} region={float(region.float().mean()):.3f} "
          f"| linear {pct(world(A @ Y0te)):6.1f}%  mean-delta {pct(world(Y0te + bmd)):6.1f}%  "
          f"identity {pct(world(Y0te)):6.1f}%")

d = load_transition_arrays(a.cfg, split="train")
evaluate("registry occ, as stored", gblur(d.occ_t, a.blur), gblur(d.occ_t1, a.blur),
         d.actions, d.episode_ids)
evaluate("registry occ, TRANSPOSED", gblur(d.occ_t.transpose(1,2).contiguous(), a.blur),
         gblur(d.occ_t1.transpose(1,2).contiguous(), a.blur), d.actions, d.episode_ids)
o0, o1, act, ep, _, _ = load_transition_fields(a.glob, 64, a.blur, "mean", a.min_push_mm,
                                         "cpu", view="mask", min_grains=1.0,
                                         cube_size=a.cube_size)
evaluate("sand_to_mask rasteriser", o0, o1, act, ep)
