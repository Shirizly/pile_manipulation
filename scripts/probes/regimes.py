"""Same code, same metric, same rasteriser: scattered monolayer vs piled cubes
vs sand. The comparison docs/sand_manipulation.md sec.8 makes across two
projection paths, made within one."""
import argparse, torch
from occupancy_foresight import load_transition_fields, BOUNDS
from fit_linear_foresight import (canonicalise, fit_operator, metrics,
                                  swept_region_mask, actions_to_pixels)
from transforms.functional import (blend_push_prediction, from_push_frame,
                                   push_frame_validity_mask)
H = W = 64
def run(name, glob, cube_size, min_push, min_grains, res, crop, blur, max_ep=None):
    o0, o1, act, ep, _, _ = load_transition_fields(glob, 64, blur, "mean", min_push, "cpu",
                                             view="mask", min_grains=min_grains,
                                             cube_size=cube_size, max_episodes=max_ep)
    s_px, e_px = actions_to_pixels(act, (BOUNDS["x_min"], BOUNDS["y_min"]),
                                   (BOUNDS["x_max"], BOUNDS["y_max"]), (H, W))
    eps = ep.unique(); g = torch.Generator().manual_seed(0)
    val = set(eps[torch.randperm(len(eps), generator=g)][:max(1, len(eps)//4)].tolist())
    te = torch.tensor([int(e) in val for e in ep]); tr = ~te
    Y0 = canonicalise(o0[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    Y1 = canonicalise(o1[tr], s_px[tr], e_px[tr], res, crop).reshape(int(tr.sum()), -1).T
    A = fit_operator(Y0, Y1, 1.0, toward_identity=True)
    bmd = (Y1 - Y0).mean(1, keepdim=True)
    ote, o1te, ste, ete = o0[te], o1[te], s_px[te], e_px[te]
    plate = 0.04 / 0.128 * W
    region = swept_region_mask(ste, ete, (H, W), 0.5*plate+2.0, 0.5*plate)
    base = metrics(ote, o1te, ote, region=region)["rms"]
    Y0te = canonicalise(ote, ste, ete, res, crop).reshape(int(te.sum()), -1).T
    msk = push_frame_validity_mask(ste, ete, (H, W), (res, res), crop)
    def world(Yp):
        back = from_push_frame(Yp.T.reshape(-1, res, res), ste, ete, (H,W), crop)
        return blend_push_prediction(back, ote, msk).clamp_min(0.0)
    ex = lambda P: 1 - metrics(P, o1te, ote, region=region)["rms"] / base
    el, em = ex(world(A @ Y0te)), ex(world(Y0te + bmd))
    print(f"{name:38s} M={o0.shape[0]:5d}  mean-delta {em:6.3f}  linear {el:6.3f}  "
          f"operator margin {el-em:+6.3f}")

for res, crop in [(64, 0.5), (64, 1.0)]:
    print(f"\n--- res={res} crop={crop} blur=1.0, mask view, ridge->I ---")
    run("cubes n50 SCATTERED monolayer, blind", "Genesis/data/foresight/L040/**/*_data.pt",
        0.005, 39.0, 1.0, res, crop, 1.0)
    run("cubes n50 scattered, CONTACT-sampled", "Genesis/data/foresight/scatter_contact/**/*_data.pt",
        0.005, 19.0, 1.0, res, crop, 1.0)
    run("cubes n30 PILED (heap)", "Genesis/data/foresight/pile30_L020/**/*_data.pt",
        0.005, 19.0, 1.0, res, crop, 1.0)
    run("cubes n20 PILED (2 layers)", "Genesis/data/cube_spectrum/n20/*_data.pt",
        0.005, 19.9, 1.0, res, crop, 1.0)
    run("sand varied (40 eps)", "Genesis/data/sand/varied/*_data.pt",
        None, 19.9, 2.0, res, crop, 1.0, max_ep=40)
