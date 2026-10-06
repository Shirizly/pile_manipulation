"""Ceiling scoring: a perfect-physics predictor (the re-simulated state, optionally from a perturbed start) scored as if it were a
MODEL PREDICTION of the recorded DS-0016 outcome, in each model's frame. Reads artifacts/ceiling/resim_*.pt, writes
results/ceiling/ceiling_scores.json (rewritten atomically per tag; skips tags already done)."""
import json, os, sys, glob, numpy as np, torch
sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code")
from ceiling_lib import *
import score_zoom as Z
ART = "experiments/EXP-0072-zoom-window-nfd/artifacts/ceiling/"; FN = OUT + "ceiling_scores.json"
c = load_chains(); S0, S1 = c["states"].float(), c["states_"].float(); P0, P1 = c["p_starts"][:, :2].numpy(), c["p_stops"][:, :2].numpy()
act = torch.cat([c["p_starts"][:, :2], c["p_stops"][:, :2]], 1).float(); kind = c["kind"]; step = c["step"].numpy()
SP = {64: WindowSpec(res=64), 128: WindowSpec(res=128)}
O0, O1 = occ_from_particles(S0).cpu(), occ_from_particles(S1).cpu(); R64 = swept_region(act, "cpu"); R128 = region128(act)
Xw = {r: window_batch(S0, SIZES, P0, P1, SP[r]) for r in SP}; Yw = {r: window_batch(S1, SIZES, P0, P1, SP[r]) for r in SP}
Rw = {r: torch.stack([swept_region_window(P0[b], P1[b], SP[r]) for b in range(len(S0))]) for r in SP}
X128, Y128 = raster128(S0, SIZES), raster128(S1, SIZES)
W128 = Z.World128Model.__new__(Z.World128Model)


def frames(Sp):
    """dict frame -> (P,T,O,R) for predicted next states Sp (rows aligned to chains)."""
    f = {"w64_native": (occ_from_particles(Sp).cpu(), O1, O0, R64)}
    f["w128_native"] = (raster128(Sp, SIZES), Y128, X128, R128)
    for r in (64, 128):
        Pw = window_batch(Sp, SIZES, P0, P1, SP[r])
        f[f"window{r}"] = (Pw, Yw[r], Xw[r], Rw[r])
        f[f"paste_zoom{r}"] = (torch.stack([paste_delta(Pw[b], Xw[r][b], O0[b], P0[b], P1[b], SP[r]) for b in range(len(Sp))]), O1, O0, R64)
    P128 = raster128(Sp, SIZES)
    f["paste_world128"] = (torch.stack([W128.paste((P128, X128), None, None, b, O0[b], P0[b], P1[b], None) for b in range(len(Sp))]), O1, O0, R64)
    return f


def pooled(fr, mask=None):
    out = {}
    for k, (P, T, O, R) in fr.items():
        if mask is not None: P, T, O, R = P[mask], T[mask], O[mask], R[mask]
        out[k] = acc(P, T, O, R)
    return out


def mm(Sp, T, mask=None):
    d = (Sp[..., :2] - T[..., :2]).norm(dim=-1) * 1000
    if mask is not None: d = d[mask]
    return {"mm_mean": float(d.mean()), "mm_rms": float((d ** 2).mean().sqrt()), "frac_cubes_gt1mm": float((d > 1).float().mean()), "frac_cubes_gt2.5mm": float((d > 2.5).float().mean())}


res = json.load(open(FN)) if os.path.exists(FN) else {}
S0p, S1pool, PP0, PP1 = load_pools()
def pool_frames(Sq):
    Pp = {"w64_native": occ_from_particles(Sq).cpu()}
    for rr in (64, 128):
        Pw = window_batch(Sq, SIZES, PP0, PP1, SP[rr])
        Pp[f"paste_zoom{rr}"] = torch.stack([paste_delta(Pw[b], window_batch(S0p[b // 64][None], SIZES, PP0[b:b+1], PP1[b:b+1], SP[rr])[0], occ_from_particles(S0p[b // 64][None]).cpu()[0], PP0[b], PP1[b], SP[rr]) for b in range(len(Sq))])
    return Pp
if "_true_label_cap" not in res:      # the TRUE next state rendered in each frame (box-vs-disc raster-style cap), no noise
    res["_true_label_cap"] = {"onestep": pooled(frames(S1)), "pools_slateN": {k: float(np.nanmean(np.nanmean(pool_caps(P, S0p, S1pool), 0))) for k, P in pool_frames(S1pool).items()}}
    json.dump(res, open(FN + ".tmp", "w")); os.replace(FN + ".tmp", FN); print("true-label cap", res["_true_label_cap"], flush=True)
for f in sorted(glob.glob(ART + "resim_*.pt")):
    tag = os.path.basename(f)[6:-3]
    if tag in res: continue
    d = torch.load(f, weights_only=False); r = {"level_mm": d["level_mm"]}
    # ---- one step
    Sp = d["onestep"]; fr = frames(Sp); r["onestep"] = {"all": pooled(fr), "mm": mm(Sp, S1)}
    for k in ("scatter", "clump"): r["onestep"][k] = pooled(fr, torch.from_numpy(kind == k))
    # ---- chains: pushes 1..8, prediction of push t = resim chain state after t pushes (perturbed once at chain start)
    nc = len(S0) // 8; Sc = d["chain"].reshape(nc * 8, 20, 7)   # rows sorted (chain, step) as load_chains
    frc = frames(Sc); r["chain"] = {}
    for t in range(8):
        m = torch.from_numpy(step == t); r["chain"][str(t + 1)] = {"acc": pooled(frc, m), "mm": mm(Sc, S1, m)}
    # ---- pools (slateN of a perfect-physics predictor): w64 hard raster and the three paste frames
    Pp = pool_frames(d["pools"])
    r["pools_slateN"] = {}; r["pools_caps"] = {}
    for k, P in Pp.items():
        cp = pool_caps(P, S0p, S1pool); r["pools_caps"][k] = np.where(np.isnan(cp), -9, cp).tolist()
        r["pools_slateN"][k] = float(np.nanmean(np.nanmean(cp, 0)))     # mean over goals of mean over pools
    res[tag] = r; json.dump(res, open(FN + ".tmp", "w")); os.replace(FN + ".tmp", FN)
    print(tag, {k: round(v, 3) for k, v in r["onestep"]["all"].items()}, {k: round(v, 3) for k, v in r["pools_slateN"].items()}, flush=True)
