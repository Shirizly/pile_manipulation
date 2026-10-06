"""PILOT (EXP-0072 active-collection plan, 2026-10-06). Do the cheap uncertainty proxies predict the REALISED per-row
error of the zoom-64 NFD? Rows: DS-0016 test_chains_v2_clean + test_pools_v2 (valid only), ground truth already simulated.
Error is measured in the model's own window frame (1 mm/px, 64x64): e_sw = SE inside eval_narrow's swept region,
e_win = SE over the whole window, e_chg = SE where truth or prediction differs from the input by > 0.5.
Proxies (all computed WITHOUT the truth): ensemble disagreement (ft100/ft300/scratch100 [+world128_300]), perturbation
variance (action jitter / input jitter / both), input-gradient sensitivity, heuristics, and the trivial control
`pred_mass` = sum|pred - x| (error scales with how much the model thinks moves).
Outputs runs/active_pilot/proxy_rows.npz + results/active_proxy_corr.json.   python -u, PYTHONPATH=.
"""
import sys, json, glob, math, time
import numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from score_zoom import load_unet, World128Model, rows, SIZES, D
from model.zoom_nfd.window import *
from model.zoom_nfd.window import world_to_window_idx, push_axes, quat_yaw
from model.retrieval.frame import yaw_from_quat, yaw_to_quat

R = "experiments/EXP-0072-zoom-window-nfd/runs/"
AP = "experiments/EXP-0073-active-data-mining/runs/active_pilot/"
OUT = AP
spec = WindowSpec(res=64)
dev = "cuda"


def load_rows():
    out = {k: [] for k in ("S", "S_", "P0", "P1", "kind", "step", "pool", "src", "ps3", "pe3", "ang")}
    for d in rows(D + "test_chains_v2_clean/_*_data.pt"):
        n = len(d["states"]); out["S"].append(d["states"].float()); out["S_"].append(d["states_"].float())
        out["ps3"].append(d["p_starts"].float()); out["pe3"].append(d["p_stops"].float()); out["ang"].append(d["angles"].float())
        out["P0"].append(d["p_starts"][:, :2].numpy()); out["P1"].append(d["p_stops"][:, :2].numpy())
        out["kind"] += list(d["start_kind"]); out["step"].append(d["chain_step"].numpy()); out["pool"].append(-np.ones(n, int)); out["src"] += ["chain"] * n
    off = 0
    for d in rows(D + "test_pools_v2/pools_*.pt"):
        v = d["valid"].bool(); n = int(v.sum())
        out["S"].append(d["states"].float()[v]); out["S_"].append(d["states_"].float()[v])
        out["ps3"].append(d["p_starts"].float()[v]); out["pe3"].append(d["p_stops"].float()[v]); out["ang"].append(d["angles"].float()[v])
        out["P0"].append(d["p_starts"][v][:, :2].numpy()); out["P1"].append(d["p_stops"][v][:, :2].numpy())
        out["kind"] += [k for k, vv in zip(d["start_kind"] if isinstance(d["start_kind"], list) and len(d["start_kind"]) == len(v) else ["pool"] * len(v), v) if vv]
        out["step"].append(-np.ones(n, int)); out["pool"].append(d["pool_idx"][v].numpy() + 1000 * off); out["src"] += ["pool"] * n; off += 1
    return dict(S=torch.cat(out["S"]), S_=torch.cat(out["S_"]), P0=np.concatenate(out["P0"]), P1=np.concatenate(out["P1"]),
                kind=np.array(out["kind"]), step=np.concatenate(out["step"]), pool=np.concatenate(out["pool"]), src=np.array(out["src"]), ps3=torch.cat(out["ps3"]), pe3=torch.cat(out["pe3"]), ang=torch.cat(out["ang"]))


@torch.no_grad()
def fwd(net, x):
    return torch.cat([torch.sigmoid(net(x[i:i + 256].to(dev))).squeeze(1).cpu() for i in range(0, len(x), 256)])


def build_x(S, P0, P1):
    X = window_batch(S, SIZES, P0, P1, spec); A = plates_batch(P0, P1, spec)
    return X, torch.cat([X[:, None], A], 1)


def jitter_states(S, rng, xy_mm, yaw_deg):
    out = S.clone()
    out[..., :2] += torch.from_numpy(rng.normal(0, xy_mm / 1000, out[..., :2].shape)).float()
    yaw = yaw_from_quat(out[..., 3:7]) + torch.from_numpy(rng.normal(0, np.deg2rad(yaw_deg), out.shape[:2])).float()
    out[..., 3:7] = yaw_to_quat(yaw); return out


def jitter_action(P0, P1, rng, xy_mm, ang_deg):
    B = len(P0); sh = rng.normal(0, xy_mm / 1000, (B, 2)); a = rng.normal(0, np.deg2rad(ang_deg), B)
    d = P1 - P0; c, s = np.cos(a), np.sin(a)
    d2 = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1)
    return P0 + sh, P0 + sh + d2


def errs(Pw, Xw, Yw, Rw):
    se = (Pw - Yw) ** 2
    chg = ((Yw - Xw).abs() > .5) | ((Pw - Xw).abs() > .5)
    return dict(e_sw=(se * Rw).sum((1, 2)).numpy(), e_win=se.sum((1, 2)).numpy(), e_chg=(se * chg).sum((1, 2)).numpy(),
                e_pers_sw=(((Xw - Yw) ** 2) * Rw).sum((1, 2)).numpy())


def heuristics(S, P0, P1, Rw):
    B = len(S); h = {k: np.zeros(B) for k in ("n_in_win", "n_swept", "n_contacts_win", "gap_first", "wall_min", "n_swept_contact")}
    for b in range(B):
        xy = S[b, :, :2].numpy().astype(float); u, v, L = push_axes(P0[b], P1[b])
        ij = world_to_window_idx(xy, P0[b], u, v, spec)
        inwin = (ij[:, 0] >= 0) & (ij[:, 0] < 64) & (ij[:, 1] >= 0) & (ij[:, 1] < 64)
        s = (xy - P0[b]) @ u; t = (xy - P0[b]) @ v
        swept = (np.abs(t) <= 0.024) & (s >= -0.02) & (s <= L + 0.0025)
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) + np.eye(len(xy)) * 1
        con = d < 0.0056                              # centres closer than ~ cube edge + 0.6 mm = touching
        h["n_in_win"][b] = inwin.sum(); h["n_swept"][b] = swept.sum()
        h["n_contacts_win"][b] = (con[inwin][:, inwin]).sum() / 2
        h["n_swept_contact"][b] = con[swept].any(1).sum()
        fs = s[swept & (np.abs(t) <= 0.021)]
        h["gap_first"][b] = (fs[fs > 0].min() - 0.0025 - 0.001 if (fs > 0).any() else 0.03)
        pts = np.stack([P0[b], P1[b]]); h["wall_min"][b] = (0.064 - np.abs(pts)).min()
    return h


def main():
    t0 = time.time(); rng = np.random.default_rng(0); torch.manual_seed(0)
    Dd = load_rows(); S, S_, P0, P1 = Dd["S"], Dd["S_"], Dd["P0"], Dd["P1"]; B = len(S)
    print("rows", B, {k: int((Dd["src"] == k).sum()) for k in ("chain", "pool")}, flush=True)
    Xw, x = build_x(S, P0, P1); Yw = window_batch(S_, SIZES, P0, P1, spec)
    Rw = torch.stack([swept_region_window(P0[b], P1[b], spec) for b in range(B)]).float()
    print("rasterised", round(time.time() - t0), flush=True)
    nets = {n: load_unet(R + f"{n}/unet_best.pth").to(dev) for n in ("ft100", "ft300", "scratch100")}
    pred = {n: fwd(net, x) for n, net in nets.items()}
    w128 = World128Model(load_unet(R + "world128_300/unet_best.pth"))
    pred["world128_300"] = torch.cat([w128.windows(S[i:i + 256], P0[i:i + 256], P1[i:i + 256], spec)[1] for i in range(0, B, 256)])
    res = {}
    for n in pred: res[f"err_{n}"] = errs(pred[n], Xw, Yw, Rw)
    P = pred["ft100"]; E = errs(P, Xw, Yw, Rw)
    P3 = torch.stack([pred[n] for n in ("ft100", "ft300", "scratch100")]); P4 = torch.stack(list(pred.values()))
    prox = {}
    def dis(Pm, mask=None):
        v = Pm.var(0)
        return (v * (Rw if mask is None else mask)).sum((1, 2)).numpy()
    prox["ens3_var_sw"] = dis(P3); prox["ens3_var_win"] = dis(P3, torch.ones_like(Rw))
    prox["ens4_var_sw"] = dis(P4); prox["ens4_var_win"] = dis(P4, torch.ones_like(Rw))
    prox["ens3_chg_var_sw"] = (( (P3 - Xw[None]).var(0)) * Rw).sum((1, 2)).numpy()
    prox["ens3_maxstd"] = P3.std(0).flatten(1).max(1).values.numpy()
    # ensemble-mean error (what you'd get from averaging) for reference
    res["err_ens3mean"] = errs(P3.mean(0), Xw, Yw, Rw)
    prox["pred_mass"] = ((P - Xw).abs() * Rw).sum((1, 2)).numpy()
    prox["pred_mass_win"] = (P - Xw).abs().sum((1, 2)).numpy()
    prox["pred_entropy_sw"] = (-(P.clamp(1e-4, 1 - 1e-4).log() * P + (1 - P).clamp_min(1e-4).log() * (1 - P)) * Rw).sum((1, 2)).numpy()
    prox["pred_grayness_sw"] = ((P * (1 - P)) * Rw).sum((1, 2)).numpy()      # mass of unresolved (blurred) pixels
    # ---- perturbation variance (K draws), ft100
    K = 8
    for tag, (axy, aang, ixy, iyaw) in dict(act=(1.0, 1.0, 0, 0), inp=(0, 0, 0.5, 1.0), both=(1.0, 1.0, 0.5, 1.0)).items():
        Ps = []
        for k in range(K):
            S2 = jitter_states(S, rng, ixy, iyaw) if ixy else S
            a0, a1 = jitter_action(P0, P1, rng, axy, aang) if axy else (P0, P1)
            _, xk = build_x(S2, a0, a1); Ps.append(fwd(nets["ft100"], xk))
        Ps = torch.stack(Ps)
        prox[f"pert_{tag}_var_sw"] = (Ps.var(0) * Rw).sum((1, 2)).numpy()
        prox[f"pert_{tag}_var_win"] = Ps.var(0).sum((1, 2)).numpy()
        prox[f"pert_{tag}_shift_sw"] = (((Ps - P[None]).abs()) * Rw).sum((2, 3)).mean(0).numpy()
        print("pert", tag, round(time.time() - t0), flush=True)
    # ---- input-gradient sensitivity (ft100): d sum_sw(pred) / d x
    g_img = np.zeros(B); g_plate = np.zeros(B); g_all = np.zeros(B); net = nets["ft100"]
    for i in range(0, B, 128):
        xb = x[i:i + 128].to(dev).clone().requires_grad_(True)
        o = torch.sigmoid(net(xb)).squeeze(1); (o * Rw[i:i + 128].to(dev)).sum().backward()
        g = xb.grad.cpu(); g_img[i:i + 128] = g[:, 0].flatten(1).norm(dim=1).numpy(); g_plate[i:i + 128] = g[:, 1:].flatten(1).norm(dim=1).numpy()
        g_all[i:i + 128] = g.flatten(1).norm(dim=1).numpy()
    prox["grad_img"], prox["grad_plate"], prox["grad_all"] = g_img, g_plate, g_all
    prox.update({"heur_" + k: v for k, v in heuristics(S, P0, P1, Rw).items()})
    prox["heur_chain_step"] = Dd["step"].astype(float)
    prox["heur_clump"] = (Dd["kind"] == "clump").astype(float)
    np.savez_compressed(OUT + "proxy_rows.npz", **{f"proxy_{k}": v for k, v in prox.items()},
                        **{f"{m}_{k}": v for m, e in res.items() for k, v in e.items()}, **{f"ft100_{k}": v for k, v in E.items()},
                        kind=Dd["kind"], src=Dd["src"], pool=Dd["pool"], step=Dd["step"])
    print("saved", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
