"""EXP-0048: does the material physics set change 20 mm push outcomes more than
imperceptible state/action perturbations (and the simulator repeat floor)?

One GenesisOracleEnv (n_envs=32), physics switched at runtime with
apply_physics + env._real_settle_steps. 12 DS-0006 states x 32 perpendicular
20 mm pushes aimed at a random cube, every condition on the same states/actions.
Raw final positions are checkpointed (atomic) after every state; `--analyze`
recomputes results/sensitivity.json from the checkpoint.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
EXP = REPO / "experiments/EXP-0048-physics-sensitivity"
RAW = EXP / "artifacts/raw.pt"
RES = EXP / "results/sensitivity.json"
CORPUS = "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys"
CFG = REPO / "simple_mpc/config/config_oracle.yaml"
K = 32
SLATES = list(range(100, 112))
LEN = 0.020
P_TRAIN = dict(particle_friction=0.7, particle_density=450.0, box_friction=0.5, settle_steps=3000)
P_BINNED = dict(particle_friction=0.3, particle_density=1000.0, box_friction=0.3, settle_steps=3000)
P_ORACLE = dict(particle_friction=0.25, particle_density=750.0, box_friction=0.3, settle_steps=100)
# name -> (physics, state jitter mm, yaw jitter deg, action shift mm, action rot deg)
CONDS = {
    "ref":          (P_TRAIN, 0, 0, 0, 0),
    "repeat":       (P_TRAIN, 0, 0, 0, 0),
    "P_binned":     (P_BINNED, 0, 0, 0, 0),
    "P_oracle":     (P_ORACLE, 0, 0, 0, 0),
    "ref_back":     (P_TRAIN, 0, 0, 0, 0),   # after switching back: switch is reversible
    "state_0.25mm": (P_TRAIN, 0.25, 1.0, 0, 0),
    "state_0.5mm":  (P_TRAIN, 0.5, 1.0, 0, 0),
    "state_1.0mm":  (P_TRAIN, 1.0, 1.0, 0, 0),
    "act_0.5mm":    (P_TRAIN, 0, 0, 0.5, 0),
    "act_1.0mm":    (P_TRAIN, 0, 0, 1.0, 0),
    "act_rot1deg":  (P_TRAIN, 0, 0, 0, 1.0),
}
LO, HI = -0.064 + 0.004, 0.064 - 0.004


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    if str(path).endswith(".json"):
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def sample_actions(pos, rng):
    """32 perpendicular 20 mm pushes: blade starts 8 mm (1 cube + 3 mm) before a
    random cube centre, travelling toward it; start/end inside the tray minus 4 mm;
    rejected if any cube centre lies under the blade's touchdown footprint."""
    xy = pos[:, :2].numpy()
    acts = []
    while len(acts) < K:
        c = xy[rng.integers(len(xy))]
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)]); n = np.array([-u[1], u[0]])
        s = c - 0.008 * u; e = s + LEN * u
        if not (np.all((s > LO) & (s < HI)) and np.all((e > LO) & (e < HI))):
            continue
        d = xy - s
        under = (np.abs(d @ u) < 0.001 + 0.0045) & (np.abs(d @ n) < 0.020 + 0.0035)
        if under.any():
            continue
        acts.append(np.concatenate([s, e]))
    return torch.tensor(np.array(acts), dtype=torch.float32, device="cpu")


def perturb_state(st, s_mm, yaw_deg, g):
    st = st.clone()
    P = st.shape[0]
    st[:, :2] += torch.randn(P, 2, generator=g, device="cpu") * s_mm * 1e-3
    a = torch.randn(P, generator=g, device="cpu") * math.radians(yaw_deg) / 2
    qz = torch.stack([a.cos(), torch.zeros_like(a), torch.zeros_like(a), a.sin()], 1)  # wxyz
    q = st[:, 3:7]
    w1, x1, y1, z1 = qz.unbind(1); w2, x2, y2, z2 = q.unbind(1)
    st[:, 3:7] = torch.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                              w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)
    return st


def perturb_actions(A, s_mm, rot_deg, g):
    A = A.clone()
    if s_mm:
        A += torch.randn(A.shape, generator=g, device="cpu") * s_mm * 1e-3
    if rot_deg:
        sgn = torch.where(torch.rand(len(A), generator=g, device="cpu") < 0.5, -1.0, 1.0)
        a = sgn * math.radians(rot_deg)
        d = A[:, 2:] - A[:, :2]
        d = torch.stack([d[:, 0] * a.cos() - d[:, 1] * a.sin(), d[:, 0] * a.sin() + d[:, 1] * a.cos()], 1)
        A[:, 2:] = A[:, :2] + d
    return A


def simulate():
    from simple_mpc.genesis_oracle import GenesisOracleEnv
    from simple_mpc.learned_mpc import apply_physics, oracle_config_with_physics
    from simple_mpc.oracle_mpc import load_oracle_config
    from Genesis.binned_slate_dataset import BinnedSlateCorpus
    from utils import git_provenance
    tr = BinnedSlateCorpus.load(REPO / CORPUS).step(0)
    raw = torch.load(RAW, weights_only=False) if RAW.exists() else {
        "states": {}, "provenance": git_provenance(), "conds": {k: list(v[1:]) + [v[0]] for k, v in CONDS.items()}}
    cfg = oracle_config_with_physics(load_oracle_config(str(CFG)), P_TRAIN)
    cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K)
    import genesis as gs
    for s in SLATES:
        if s in raw["states"]:
            continue
        t0 = time.time()
        row = int((tr.slate_idx == s).nonzero()[0, 0])
        st0 = tr.states[row].float().cpu()
        rng = np.random.default_rng(1000 + s)
        g = torch.Generator(device="cpu").manual_seed(2000 + s)
        A0 = sample_actions(st0, rng)
        rec = {"state": st0, "actions": A0, "cond": {}}
        for name, (phys, sj, yj, asj, ar) in CONDS.items():
            st = perturb_state(st0, sj, yj, g) if sj else st0
            A = perturb_actions(A0, asj, ar, g) if (asj or ar) else A0
            apply_physics(env, phys); env._real_settle_steps = int(phys["settle_steps"])
            snap = {"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
            fin = env.rollout_candidates(A[:, None, :], snap, use_rollout_fidelity=False, record=False)
            rec["cond"][name] = {"start": st, "actions": A, "final": fin.float().cpu()}
        raw["states"][s] = rec
        save_atomic(raw, RAW)
        print(f"slate {s}: {len(CONDS)} conditions ({time.time() - t0:.0f}s)", flush=True)
        analyze()
    env.destroy()


def analyze():
    sys.path.insert(0, str(REPO / "experiments/EXP-0029-state-vs-pool-split/code"))
    from split_test import goal_tensors, values
    from simple_mpc.adapters import occ_for_scoring
    from scipy.stats import spearmanr
    raw = torch.load(RAW, weights_only=False)
    masks, dists = goal_tensors("cpu")
    ref_all = {}
    out = {"n_states": len(raw["states"]), "n_actions": K, "conditions": {}}
    per = {c: {"d_all": [], "d_moved": [], "pair_max": [], "frac2": [], "img": [], "dvr": [],
               "rho": [], "top1": [], "percept": [], "regret": [], "push_mean": [], "state_med": [], "dd_moved": []} for c in CONDS if c != "ref"}
    for s, rec in raw["states"].items():
        R = rec["cond"]["ref"]
        occ0 = occ_for_scoring(R["start"][None])
        occR = occ_for_scoring(R["final"])
        massR = occR.flatten(1).sum(1)
        dvR = values(occR, masks, dists)[..., 0] - values(occ0, masks, dists)[..., 0]  # (K,G)
        sdR = dvR.std(0).clamp_min(1e-9)
        movedR = (R["final"][..., :2] - R["start"][None, :, :2]).norm(dim=-1) > 1e-3   # (K,P)
        for c, v in per.items():
            C = rec["cond"][c]
            d = (C["final"][..., :2] - R["final"][..., :2]).norm(dim=-1) * 1e3          # mm (K,P)
            v["d_all"].append(d.flatten()); v["d_moved"].append(d[movedR])
            dd = ((C["final"][..., :2] - C["start"][None, :, :2]) - (R["final"][..., :2] - R["start"][None, :, :2])).norm(dim=-1) * 1e3
            v["dd_moved"].append(dd[movedR])
            v["push_mean"].append(d.mean(1)); v["state_med"].append(float(d[movedR].median()) if movedR.any() else 0.0)
            v["pair_max"].append(d.max(1).values); v["frac2"].append((d > 2).float().mean(1))
            occC = occ_for_scoring(C["final"])
            v["img"].append((occC - occR).abs().flatten(1).sum(1) / massR)
            occC0 = occ_for_scoring(C["start"][None])
            v["percept"].append(float((occC0 - occ0).abs().sum() / occ0.sum()))
            dvC = values(occC, masks, dists)[..., 0] - values(occC0, masks, dists)[..., 0]
            v["dvr"].append(((dvC - dvR).abs() / sdR).flatten())
            for gi in range(dvR.shape[1]):
                rho = spearmanr(dvR[:, gi].numpy(), dvC[:, gi].numpy()).correlation
                v["rho"].append(float(rho))
                v["top1"].append(float(int(dvR[:, gi].argmin()) == int(dvC[:, gi].argmin())))
                v["regret"].append(float((dvR[int(dvC[:, gi].argmin()), gi] - dvR[:, gi].min()) / sdR[gi]))
    q = lambda x, p: float(np.quantile(x, p))
    for c, v in per.items():
        da, dm = torch.cat(v["d_all"]).numpy(), torch.cat(v["d_moved"]).numpy()
        pm, img, dvr = torch.cat(v["pair_max"]).numpy(), torch.cat(v["img"]).numpy(), torch.cat(v["dvr"]).numpy()
        out["conditions"][c] = {
            "cube_diff_mm_all": {"median": q(da, .5), "p90": q(da, .9), "max": float(da.max())},
            "cube_diff_mm_moved_in_ref": {"n": int(len(dm)), "median": q(dm, .5), "p90": q(dm, .9)},
            "displacement_diff_mm_moved_in_ref": {"median": q(torch.cat(v["dd_moved"]).numpy(), .5),
                                                  "p90": q(torch.cat(v["dd_moved"]).numpy(), .9)},
            "per_push_max_cube_diff_mm": {"median": q(pm, .5), "p90": q(pm, .9)},
            "frac_cubes_diff_gt_2mm": float(torch.cat(v["frac2"]).mean()),
            "img_L1_over_mass": {"median": q(img, .5), "p90": q(img, .9)},
            "input_perceptibility_L1_over_mass": float(np.mean(v["percept"])),
            "abs_dv_diff_over_between_action_sd": {"median": q(dvr, .5), "p90": q(dvr, .9)},
            "spearman_dv_across_actions": {"median": float(np.nanmedian(v["rho"])),
                                           "p10": float(np.nanquantile(v["rho"], .1))},
            "top1_same": float(np.mean(v["top1"])),
            "top1_regret_over_sd": {"mean": float(np.mean(v["regret"])), "p90": q(np.array(v["regret"]), .9)},
            "per_state_median_moved_cube_diff_mm": v["state_med"],
        }
    rep = torch.cat(per["repeat"]["push_mean"])
    for c, v in per.items():
        out["conditions"][c]["frac_pushes_mean_diff_gt_repeat"] = float((torch.cat(v["push_mean"]) > rep).float().mean())
    # reference movement scale
    mv = []
    for rec in raw["states"].values():
        R = rec["cond"]["ref"]
        mv.append((R["final"][..., :2] - R["start"][None, :, :2]).norm(dim=-1).flatten() * 1e3)
    mv = torch.cat(mv).numpy()
    out["ref_displacement_mm"] = {"median": q(mv, .5), "p90": q(mv, .9), "max": float(mv.max()),
                                  "frac_moved_gt_1mm": float((mv > 1).mean())}
    save_atomic(out, RES)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        print(json.dumps(analyze(), indent=1))
    else:
        simulate()
