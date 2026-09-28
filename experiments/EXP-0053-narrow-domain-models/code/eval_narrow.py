"""EXP-0053 offline evaluation on the clean narrow-domain test set (DS-0009).

For each model (OCC_ADAPTERS id), on the planner's 64 px slate grid (occ_from_particles):
  * accuracy_1: METRICS.md `accuracy` = 1 - rms(pred - truth) / rms(prev - truth), pooled over
    rows, swept region (push rectangle: plate half-width 20 mm + 4 mm, 20 mm pad along the push;
    rasterised with the same function as the images, so no axis-convention risk); split by
    start kind (scatter / clump);
  * rollout accuracy_k, k = 1..4, on the test chains: the model is fed its OWN previous
    prediction, truth = the recorded state after k pushes; region = union of the k swept regions;
  * slateN on the test pools (same-state, 64 pushes each), soft truth (occ_for_scoring),
    lyapunov, 12-goal set (EXP-0046 recommendation) + two_squares; capture per (pool, goal).
Results JSON rewritten after every model (resumable: finished models are skipped).
"""
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments/EXP-0043-batched-closed-loop/code"))
from batched_closed_loop import goal_mask
from Baselines.common.goals import dist_field_from_mask
from simple_mpc.adapters import make_occ_adapter, occ_for_scoring, occ_from_particles
from simple_mpc.learned_mpc import lyap

GOALS = ["letter_O", "letter_T", "letter_F", "letter_S", "letter_X", "letter_M", "letter_W", "letter_J",
         "letter_L", "letter_I", "quadrant_0", "quadrant_1", "two_squares"]
D = REPO / "Genesis/data/narrow_l20_n20"
RES = REPO / "experiments/EXP-0053-narrow-domain-models/results/offline_eval.json"


def swept_region(act, dev):
    """(B,4) metres -> (B,64,64) bool: dense points over the push rectangle rasterised
    with occ_from_particles (same grid/convention as the images)."""
    B = len(act)
    s, e = act[:, :2], act[:, 2:]
    d = e - s; L = d.norm(dim=1, keepdim=True).clamp_min(1e-6); u = d / L; n = torch.stack([-u[:, 1], u[:, 0]], 1)
    a = torch.linspace(-0.02, 1.0, 26)                      # along (fraction of L, plus pad handled below)
    al = torch.cat([torch.linspace(-0.020, 0.0, 6), torch.linspace(0.0, 1.0, 21)])   # -20 mm pad + push
    lat = torch.linspace(-0.024, 0.024, 25)
    pts = []
    for i in range(B):
        A_ = torch.where(al < 0, al, al * L[i, 0])[:, None, None]
        P = s[i] + A_ * u[i] + lat[None, :, None] * n[i]
        pts.append(P.reshape(-1, 2))
    P = torch.stack(pts)                                  # (B, M, 2)
    parts = torch.cat([P, torch.full((B, P.shape[1], 1), 0.0125)], 2)
    return occ_from_particles(parts, dev) > 0


def acc(pred, truth, prev, region):
    r = region.float()
    num = (((pred - truth) ** 2) * r).sum(); den = (((prev - truth) ** 2) * r).sum()
    return float(1 - torch.sqrt(num / den.clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["nfd_3ch_narrow_l20", "nfd_3ch_narrow_l20_wide",
                                                    "linear_narrow_l20_res32", "linear_narrow_l20_res64",
                                                    "nfd_3ch_randlen", "nfd_residual_worldframe_noaug_ep43",
                                                    "linear_switched_soft"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = json.loads(RES.read_text()) if RES.exists() else {}
    RES.parent.mkdir(parents=True, exist_ok=True)
    cpu = lambda d: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in d.items()}   # collector saved some on cuda
    ch = [cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in sorted(glob.glob(str(D / "test_chains/_*_data.pt")))]
    pools = [cpu(torch.load(f, map_location="cpu", weights_only=False)) for f in sorted(glob.glob(str(D / "test_pools/pools_*.pt")))]
    Dist = {g: torch.from_numpy(dist_field_from_mask(goal_mask(g))).float() for g in GOALS}
    for m in a.models:
        if m in out:
            continue
        try:
            ad = make_occ_adapter(m, dev, "corner")
        except KeyError:
            print("skip (not registered/trained):", m); continue
        r = {}
        # --- 1-step accuracy on all chain rows
        preds, truths, prevs, regs, kinds = [], [], [], [], []
        for d in ch:
            act = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
            o0 = occ_from_particles(d["states"].float(), dev); o1 = occ_from_particles(d["states_"].float(), dev)
            with torch.no_grad():
                p = torch.cat([ad.predict_step(o0[i:i + 128], act[i:i + 128].to(dev)) for i in range(0, len(act), 128)])
            preds.append(p.float().cpu()); truths.append(o1.float().cpu()); prevs.append(o0.float().cpu())
            regs.append(swept_region(act, "cpu").cpu()); kinds += list(d["start_kind"])
        P, T, O, R = map(torch.cat, (preds, truths, prevs, regs))
        kinds = np.array(kinds)
        r["accuracy_1"] = acc(P, T, O, R)
        for k in ("scatter", "clump"):
            ix = torch.from_numpy(np.nonzero(kinds == k)[0])
            r[f"accuracy_1_{k}"] = acc(P[ix], T[ix], O[ix], R[ix])
        # --- rollout accuracy on chains (own predictions fed back)
        roll = {kk: [] for kk in range(1, 5)}
        for d in ch:
            E = int(d["chain_env"].max()) + 1
            for e in range(E):
                rows = [int(i) for i in torch.nonzero(d["chain_env"] == e)[:, 0]]
                rows = sorted(rows, key=lambda i: int(d["chain_step"][i]))[:4]
                cur = occ_from_particles(d["states"][rows[0]][None].float(), dev)
                start = cur.clone(); reg = torch.zeros(1, 64, 64, dtype=torch.bool)
                for kk, i in enumerate(rows, 1):
                    act = torch.cat([d["p_starts"][i, :2], d["p_stops"][i, :2]])[None].float()
                    with torch.no_grad():
                        cur = ad.predict_step(cur, act.to(dev)).float()
                    reg = reg | swept_region(act, "cpu").cpu()
                    truth = occ_from_particles(d["states_"][i][None].float(), dev)
                    roll[kk].append((cur.cpu(), truth.cpu(), start.cpu(), reg.clone()))
        for kk, L_ in roll.items():
            if L_:
                Pk, Tk, Ok, Rk = (torch.cat([x[j] for x in L_]) for j in range(4))
                r[f"rollout_accuracy_{kk}"] = acc(Pk, Tk, Ok, Rk)
        # --- slateN on pools (soft truth, lyapunov, GOALS)
        caps = {g: [] for g in GOALS}
        for d in pools:
            act_all = torch.cat([d["p_starts"][:, :2], d["p_stops"][:, :2]], 1).float()
            for pi in torch.unique(d["pool_idx"]):
                ix = torch.nonzero(d["pool_idx"] == pi)[:, 0]
                s0 = d["states"][ix[0]][None].float()
                o0 = occ_from_particles(s0, dev)
                truth_occ = occ_for_scoring(d["states_"][ix, :, :3].float()); t0 = occ_for_scoring(s0[:, :, :3])
                with torch.no_grad():
                    po = ad.predict_step(o0.expand(len(ix), -1, -1).contiguous(), act_all[ix].to(dev)).float().cpu()
                for g in GOALS:
                    vt = (lyap(truth_occ, Dist[g]) - lyap(t0, Dist[g])).numpy()
                    vp = (lyap(po, Dist[g]) - lyap(o0.cpu(), Dist[g])).numpy()
                    den = vt.mean() - vt.min()
                    if den > 1e-9:
                        caps[g].append(float((vt.mean() - vt[vp.argmin()]) / den))
        r["slateN_per_goal"] = {g: float(np.mean(v)) for g, v in caps.items() if v}
        r["slateN"] = float(np.mean([np.mean(v) for v in caps.values() if v]))
        r["n_test_rows"] = int(len(P)); r["n_pools"] = int(sum(len(torch.unique(d["pool_idx"])) for d in pools))
        out[m] = r
        tmp = Path(str(RES) + ".tmp"); tmp.write_text(json.dumps(out, indent=1)); os.replace(tmp, RES)
        print(f"{m:36s} acc1 {r['accuracy_1']:+.3f} (scatter {r['accuracy_1_scatter']:+.3f}, clump {r['accuracy_1_clump']:+.3f}) "
              f"rollout " + " ".join(f"{r.get(f'rollout_accuracy_{k}', float('nan')):+.3f}" for k in range(1, 5)) +
              f"  slateN {r['slateN']:.3f}", flush=True)
        del ad; torch.cuda.empty_cache()
    print("wrote", RES)


if __name__ == "__main__":
    main()
