"""EXP-0031: do the corpus-collection path and GenesisOracleEnv.rollout_candidates
(full fidelity) give the same outcome for the same (state, action), once truth is
scored with the soft rasteriser? (EXP-0023 saw r = 0.959 with IMAGE scoring.)

DS-0006 slates 0-7 (training-matched physics; the env is given the SAME physics), 64 corpus actions each (the first 64 of EXP-0023's pool
order for slates 0-9's generator is NOT needed -- any corpus rows do). Rollout
batches are grouped by push length (a batch runs as long as its longest push).
Outputs results/crosspath.json, rewritten after every slate.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))
from control_utility_test import lyapunov_weights
from simple_mpc.adapters import occ_for_scoring, occ_from_particles, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import load_oracle_config
from Genesis.binned_slate_dataset import BinnedSlateCorpus
from stage1_optimise import project
from simple_mpc.learned_mpc import oracle_config_with_physics, apply_physics

RES = REPO / "experiments/EXP-0031-crosspath-agreement/results/crosspath.json"
ART = REPO / "experiments/EXP-0031-crosspath-agreement/artifacts/RUN-0001"
ART.mkdir(parents=True, exist_ok=True); RES.parent.mkdir(parents=True, exist_ok=True)
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_randlenphys")).step(0)
K, N_PER = 32, 64
cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
cfg["dataset"]["record_transitions"] = False; cfg["mpc"]["n_envs"] = K
cfg = oracle_config_with_physics(cfg)          # the corpus (DS-0006) physics
env = GenesisOracleEnv(cfg, n_envs=K)
apply_physics(env)
import genesis as gs
dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
def V(occ): f = occ.reshape(len(occ), -1); return (f @ dw.reshape(-1)) / f.sum(1)
res = {"slates": {}, "note": "corpus = binned_slate_collection outcome; rollout = rollout_candidates full fidelity"}
rng = np.random.default_rng(0)
for s in range(8):
    t0 = time.time()
    idx = np.nonzero(rows.slate_idx.numpy() == s)[0]
    pick = np.sort(rng.choice(idx, N_PER, replace=False))
    st = rows.states[pick[0]].float()
    act = torch.cat([rows.p_starts[pick, :2], rows.p_stops[pick, :2]], 1).float()
    act_run, hit = project(act)                       # what the planner path would run
    order = np.argsort((act_run[:, 2:] - act_run[:, :2]).norm(dim=1).numpy())   # batch by length
    snap = {"pos": st[None, :, :3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
    fin = torch.zeros(N_PER, st.shape[0], 3, device="cpu")   # Genesis sets the default device to cuda
    for i in range(0, N_PER, K):
        o = order[i:i + K]
        pos = env.rollout_candidates(act_run[o][:, None, :], snap, use_rollout_fidelity=False, record=False)
        fin[o] = pos.float().cpu()[:len(o)]
    keep = ~hit.any(1)                                # compare only actions the projection left untouched
    corp = rows.states_[pick][:, :, :3].float()
    fin_all, corp_all = fin, corp
    fin, corp = fin[keep], corp[keep]
    disp = (fin - corp).norm(dim=-1)                  # (N, P) metres
    v0s, v0i = V(occ_for_scoring(st[None])), V(occ_from_particles(st[None]))
    dv_soft_r, dv_soft_c = V(occ_for_scoring(fin)) - v0s, V(occ_for_scoring(corp)) - v0s
    dv_img_r, dv_img_c = V(occ_from_particles(fin)) - v0i, V(occ_from_particles(corp)) - v0i
    projected = int(hit.any(1).sum())
    rec = dict(n=int(keep.sum()), n_actions_changed_by_projection=projected,
               particle_disp_mm=dict(median=float(disp.median() * 1e3), p95=float(disp.quantile(.95) * 1e3), max=float(disp.max() * 1e3)),
               soft=dict(r=float(np.corrcoef(dv_soft_r, dv_soft_c)[0, 1]), sd_diff=float((dv_soft_r - dv_soft_c).std()),
                         sd_between=float(dv_soft_c.std())),
               image=dict(r=float(np.corrcoef(dv_img_r, dv_img_c)[0, 1]), sd_diff=float((dv_img_r - dv_img_c).std()),
                          sd_between=float(dv_img_c.std())))
    res["slates"][s] = rec
    torch.save({"pick": pick, "final_rollout": fin_all, "final_corpus": corp_all, "unprojected": keep}, ART / f"slate{s}.pt")
    tmp = Path(str(RES) + ".tmp"); tmp.write_text(json.dumps(res, indent=1)); os.replace(tmp, RES)
    print(f"slate {s}: disp median {rec['particle_disp_mm']['median']:.2f} mm p95 {rec['particle_disp_mm']['p95']:.1f} "
          f"max {rec['particle_disp_mm']['max']:.1f}; soft r {rec['soft']['r']:.4f} sd_diff/sd_between "
          f"{rec['soft']['sd_diff'] / rec['soft']['sd_between']:.3f}; image r {rec['image']['r']:.4f} "
          f"{rec['image']['sd_diff'] / rec['image']['sd_between']:.3f}; projected {projected} ({time.time() - t0:.0f}s)", flush=True)
env.destroy()
