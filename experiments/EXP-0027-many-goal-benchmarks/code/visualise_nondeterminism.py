"""Make the rollout non-repeatability VISIBLE: record every physics step of the
probe sequence (EXP-0027 probe_batch_dependence.py) and render the two copies
of the SAME push, from the SAME restored state, whose outcomes differ most.

Nothing in the physics path is changed: SandboxManipulation.execute_action,
update_material_state and set_particle_state are wrapped only to record state
(all env slots' particle positions and the plate pose, every `--every` steps;
the env-0 camera frame at the same steps). rollout_candidates(use_rollout_
fidelity=False, record=False) is called exactly as in the probes.

Call sequence (slate 0 of DS-0001, 32 env slots):
  call 1 "A1": 8 test pushes in slots 0-7, slots 8-31 = copies of test push 0
  call 2 "A2": identical to A1
  call 3 "B" : pool pushes in slots 0-23, the 8 test pushes reversed in 24-31
  call 4 "C" : all 32 slots = test push 0
Test push 0 therefore runs 25 + 25 + 1 + 32 = 83 times from the same state.

Outputs (artifacts/RUN-0005-nondeterminism-video/):
  divergence.json   dv and final positions of every copy of test push 0; the
                    chosen pair; per-step particle and plate separation
  before_after.png  shared start, both finals, overlay with per-particle arrows
  pair_topdown.mp4  side-by-side top-down replay (exact simulator positions)
  cam_env0_callX.mp4 the renderer's view of env slot 0 for each call where slot
                    0 ran test push 0 (A1, A2, C)
"""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(next((REPO / "experiments").glob("EXP-0023-*")) / "code"))
from control_utility_test import lyapunov, lyapunov_weights
from simple_mpc.adapters import occ_from_particles, OCC_GRID
from simple_mpc.genesis_oracle import GenesisOracleEnv
from simple_mpc.oracle_mpc import load_oracle_config
from stage1_optimise import project

OUT = REPO / "experiments/EXP-0027-many-goal-benchmarks/artifacts/RUN-0005-nondeterminism-video"
EVERY = 2
PLATE_L, PLATE_W = 0.040, 0.002


def yaw_of(q):                       # genesis quat (w, x, y, z)
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = torch.load(glob.glob(str(REPO / "experiments/EXP-0023-*/artifacts/stage1_actions.pt"))[0],
                   weights_only=False)
    s = 0
    st = r["states0"][s].float()
    test = project(torch.stack([r["arms"][a][k][s] for a in list(r["arms"])[:4]
                                for k in ("a_rank", "a_grad")]))[0]
    pool = project(r["pool_actions"][s].float())[0]
    K = 32
    cfg = load_oracle_config(str(REPO / "simple_mpc/config/config_oracle.yaml"))
    cfg["dataset"]["record_transitions"] = False
    cfg["mpc"]["n_envs"] = K
    env = GenesisOracleEnv(cfg, n_envs=K)
    import genesis as gs
    sim = env._sim
    snap = {"pos": st[None, :, 0:3].to(gs.device), "quat": st[None, :, 3:7].to(gs.device)}
    dw = lyapunov_weights((OCC_GRID, OCC_GRID), "corner", "cpu")
    v0 = lyapunov(occ_from_particles(st[None], "cpu"), dw)

    # ---- recording wrappers (record only; physics untouched) --------------
    rec = {"frames": [], "cam": [], "film": False}

    def grab(phase, step):
        if step % EVERY and phase != "restore":
            return
        rec["frames"].append(dict(phase=phase, step=int(step),
                                  pos=sim._get_particle_positions().float().cpu().clone(),
                                  plate_pos=sim.plate.get_pos().float().cpu().clone(),
                                  plate_quat=sim.plate.get_quat().float().cpu().clone()))
        if rec["film"]:
            rgb = env._cam.render(rgb=True, depth=False, segmentation=False)[0]
            rec["cam"].append((phase, int(step), np.asarray(rgb, dtype=np.uint8)[..., :3].copy()))

    o_exec, o_settle, o_set = sim.execute_action, sim.update_material_state, sim.set_particle_state

    def exec_w(p_start, p_stop, angle, on_phase=None, on_step=None):
        return o_exec(p_start, p_stop, angle, on_phase=on_phase, on_step=lambda ph, k: grab(ph, k))

    def settle_w(store_other=False, on_step=None):
        return o_settle(store_other=store_other, on_step=lambda k: grab("settle", k))

    def set_w(pos, quat):
        o_set(pos, quat)
        grab("restore", 0)

    sim.execute_action, sim.update_material_state, sim.set_particle_state = exec_w, settle_w, set_w

    A = torch.cat([test, test[:1].expand(K - 8, -1)])
    B = torch.cat([pool[10:34], test.flip(0)])
    C = test[:1].expand(K, -1).clone()
    calls = [("A1", A, True), ("A2", A, True), ("B", B, False), ("C", C, True)]
    runs = {}
    for name, batch, film in calls:
        rec.update(frames=[], cam=[], film=film)
        pos = env.rollout_candidates(batch[:, None, :].cpu(), snap, use_rollout_fidelity=False, record=False)
        dv = (lyapunov(occ_from_particles(pos.float().cpu(), "cpu"), dw) - v0)
        runs[name] = dict(batch=batch, final=pos.float().cpu(), dv=dv, frames=rec["frames"], cam=rec["cam"])
        print(f"call {name}: {len(rec['frames'])} recorded steps, dv range of test-push-0 copies "
              f"{float(dv[(batch == test[0]).all(1)].min()):+.4f} .. {float(dv[(batch == test[0]).all(1)].max()):+.4f}",
              flush=True)
    env.destroy()
    torch.save({nm: dict(batch=runs[nm]["batch"], final=runs[nm]["final"], dv=runs[nm]["dv"]) for nm in runs}
               | {"state0": st, "test0": test[0]}, OUT / "finals.pt")

    # ---- every copy of test push 0 ------------------------------------------
    copies = [(n, k, float(runs[n]["dv"][k])) for n, _, _ in calls
              for k in range(K) if bool((runs[n]["batch"][k] == test[0]).all())]
    lo = min(copies, key=lambda c: c[2]); hi = max(copies, key=lambda c: c[2])
    print(f"{len(copies)} copies of test push 0; dv from {lo[2]:+.4f} ({lo[0]} slot {lo[1]}) "
          f"to {hi[2]:+.4f} ({hi[0]} slot {hi[1]})")
    fa, fb = runs[lo[0]]["final"][lo[1]], runs[hi[0]]["final"][hi[1]]
    sep_final = (fa - fb).norm(dim=-1)

    # restore check: every slot's recorded post-restore state vs the snapshot
    restore_err = {n: float((runs[n]["frames"][0]["pos"] - st[None, :, :3]).abs().max()) for n in runs
                   if runs[n]["frames"] and runs[n]["frames"][0]["phase"] == "restore"}

    # per-step separation between the two chosen copies (only if same #frames)
    fr_a, fr_b = runs[lo[0]]["frames"], runs[hi[0]]["frames"]
    n = min(len(fr_a), len(fr_b))
    timeline = []
    for i in range(n):
        a, b = fr_a[i], fr_b[i]
        timeline.append(dict(phase=a["phase"], step=a["step"], phase_b=b["phase"],
                             max_particle_sep_mm=1000 * float((a["pos"][lo[1]] - b["pos"][hi[1]]).norm(dim=-1).max()),
                             plate_sep_mm=1000 * float((a["plate_pos"][lo[1]] - b["plate_pos"][hi[1]]).norm()),
                             plate_yaw_diff_deg=math.degrees(yaw_of(a["plate_quat"][lo[1]].tolist())
                                                             - yaw_of(b["plate_quat"][hi[1]].tolist()))))
    first_div = next((t for t in timeline if t["max_particle_sep_mm"] > 0.1), None)
    json.dump(dict(state="DS-0001 slate 0 (EXP-0023 states0[0])", action_test0=test[0].tolist(),
                   copies=[dict(call=c[0], slot=c[1], dv=c[2]) for c in copies],
                   pair=dict(low=dict(call=lo[0], slot=lo[1], dv=lo[2]), high=dict(call=hi[0], slot=hi[1], dv=hi[2])),
                   final_particle_sep_mm=dict(max=1000 * float(sep_final.max()), mean=1000 * float(sep_final.mean()),
                                              n_moved_gt_1mm=int((sep_final > 1e-3).sum())),
                   restore_err_m=restore_err, first_divergence=first_div,
                   timeline=timeline), open(OUT / "divergence.json", "w"), indent=1)
    print("restore error per call (max |pos - snapshot|, m):", restore_err)
    print("first step where the two copies differ by > 0.1 mm:", first_div)
    print(f"final: max particle separation {1000 * float(sep_final.max()):.1f} mm, "
          f"{int((sep_final > 1e-3).sum())}/{len(sep_final)} particles moved > 1 mm")

    # ---- before / after image ---------------------------------------------
    col = plt.cm.tab20(np.arange(st.shape[0]) % 20)
    a0 = test[0].numpy()
    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    for axi, (title, P) in zip(ax[:3], [("start (restored snapshot, shared)", st[:, :3]),
                                        (f"after: {lo[0]} slot {lo[1]}  dv {lo[2]:+.4f}", fa),
                                        (f"after: {hi[0]} slot {hi[1]}  dv {hi[2]:+.4f}", fb)]):
        axi.scatter(P[:, 0] * 1000, P[:, 1] * 1000, c=col, s=60, marker="s", edgecolors="k")
        axi.set_title(title, fontsize=10)
    ax[3].scatter(fa[:, 0] * 1000, fa[:, 1] * 1000, c=col, s=60, marker="s", edgecolors="k", label=f"{lo[0]}/{lo[1]}")
    ax[3].scatter(fb[:, 0] * 1000, fb[:, 1] * 1000, c=col, s=60, marker="o", edgecolors="k", label=f"{hi[0]}/{hi[1]}")
    for i in range(st.shape[0]):
        ax[3].plot([fa[i, 0] * 1000, fb[i, 0] * 1000], [fa[i, 1] * 1000, fb[i, 1] * 1000], "k-", lw=1)
    ax[3].set_title("overlay: square vs circle, line = same particle", fontsize=10); ax[3].legend(fontsize=8)
    for axi in ax:
        axi.arrow(a0[0] * 1000, a0[1] * 1000, (a0[2] - a0[0]) * 1000, (a0[3] - a0[1]) * 1000,
                  width=0.4, color="r", alpha=0.5, length_includes_head=True)
        axi.set_xlim(-64, 64); axi.set_ylim(-64, 64); axi.set_aspect("equal"); axi.grid(alpha=0.3)
        axi.set_xlabel("x (mm)"); axi.set_ylabel("y (mm)")
    fig.suptitle("Same restored state, same push (red arrow = plate path), two different outcomes", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "before_after.png", dpi=110); plt.close(fig)

    # ---- side-by-side top-down replay from exact simulator states ------------
    def plate_poly(pp, pq):
        yaw = yaw_of(pq.tolist()); c, s_ = math.cos(yaw), math.sin(yaw)
        hx, hy = PLATE_L / 2, PLATE_W / 2 + 0.001
        pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        R = np.array([[c, -s_], [s_, c]])
        return (pts @ R.T + pp[:2].numpy()) * 1000

    vw = None
    for i in range(n):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
        for axi, (fr, slot, lab) in zip(ax[:2], [(fr_a[i], lo[1], f"{lo[0]} slot {lo[1]}"),
                                                 (fr_b[i], hi[1], f"{hi[0]} slot {hi[1]}")]):
            P = fr["pos"][slot]
            axi.scatter(P[:, 0] * 1000, P[:, 1] * 1000, c=col, s=60, marker="s", edgecolors="k")
            axi.add_patch(plt.Polygon(plate_poly(fr["plate_pos"][slot], fr["plate_quat"][slot]),
                                      color="r", alpha=0.7))
            axi.set_title(f"{lab}  [{fr['phase']} step {fr['step']}]", fontsize=10)
        Pa, Pb = fr_a[i]["pos"][lo[1]], fr_b[i]["pos"][hi[1]]
        ax[2].scatter(Pa[:, 0] * 1000, Pa[:, 1] * 1000, c=col, s=50, marker="s", edgecolors="k")
        ax[2].scatter(Pb[:, 0] * 1000, Pb[:, 1] * 1000, c=col, s=50, marker="o", edgecolors="k")
        for j in range(Pa.shape[0]):
            ax[2].plot([Pa[j, 0] * 1000, Pb[j, 0] * 1000], [Pa[j, 1] * 1000, Pb[j, 1] * 1000], "k-", lw=1)
        ax[2].set_title(f"overlay; max particle sep {timeline[i]['max_particle_sep_mm']:.2f} mm, "
                        f"plate sep {timeline[i]['plate_sep_mm']:.2f} mm", fontsize=10)
        for axi in ax:
            axi.set_xlim(-64, 64); axi.set_ylim(-64, 64); axi.set_aspect("equal"); axi.grid(alpha=0.3)
        fig.tight_layout(); fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.close(fig)
        if vw is None:
            vw = cv2.VideoWriter(str(OUT / "pair_topdown.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                 (img.shape[1], img.shape[0]))
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if vw is not None:
        vw.release()

    # ---- renderer view of slot 0 --------------------------------------------
    for name in ("A1", "A2", "C"):
        cam = runs[name]["cam"]
        if not cam:
            continue
        h, w = cam[0][2].shape[:2]
        cw = cv2.VideoWriter(str(OUT / f"cam_env0_call{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        for ph, k, im in cam:
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            cv2.putText(im, f"call {name} slot 0 dv(final) {float(runs[name]['dv'][0]):+.4f}  {ph} {k}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cw.write(im)
        cw.release()
        cv2.imwrite(str(OUT / f"cam_env0_call{name}_final.png"), cv2.cvtColor(cam[-1][2], cv2.COLOR_RGB2BGR))
    print("slot 0 dv per call:", {nm: float(runs[nm]["dv"][0]) for nm in runs})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
