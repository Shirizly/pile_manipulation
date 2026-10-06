"""PILOT (EXP-0072 active plan): re-simulation repeatability / chaos probe on 96 DS-0016 rows (48 random + 48 from the top-20% ft100
swept-region-error rows). Per row: 1 exact repeat from the recorded state, and K=3 re-sims from the state perturbed by xy N(0, 0.25 mm)
+ yaw N(0, 0.5 deg) (below the 1 mm/px raster the model sees), + 1 re-sim at 1 mm / 1 deg. 5 x 96 = 480 sims, 15 batches of 32.
Saves runs/active_pilot/resim_rows.npz (resim particle states) atomically after every batch; resumable.  python -u, PYTHONPATH=."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, "."); sys.path.insert(0, "experiments/EXP-0072-zoom-window-nfd/code"); sys.path.insert(0, "experiments/EXP-0073-active-data-mining/code")
from active_proxies import load_rows, jitter_states
from active_perturb import build_sim
OUT = "experiments/EXP-0073-active-data-mining/runs/active_pilot/resim_rows.npz"
LEVELS = [("repeat", 0, 0), ("p025_a", .25, .5), ("p025_b", .25, .5), ("p025_c", .25, .5), ("p1", 1.0, 1.0)]

def main():
    Dd = load_rows(); Z = np.load("experiments/EXP-0073-active-data-mining/runs/active_pilot/proxy_rows.npz", allow_pickle=True)
    e = Z["ft100_e_sw"]; rng = np.random.default_rng(1); B = len(e)
    top = np.nonzero(e >= np.quantile(e, 0.8))[0]; rest = np.setdiff1d(np.arange(B), top)
    sel = np.concatenate([rng.choice(rest, 48, replace=False), rng.choice(top, 48, replace=False)])   # first 48 = random-ish (80% non-top), last 48 = top-20%
    state = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
    env, sim = build_sim(32)
    import genesis as gs
    for lv, (name, xy, yaw) in enumerate(LEVELS):
        res = state.get(name, np.zeros((96, 20, 7), np.float32)); done = state.get(name + "_done", np.zeros(3, bool))
        for c in range(3):
            if done[c]: continue
            t0 = time.time(); ix = sel[c * 32:(c + 1) * 32]
            S = Dd["S"][ix]; S2 = jitter_states(S, np.random.default_rng(100 * lv + c), xy, yaw) if xy else S
            sim.set_particle_state(S2[:, :, :3].to(gs.device).contiguous(), S2[:, :, 3:7].to(gs.device).contiguous()); sim.update_material_state()
            sim.execute_action(Dd["ps3"][ix].to(gs.device), Dd["pe3"][ix].to(gs.device), Dd["ang"][ix].to(gs.device)); sim.update_material_state()
            res[c * 32:(c + 1) * 32] = sim._particle_state[:, :, :7].detach().float().cpu().numpy(); done[c] = True
            state[name] = res; state[name + "_done"] = done; state["sel"] = sel
            np.savez_compressed(OUT + ".tmp.npz", **state); os.replace(OUT + ".tmp.npz", OUT)
            print(name, c, f"{time.time() - t0:.0f}s", flush=True)
    env.destroy()

if __name__ == "__main__": main()
