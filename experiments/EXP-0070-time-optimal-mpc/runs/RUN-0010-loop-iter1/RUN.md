# RUN-0010 — user's loop, iteration 1 (map m0 = MC/TD bootstrap from recorded trajectories)

- `time_mpc.py --tag iter1 --cells lyap,blend0.5,blend1.0 --starts 40..45 --map2 results/etc_map_m0.json --map-key "gap|td"` (run_probe tag exp0070_iter1), 2026-10-06 02:01
- blend<lam>: CEM on model lyap dV (0.5 s, pop 1024, elite 0.25), pool = its 64 best evaluated pushes, execute argmin (1-lam) z(dV) + lam z(J), J = t(a) + f(r') - f(r), r = mass gap max(0, theta - mass_rel), r' read through the model with the calibrated offset (etc_v2_calib)
- smoke before it: `--tag smoke2` (2 steps, 6 episodes; not analysed)
- commit a5ccb680 dirty
