# RUN-0012 — ablation: time term vs mass-gap re-rank alone

- `code/queue_ablation.sh` after RUN-0011: `time_mpc.py --tag ablation --cells lyap,blend1.0,blendnt1.0 --starts 40..45 --map2 results/etc_map_m0.json --map-key "gap|td"`
- blendnt1.0 = blend1.0 with t(a) removed from J (J = f(r') - f(r)): same search, pool, readout, offset and map; isolates the time term. lyap and blend1.0 re-run in the same chunks (replicates RUN-0010's cells under a fresh run)
- added after RUN-0010's result was seen (exploratory, not pre-declared)
- commit a5ccb680 dirty
