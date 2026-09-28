# DS-0005 — PARTIAL, mismatched physics: n20 scatter, planned 160 x 128, stopped at 7,616 pushes

**Status:** partial (collection deliberately stopped 2026-09-24)
**Payload:** `Genesis/data/slates_binned/n20_scatter_s160a128_L20-70mm_fric03_dens1000_PARTIAL/`
(`step0.partial.pt` + `manifest.json`; gitignored)

Started as the large-pool benchmark test set and stopped after 60 of 163 batches when
it was found that the collector's defaults (particle and box friction 0.3, density
1000) do not match the models' training physics (overnight_randlen: 0.7 / 0.5 / 450;
invariant `benchmark-physics-matches-training`). Kept, not deleted, as a record of the
mismatched setting: 7,616 of 20,480 chains simulated (the `simulated` mask in the
partial file marks which); because pushes are simulated in push-length-bin order, the
simulated subset covers the underflow bin, bin 0 and part of bin 1 only -- NOT a
uniform sample of lengths or of states. Not usable as a benchmark as-is.

Command: see `runs/ds0005_collect.json` (seed 1, 160 states, 128 actions, drop spawn,
collector defaults for physics). Successor: DS-0006 (same design, matched physics).
