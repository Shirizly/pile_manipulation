# RUN-0004-regret-massgoals-rescoring

Directly executes the "What would change the verdict" cheapest-check idea
from this record's `EXPERIMENT.md`: re-scores the SAME horizon-3
closed-loop chains (RUN-0001's 4 image models + RUN-0003's 4 fine-tuned
NFD checkpoints, `nfd_lam{0.3,0.5,0.7,0.9}.pth`, reused verbatim, no
retraining) under `mass_in_region` and `signed_mass_in_region` instead of
only `lyapunov` — because `lyapunov` was measured near-ceiling untrained
(0.967, 15/0/0 wins vs init) so the accuracy gain from RUN-0003 could not
register as a control gain there. Unlike RUN-0001 (which pooled all 50
slates), every model here — including the untrained baselines — is scored
on identically the SAME held-out 15-slate test split (seed 0, 35/15, the
same split RUN-0002/RUN-0003 trained against), asserted disjoint and
asserted to cover the full `slate_idx` range on both datasets.

**Result: the premise holds (untrained NFD IS off-ceiling under
`mass_in_region`, 0.821/0.856 vs 0.967/0.935 lyapunov) but the extra
headroom does not resolve into a statistically distinguishable fine-tuning
benefit at this sample size.** Paired per-slate capture difference
(fine-tuned − untrained NFD, closed-loop, pooled n=30 across both
datasets): every lambda/value-fn cell is positive-signed but none clears
2×sem (best case `mass_in_region` lam=0.7: +0.048±0.034, ≈1.4 sem).
Per-dataset the signal is inconsistent (n20_L20mm alone: +0.09±0.055,
≈1.6 sem; n20_L40mm alone: ≈0). This is a genuine null / power-limited
result at n=15-30 slates, not a clean resolution either way — it updates
this record's "unmeasured" framing (RUN-0003's Numbers table) to
"measured, but not distinguishable from noise at this n", and is the
correct honest update given the task explicitly asked not to oversell.

- **Command**: see `COMMAND.txt`.
- **Commit**: `0ddab20f` (dirty tree — same pre-existing, unrelated
  docs/skills reorganisation as RUN-0001/0002/0003).
- **Device**: single GPU, ~10s wall clock (only inference passes — no
  training; reuses RUN-0001's raw-file loader and RUN-0003's fine-tuned
  checkpoints as-is).
- **Data / split**: same raw-file loader (`rollout.py::load_dataset`) and
  split convention as RUN-0002/RUN-0003 (`train_nfd_multistep.py::make_split`,
  seed 0, 35 train / 15 test of 50 slates, `n20_L20mm` + `n20_L40mm`,
  `n20_L10mm` excluded); disjointness AND slate_idx-range-coverage asserted
  in-script (the latter is new here — needed to trust that the index-based
  split lines up with actual slate identity in both corpora, not just
  assumed as in RUN-0002/RUN-0003).
- **Degeneracy check** (new requirement for this run, not present in
  RUN-0001/RUN-0002/RUN-0003): frac(dv_true==0) per (dataset, value-fn),
  all ≤0.067 — no cell excluded as degenerate.
- **Wins/losses/ties**: uses METRICS.md's cross-model-agreement definition
  throughout (RUN-0002's corrected `wlt_fixed` sense, not the strict
  argmax-match definition RUN-0001 used for its own ties column) —
  reported both vs. `persistence` (all image models clearly beat it) and,
  the load-bearing comparison, each fine-tuned NFD lambda vs. untrained
  NFD (small, noisy at n=15; supplemented with a paired capture-difference
  test for more power, see `results/RUN-0004-...-RESULTS.md`).
- **Status**: completed, no errors.
- **Outputs**: `artifacts/RUN-0004-regret-massgoals-rescoring/
  results_regret_massgoals.json` (raw), `code/multistep-regret__rescore_regret.py`,
  `results/RUN-0004-regret-massgoals-rescoring-RESULTS.md` (curated, copied
  verbatim from the source `experiments/temp/multistep-regret/RESULTS.md`,
  which remains available under `experiments/temp/multistep-regret/` and
  is not deleted by this fold-in).
