# MODEL-0002 test history

- held-out `slateN`/`slate32` control-ranking via point-mass value readout, `corner` goal, `lyapunov` value function, `slates_multistep` → EXP-0006 → results
- extended to `stripe`/`random_quadrant`/`ring_O`/`T` goal shapes and `mass_in_region`/`signed_mass_in_region` value functions, on `slates_multistep` (L10mm excluded) AND the 43-file overnight holdout (never trained on) → EXP-0008 → results
- corpus-specificity finding: survives (weaker) in-corpus, fails and reverses sign out-of-corpus on annular/thin targets → EXP-0008 → see MODEL.md's "Corpus-specificity finding"
- near-zero `slateN` on `Genesis/data/slates_binned` (n20, scatter-spawn) diagnosed as genuine corpus difficulty (no scoring/rasteriser/pool-size/cross-bin-calibration defect found) → EXP-0012 → results
- compared (via a single small MLP on the same 94-dim basis) on held-out one-step `descriptor_accuracy`: this switched-linear operator (0.435) beats the MLP (0.131) → EXP-0011 → results
- action ranking on DS-0001, `corner`/`lyapunov`: `slateN` -0.090 (sem .030), indistinguishable from `random` (-0.141 sem .101); DIAGNOSED MECHANISM — 26% of its point-mass predictions are exactly 0 (predicted COM inside the target region, where the distance field is identically zero), so ~26% of each pool ties at the minimum → EXP-0014 → results
- descriptor-space value readout: a linear map from descriptor differences to value is refuted, but capacity fixes it (MLP held-out-goal R² 0.69-0.89); goal-as-legal-configuration instead of goal-as-solid-mask cuts the held-out-goal failure by 2-3 orders of magnitude → EXP-0015 → results
- slateN vs pool size K (25-1000) on DS-0001, paired power → EXP-0026 / RUN-0003 → results/a3_pool_vs_states.json
