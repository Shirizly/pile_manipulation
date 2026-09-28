# MODEL-0001 test history

- held-out image-space `accuracy` (32x32 canonical, swept region) → EXP-0005 → results
- held-out `slateN`/`slate32` control-ranking on `slates_multistep` real same-state pools, `corner`/`stripe`/`center` goals → EXP-0006 → results
- compared against its own global/unswitched counterpart and the stage-3 latent+descriptor family → EXP-0006 → results
- action ranking on DS-0001 (`slates_binned` n20 scatter, 20x1000), `corner`/`lyapunov`: `slateN` +0.740 (sem .055), best of the three promoted instances → EXP-0014 → results
- slateN vs pool size K (25-1000) on DS-0001, paired power → EXP-0026 / RUN-0003 → results/a3_pool_vs_states.json
