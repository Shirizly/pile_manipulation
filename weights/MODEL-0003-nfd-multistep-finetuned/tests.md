# MODEL-0003 test history

- closed-loop 3-step rollout accuracy + terminal slateN (own training
  corpus, held-out 15/50 slates) -> EXP-0010 / RUN-0003 -> results/
  RUN-0003-nfd-multistep-finetune-RESULTS.md
- action ranking on DS-0001, `corner`/`lyapunov`: `slateN` +0.516 (sem .066) — real skill, but below MODEL-0001's +0.740 by ~2.6 sem → EXP-0014 → results
- slateN vs pool size K (25-1000) on DS-0001, paired power → EXP-0026 / RUN-0003 → results/a3_pool_vs_states.json
