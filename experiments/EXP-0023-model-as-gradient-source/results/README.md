# EXP-0023 results

| file | what |
|---|---|
| `metrics.json` | every per-(state, arm) row and the per-arm summary, the floors, the power check, and the cache-reproduction check. Written by `code/stage3_analyse.py`. |
| `../artifacts/grad_check.json` | RUN-0000's per-arm gradient report (per-component magnitudes, directional finite-difference agreement). |
| `../artifacts/stage1_actions.pt` | the shared seed pool, its row ids, the 10 slate states, and every arm's `a_rank`/`a_grad` plus its optimisation trajectory and bound-hit fractions. **This is the fitted-object record: every optimised action is kept.** |
| `../artifacts/stage2_genesis.pt` | every Genesis execution: labels, true `dv`, the CEM winner and its per-iteration history, and the state-restore error per slate. |
| `../artifacts/stage2.log` | the raw stdout of the Genesis run. |

`metrics.json`'s `sign_convention` field records, in the file, the one thing
that would invert every conclusion here.
