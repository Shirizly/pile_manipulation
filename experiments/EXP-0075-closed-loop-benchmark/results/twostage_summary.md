# Two-stage model-switching planner (vanilla64 pool -> ens128 GD), 10 tasks, 4 pushes, true value change in the simulator (lower = better)

Stage 1: vanilla64, ~4615 H=4 sequences (0.6 s unit cost); stage 2: ens128 re-score of top 8 + GD (21 steps, 8 sequences); measured wall time per planning call (exclusive GPU, after warm-up) ~1.3-1.6 s.

| mode | true value | vs open (paired) |
|---|---|---|
| greedy H=1 closed loop (re-plan each push) | -0.819 [-0.916, -0.726] | -0.179 [-0.262, -0.104] |
| H=4 open loop (plan once) | -0.640 [-0.769, -0.520] | - |
| hybrid: H=4 once, each later push re-optimised on the real state | -0.770 [-0.891, -0.654] | -0.131 [-0.203, -0.076] |

hybrid - greedy: +0.048 [-0.022, +0.129]

Plan optimism (predicted - true, open loop): -0.087 [-0.159, -0.020]  (predicted -0.727)
Stage-1 vanilla64 best own-cost -> ens128 cost of the same candidates -> after GD: -0.689 -> -0.594 -> -0.727

## Divergence from the plan (hybrid vs open, from the SAME real state)

* push 2 (first divergence): improvement (hybrid dv - open dv; negative = refinement better): -0.036 [-0.099, +0.026]; better in 5/10 tasks
* push 3 dv difference (states already diverged): -0.035 [-0.092, +0.015]
* push 4 dv difference (states already diverged): -0.058 [-0.142, +0.027]
* terminal: -0.131 [-0.203, -0.076]

Mean per-push dv: greedy: -0.297, -0.206, -0.171, -0.144; open: -0.125, -0.084, -0.071, -0.360; hybrid: -0.127, -0.121, -0.106, -0.418

Per task (greedy / open / hybrid / plan prediction): T/40: -1.06/-1.01/-1.07/-0.93; O/41: -0.60/-0.42/-0.57/-0.65; S/42: -0.90/-0.72/-0.82/-0.75; X/43: -1.04/-0.91/-1.07/-0.94; T/44: -0.84/-0.45/-0.86/-0.75; O/45: -0.61/-0.50/-0.61/-0.52; S/46: -0.64/-0.63/-0.78/-0.60; X/47: -0.84/-0.61/-0.68/-0.74; O/40: -0.86/-0.78/-0.79/-0.84; X/41: -0.79/-0.37/-0.45/-0.55