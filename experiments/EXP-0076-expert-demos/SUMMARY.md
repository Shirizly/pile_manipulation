# EXP-0076 — state of the project (final for the 2026-10-16/17 overnight run)
Brief: BRIEF.md. Full history (every run, its result): LOG.md. Record: EXPERIMENT.md (C-088, C-089 in REGISTER.md).
Demo counts: results/demo_counts.md. Dataset: datasets/DS-0024-expert-demos (INDEX.json = best demo per scenario + replay flags).
GIFs: figures/gifs_median/ (median-time demo of every stratum), figures/gifs/ (fastest expert demo per n20 stratum).

## Demonstrations collected (2026-10-17 06:50)
**272 of the 540 generated scenarios have a solved demo** (in-goal mass >= 0.95, legal pushes, fastest physical time found per
scenario across pipelines); 233 of them in < 10 pushes. Best demo from the expert pipeline in 51, from the feasible pipeline in the rest.

| n | init | disc | square | letter | median best pushes (disc / square / letter) |
|---|---|---|---|---|---|
| 20 | scattered | 21/30 | 25/30 | 30/30 | 8 / 8 / 7 |
| 20 | inbetween | 28/30 | 27/30 | 29/30 | 6 / 6 / 6 |
| 20 | piled | 29/30 | 28/30 | 25/30 | 3 / 3 / 2 |
| 50 | scattered | 11/15 tried | 9/15 tried | 2/15 tried | 9 / 11 / 14 |
| 50 | inbetween | 2/2 tried | 2/2 tried | 0/2 tried | 5 / 9 / - |
| 50 | piled | 2/2 tried | 2/2 tried | 0/2 tried | 5 / 2 / - |

n20 total 242 / 270 (the 28 left: tight goals with rho > 0.45 and 3 starts with cubes jammed in a tray corner, likely tool-infeasible;
results/ids_corner_jammed.json). n50 total 30 of 63 attempted (n50 is simulation-bound, see below).
**Mined separately** (as asked; never mixed with the generated set): 1,558 qualifying legal episodes from earlier EXP records
(n20 DS-0006 scatter starts, named letter / quadrant goals) = 173 unique (goal, start) tasks, median 7 pushes / 9.4 s; data/mined/.

## Best pipelines found
**Expert (anything goes):** beam search over TRUE simulator states -- each of B=4 nodes is planned by the learned planner (ens2bal,
fetch + legal scoring, CEM + GD), its 8 best distinct legal pushes are simulated, the 4 best true children (value + 0.05 x elapsed
seconds) continue; stop at the first child with mass >= 0.95 (fastest such). Pilot: 18/18 solved vs 14/18 for the planner alone,
-0.86 s [-1.73, -0.05] on jointly solved; vs simulating the planner's top-16 and taking the best: -0.66 s [-1.02, -0.32]. On the 68 n20
scenarios the planner alone could not solve it solved 37. Beam shape (8x4 vs 4x8) does not matter at equal sims; time weighting
-0.23 s (n.s.). Cheap complement: **best-of-N feasible restarts** (portfolio) -- best of 4 configs solved 32/36 vs 27/36 and saved
0.77 s; on 78 slow scenarios best-of-3 cut mean time 10.80 -> 9.86 s. Use the beam for unsolved / hard scenarios, restarts to polish.

**Feasible (real robot, <= 3 s / decision):** the EXP-0075 planner -- ens2bal, H=1, fetch proposals + legalise-before-scoring,
CEM 1,280 + GD 20, 1.3 s / decision -- solves 196/270 n20 (73 %) and 22/45 n50 scattered within 12 / 16 pushes. On the 36-scenario
ablation set nothing beat it (2x budget, H=2 OT pool, H2->H1 schedule, time-aware re-ranking, tail rule: all inside noise; seed replicate
+-1-2 solved, +-0.45 s).
**Model accuracy matters in both phases (powered follow-up, 2026-10-17):** the cheap vanilla64 (4x budget, 0.4 s / decision) looked
'inside noise' on 36 scenarios, but over all 270 it solves 166 vs 196 and is slower to mass 0.8 by +0.58 s [+0.40, +0.77] (bulk), and in
96 paired tail continuations from identical recorded 0.8-mass states it is slower by +0.61 s [+0.28, +0.94] with ~6 pp fewer solves
(2 seeds; same-model seed floor < 0.1 s). Bulk cost is largest on scattered starts (+1.4 s), smallest on piled (+0.1-0.3 s).
That is on PHYSICAL time only. **On TOTAL time (planning + physical; user's criterion for switching decisions) the verdict flips**
(results/total_time.md): ens planning costs 1.2-1.3 s per decision vs 0.38 s for vanilla64, which outweighs its 0.15-0.4 s physical gain
per push. v64 - ens total: bulk to 0.8 -2.64 s [-2.99, -2.31]; tail -2.90 s [-3.60, -2.18]; on pairs both solve -2.0..-3.5 s. Composite
estimates (bulk + tail): ens/ens 19.5 s, ens/v64 16.7, v64/ens 16.1, v64/v64 13.0 -> the cheap model throughout is fastest; switching to
the ensemble in either phase costs time. Its remaining cost is reliability within the push cap: 166 vs 196 of 270 solved in 12 pushes.
(results/tail_switch.md, results/bulk_compare.md, code/phase_split.py)

## What determines difficulty / time (results/structure_analysis.md)
- Goal tightness rho (cube area / goal area): n20 scattered disc/square solved 13/16 at rho < 0.3 vs 6/25 at rho > 0.45 (feasible);
  even rho 0.6 goals hold 99.4 % in principle (packing problem). Init: piled 2-3 pushes, inbetween ~6, scattered ~8.
- The 0.8 -> 0.95 tail is 40 % of solved time; failures are near-miss stalls (64 % end >= 0.90); transit is 45 % of physical time.

## Caveats that matter for imitation learning
- **Open-loop reproducibility.** Replaying a demo's actions in a fresh continuous env keeps it >= 0.95 for 158/203 feasible but only
  19/55 expert demos; by recorded final mass, demos ending >= 0.98 replay 78/81. The variation is Genesis call-history dependence
  (identical envs in one batch agree exactly; separate processes differ ~1.3 mm per cube) and accumulates over the sequence;
  re-simulating the selected children (re-check) does not fix it. Each stored transition is a genuine simulator transition; INDEX.json
  carries a replay flag per demo. **Margin fix:** an expert beam that accepts success only at >= 0.97 produced demos of which 7/9 replay
  >= 0.95 open loop (0.95-acceptance beam: 3/12); cost +0.5 s / +0.7 push and ~1/5 fewer solves. 24 such robust demos are in
  data/expert_robust/ (INDEX.json key "robust"). For IL, prefer replay-flagged or robust demos.
- One learned model instance; planner seed 0 except the replicates; 36-scenario ablation set.
- n50: dense starts (inbetween / piled) cost 5-15 min of GPU per push batch (contact islands of pushed piles), so only 12 were tried;
  n50 letters were never solved (0/4 dense, 2/15 scattered).
