# Active / hard-example data collection for the zoom-window NFD -- PLAN + PILOT groundwork (2026-10-06)

Status: PLAN for user review. Nothing here is a register claim. Everything measured is **pilot-grade** (narrow n20 exact-20 mm task, DS-0015/DS-0016, one
fitted model family, 1 seed unless stated). No existing dataset or library file was modified; pilot artefacts are in
`experiments/EXP-0073-active-data-mining/runs/active_pilot/` (labelled PILOT) and `results/active_*.json`; code `code/active_*.py`.
Not filed in COMMANDS.jsonl / no EXPERIMENT.md (EXP-0072 and EXP-0073 have none yet; moved here from EXP-0072 on 2026-10-06) -- file them when the user approves the plan.

## v2 (2026-10-06, later): reducible-error refinement + retrospective loop -- READ THIS FIRST
Sections 0-7 below are the v1 plan; where v2 changes them it is marked **[v2]**. New code: `active_common.py` (EvalSet, train_unet), `active_reducible.py`, `active_retro.py`, `active_retro_report.py`,
`active_rowcurves.py`, `active_optimism.py`. New results: `results/active_reducible.json`, `active_retro.json`, `active_rowcurves.json`, `active_optimism.json`; models/evals `../EXP-0073-active-data-mining/runs/active_pilot/retro/`.
All pilot-grade: narrow n20, window-frame metrics, one seed set / one scoring ensemble, 3 retrain seeds per arm. No new sims were run (the ceiling agent's re-sim artifacts
`artifacts/ceiling/resim_{0,0,0.01,0.05,0.25,0.5,1}mm` were reused as they appeared; the 1 mm file is excluded from the noise floor).

### v2.1 De-noised target, reducible error, proxy validity (2944 DS-0016 rows)
Replicates per row = recorded outcome + ceiling re-sims at 0 (x2), 0.01, 0.05, 0.25, 0.5 mm (K = 7; the 0.25/0.5 mm draws are below the 1 mm/px raster, i.e. unobservable to the model).
`n_i` = unbiased pixel variance across replicates (swept region); M = replicate mean (de-noised target); `e_raw` = error vs the recorded run; **`e_red = ||P - M||^2 - n_i/K`** (unbiased estimate of error vs the conditional mean = what more data could in principle remove).
* ft100: mean e_raw 1.75, noise floor n_i 0.73 (**42 % of e_raw**; top-20 % rows: 2.75 of 8.17 = 34 %), e_red 1.20 (69 % of e_raw). The share is level dependent: with only the near-zero replicates (<= 0.01 mm: pure re-run nondeterminism) the noise is just 16 % of e_raw; it grows with the jitter scale (mean cube diff to recorded: 0.13 mm at 0, 0.44 at 0.25, 0.73 at 0.5 mm).
  Noise is a **property of the model's resolution**: sub-pixel state noise is unobservable, so I count <= 0.5 mm as irreducible.
* Both `e_red` and `n_i` live in the same ~20 % of rows: top-20 % rows hold 94 % of e_red; rho(e_raw, n_i) = 0.70, rho(e_red, n_i) = 0.61, rho(e_raw, e_red) = 0.89.
* A 3-seed independent scratch ensemble mean has e_red 1.17 vs 1.35 for a single scratch model (-13 %): **plain ensembling removes about as much reducible error as ~2-3x more random data** (see v2.3) and is free of sim cost.
Proxy validity (Spearman over all rows; "partial" = rank-partial given n_i):
| proxy | rho e_raw | rho n_i (noise) | **rho e_red** | partial e_red given noise | lift (e_red top-10 %) |
|---|---|---|---|---|---|
| model self-variance sum p(1-p) (ft100) | 0.97 | 0.70 | **0.87** | 0.88 | 6.0 |
| predictive entropy (ft100) | 0.96 | 0.70 | 0.86 | 0.86 | 6.0 |
| input-gradient norm | 0.85 | 0.66 | 0.77 | 0.66 | 5.0 |
| independent-seed ensemble variance (3 scratch) | 0.81 | 0.76 | 0.73 | 0.55 | 5.3 |
| independent-seed mean p(1-p) | 0.79 | 0.77 | 0.70 | 0.50 | 6.0 |
| ft-init ensemble variance (ft100/ft300/scratch) | 0.79 | 0.75 | 0.71 | 0.49 | 5.4 |
| prediction shift under input jitter | 0.81 | 0.83 | 0.71 | 0.55 | 5.1 |
| epistemic ratio var/(gray+0.5) (indep) | 0.80 | 0.74 | 0.71 | 0.52 | 4.1 |
| prediction variance under perturbation | 0.70-0.73 | 0.78 | 0.49-0.59 | 0.31-0.36 | 2.7 |
| heuristics: n swept cubes / swept cubes in contact | 0.35 / 0.25 | 0.13 / 0.03 | 0.33 / 0.26 | 0.32 / 0.29 | 1.8 |
Reading (corrects my v1 guess): all learned proxies survive against e_red (rho 0.70-0.87), but **ensemble disagreement and perturbation shift track noise (rho 0.75-0.83) at least as strongly as they track e_red; the model's own p(1-p) tracks noise least (0.70) and e_red most.**
Caveat: ft100's p(1-p) and e_red both use ft100's P, a partial mechanical coupling; the independent-seed p(1-p) (0.70 / 0.50 partial) is the cleaner number. No proxy isolates the epistemic part; the partial correlations (0.5-0.9) say they carry information beyond noise, which the retrospective loop then tests directly (v2.2).
Heuristics are weak and are only informative for contact-rich pushes. Novelty (distance to nearest training window) remained uninformative (v1, rho +0.01).

### v2.2 RETROSPECTIVE active-learning loop on DS-0015 (zero sim cost) -- the gating experiment
Seed set 2000 random `train_v2` rows (fixed), reservoir = other 8,653 rows with known outcomes. Scoring ensemble = 4 members trained ONLY on the seed set (2 pretrained-init, 2 scratch, different seeds). Each arm adds 500 rows
(`rand1000`: 1000) chosen by its score; every arm model = identical pretrained-init fine-tune, 150 ep, retrain seeds 0/1/2 (mean +- sd over those 3). Evaluated on DS-0017 (selection set; headline) and DS-0016 (shown for the noise stratification, no decision used it). Window frame, swept region.
| arm (rows) | **window slateN** DS-0017 | window acc1 DS-0017 (pools) | e_sw all (DS-0017) | e_sw hard-20 % (DS-0017) | DS-0016: window slateN | e_sw all | pasted slateN (DS-0016) | pasted acc1 |
|---|---|---|---|---|---|---|---|---|
| base (2000) | 0.954+-0.003 | 0.9765+-0.0003 | 2.171+-0.029 | 9.83+-0.12 | 0.932+-0.000 | 2.108+-0.016 | 0.745+-0.005 | 0.606+-0.005 |
| random +500 | 0.949+-0.005 | 0.9774+-0.0006 | 2.088+-0.052 | 9.35+-0.14 | 0.934+-0.003 | 2.023+-0.026 | 0.746+-0.002 | 0.610+-0.002 |
| random +1000 | 0.950+-0.002 | 0.9781+-0.0002 | 2.025+-0.015 | 9.03+-0.07 | 0.937+-0.003 | 1.962+-0.010 | 0.747+-0.002 | 0.611+-0.002 |
| **ens-var +500** (indep. ensemble) | 0.950+-0.002 | 0.9787+-0.0002 | **1.971+-0.020** | **8.82+-0.07** | 0.936+-0.004 | 1.909+-0.007 | 0.745+-0.008 | 0.613+-0.006 |
| **p(1-p) +500** (model-own variance) | 0.946+-0.005 | 0.9789+-0.0002 | **1.952+-0.020** | **8.67+-0.09** | 0.937+-0.003 | 1.900+-0.016 | 0.741+-0.004 | 0.612+-0.001 |
| epistemic ratio +500 | 0.947+-0.001 | 0.9783+-0.0001 | 2.002+-0.008 | 8.97+-0.04 | 0.939+-0.003 | 1.930+-0.011 | -- | -- |
| decision (largest predicted dv gain) +500 | 0.945+-0.006 | 0.9778+-0.0004 | 2.051+-0.038 | 9.16+-0.15 | 0.932+-0.004 | 1.967+-0.035 | -- | -- |
| diversity only (FPS) +500 | 0.950+-0.002 | 0.9777+-0.0002 | 2.063+-0.018 | 9.21+-0.09 | 0.934+-0.002 | 2.021+-0.008 | -- | -- |
| oracle (largest TRUE error, labels used) +500 | 0.945+-0.003 | 0.9783+-0.0002 | 2.003+-0.017 | 8.78+-0.06 | 0.932+-0.003 | 1.929+-0.014 | -- | -- |
(Window-frame accuracy on pool rows is not comparable with the chain-row window accuracy 0.84 of EXP-0072 NOTES; pasted numbers are the NOTES-comparable DS-0016 ones, only run for 5 arms. sd = across 3 retrain seeds; the SELECTION is the same for all seeds except `rand`.)
Findings:
1. **slateN does not move**: window slateN is 0.945-0.954 and pasted slateN 0.741-0.747 for every arm, differences <= 0.006 with sd 0.002-0.008. Even +1000 random rows is invisible. slateN is saturated/underpowered at this scale -- it can only serve as a no-harm check.
2. **Swept-region error and accuracy do move**: ens-var +500 gives e_sw -9 % vs base (random +500: -4 %, +1000: -7 %); the paired delta vs `rand` is -0.117 (ens) / -0.136 (p(1-p)) with seed sd 0.02-0.05, i.e. 2-4 sd; hard subset -0.53 / -0.68 vs rand (-5.7 % / -7.3 %). Window accuracy: 0.9787 / 0.9789 vs 0.9774 (rand) / 0.9781 (rand1000), sd 0.0002-0.0006. **500 mined rows beat 1000 random rows** (data-efficiency about 2.5-3.5x on e_sw, by interpolation of the random curve) -- on both DS-0017 and DS-0016 (1.909 / 1.900 vs 1.962).
3. **Which scores work:** p(1-p) >= independent-ensemble variance > epistemic ratio > oracle-by-true-error (!) ~ ens > decision score > diversity ~ random. The label-using oracle is **not** better than ens-var: true error includes noise rows; mining raw error is no better than the proxies. Diversity-only and the decision score give nothing over random (dec gain 0.04, div 0.02). Overlap of the 500 sets: ens-gray 209, ens-epi 418, ens-oracle 188, dec-others <= 77, div-others <= 57.
4. **Stratified:** DS-0016 e_sw by noise-floor tercile (low/mid/high; noisiest 10 %), seed means: base 0.071/0.950/6.05 (11.7), rand 0.075/0.902/5.79 (11.3), rand1000 0.069/0.879/5.62 (10.9), ens 0.075/0.855/5.46 (10.7), gray 0.067/0.879/5.44 (10.5). Low-noise rows are flat in every arm (nothing to learn); gains sit in mid/high-noise rows. Because the hardest rows are also the noisy ones, mean e_sw improvements are mostly a hard-row effect, and there is no gain hidden in the mean.
5. Seed noise: retrain sd of e_sw 0.01-0.05 (0.5-2.5 %), of slateN 0.002-0.008; the selection set is a single draw (not replicated: selection variance NOT measured -- caveat).

### v2.3 Per-row learning curves (item 3)
Same architecture, 25 / 50 / 100 % of DS-0015 (scratch, equal steps, 2-3 seeds at 100 %), 2944 DS-0016 rows, an improvement counts only if the 25 %-> 100 % drop exceeds 2 per-row seed-sd + 0.05.
mean e_sw 2.42 (25 %) -> 2.04 -> 1.87 (100 %). Top-20 % rows: 10.7 -> 9.2 -> 8.6; **32 % of top-error rows improve with 4x data, 63 % plateau, 4 % get worse**; top rows carry 78 % of the total reduction. Improvement is weakly related to noise (rho +0.23 with n_i, +0.27 with e_red; within the top rows +0.1) -- the plateau is not only noise: capacity/bias or input resolution also bind. rho(per-row error, 25 % vs 100 % model) = 0.89: the same rows stay hard.

### v2.4 Decision relevance / optimism (item 4)
DS-0016 pools, the model's top-1 push per (pool, goal) (416 picks per model): picked push true-rank percentile 0.02 (excellent ranking), mean optimism (pred dv - true dv) +0.001 (sd 0.007; 40-43 % of picks optimistic) -- **no systematic optimism** at this window dv scale, magnitude weakly tied to proxies (|rho| <= 0.10) and noise (0.04).
Picked rows have 1.6x the average error (e_sw 2.6-3.0 vs 1.6-1.8; 29-31 % in the top error quintile vs 20 %) -- the planner favours moving pushes, which are the error-dense ones -- but a `dec` selection score (largest predicted dv gain) did not beat random in the retrospective loop. Evidence that on-policy error matters for closed loop (EXP-0039) is not contradicted, but it is not exploited by this score; a real test needs the closed-loop pools (DS-0023 style), not offline rows.

### v2.5 Go / no-go (item 6) and revised cost
What the evidence says: (i) random data is a flat lever (4x data: e_sw -23 %, window acc +0.009, pasted slateN +0.01), (ii) mined data is 2.5-3.5x as efficient as random for swept-region error/accuracy in the retrospective loop, (iii) slateN does not respond at all (saturated / below noise), (iv) 40 % of raw error is a noise floor at this resolution, and only ~1/3 of the hard rows improve with 4x data, (v) a 3-seed ensemble removes ~13 % of e_red for free.
Verdict: **NO-GO for the ~6 h, 15k-sim 3-arm study as designed in v1** (its headline slateN cannot show a gain; its error/accuracy gain would be a few %); **GO only for a small live validation round if the user wants error/accuracy gains** (see below). If the user's goal is slateN or closed-loop control, spend the effort on ensembling and on the control-relevant eval instead of more data.
Smaller first live round (replaces v1 section 5 cost): from the full 10.7k DS-0015 model; B = 2,000 new sims per arm (~19 % more data -> expected random effect -3..4 % e_sw, mined -7..10 % if the retrospective effect transfers), arms C0 random (DS-0015 recipe), M mined (indep.-ensemble variance + p(1-p), 25 % random quota, pile-aware legal candidates, no CEM), optional C1 (perturbed + random). Sims ~0.6 s x 2000 ~ 20 min/arm (observed, shared GPU); retrain 3 seeds x 4.5 min/arm; ensemble for scoring 3 x 4.5 min; eval 5 min => ~2 h total for C0+M (3 h with C1), no replicates.
**Go criterion for a second, larger live study:** on DS-0017 (3 retrain seeds, paired by seed) M beats C0 by >= 4 % in e_sw (all rows) AND >= 6 % in the hard-20 % subset AND window accuracy >= +0.002, with M-C0 >= 2 paired seed-sd, AND window/pasted slateN not lower than C0 by > 0.01; else stop and report. Re-estimate rho(score, e_red) on the current model each round; < 0.5 -> random fallback. DS-0016 only for the final report.

### v2.6 What changed in v1 sections
* Section 3 [v2]: the 45 %/55 % chaos/excess split came from 96 stratified rows at 0.25 mm; the 7-replicate de-noised estimate gives 34 % noise in the top rows (42 % overall). Re-sims from the identical recorded state are not bit-identical (0.13 mm mean cube difference).
* Section 2 [v2]: p(1-p)/entropy are NOT mainly chaos trackers (rho with noise 0.70 vs 0.87 with e_red); ensemble disagreement and jitter shift are the more noise-contaminated ones.
* Section 4 [v2]: scratch seed 2 under-trained (val MSE 0.0170 vs 0.0122 for seeds 0/1/3 at 100 ep) -- one of four scratch seeds failed to converge at 100 epochs; use >= 150 epochs or pretrained init for ensemble members.
* Section 5 [v2]: CEM action optimisation, perturbation ops and the 25 % exploration quota are untested live; the retrospective loop used selection among existing sampler draws only. Dedup/diversity-only selection gave nothing. Open questions are in section 6 [v2] (rewritten at the end of this file).

---

## 0. TL;DR (v1; see v2 above for what changed)

1. **Perturbation** of existing states is cheap and can be made 100 % legal (SAT, in tray, still layer 0 after an in-sim re-settle). Six ops built and checked (section 1).
2. **Error is extremely heavy-tailed**: the zoom-64 model's swept-region error has median 0 and the top 20 % of rows hold **93 %** of the total (top 10 % = 68 %). Random collection spends
   most sims on pushes that move almost nothing (easy). So there is real room for targeting -- *if* the hard rows are fixable by data.
3. **Cheap proxies predict the realised error well** (Spearman 0.8-0.97 with swept-region error, lift 5-6x in the top decile), but **mostly because they are the model's own
   predicted-variance**, and the trivial heuristic "number of cubes swept / in contact" is already worth rho 0.35-0.65 (section 2).
4. **The error is NOT a coverage problem in the model's input space**: distance of a test window to its nearest training window has rho **+0.01** with error (section 3). Hard rows
   are contact-rich pushes (more swept cubes, ~2x contacts), not unseen configurations.
5. **A large part of the hard-row error is chaos (aleatoric)**: on 96 re-simulated rows, in the top-20 %-error rows the sim's own outcome variance under 0.25 mm / 0.5 deg
   state jitter accounts for ~45 % of the model's error; ~55 % is excess (epistemic or bias). Even an *exact* re-sim differs from the recorded outcome by 0.4 mm/cube (not bit-identical), and
   0.25 mm of state noise is amplified to ~1 mm of outcome spread (section 3). Proxies track chaos about as well as they track error, so **proxy alone cannot separate epistemic from aleatoric**.
6. **Random data is weak**: halving/quartering DS-0015 costs only ~0.005/0.009 pasted accuracy and ~0.01-0.02 pasted slateN (section 4); the pasted-accuracy ceiling is 0.651 and the model is at 0.615.
   Mining therefore has to beat a very flat baseline curve, and the headline metrics must be the WINDOW ones and a hard-subset error, not pasted accuracy.
7. Recommendation: run the loop as a **3-arm controlled study** (C0 random states+actions, C1 perturbed states + random actions, M mined) with chaos-aware bookkeeping (jittered re-sim replicates
   at collection time, which also gives the aleatoric estimate for free), a 25 % random exploration quota, and a go/no-go gate after round 2. Cost ~40 min/round/arm (section 5).

## 1. Seams and what "largish perturbation of an existing state" can mean

### Reuse (already in the repo)
| need | seam |
|---|---|
| start states from a corpus | `Genesis/data/narrow_l20_n20/train_v2_clean/_k_data.pt` -> `states` (all chain steps), `start_kind`, `valid`; or `chain_collection.py --starts file --starts-file X.pt` (tensor (N,20,7)) |
| write arbitrary state + settle | `sim.set_particle_state(pos, quat)` + `sim.update_material_state()` (`chain_collection.py::set_states`); `clump_states.make_clump_states` is the worked example incl. layer-0/in-tray rejection |
| legal pile-aware actions | `Genesis/action_sampling.py::pile_aware_action_batch` (Genesis-free, exact SAT redraw, `start_gap_range [0.005,0.005]`, exact 20 mm) via `sim.generate_action_samples(1, pile_aware=True, min_swath_particles=3, push_length=0.02, start_gap_range=[0.005,0.005])` = DS-0015/16 recipe; `chain_collection.check_actions` for exact length/perpendicular |
| project an optimised push to legal | `action_sampling.legalize_pushes` (slides back along the push, exact SAT `cube_overlap.overlaps_rect_pairs`; returns `ok=False` when impossible -> drop, never execute) |
| simulate + write | `chain_collection.py --mode chains/pools` (atomic per-chunk, resumable, manifest); pools mode = same-state pools |
| validity | per row `valid`, `gap_out_of_window`, `single_layer` (+ `PileSweepData(exclude_flagged=True)` loader) |
| chaos / perturbation precedent | `EXP-0059/code/chaos_floor.py::perturb` + re-sim recipe (what `active_resim.py` copies), `perturbed_sim_zoo.py` |

New code (pilot, `code/`): `active_perturb.py` (ops + legality + `settle_in_sim`), `active_proxies.py` / `active_proxy_corr.py` (proxy vs error), `active_resim.py` /
`active_resim_analyse.py` (chaos probe), `active_novelty.py`, `active_perturb_viz.py`. `load_rows` in `active_proxies.py` is the DS-0016 reader reused by all of them.

### Physically meaningful perturbation ops (all keep n=20, layer 0)
| op | what it does | legal? | distribution-preserving? |
|---|---|---|---|
| `jitter_small/large` | per-cube xy N(2/6 mm) + yaw N(0.3/0.8 rad), sequential rejection | yes | keeps density/wall-gap stats; **kills the few existing contacts** (scatter contacts 0.48 -> 0.02-0.04), so it generates *looser* scenes than the corpus. Good for action/geometry diversity, bad for contact diversity |
| `relocate3/6` | move 3/6 cubes, 50 % to a free spot, 50 % to touch another cube (0.3-1 mm gap) | yes | **creates contacts** (0.48 -> 0.73 / 1.09 mean contacts); nn-distance stats shift slightly (16.5 -> 16.1 / 15.7 mm). The main op for generating hard (contact-rich) scenes |
| `cluster_shift` | rigid translate/rotate one contact component | yes | preserves contact topology; weak on scatter (components are mostly single cubes, mean move 1.2 mm) -- useful on clump sources |
| `rigid` | rotate the whole pile by a random angle + translate inside the tray | yes | preserves topology exactly; changes wall relation (min wall gap 4.9 -> 4.5 mm, bbox +4 %). By symmetry the physics is unchanged, so the *model* (flip/D4-augmented) gains little; only wall-adjacent contexts are new |
| D4 flip/rotate of the whole state | exact tray symmetry | yes | **not useful** for data: the model is already trained with it |
| add / remove cubes | changes n | -- | **not distribution-preserving** for the fixed-n20 pilot; skip (revisit only for the broad task, where n varies) |
| re-settle in sim | `settle_in_sim` after any op | -- | mandatory: resolves contact geometry and gives the exact state the collector would record |

Sanity (256 DS-0015 `train_v2_clean` scatter states, `results/active_perturb_sanity.json`, figure `figures/active_perturb_examples.png`):
all six ops succeed 256/256 with **0 % illegal after the op**; after a 32-env sim re-settle: 0 % overlap, 0 % out of tray, drift 0.004-0.03 mm mean (0.06-0.6 mm max), layer-0 100 %
except `jitter_large` 96.9 % (1/32 cube popped out of layer 0 -> reject on the post-settle check, built into the report). Legality definition: moved cube keeps >= 0.1 mm SAT gap; whole state
passes SAT with squares shrunk by 0.2 mm, because the *source* sim states themselves touch/interpenetrate by up to 0.2 mm (only 52 % of source scatter states pass a strict 0.2 mm *gap* test;
a naive strict test would silently reject real data). Scatter sources are sparse (mean 0.5 contacts) -- push outcomes there are mostly single-cube; clump sources or `relocate` are where interactions live.

## 2. Do cheap uncertainty proxies predict the realised error?

Setup (`active_proxies.py`, `results/active_proxy_corr.json`): DS-0016 `test_chains_v2_clean` (896) + `test_pools_v2` valid (2048) = 2944 rows with simulated truth. Error is measured **in the zoom window**
(1 mm/px, ft100 model): `e_sw` = squared error summed over eval_narrow's swept region (what `accuracy` integrates), `e_chg` = error on pixels the truth or prediction changes, `e_win` = whole window. Proxies never see the truth.
Ensembles: ft100 / ft300 / scratch100 (zoom-64) [+ world128_300]. Perturbation variance: K=8 draws, action jitter (1 mm, 1 deg), input jitter (0.5 mm, 1 deg), or both.
Gradient: norm of d(sum swept pred)/d(input) for the image / plate channels. Heuristics from particles. Control: `pred_mass` = sum|pred - input| in the swept region.

Spearman rho with `e_sw` (all 2944 rows | within-state, pool rows only | partial given `pred_mass` | lift = share of total error in the top-10 % by score / 0.10; random = 1.0):

| proxy | rho | within-state | partial | lift |
|---|---|---|---|---|
| model self-variance in swept region (sum p(1-p)) "grayness" | **+0.97** | +0.92 | +0.97 | 6.1 |
| predictive entropy (swept) | +0.96 | +0.90 | +0.95 | 6.1 |
| input-gradient norm (plate ch / all / image ch) | +0.86 / +0.85 / +0.85 | +0.82 | +0.85 | 5.0 |
| perturbation: mean |shift| of prediction, input jitter | +0.81 | +0.66 | +0.80 | 5.1 |
| ensemble-3 variance (swept) | **+0.79** | +0.82 | +0.75 | 5.5 |
| ensemble-4 variance (+ world128) | +0.76 | +0.63 | +0.74 | 5.1 |
| perturbation: variance (act / inp / both) | +0.69 / +0.73 / +0.70 | +0.40-0.43 | +0.74-0.76 | 2.6-2.7 |
| ensemble variance over whole window | +0.45 | +0.37 | +0.28 | 3.6 |
| heuristic: n cubes swept | +0.35 | +0.23 | +0.11 | 1.8 |
| heuristic: n swept cubes in contact / contacts in window | +0.25 / +0.23 | +0.14 / -0.10 | ~0 | 1.8 / 0.8 |
| predicted mass moved (control) | +0.36 | +0.19 | -- | 1.6 |
| clump start, wall distance, gap to first cube, chain step | +0.14 / +0.10 / -0.07 / +0.04 | ~0 | ~0 | 0.5-1.7 |

(On `e_chg`/`e_win`, the whole-window proxies and heuristics rank higher -- n contacts +0.67, window ensemble variance +0.81 -- because those errors scale with how much is in the window; swept-region
proxies are the ones that matter for the metric the user cares about (swept region = `accuracy`'s region).)

Reading:
* **Which proxies work:** ensemble variance (swept), grayness/entropy, input gradient, input-jitter prediction shift. All have rho >= 0.75 and lift >= 5. Perturbing the *variance* version is weaker than the *shift* version.
* **Caveat that matters:** grayness/entropy are the model's own predicted variance -- an MSE regressor that is calibrated has error ~ p(1-p), so this is rho~1 by construction and it measures *how
  unresolved the model considers the outcome* (aleatoric-flavoured), not what it has not learned. Ensemble disagreement is the more epistemic signal, but our members share an init (ft100 vs ft300 error rho 0.94, scratch100 vs ft100 0.80), so true
  independence needs different seeds/inits (cost: 3 trainings per round, ~12 min).
* Ensemble mean beats the single model slightly (mean e_sw 1.66 vs 1.75; ft300 1.67, scratch100 1.93, world128 2.78) -- an ensemble is also the cheapest accuracy gain, independent of data.
* Heuristics alone (n swept cubes/contacts) are far weaker (lift ~1.8) -- the model-based signals add real information.
* **Proxies must be re-validated on DS-0017 (val pools)** before use: I scored them on DS-0016, the test set, which is only acceptable for this feasibility check; final numbers of the study must come from untouched pools.

## 3. Epistemic vs aleatoric: re-simulation probe (96 rows, 480 sims) and coverage

`active_resim.py`: 96 rows (48 from the lower 80 % of error, 48 from the top 20 %), each re-simulated 1x exactly from the recorded state, 3x with xy N(0.25 mm)+yaw N(0.5 deg), 1x at 1 mm / 1 deg
(batches of 32 envs; 14-35 s per batch on a shared GPU -- not a benchmark). `results/active_resim.json`.

* **A re-sim from the identical recorded state is NOT bit-identical** (EXP-0024's "near-bit-identical" holds for its snapshot-restore seam, not for re-writing a recorded state into a fresh batch):
  per-cube xy diff to the recorded outcome averages 0.42 mm on moved cubes (median row-max 0.55 mm; 93/96 rows > 0.1 mm; unmoved cubes 0.006 mm). With 0.25 mm / 0.5 deg input noise the outcome
  spread between re-sims is ~1.0 mm (mean on moved cubes; 44/96 rows have a cube spreading > 1 mm), and at 1 mm noise 95/96 rows deviate > 1 mm. i.e. the sim is chaotic at the model's input resolution (1 mm/px).
* **Aleatoric share:** let `alea` = pixelwise variance of the 3 jittered re-sim windows summed over the swept region (the irreducible error of a conditional-mean predictor that cannot resolve sub-pixel state).
  Random-ish rows: mean e_model 0.20 vs alea 0.39 (the model is already below the chaos floor; nothing to learn there). Top-20 % rows: e_model 8.7 vs alea 3.9 -> **~55 % excess, ~45 % chaos**;
  58 % of top rows have e_model > 2 alea + 0.5. rho(e_model, alea) = 0.75 (0.44 within the top group) -- hard rows are mostly *also* chaotic rows.
* **Proxies track chaos about as well as error** (e.g. ens3 variance: rho 0.88 with e_model, 0.77 with alea, 0.66 with excess = e_model - alea; grayness 0.96 / 0.79 / 0.71; gradient 0.86 / 0.72 / 0.63;
  heuristics 0.3-0.4 with excess). Partial rho with e_model given alea: grayness +0.89, ensemble-3 +0.62, gradient +0.60, input-shift +0.54. These are 96 stratified rows (3 re-sims each): treat as indicative.
* **Coverage test:** rho(error, distance to the nearest DS-0015 train window) = +0.013 (5th-NN: +0.045; within-state +0.067; error in the farthest decile 2.5 vs 1.3-1.9 elsewhere, non-monotone). In this crude feature space (avg-pooled 16x16 window + plates) **novelty does not predict error**: more
  data "near" the hard rows is already available; what is hard is the outcome (contact-rich chaotic pushes), not the input. Density/novelty-based diversity is therefore *not* a substitute for the error proxy, and
  "fill the sparse regions" is not expected to help.

Consequence for the plan: the loop must (i) log a chaos estimate per collected row (jittered replicates, which are themselves legitimate training transitions), (ii) report the metric on a fixed *hard subset* and on all rows, and
(iii) include the random-collection and perturbed-random controls, because the only thing that proves epistemic gain is that mined data beats equal-size random data. A conditional-mean regressor cannot go below the chaos floor; the achievable
gain from data on the top rows is bounded by ~55 % of their error (upper bound, one-sided), i.e. at most ~half of the 93 %-of-error tail.

## 4. How much does plain data help? (learning curve, zoom-64 scratch, same steps)

DS-0015 train rows subsampled (seed 0), UNet [4,8,16] scratch, equal gradient steps (`../EXP-0072-zoom-window-nfd/code/train_zoom.py`, `../EXP-0073-active-data-mining/runs/active_pilot/lc_*`, `score_*.json`), scored on DS-0016 with `score_zoom.py`:

| train rows | val MSE | pasted acc1 / slateN | window acc1 / slateN |
|---|---|---|---|
| 2.7k (25 %, 400 ep) | 0.01376 | 0.606 / 0.734 | 0.823 / 0.850 |
| 5.3k (50 %, 200 ep) | 0.01285 | 0.611 / 0.745 | 0.832 / 0.917 |
| 10.7k (100 %, 100 ep) seed 1 | 0.01215 | 0.615 / 0.746 | 0.839 / 0.934 |
| 10.7k seed 0 (EXP-0072 NOTES) | 0.01235 | 0.615 / 0.761 | 0.836 / 0.925 |
| ft300 (pretrained init, 300 ep) | 0.01156 | 0.622 / 0.765 | 0.846 / 0.947 |

Per data doubling: +0.004 pasted acc1, +0.01 pasted slateN, +0.008 window acc1; seed sd ~0.003 (acc) / ~0.01 (slateN). The pasted accuracy ceiling (true label pasted) is 0.651, so pasted acc1 has ~0.03 headroom; window
metrics have ~0.15. **A mined-data gain must be measured against this flat curve**: e.g. 4k mined rows are worth more than 4k random rows only if window slateN/acc1 and the hard-subset error improve by clearly more than ~+0.01 / ~+0.008, with >= 3 seeds per arm (seed sd comparable to the effect).

## 5. The loop (proposal)

Notation: B = new sims per round per arm, K = candidates scored per kept sim.

**Round r (r = 0 is the existing DS-0015 model + ensemble):**
1. *States.* Draw 4,000 parents from DS-0015 `train_v2_clean` (scatter-only first, as the user asked; add clump later; parents from DS-0015 only so DS-0016/17 stay clean). Apply a recipe mix (relocate3 30 %, relocate6 20 %,
   jitter_large 15 %, rigid 15 %, cluster_shift 10 %, nothing 10 %) -> legal states; settle in sim in batches of 32 (~2 s per batch; drop states that leave layer 0 / overlap post-settle).
2. *Actions.* Per state draw 64 legal pile-aware candidates with the DS-0016 sampler (Genesis-free `pile_aware_action_batch`, CPU: no sim needed for sampling) -> 256k (state, action) pairs.
3. *Score (stage A, cheap).* One window raster per pair (~1 ms CPU, ~4 min for 256k; the GPU raster `window_gpu.py` if that dominates) + 3-member ensemble forward (~1 ms per 32 rows on GPU) -> `ens_var_sw`. Keep the top 5 % plus
   the grayness/gradient columns for the report. K=8 perturbation variance is 9x the raster cost: use only in stage B.
4. *Optimise (stage B).* For the top ~3,000 pairs run 3 CEM iterations (population 128 around the pair's action: heading +-15 deg, lateral +-6 mm, gap in [5, 15] mm; elite 20 %), objective `ens_var_sw` (+ optional 0.5 * `pert_input_shift`),
   **project every candidate with `legalize_pushes` (box = tray minus 4 mm) and re-check exact length/perpendicular with `check_actions`; drop `ok=False`** -> ~384k extra model evals, ~2-3 min. This is "optimise actions/states for worst quality" with legality hard-wired. State-level optimisation = take the
   state's best action score (max); we do not backprop to the state (its raster is non-differentiable and the gradient proxy is only a ranking signal).
5. *Select B = 1,536 (48 sim batches).* 50 % top-scoring after dedup, 25 % diversity (farthest-point among the top 30 %), **25 % uniformly random legal pairs from the same perturbed pool (exploration/safety against proxy failure)**.
   Dedup: reject pairs whose (16x16 pooled window + plates) feature is within the 2nd percentile of train-window NN distances of an already chosen or training row; max 2 actions per perturbed state, max 8 children per DS-0015 parent
   (tag `parent_id` so train/val splits are by parent -- the DS-0002 grouping rule: children of one parent never straddle a split).
6. *Simulate.* Through `chain_collection.py --starts file` with an actions-file seam (the one new driver feature: `--actions-file`, replacing `draw_valid`; same atomic chunk writes, manifest, `valid`, `gap_out_of_window`, `single_layer`). **Each selected pair is run once exactly and twice more from
   the 0.25 mm-jittered state** (replicates are legitimate transitions of nearby states; they give `alea` per row; cost x3 -> optional, see open question 4). Row columns added: `arm, round, parent_id, recipe, score_ens, score_gray, score_grad, chaos_var, source=mined|explore|diverse`.
7. *Validate rows.* Drop `valid==False`, `gap_out_of_window`, not `single_layer`, cube escaped/out of tray, SAT-illegal touchdown (must be 0), null rows if > 0.5 % of a batch (DS-0015 has 2.3 %, all inside `valid==False`); report counts per round.
8. *Retrain.* Zoom-64 fine-tune from the EXP-0022 UNet, 300 epochs (~4 min) on DS-0015 + all rounds so far, same recipe, same seed schedule for every arm; 3-member ensemble for next round's scoring (ft/scratch/seed variants, 12 min). Zoom-128 (~10 min) only at round 0 and the final round.
9. *Evaluate (fixed).* DS-0016 chains + pools (no tuning on it) and DS-0017 val pools for model/round selection. Report per round, in this order: **window slateN** and **pasted slateN** (13 goals, `score_zoom.py`, same code path), then pasted/window `accuracy_1`, then
   `mean e_sw` on the fixed hard subset (top-20 % rows by the *round-0 ensemble-mean* error, never re-selected), `mean e_sw` on all rows, and the chaos-normalised excess on the 96-row re-sim set. Always the persistence (0) and previous-round rows.

**Arms (equal B, equal retrain recipe, shared parents):**
* C0 -- random: unperturbed random scatter/clump states + sampler actions (= the DS-0015 recipe).
* C1 -- random actions on **perturbed** states (separates "perturbed states help" from "mining helps").
* M -- mined (steps 1-8).
* (free) W -- no new sims: reweight DS-0015's own hard rows (loss weight proportional to ensemble error) -- tells whether tail emphasis alone helps before spending sims.
Final comparison: 3 seeds per arm at the final round, paired bootstrap over DS-0016 states (`Baselines/common/paired_stats.py`; state is the replication unit), report deltas M-C1 and M-C0 with CI.

**Stopping / gates.** Gate after round 2 (3,072 rows/arm, ~+30 % data): continue only if M beats C1 by more than the seed noise on window slateN OR the hard-subset error (>= ~0.01 slateN or >= 10 % relative hard-subset error), otherwise stop and report the null (a very plausible outcome given section 3: the tail is mostly
chaos and the non-chaos tail is not coverage-driven). Hard stop at 8 rounds (~12k new rows ~ DS-0015 doubled) or when the pasted accuracy is within 0.01 of the 0.651 ceiling. Re-validate the proxy (rho with realised error of the *current* model on DS-0017) every round; if rho(score, error) drops < 0.5, fall back to random.

**Cost per round per arm (shared GPU; measured ~0.5-0.8 s per sim, 14-35 s per 32-env batch; not a benchmark):** state perturb+settle 3 min; scoring + CEM ~10 min; sims 1,536 x 0.6 s ~ 15 min (x3 with replicates ~ 45 min); retrain 4 min (+12 for the ensemble); eval ~3 min.
~35-40 min/round without replicates, ~70 min with. Control arms need no scoring (~25 min). Full study: 3 arms x 4 rounds ~ 5-6 h wall, plus final 3 seeds x 3 arms x (4+3) min ~ 1 h.

**Failure modes and guards**
* Proxy-gaming: ensembles agree on garbage / extreme chaotic states -> 25 % exploration quota, chaos replicate column, proxy re-validation each round, hard-subset metric defined once.
* Distribution shift: `rigid`/`relocate` states may not occur in closed-loop use; compare perturbed-state stats with DS-0023 visited states (nn-distance, contacts, wall gap) before round 1.
* Illegal actions: CEM output only enters via `legalize_pushes` + exact SAT assert; the audit `EXP-0059 audit_tool_placement.py` must read 0 % per round.
* Escaped / popped cubes (3 % under `jitter_large` in the pilot), null transitions, `valid==False`, `gap_out_of_window`: filtered and counted (see `PileSweepData exclude_flagged`).
* Near-duplicates: dedup + per-parent cap; split by parent.
* Retrain noise: seed sd is the same size as the expected effect -> 3 seeds at gates; do not rank single-seed round-to-round differences.
* Heavy tail makes mean-error metrics noisy: report per-state paired bootstrap, not row SEM.

## 6. Open questions for the user (v1 list; superseded by the v2 list at the end of the file)
1. **Target metric.** Pasted accuracy_1 is within 0.03 of its raster ceiling; do we optimise window slateN / hard-subset error (where headroom exists) and accept that pasted numbers will barely move?
2. **Budget and approval:** is a ~6 h, ~15k-sim study (3 arms x 4 rounds) acceptable on the shared GPU, and is the "stop at gate 2 if no gain" rule right?
3. **Is the perturbed-state distribution acceptable?** `relocate` (contact creation) and `rigid` (pile against walls) generate scenes the closed loop may never visit; use DS-0023 visited states as the in-distribution reference, or restrict to visited-like perturbations?
4. **Chaotic rows:** keep them (the regressor learns the conditional mean; 3 jittered replicates make that mean better estimated at 3x sim cost), drop them, or down-weight? The replicate design triples sims per selected pair.
5. **Epistemic target:** accept ensemble variance (needs 3 independently seeded members per round, +12 min) as *the* score, or add a different model family (retrieval banks / world128) as disagreement partner?
6. **Scatter-only vs scatter+clump** parents, and whether the first pilot round should be scatter-only (sparse, mostly 1-cube pushes: weaker interactions) as proposed.
7. **Evaluation honesty:** the proxies here were scored on DS-0016 (test); are we allowed to use DS-0017 for proxy/round selection and keep DS-0016 untouched for the final report, or should a fresh pool set (new seed) be collected?
8. **Pilot vs broad:** narrow data is a pilot instrument; any positive result from mining must be replicated on broad data before it is a project claim -- should the plan include a broad replicate, or stay as a cheap R&D loop?
9. Use cube-position error (mm, per cube) as an additional metric? Window-image error mixes position and shape.

## 7. Provenance and limits
* All numbers: DS-0015/DS-0016, zoom-64 NFD (ft100 for errors; ft300/scratch100/world128_300 as ensemble), one training seed each, DS-0016 rows with `valid` only (2944), swept-region error in the 64 mm window frame.
* The 96-row chaos probe is stratified (48 random-ish, 48 top-20 %), 3 jittered re-sims, jitter 0.25 mm/0.5 deg; shares of "chaos vs excess" depend on that jitter scale (a larger jitter raises the aleatoric share; 1 mm re-sims also were run but only one sample each).
* No timing benchmark was run (GPU shared); sim seconds are observed wall times.
* Files: `results/active_proxy_corr.json`, `active_resim.json`, `active_novelty.json`, `active_perturb_sanity.json`; arrays `../EXP-0073-active-data-mining/runs/active_pilot/{proxy_rows,resim_rows,perturbed_scatter_pilot}.npz`; checkpoints `runs/active_pilot/lc_*`.

## 8. Open questions [v2, current list]
1. **Target metric / is this worth doing?** The retrospective loop shows a ~10 % swept-region-error and +0.001-0.002 window-accuracy gain over random from mined data but no measurable slateN change (saturated at ~0.95 window / 0.745 pasted). Do you want error/accuracy improvements for their own sake, or is slateN / closed-loop control the target (then data mining is the wrong lever; ensembling is)?
2. **Go for the ~2-3 h live validation round (2,000 sims/arm, C0 vs M) or stop at the retrospective result?** (The 6 h study is not recommended.)
3. **Which noise scale counts as irreducible?** I used <= 0.5 mm (below the raster). A different convention (e.g. 1 mm = observation noise, or 0.01 mm = determinism only) changes the noise share from 16 % to 40 %+.
4. **Chaotic rows**: keep (the regressor learns the conditional mean; the retrospective result shows high-noise rows are where mined data helps) or replicate/drop? Replicates triple sims.
5. **Epistemic score**: the simplest well-performing score is the model's own p(1-p) (best in the retro loop and on e_red) or an independent-seed ensemble variance (more epistemic, more noise-tracking, needs 3-4 trained members per round). Which do you prefer, or both?
6. **Perturbed states / contact creation**: still unvalidated live (the retro loop only re-uses existing sampler draws). Should the first live round use unperturbed fresh states (cleaner comparison) or the perturbed pool?
7. **Ensembling as a deliverable**: a 3-seed ensemble gives -13 % e_red at zero data cost -- do you want ensemble models counted as part of the "best prediction quality" target?
8. **Evaluation honesty**: DS-0017 was the selection set here; DS-0016 was looked at for noise stratification only. Should a fresh pool set be collected (new seed) for the final report?
9. Per-cube mm error and pasted slateN on more arms (only 5 arms have pasted scores) -- do you want them for all arms?
10. Narrow pilot only: any positive live result needs a broad-data replicate before being a project claim.
