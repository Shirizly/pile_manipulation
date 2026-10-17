
## Verdicts G3 / P0-2 remainder / G5 (pre-registration LOG.md 2026-10-15 02:44; all 21 cells rc 0, 32 paired episodes each, one planner seed)

- **G3 seed gate -> ONE TIER.** ens - vanilla128 mean rel 1-10: seed 0 +.008 [-.012, +.028], seed 1 +.031 [+.009, +.052] (< .04 rule); rel@10 n.s. at both seeds. Seed spread |s0 - s1| mean rel 1-10: vanilla128 .014, zoom128 .023, ens128 .009 -- same size as the model gaps. Embarrassing check not triggered: ens s1 - vanilla128 s0 +.016 [+.001, +.032].
- **P0-2 (i) -> crossover claim SUPPORTED on the accurate side.** ens128 cem 0.3 -> 3.0 s +.041 [+.021, +.064] (rel@10 +.042 [+.013, +.074]); vanilla64 cem flat (G2). Most of the rise is below 1 s (0.1 -> 0.3 +.047, 0.3 -> 1.0 +.024, 1.0 -> 3.0 +.017 n.s.).
- **P0-2 (ii) equal wall time t = 10 x (b + 1 s).** CEM: ens below v64 at every matched budget (0.1 s -.072, 0.3 s -.032, 1.0 s -.012 n.s.); no crossing inside the measured range t 11-20 s; with v64 held flat past 1.0 s (assumption, G2) the curves cross at t ~ 33-36 s (b ~ 2.3-2.6 s) and ens at 3.0 s only ties v64 at 1.0 s (+.005 [-.013, +.021]). CEM+GD: ens - v64 at 0.3 s -.011 n.s., at 1.0 s +.010 n.s.; mean-curve crossing at t ~ 16 s (b ~ 0.6 s) but no point differs from 0; ens cem_gd peaks at 1.0 s (.765) and 3.0 s adds nothing (-.003). GD time share carries the +-15-30 % timing band. Equal-time GD gain: ens +.009 (0.3 s), +.020 [+.004, +.035] (1.0 s), .000 (3.0 s); v64 -.012 / -.012 / -.002 (all n.s.).
- **EMBARRASSING (stated plainly):** ens128 at 3.0 s/decision (30x the time) is barely above vanilla64 at 0.1 s: mean rel +.016 [-.000, +.032], rel@10 +.027 [+.005, +.052]. LF is far below both and flat (cem 0.1/0.3/1.0 s mean rel .632/.635/.648; -.11 vs v64; leaving 3.2-3.6 per episode).
- **G5 -> fetch+legal helps the cheap model as much as the accurate one.** fl - base: vanilla64 +.072 [+.055, +.089] (rel@10 +.073), ens128 +.081 [+.056, +.105] (rel@10 +.081; prior +.077); interaction +.009 [-.015, +.033] n.s. fl1F reproduces (-.002 [-.006, +.002]). Under fl, ens - v64 +.022 [+.010, +.033] mean rel, rel@10 +.005 n.s. (both near ceiling .965-.969). Phase A of P0-8 can use the fetch bank with vanilla64.
- **Spill filter 1.0 -> does NOT matter (pre-registered rule not met)**: rel@10 +.001 [-.005, +.007], leaving -.125 [-.281, +.032] (0.25 -> 0.12 per episode, n.s.), and it COSTS early progress: mean rel 1-10 -.017 [-.026, -.008]. It rejects 75 % of the initial pool on average (27 % at push 1 -> 94 % at push 10) and falls back (all rejected) in 35/320 decisions.

## Verdict G6 (pre-registration LOG.md 2026-10-15 05:45; 9 cells rc 0, 32 paired episodes each, one planner seed; rel vs each width's own V*)

- **Model effect appears with headroom -> SUPPORTED (rule met in both w10 cells).** ens - vanilla64 at 1,280 evals, mean rel 1-10 / rel@10 [95 % CI]:
  w10 base +.017 [-.003, +.036] / **+.033 [+.005, +.062]** (21/11 episodes); w10 fl **+.039 [+.025, +.054] / +.030 [+.011, +.048]** (26/6, 25/7);
  w15 base +.015 [-.004, +.033] / +.009 n.s.; w15 fl **+.036 [+.024, +.049]** / +.024 [+.011, +.037]. Reference w20: base +.013 n.s. / -.003 n.s.; fl +.022 [+.010, +.033] / +.005 n.s.
  The gap is clearest under fl (fetch + legal scoring) and at w10; under base CEM only rel@10 at w10 clears +.03.
- **fl amplifies the model gap at thin widths**: fl gain ens - fl gain v64, mean rel 1-10: w10 +.023 [+.004, +.040], w15 +.020 [+.000, +.042] (w20 G5: +.009 n.s.); rel@10 n.s.
- **Equal time (w10)**: ens base 1,280 (~1 s/decision) - vanilla64 cem 0.1 s (2,625 evals): mean rel +.007 [-.013, +.030], rel@10 +.031 [+.002, +.065] (CI barely excludes 0). vanilla64 1,280 -> 0.1 s: +.010 / +.002 n.s. (flat, as at w20).
- **Embarrassing outcome NOT triggered**: vanilla64 does not beat ens at w10 at equal evaluations (ens better in 21/32 base, 26/32 fl episodes).
- Cells (mean rel 1-10 / rel@10 / done@0.9 / done@0.8 / optimism exec / leaving per ep): w10 v64 base .576/.739/0/6/+.025/1.72; ens base .593/.772/2/14/+.023/1.78; v64 fl .641/.817/1/22/+.026/1.78; ens fl .680/.847/5/30/+.025/1.81; v64 0.1 s .585/.741/0/7/+.027/2.06. w15 v64 base .664/.830/8/22/+.020/1.47; ens base .680/.839/11/23/+.022/2.19; v64 fl .736/.920/24/32/+.022/0.69; ens fl .771/.944/29/32/+.022/0.72. Optimism and leaving do not differ between models (all paired CIs include 0).
- Caveats: one planner seed and one model seed per model; the gaps (+.03-.04) are ~2x the w20 seed spread (.01-.02, G3), so a second planner seed at w10 fl is the cheapest confirmation.

## Verdict G4 oracle pilot (pre-registration LOG.md 2026-10-15 05:45; results/oracle_g4/g4_pilot_O_w20_s40.json, cost128/; NON-CLEAN-irrelevant: only GPU job)

- **Cost**: legal CEM oracle, pop 128 x 5 iters = 631-640 sims/decision (unlegalisable rows skipped: 0-1.4 %). 32 envs: 200 s/decision (168-214), 3.19 sims/s. 128 envs (2-push cost check, same start): 120 s/decision, 5.3 sims/s. P0-6 (T + O at w20, 8 starts each = 16 episodes x 10 pushes = 160 decisions, ~102k sims): **8.9 h at 32 envs, 5.4 h at 128 envs** -> under the 12 h rule; no cut needed at 128 envs (cuts if wanted: 4 starts/letter -> 2.7 h, or 320 evals -> ~2.7 h).
- **Legality**: 10/10 executed pushes re-legalise with ok and zero shift (0 % illegal); legal_fail 0, resamples 0; 15-46 % of proposed rows needed a shift.
- **rel on letter_O_w20 start 40** (rel@10 / mean rel 1-10): oracle .999 / .788; ens2bal base1F (g1_ens_cem) .773 / .554; ens2bal fl1F (g5_ens_fl) .986 / .815; vanilla64 fl (g5_v64_fl) .993 / .814. Oracle beats both learned planners on rel@10, but **its mean rel 1-10 is BELOW the fl planners' on the same episode** (slower at pushes 4-7: .772/.875 vs .844/.936 for ens fl) -- the oracle's pile-aware pool has no fetch extras (reach_frac falls .92 -> .01 by push 8). One episode; not a ceiling estimate.

## Verdict G6 planner-seed-1 replication, fl cells (pre-registration LOG.md 2026-10-15 07:18; g6s1_*.json, code/queue_g6s1.sh)

- **Seed-1 gap ens - v64, mean rel 1-10 under fl**: w10 +.036 [+.022, +.050] (25/7); w15 +.032 [+.017, +.047] (25/7). Both >= +.02 with CI excluding 0 -> **CONFIRMED** at each width. rel@10: w10 +.044 [+.022, +.066], w15 +.016 [+.004, +.031].
- **Pooled over planner seeds 0 + 1** (32 episodes x 2 seeds, bootstrap over episodes): mean rel 1-10 w10 +.038 [+.026, +.049], w15 +.034 [+.023, +.045]; rel@10 w10 +.037 [+.024, +.052], w15 +.020 [+.010, +.031].
- **Planner-seed spread |s0 - s1|** (mean rel 1-10 / rel@10): v64 w10 .004 / .003, ens w10 .001 / .017, v64 w15 .002 / .011, ens w15 .002 / .003 -- all paired s1 - s0 CIs include 0; spread is ~10x smaller than the gap on mean rel 1-10.
- Embarrassing outcome (seed-1 gap <= 0, or seed spread >= the seed-0 gap) NOT triggered.
