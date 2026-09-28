# Experimenter status (EXP-0059) -- live, overwritten after each task

Last updated: 2026-09-28 07:45 (coordinator round 2 complete: bootstrap CIs, neighbour-rank
curve, confidence model, coder's model multi-step -- all done before the 08:10 hard stop)

## Model x metric table (DS-0009 1-step; DS-0013 multi-step terminal; CI where computed)

| model | 1-step slateN_tough | multi-step terminal slateN_tough (95% CI where computed) | 1-step acc1 | 1-step rollout4 | mm |
|---|---|---|---|---|---|
| persistence | 0.050 | 0.075 [-0.003, 0.151] | 0.000 | 0.000 | n/a |
| nfd_3ch_narrow_l20 | 0.778 | 0.708 [0.637, 0.772] | 0.506 | 0.348 | n/a (occ) |
| nfd_3ch_narrow_l20_wide | 0.827 | 0.777 [0.715, 0.832] | 0.486 | 0.336 | n/a (occ) |
| nfd_residual_worldframe_noaug_ep43 (broad) | 0.749 | 0.665 [0.600, 0.730] | 0.466 | 0.335 | n/a (occ) |
| linear_narrow_l20_res64 | 0.633 | 0.715 [0.632, 0.788] (rising with horizon; NOT sig. above narrow NFD) | 0.452 | 0.324 | n/a (occ) |
| retrieval_1nn (orig 12k bank) | 0.436 | 0.480 | 0.172 | 0.009 | 5.77 |
| retrieval k5_cube_median @ orig 12k | 0.546 [0.433,0.650] | -- | 0.227 | 0.075 | 4.40 |
| retrieval k5_cube_median @ ds0012-only 12k | 0.552 [0.454,0.650] | -- | 0.210 | 0.036 | 4.33 |
| retrieval k5_cube_median @ 25k (best point est., task1) | 0.591 [0.499,0.672] | 0.546 | 0.232 | 0.060 | 4.20 |
| retrieval k5_cube_median @ 50k | 0.583 [0.481,0.669] | -- | 0.241 | 0.074 | 4.08 |
| retrieval k5_cube_median @ 98k (all) | 0.573 [0.473,0.666] | -- | 0.258 | 0.074 | 3.99 |
| retrieval_nfd_ref (coder, main, retrieval ON) | 0.686 | **0.642** | 0.480 | 0.341 | n/a (occ) |
| retrieval_nfd_random_donor (coder, control) | 0.597 | 0.553 | 0.464 | 0.326 | n/a (occ) |
| retrieval_nfd_ref_zeroed (coder, control) | **0.715** | 0.613 | 0.490 | 0.337 | n/a (occ) |

## Headline corrections this round (coordinator asked to reword based on CIs)

1. **Data scaling: "slateN_tough declines past 25k" is NOT statistically supported.** Pool-
   bootstrap 95% CIs on all 5 bank sizes overlap almost completely; paired delta (98k-25k)
   -0.018, CI [-0.067,+0.030], includes 0 (22.9% of bootstrap draws show 98k above 25k).
   Corrected statement: **flat beyond ~12-25k at this sample size (32 pools), not resolved**.
   `accuracy_1`/mm/oracle-mm still improve monotonically with more data (much larger samples,
   not bootstrapped here) -- that part of the original claim stands.
2. **Multi-step: "linear64 edges out narrow NFD at 3 steps" is NOT statistically supported.**
   Paired delta (linear64-narrowNFD, terminal) +0.007, CI [-0.034,+0.049], includes 0 (63.3% of
   draws favour linear64). Corrected: **statistically indistinguishable at 32 pools; point
   estimate mildly favours linear64**. The TREND itself (linear64 rising with horizon, every
   NFD falling) is a within-model pattern, not itself re-bootstrapped, and is a better-supported
   observation than the specific crossover.

## New results this round

- **Neighbour-rank curve** (`results/neighbour_rank_curve.json`): clean, near-monotone decline
  on acc1/mm/slateN_tough as retrieval rank r increases from 1 to random (acc1 0.197->0.006,
  mm 4.22->6.48, slateN_tough 0.516->0.302) -- retrieved information is doing real work.
- **Confidence model** (`results/confidence_model.json`): `knn_disagreement` correlates with
  error (rho +0.22 mm / +0.45 swept, within-pool +0.22/+0.35) -- usable signal.
  `top1_dist` does NOT (rho +0.06 mm, **-0.23 swept**, CI excludes 0 -- actively misleading for
  swept-region error; top-10%-least-confident-by-top1_dist decile has BELOW-average error,
  lift 0.75x). **Use `knn_disagreement`, not `top1_dist`.**
- **Coder's model**: 4 real bugs fixed, retrained on full DS-0008+DS-0010 corpus (20 epochs).
  1-step (20ep): `zeroed` (0.715) > `ref` (0.686) > `random_donor` (0.597), all < narrow NFD
  (0.778). Multi-step terminal (20ep): `ref` (0.642) > `zeroed` (0.613) > `random_donor` (0.553)
  -- ORDER REVERSED from 1-step; not CI-tested, flagged as unresolved not confirmed.
  **SUPERSEDED (past this record's own hard stop, ~08:14, relayed not independently
  re-verified):** `random_donor` was not a valid info-vs-noise control (conflates wrong-ref
  with no-ref). Coder's fair `retrieval_nfd_noref` twin (zeroed in training AND eval), full
  60-epoch schedule: `noref` 0.778 / `ref` 0.752 / `ref_zeroed` 0.775 on DS-0009 slateN_tough
  (DS-0011: `ref` 0.790 > `noref` 0.752 > `ref_zeroed` 0.777) -- coder's verdict "a wash, not a
  win." `retrieval_nfd_noref` is now the correct info-vs-noise baseline going forward;
  `random_donor` retired from that role. Not rerun through multi-step here.

`scripts/check_register.py` exits 0. All new compute this round ran with `CUDA_VISIBLE_DEVICES=`
(GPU reserved for the coder throughout).
