# Pilot eval: unwarped vs. warped vs. warped+walls NFD, L20mm-only (RUN-0004)

**This is a shakeout, not a verdict.** `n20_L20mm` is a SINGLE-push-length corpus (every push
in the training AND eval slates is ~20mm), so in the canonical push frame every push looks
nearly identical to every other push -- this is the most favourable possible case for a
push-frame model, because the warp has almost no cross-length variation to normalise away. It
says very little about whether the warp generalises across push lengths. **The comparison that
actually decides anything is the `overnight_randlen` (multi-length) run against EXP-0001's own
numbers.** Treat everything below as "did the wiring work and is the effect measurable at all",
not "does the warp help".

Harness: `Baselines/common/eval_report.py` (same code as EXP-0001), corpus `L20mm`
(`configs/dataset/genesis_slates_multistep_n20_L20mm_eval.yaml`, 7680 transitions, 60 step-0
slates of 128 candidates each). Run: `../runs/RUN-0004-pilot-eval/`. Raw JSON:
`../artifacts/RUN-0004-pilot-eval/pilot_eval.json`.

All three models (and the reference rows) ran on **CPU** -- `Baselines/common/eval_baseline.py`'s
`_predictor_batch` never moves the batch off CPU (documented trap); GPU was idle and unused for
this eval. This does not change the numbers, just where they ran.

## Lead metric: slateN capture (goal-averaged over 3 goals, per value function)

| arm | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| **random** (ranking floor) | -0.008 | -0.005 | -0.012 |
| nfd_unwarped (control) | **0.881** | 0.823 | 0.765 |
| nfd_warped (no walls) | 0.834 | **0.837** | **0.804** |
| nfd_warped_walls | 0.846 | 0.825 | 0.796 |

All three arms are far above the random floor and close to each other -- within roughly 0.03-0.05
of one another on every value function, i.e. no arm dominates all three. The unwarped control
leads on `lyapunov`; the warped arms (with or without walls) lead on `mass_in_region` and
`signed_mass`. **This does not read as a clear win or loss for the warp** at this pilot's scale
(60 slates, single push length) -- treat the ranking as noisy until the randlen run.

Per-goal breakdown (goal x value_fn), for completeness:

| arm | goal | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|---|
| unwarped | random_quadrant | 0.976 | 0.971 | 0.978 |
| unwarped | ring_O | 0.818 | 0.736 | 0.564 |
| unwarped | T | 0.848 | 0.763 | 0.752 |
| warped | random_quadrant | 0.953 | 0.926 | 0.973 |
| warped | ring_O | 0.796 | 0.811 | 0.746 |
| warped | T | 0.754 | 0.773 | 0.693 |
| warped_walls | random_quadrant | 0.971 | 0.926 | 0.959 |
| warped_walls | ring_O | 0.875 | 0.828 | 0.683 |
| warped_walls | T | 0.693 | 0.721 | 0.746 |

## Suspect metric: swept-region accuracy (report beside slateN, not instead of it)

`experiments/METRICS.md` and this repo's own record (`accuracy` has repeatedly ranked models in
an order `slateN` does not reproduce) both say this number is the one to distrust when it
disagrees with slateN -- and it does disagree here on ranking, though not by much.

| arm | accuracy | ceiling (same corpus, same `canon_res`) |
|---|---|---|
| persistence (floor) | 0.000 | -- (accuracy's own denominator) |
| nfd_unwarped (control) | **0.402** | not applicable -- pays no warp round trip |
| nfd_warped (no walls) | 0.360 | **0.664** (canon_res=64, `results/warp_accuracy_ceiling.md`) |
| nfd_warped_walls | 0.368 | **0.664** (canon_res=64, same ceiling) |

Both warped arms score meaningfully below their own 0.664 ceiling (0.36-0.37 vs 0.664), so
their dynamics-prediction error, not the warp's resampling loss, is the dominant term here --
they are nowhere near saturating what the warp allows. The unwarped control (0.402) scores
*above* both warped arms on this metric, but the control never pays the resampling ceiling at
all, so **the three accuracy numbers are not on the same footing and should not be read as a
flat three-way ranking**: the control's number is directly comparable to EXP-0001's own
world-frame NFD range (0.407-0.509), while the warped arms' numbers are bounded above by 0.664
regardless of dynamics quality. Given that both warped arms sit well under their own ceiling,
this pilot does not show the ceiling itself as the binding constraint -- their prediction
quality does.

## Reference rows

- **persistence**: accuracy = 0.0000 by construction (the metric's own denominator, exactly as
  `results/warp_accuracy_ceiling.md` already documents). Its slateN row (goal-averaged:
  lyapunov +0.067, mass_in_region -0.018, signed_mass +0.010) is **DEGENERATE as a ranker** --
  persistence predicts zero change for every candidate in a pool, so `slate_n_capture`'s own
  argmax just picks whichever pool index `torch.argmax` breaks ties toward (effectively
  arbitrary, not "no information"). Its nonzero-looking numbers are an artifact of that
  tie-break, not evidence persistence carries real ranking signal -- **`random` is the ranking
  floor to compare against, not persistence**, per the task brief and `warp_accuracy_ceiling.md`.
- **random**: estimated by feeding i.i.d. noise images through the harness's own capture
  machinery (so the induced value-function score is unrelated to any candidate's identity,
  making the argmax choice effectively a uniform random pick), averaged over 20 seeds:
  lyapunov -0.008, mass_in_region -0.005, signed_mass -0.012 -- indistinguishable from 0, which
  matches the closed-form fact that `E[slate_n_capture(uniform random pick)] = 0` exactly (the
  expectation of a uniformly drawn pool value equals the pool mean, which is the metric's own
  centring point). All three real arms clear this floor by a wide margin (0.76-0.88 vs. ~-0.01
  to -0.01).

## Ranking-robustness check: does the warp round trip preserve ranking?

`warp_accuracy_ceiling.md` argues (without evidence, by its own admission) that the warp's
resampling loss is roughly uniform across candidates in a same-state pool and so largely
cancels in a ranking metric like slateN. Tested this directly: for 3 step-0 candidate pools
(128 candidates each, slates 33/31/41), pushed the TRUE `occ1` of every candidate through
`push_frame_roundtrip(identity)` at `canon_res=64` (matching the warped pilot checkpoints) and
compared the induced `lyapunov`-to-a-random-quadrant-goal ranking against the un-round-tripped
truth ranking. Code: `../code/ranking_robustness_check.py`, log:
`../runs/RUN-0004-pilot-eval/ranking_robustness.log`.

| slate | n_pool | Spearman rho | top-1 agreement |
|---|---|---|---|
| 33 | 128 | 0.9905 | yes |
| 31 | 128 | 0.9956 | no |
| 41 | 128 | 0.9891 | yes |
| **mean** | | **0.992** | **2/3 (67%)** |

**The correlation is very high (rho ~ 0.99)** -- this supports the ceiling doc's argument that
the round-trip resampling loss is close to uniform across a pool and mostly cancels under a
rank correlation, so `slateN` is largely trustworthy against this particular failure mode.
**However, top-1 agreement is only 2/3, not 3/3** -- on slate 31 the round trip changed which
single candidate is best (true best idx 25 vs. round-tripped best idx 28), even though the
overall ranking correlation stayed at 0.996. `slate_n_capture` is driven by exactly the
argmax/argmin choice, so a near-perfect Spearman correlation does not guarantee a near-perfect
slateN score on any individual slate -- close calls near the top of the ranking are exactly
where a small, roughly-uniform resampling shift can still flip the winner. With only 3 slates
sampled this is not a precise rate estimate; it is enough to say the "largely cancels" argument
holds in aggregate but is not airtight slate-by-slate, and a wider sample (more slates, more
value functions) would be needed to bound how often top-1 flips actually happen before leaning
on `slateN` as fully warp-immune.

## Bottom line for this pilot

- The wiring works: all three arms load, score, and clear the random floor by a wide margin.
- On `slateN` (the metric to trust) the three arms are close to each other -- no clear win or
  loss for the warp at this pilot's scale, and which arm "wins" depends on which value function
  is asked.
- On `accuracy` (the suspect metric) the warped arms sit well under their 0.664 ceiling, so
  their prediction quality, not the ceiling, is the binding constraint here; the unwarped
  control's higher number is not directly comparable since it pays no ceiling at all.
- The ranking-robustness check mostly supports treating `slateN` as insulated from the warp's
  resampling loss (rho ~0.99), with a real but small caveat (top-1 flips 1/3 of the time in
  this tiny sample) that argues for widening the check before treating it as settled.
- None of this is evidence either way about whether the warp helps in general -- that question
  needs the multi-length `overnight_randlen` comparison against EXP-0001.
