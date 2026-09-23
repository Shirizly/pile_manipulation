# TODO — prioritised task list

Written so an **orchestrator agent can work straight down it** without asking
the user anything. Each task states: what to do, why it matters, what would
count as done, and the known traps. Priorities are HIGH / MID / LOW.

**Standing rules for every task here** (they have each been violated at least
once in this project):
- Lead with `slateN`; `accuracy` is a diagnostic, not a verdict
  (`experiments/METRICS.md`). Use `random` as the ranking floor, never
  `persistence` — it predicts `dv=0` for every candidate and is degenerate.
- `accuracy` is a **ratio of population means**, not a mean of per-row ratios.
- **Reproduce a known number before trusting a new pipeline.**
- Match training arms by **gradient steps**, measured, not by epoch count. The
  loader batch is `batch_size // augmentation_factor`, so an "epoch" means
  different things under different augmentation settings.
- One GPU, 8 GB, shared machine: run training **sequentially**, never parallel.
- Persist every fitted object with its resolved config. Record runs through
  `experiments/COMMANDS.jsonl` and the `experiment-log` structure.
- Never commit unless the user asks.

---

## HIGH

### H1. Seed-level noise floor for `slateN`
**The single largest hole in everything measured so far.** Every arm in
EXP-0022/0025 is a single training run, so margins of 0.01-0.05 cannot be
ordered, and several verdicts rest on "the direction is consistent across three
corpora" rather than on any margin. Train **3 seeds** of the world-frame NFD
baseline config (`Baselines/NFD/configs/nfd_train_3ch_randlen.yaml`) and 3 of
one warped arm, score all six through `Baselines/common/eval_report.py`, and
report the per-cell standard deviation of `slateN` and `accuracy`.
**Done when:** a seed-spread number exists that later records can cite, and
EXP-0022's `noise_floor` field can be filled in with a measurement instead of a
disclaimer. **Cost:** ~6 training runs, the largest item on this list — but it
determines whether any of the close calls mean anything.

### H2. Decide the fate of the flow/advection line
`experiments/EXP-0025-flow-warp-nfd-pilot/` measured a **model-free ceiling**:
warping `occ0` by the GROUND-TRUTH particle displacement field scores
`accuracy` 0.167 and `slateN` 0.813/0.770/0.512 on the L20mm eval cell.
**Both are below what existing baselines already achieve on that same corpus**
(NFD: 0.407 accuracy, 0.853 lyapunov; LinearForesight switched res32: 0.262,
0.902). A parameterisation whose *perfect* version loses to a *trained*
baseline is a dead end as formulated.
Either **drop the line**, or **modify it to raise the ceiling and re-measure the
ceiling FIRST** (it is model-free and cheap, so never train before checking it).
Ceiling limitations identified, in likely order of impact: (a) the target field
is rigid translation only, no rotation; (b) bilinear `grid_sample` blurs a
near-binary occupancy field even at ground truth — worst in the near-static
regime where persistence error is already tiny; (c) stacked cubes collide in the
2D projection. A hybrid "warp **plus** a small additive residual correction"
breaks the pure-advection constraint and should lift the ceiling most.
**Done when:** either the line is closed out in the record, or a modified
parameterisation has a measured ceiling above the baselines above.

### H3. World-frame NFD + residual — finish and interpret
RUN-0022 (no augmentation, 240 epochs) and RUN-0023 (flip-only, 120 epochs) are
**already running**; epochs are matched to the baseline's 668,100 gradient
steps. Score both against the world-frame baseline and against the warped
residual arm (RUN-0019).
Expect a **smaller** gain than the warped arm's, because
`UNetModels_modular.UNet(residual=True)` already adds raw `occ0` into the
pre-sigmoid logit and every stock NFD config enables it — part of the effect is
already banked. `residual: false` is set in these configs to avoid
double-counting.
**Blocked on M1 for a clean read:** the baseline used FULL x8 augmentation, so
neither new arm has an augmentation-matched control yet.

---

## MID

### M1. Augmentation-matched world-frame NFD controls
Train a plain (non-residual) world-frame NFD on `overnight_randlen` at
`augmentation: false` (240 epochs) and `augmentation: flip` (120 epochs),
matching H3's arms. Without these, H3's residual arms can only be compared to a
full-x8-augmented baseline, which confounds the residual effect with an
augmentation effect. **This is what unblocks a clean read of H3.**

### M2. EXP-0023 statistical power
`experiments/EXP-0023-model-as-gradient-source/` answered its question about the
action pool but **failed its own pre-registered power check**: between-arm sd of
`gradient_gain` (0.0105) is smaller than between-state sd (0.0163), so the arm
ranking cannot be read. Re-run with **more states** (30-50 instead of 10), and/or
more goals and value functions, until between-arm variation exceeds
between-state variation. The harness exists and adding a model is one registry
entry in `simple_mpc/adapters.py::OCC_ADAPTERS`.
**Done when:** the power check passes, or the design is declared underpowered
for this quantity and the record says so.

### M3. Refit and diff `operators_res64.pt`
`Baselines/LinearForesight/runs/operators_res64.pt` was **missing from disk**
(never git-tracked) and was restored from an untracked stray file in
`experiments/temp/` to unblock scoring. Its provenance is unverified, and the
`linear_*_res64` rows in EXP-0022's frame-scoring table rest on it. Re-run
`Baselines/LinearForesight/fit_switched.py` at res=64 on the same corpus and
diff against the restored file. See `experiments/OPEN_ISSUES.md`.

### M4. Make the `dv` sign convention checkable for VALUE functions
`simple_mpc.adapters.assert_dv_convention` asserts "improving push gives
`dv < 0`", which is correct for `lyapunov` and **FALSE** for `mass_in_region` /
`signed_mass_in_region`, where an improving push RAISES the value. Its
docstring now says so, but the guard itself still only covers the COST case,
and the three difference metrics (`gradient_gain`, `pool_escape`,
`regret_vs_oracle`) have their subtraction order hard-coded for COST. Nothing
measured so far is wrong -- every raw `dv` computed in this repo to date uses
`lyapunov` alone -- but the moment anyone computes `dv` under a mass value
function, the metrics invert silently.
Give the assertion and the difference metrics a `higher_is_better` flag, the
way `Baselines/common/goals.py::slate_n_capture` already has one, so the wrong
case fails loudly instead of returning a plausible number with the wrong sign.
See the SIGN section of `experiments/METRICS.md`.

### M5. Residual head on LinearForesight / other model families
The residual parameterisation is the only effect that has transferred across
scales in this project (L20mm pilot -> randlen, warped). LinearForesight's ridge
is already regularised **toward identity**, which is the linear analogue, so it
is likely a no-op there — but that has not been checked, and the GNN/latent
families have not been checked either. Cheap reasoning task before any training.

---

## LOW

### L1. The unexplained push-length dependence
The warped arm's `accuracy` deficit versus the world-frame baseline grows
sharply at short pushes (8-11x across length bins) and is **unexplained after
four candidate mechanisms** were tested: metric artifact (accounts for 12-30%),
canonical action-channel overlap (refuted — channels are fully separated above
~15mm yet the deficit still shrinks 3.3x above that), input-side resampling
(real but magnitude wrong by 1.6-5.7x and confounded by its own control), and
the x8-augmentation handicap (refuted — the profile got *steeper*, not flatter,
under flip-only). See `experiments/EXP-0022-*/results/flipaug_length_profile.md`.
Only worth resuming if the push-frame line is revived.

### L2. `canon_res=96` warped arm
Supersampling the canonical frame removes only ~28% of the round-trip
degradation, and the warped arms score well below their ceiling anyway, so
resampling is not what binds. Deprioritised on measurement, not on principle.

### L3. Wall-channel arm
Gated OUT by `experiments/EXP-0022-*/results/wall_proximity.md`: on
`randlen_test` the deficit is non-monotonic in wall distance and goes flat once
push length is held fixed. It does replicate on L20mm/L40mm, but those are
single-push-length corpora. Revisit only if a length-independent wall-linked
residual appears on randlen.

### L4. Coarse-to-fine flow
Only if H2 keeps the flow line alive. The `flow_coarse16` cell pooled the
**flow field** but left the **image** at 64x64, so it never addressed the
capture-radius problem at all — gradient descent on a photometric loss can only
align features within about one feature width (~2-3 px for a 5 mm cube), while
measured displacements reach 20 px. The standard fix downsamples the IMAGE, so
a large displacement becomes a small one, solves there, and refines upward.
