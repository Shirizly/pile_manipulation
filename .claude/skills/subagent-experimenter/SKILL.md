---
name: subagent-experimenter
description: "How to run one contained experiment as a subagent in this repo, and how to brief one. Covers the standing rules that agents in this project repeatedly break: never park on a background watcher, always python -u, persist every fitted object, check the code map before grepping, and report slateN before accuracy. Use when you are a subagent given an experiment to run, or when you are writing the prompt for one."
argument-hint: "Optionally: the experiment you are about to run or delegate"
user-invocable: true
---

# Running a contained experiment

Companion to `experiment-log` (storage, tiers, budgets) and `register-validator`
(frontmatter). This file is about **execution** — the things that went wrong in
practice, measured across 26 subagent runs.

## The five rules, in the order they were broken

### 1. Never end your turn waiting on a background watcher
Broken **5 times**, by 5 different agents, in every case after the rule was stated
in the prompt. One agent even created a correct bounded waiter and then ended its
turn anyway.

Either read the output that already exists and write it up, or wait in the
**foreground** with a bounded timeout:

```bash
timeout 900 bash -c 'until ! ps -p <PID> >/dev/null 2>&1; do sleep 30; done'
```

Loop that until the job exits. A partial result written up is worth a great deal;
a complete one left uncollected is worth nothing. If your budget runs out, report
the cells that finished and name the ones that did not.

### 2. Always `python -u`
Python fully buffers stdout through a pipe, so a running job shows an empty log and
looks hung. Two agents lost budget deciding whether to kill a healthy job. Use
`python -u`, or `scripts/run_probe.py` (note its ledger path is `broken` — see the
code map).

### 3. Persist every fitted object, with its config
A run that saved only metrics forced a later agent to refit the same model, and a
refit is only *probably* the same object. Save the operator/checkpoint next to its
metrics with a one-line provenance note. `experiments/temp/` is gitignored, so this
costs nothing but disk. Deleting temp is the user's call, not yours.

### 4. Check `docs/CODEMAP.md` before you grep
**74 % of all tool calls in the audit were Bash, and most were navigation**: 216
`grep`, 142 `sed -n`, 125 `cat`, 94 `ls`, 78 `find`, against 180 `python`. Ten agents
each independently rediscovered `transforms/functional.py`. The code map exists to
end that. Add to it when you find something it lacks.

### 5. Lead with `slateN`, not `accuracy`
`accuracy` compares variants **within one model type** and is suspect everywhere
else. A comparison carrying only `accuracy` is not finished — repeat it under
`slateN` before concluding. See `experiment-log`'s Metrics section.

**Checkpointing is mandatory** for anything longer than a few minutes: rewrite
results/manifests atomically after every unit of work so a cut-off run loses
at most one unit. The full rule is in `project-overview`, "Every job must
survive being cut off".

## Measurement traps this repo has already hit

- **Assert the device.** `eval_baseline.py::_predictor_batch` leaves tensors on CPU, so
  some models time on CPU while others force CUDA. Introspect the device of the
  tensors actually fed to each forward and report it per row.
- **Python loops that force GPU syncs.** `if bool(mask.any())` inside a per-bin loop
  synchronises every iteration; a per-sample preprocessing loop does the same. Both made
  an architecture look slow for implementation reasons. Before reporting a model as slow,
  check whether its cost is a loop.
- **Characterise your noise floor before ranking.** Re-run the same configuration in
  separate processes. Measured here: ~5 % relative, and >30 % cross-process drift for
  fast (1–3 ms) models — which made an entire family unrankable by timing.
- **Pooling across unequal scales.** A pooled `1 − rms/rms` over dimensions whose errors
  differ by orders of magnitude is dominated by the well-predicted ones. Normalise per
  dimension, and drop degenerate blocks.
- **Descriptor-space accuracy compares models only within a shared descriptor set.** Two
  models predicting *different* descriptor vectors are being scored on different quantities,
  so a pooled `1 − rms/rms` over each one's own targets rewards whichever set is easier to
  predict, not whichever model is better. Same set, same normalisation, same held-out rows →
  valid, and useful where image-space accuracy is undefined for both. Different sets → void;
  let the control metric carry the comparison.
- **Vary the input, hold the target fixed.** Scoring each feature set against *its own*
  target rewards deleting hard-to-predict dimensions. This reversed a ranking once.
- **Bounded metrics near ceiling.** A model already at 0.967 with 15/0/0 wins cannot show
  improvement. Check headroom before concluding "no effect" — switching to a goal with
  headroom turned one null into a 2.5σ result.
- **Check goal degeneracy.** Report `frac(dv_true == 0)`; a goal where nothing
  discriminates produces universal ties that read as "no difference."
- **Mask the loss where the metric reads.** A swept-region-masked training objective left
  a model unconstrained exactly where the value function integrates, and training made
  control *worse* while the loss fell.
- **Characterise a corpus from the whole corpus, not one file.** Several corpora here are
  *banded*: each file covers its own slice of the parameter range. Reading `glob(...)[0]` and
  reporting its min/max describes that band, not the dataset — this produced a confident,
  wrong "the data only spans 20-30 mm" when the corpus actually spanned 0-70 mm in five
  near-equal bands. Aggregate over every file, and report a histogram, not a range.
- **Check the data supports the model.** Single-push-length corpora gave 4 of 6 length
  bins zero gradient; those operators stayed byte-identical to their initialisation.

## Name the population you actually tested

Before writing "every model", "the corpus", "all X" — **list the members you tested**, and
scope the claim to them. This has gone wrong twice here, both times recorded before anyone
checked:

- "Every model scores near zero on `slates_binned`" became an invariant tagged `holds`. Every
  model tested was a **descriptor model with a point-mass readout**; no image-space model had
  been tried. The first one that was scored 0.39-0.87 across all nine cells. The corpus is hard
  for that *readout*, not in general.
- "The pool mixes push-length bins, so the operator must rank across them" — the operator bins
  per candidate, one grep away.

A claim's scope is part of the claim. `experiment-log` says an unscoped claim that later needs
narrowing "reads as refuted for no reason but its own over-reach" — so fix the scope when you
write it, not after. If you tested one member of a class, name that member in the claim.

## Verify a mechanism in the code before you assert it

The traps above are about measurement. This one is about **explanation**, and it has bitten
twice: a run ruled out its rival hypotheses rigorously, then appended a causal story that
contradicted code it already had open ("the pool mixes push-length bins, so the per-bin
operator must rank across bins simultaneously" — the operator assigns a bin *per candidate*,
one grep away). The coordinator repeated it without checking, and it was one `grep` from
being caught.

So: **an explanation of *why* a number came out that way is a claim about the code or the
data, and needs the same evidence as the number.** Before writing "because X":

- grep the function and read it, or
- run the cheap discriminating test (here: score within a single bin and compare), or
- label it explicitly as an untested hypothesis and say what would test it.

A result with an honest "cause not isolated" beats a result with a confident wrong cause,
because the wrong cause is what the next person builds on.

## Briefing a subagent (if you are the one delegating)

Include, every time:
1. **Budget in wall-clock and tokens**, declared as HARD, with "report partial rather
   than overrun silently."
2. **The five rules above**, or a pointer to this skill.
3. **What to reuse, by path** — the existing harness, the saved operators, the split.
   Comparability usually depends on reusing a split, not re-deriving one.
4. **Every cell to report, not just the winner**, and the mandatory baselines
   (`persistence` for prediction, `random` for ranking).
5. **The one result that would embarrass the design**, and an explicit instruction to
   report it plainly if found. Null and negative results have been the most valuable
   outputs in this project.
6. **A word cap on the final report.** Long reports cost tokens and bury the number.

## Reporting back

Lead with the number and its uncertainty. State the cells you could not run. Name any
difference that sits inside the noise floor as unresolved rather than ranking it. Do not
oversell a small improvement — "indistinguishable at this power" is a result, and so is
"this objective is refuted, and here is the diagnosed cause."
