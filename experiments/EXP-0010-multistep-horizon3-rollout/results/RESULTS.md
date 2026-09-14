# Results index — EXP-0010

The per-run `RESULTS.md` files below are copied verbatim from each source
`experiments/temp/` directory and remain each run's own authoritative
detail (full sweeps, full derivations, "Answers" sections). `EXPERIMENT.md`'s
"Numbers" section is the cross-run synthesis that answers this record's
claim; it does not duplicate these files' numbers, only the subset needed
for the claim.

- `RUN-0001-rollout-eval-RESULTS.md` — closed-loop 3-step rollout
  evaluation of existing models (no training); dataset/trajectory-identity
  verification; per-step accuracy; terminal slateN.
- `RUN-0002-linear-multistep-train-RESULTS.md` — linear operator (switched +
  global) gradient-trained through the closed-loop rollout objective,
  swept-region-masked MSE, lambda sweep. **Negative result** — held-out
  terminal slateN collapses.
- `RUN-0003-nfd-multistep-finetune-RESULTS.md` — NFD fine-tuned through the
  same objective, full-image unmasked MSE, lambda sweep. **Positive on
  accuracy, null on control** (terminal slateN was already at ceiling
  untrained on this corpus).
