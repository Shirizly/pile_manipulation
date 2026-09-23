# RUN-0017 -- four-arm residual eval (R1/R2 vs RUN-0001/RUN-0002)

Harness: `Baselines/common/eval_report.py` (same code as the pilot_eval.md report),
corpus `L20mm` (7680 transitions, 60 step-0 slates of 128 candidates each). Models:
`nfd_unwarped_L20mm_pilot` (RUN-0001), `nfd_warped_L20mm_pilot` (RUN-0002),
`nfd_residual_unwarped_L20mm_pilot` (R1/RUN-0015), `nfd_residual_warped_L20mm_pilot`
(R2/RUN-0016) -- the last two registered as additive `MODELS` dict entries in
`eval_report.py` in this task, nothing existing removed/mutated. Raw JSON:
`../artifacts/RUN-0017-residual-eval/residual_eval.json`. Command: `COMMAND.txt`.

All four models ran on **CPU** (`Baselines/common/eval_baseline.py`'s `_predictor_batch`
never moves the batch off CPU -- documented trap; GPU had RUN-0010 running throughout
this eval and was otherwise idle for it).

**Reproduction check (methodology rule #2, per the task brief): RUN-0001's and
RUN-0002's numbers reproduce their `pilot_eval.md` published values exactly**
(`nfd_unwarped_L20mm_pilot` accuracy 0.4018 vs published 0.402; slateN averaged
0.881/0.823/0.765 vs published identical; `nfd_warped_L20mm_pilot` accuracy 0.3596 vs
published 0.360; slateN 0.834/0.837/0.804 vs published identical) -- confirms this eval
run is on the same footing as the existing record before trusting the new numbers.

See `../results/residual_pilot.md` for the full four-arm table, the random-floor row, and
interpretation.
