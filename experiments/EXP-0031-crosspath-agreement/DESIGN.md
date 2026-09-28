# EXP-0031 — do the corpus collector and rollout_candidates agree, once physics match?

Written 2026-09-24 before running. EXP-0023 saw r = 0.959 between DS-0001 corpus dv
and rollout_candidates re-execution; since then: (a) truth scoring was image-based
(noisy, fixed), (b) DS-0001 (0.3 / 1000) and the oracle env (0.25 / 750) used
different physics. Here both sides use the training physics (DS-0006 + the env
configured with TRAINING_PHYSICS) and soft scoring.
Prediction: with matched physics, soft-scored dv agrees with sd(diff)/sd(between)
<= 0.1 (a physics-free code-path difference is small). Refuted if > 0.3 (then the
paths genuinely differ -- e.g. settle budget, plate control -- and the gradient
benchmark must not mix corpus pools with rollout-evaluated actions).
