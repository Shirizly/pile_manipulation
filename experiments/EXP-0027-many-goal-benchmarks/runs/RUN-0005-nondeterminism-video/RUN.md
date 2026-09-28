# RUN-0005 — visualise the non-repeatability (videos, before/after, finals)

- **Commit:** 3bae8cd7, dirty. Via run_probe: exp0027_nondet_video.json (argv). Run twice: the second run added finals.pt (all copies' final positions) and reproduced the first run's values exactly.
- code/visualise_nondeterminism.py; outputs in artifacts/RUN-0005-nondeterminism-video/. ~100 s GPU.
- The particle-based vs image-based decomposition was run inline (not as a script) on finals.pt; its numbers are in EXPERIMENT.md.
