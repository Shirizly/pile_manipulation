# RUN-0003 — stage 2 smoke test (slate 0), C6 check, Genesis determinism probes

- **Commit:** 3bae8cd7, dirty (see EXPERIMENT.md). Python: /home/alon/anaconda3/envs/pme/bin/python.
- Run in the foreground WITHOUT run_probe (reconstructed commands, not recorded argv):
  - code/stage2_genesis_bank.py --stage1 artifacts/RUN-0002-grad-stage1/stage1.pt --out <scratchpad>/s2_smoke.pt --slates 0  (3 attempts: CUDA OOM while item 1 held the GPU; gt_bank device bug; success)
  - code/check_c6.py <scratchpad>/s2_smoke.pt  -> 1.55e-02
  - code/probe_batch_dependence.py ; code/probe_reset_fix.py (run twice: the second version added the reset-vs-reset comparisons)
- The 175 rows the smoke test banked in DS-0004 (sim_path oracle_rollout_full) were deleted.
- Outputs printed to the terminal only; the numbers are transcribed in EXPERIMENT.md. Re-running the probe scripts regenerates them (~1.5 min GPU each).
