# RUN-0002 — stage 2, ground truth and the oracle (Genesis)

## Exact argv
```
OMP_NUM_THREADS=4 PYTHONPATH=. python -u \
  experiments/EXP-0023-model-as-gradient-source/code/stage2_genesis.py \
  --stage1 .../artifacts/stage1_actions.pt \
  --out    .../artifacts/stage2_genesis.pt
```
Defaults: `--n-envs 32 --cem-iters 4 --goal corner --seed 0`.
Config: `simple_mpc/config/config_oracle.yaml` with `record_transitions`
forced OFF (this run must not write into `Genesis/data/mpc_runs`).

## What happens per state
1. The slate's own particle state is pushed in as a frozen snapshot
   (`GenesisOracleEnv.restore_snapshot`). **`restore_err` — the max abs
   difference between what was asked for and what the env then holds — is
   exactly 0.0 on every state.**
2. **CEM oracle**, 4 iterations x 32 envs. Iteration 0's population is a
   random 32-subset of THE SAME 100-action seed pool stage 1 ranked, so the
   oracle starts where the arms start; iterations 1–3 sample from the
   refitted Gaussian. Every candidate goes through the **identical legality
   projection** stage 1 uses, so the oracle searches the same legal set.
   CEM and only CEM, per `DESIGN.md` — one optimiser held fixed so arms
   differ only in the model. Planning rollouts use the reduced-fidelity
   settle budget; the winner is re-executed at full fidelity in step 3.
3. One **full-fidelity** batch of exactly 32 executions: 6 arms x
   {`a_rank`, `a_grad`} = 12, the CEM winner, and **19 seed-pool rows whose
   true `dv` is already cached** by `scripts/probes/binned_pool_cache.py` —
   the known-number reproduction check for this entire pipeline (190 rows
   over 10 states).

`dv = lyapunov(occ_after) - lyapunov(occ_before)` with occupancies built by
`simple_mpc.adapters.occ_from_particles`, i.e. bit-for-bit the convention
`binned_pool_cache.py` used (±0.064 m, 64 px, footprint radius 1.25 vox).

## Why no resimulation-noise term
Invariant `genesis-snapshot-restore-repeat-determinism` (EXP-0024): this
exact seam returns bit-identical terminal positions on repeat. Every `dv`
here is a deterministic function of (state, action), not a noisy draw.

## Timing
~78 s per state (4 CEM batches at reduced fidelity + 1 full-fidelity batch,
each 32 parallel envs), ~13 min total, plus ~100 s of Genesis scene build and
kernel compilation. Run under 3-way GPU contention — **inflated, not a
benchmark.**
