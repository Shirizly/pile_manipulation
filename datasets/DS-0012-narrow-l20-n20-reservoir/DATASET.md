---
id: DS-0012
title: Narrow-domain CANDIDATE-POOL RESERVOIR ("DS-C") -- n20 single layer, chain-shaped, exact 20 mm perpendicular, training physics
status: defective  # 2026-09-28, ISS-010 illegal touchdowns; no clean replacement yet
date: 2026-09-28
final_count: "84 chunks x 1,024 rows/chunk = 86,016 chain transitions (all `valid`); stopped
  deliberately at 05:31 (2026-09-28) to free the GPU for the coder's NFD-with-reference
  training, not because it reached its `--n-chunks 500` cap or ran out of the ~5h wall-clock
  budget -- manifest.json's `complete` field is `false` (a deliberate stop, not a finish) and
  `chunks` has 84 entries (indices 0..83, none skipped)."
path: Genesis/data/narrow_l20_n20/reservoir_dsC (_{k}_data.pt, k = 0..)
producer: Genesis/chain_collection.py --mode chains --seed 301 (new seed, disjoint from DS-0008/9/10/11's 1/101/102/201)
---
- Chain-shaped reservoir, same recipe as DS-0008 (pile-aware sampler, min_swath 3, push_length
  0.02, exact-length + perpendicular validity checks, TRAINING_PHYSICS 0.7/0.5/450 settle 3000,
  mixed scatter/clump starts via `Genesis.clump_states:clump_starts`), but at `--n-envs 128`
  (vs DS-0008's 32) per the designer's "use 128 envs if stable" note, to raise chunk throughput
  toward the 150-250k-transition target. `--steps 8` (8 chained pushes/env/chunk, same as
  DS-0008) x `--n-chunks 500` (generous upper bound; actual chunk count reached depends on the
  ~5h wall-clock budget the designer set, not on hitting 500) -- **capped by time, not row
  count**: 128 envs x 8 steps = 1,024 rows/chunk, so 500 chunks would be 512,000 rows if it ran
  to completion, well past the 150-250k target; the job is expected to be stopped (or left to
  self-checkpoint) well before chunk 500.
- Command: `Genesis/chain_collection.py --mode chains --out
  Genesis/data/narrow_l20_n20/reservoir_dsC --n-chunks 500 --steps 8 --n-envs 128 --starts
  mixed --clump-fn Genesis.clump_states:clump_starts --sampler '{"pile_aware": true,
  "min_swath_particles": 3, "push_length": 0.02}' --seed 301`.
- **Simplification vs the designer's full DS-C spec**, per the coordinator's instruction that
  folded this in: this launch is CHAIN-SHAPED ONLY (`same recipe as DS-0008`). The design doc's
  additional "pool-shaped share: same-state branches of 16-64 pushes" and "near-wall states"
  mix are NOT included in this launch -- noted as a gap against the full DS-C spec, not a
  silent omission (docs/experimental_design/retrieval_based_modeling.md section 2.1).
- Checkpointing: `chain_collection.py` writes `_{k}_data.pt` + rewrites `manifest.json` after
  EVERY chunk (atomic, `os.replace`), and skips any chunk index already present in
  `manifest.json["chunks"]` on restart -- so this reservoir survives being killed and restarted,
  or simply left running past this session, without an external checkpoint wrapper.
- **Final status (2026-09-28, fresh-instance update): STOPPED, not complete.** 84 of the
  planned (up to) 500 chunks finished (86,016 rows, all `valid` per `bank.py::_load_dir`'s
  filter -- every DS-0012 row loaded as `valid=True`, i.e. this reservoir's own exact-20mm/
  perpendicular acceptance check already filtered at collection time, same as DS-0008). Stopped
  deliberately at 05:31 to free the GPU for the coder's concurrent NFD-with-reference training,
  per the coordinator's overnight schedule -- an intentional early stop, not a crash or a budget
  overrun (`manifest.json`'s `complete: false` reflects the stop, not a failure). This record's
  data-scaling task (EXP-0059) uses these 86,016 rows as the pool to subsample/add to the
  existing DS-0008+DS-0010 bank for 25k/50k/"all ~98k" bank-size comparisons -- see EXP-0059's
  `results/data_scaling.json`.

**KNOWN DEFECT (2026-09-28, ISS-010):** same `_pile_aware_stops` clamp defect as DS-0008.
Measured: 45.8% of rows illegal at touchdown (0mm SAT overlap; 70.2% at 1mm margin); worse for
clump (0.535) than scatter (0.380); falls across chain steps (0.513 -> 0.409, step 0 -> 7). Per-row
flags: `Genesis/data/narrow_l20_n20/reservoir_dsC/_{k}_data_legality.pt`. Full writeup:
`experiments/OPEN_ISSUES.md` ISS-010.
