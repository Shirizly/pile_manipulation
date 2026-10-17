# RUN-0001 -- EXP-0069 letter_T diagnosis (zero simulation, CPU only)

- **Started / ended:** 2026-10-05 ~09:00-10:15 local; three foreground CPU jobs, each < 2 min.
- **Code:** commit a5ccb680, **dirty** (other agents' uncommitted edits to `simple_mpc/`, `Baselines/LinearForesight/`,
  EXP-0039/0043 code, docs; none of them in the read-only scoring path used here: `simple_mpc.adapters.occ_for_scoring`,
  `Baselines/common/goals.letter_mask` / `mass_in_region`).
- **Interpreter:** `/home/alon/anaconda3/envs/pme/bin/python`, `CUDA_VISIBLE_DEVICES=` (empty), `OMP_NUM_THREADS=4`.
- **Inputs (read only):** EXP-0066 `results/{nfd_s0,nfd_s1,nfd_s2,lin_orig,lin_b1,lin_b2,broad_s0,broad_s1}_ps0.json`
  (recorded `states` per push), EXP-0039 `results/run0002/hm_{cem,rank}_*.json` (16 files), EXP-0046 `results/vstar.json`.
- **Commands** (`COMMAND.txt`; ledger `experiments/COMMANDS.jsonl` run_ids `EXP-0069/RUN-0001/diagnose`, `.../mechanism`):
  1. `code/diagnose_T.py` -> `results/diagnosis.{json,md}`, `results/figures/f1..f4` (`stdout.log`).
  2. `code/mechanism_8letters.py` -> `results/mechanism.{json,md}`, `results/figures/f5` (`stdout_mechanism.log`).
  3. inline heredoc (not a file; reconstructed in `code/push_direction_effect_inline.py` from the transcript):
     per-push gain to mask / crossbar / stem by direction class -> `results/push_direction_effect.json`.
- **Status:** all three exit 0; no cell missing.
- **Deviation from the brief:** the brief's figure list asked for 2-4 figures; 5 were written (f5 was added for the
  8-letter blob check).
