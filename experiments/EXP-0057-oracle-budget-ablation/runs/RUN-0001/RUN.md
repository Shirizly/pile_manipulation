# RUN-0001 — throughput test, smoke test, and the launched main/extra queues

- Commit 3bae8cd7, dirty tree (pre-existing overnight-branch changes; see git status).
- Throughput test: `code/throughput_test.py --n-envs {32,64,128,256} --n-batches 3`, each in
  its own process (Genesis allows `gs.init()` once per process). Results:
  `results/throughput.json`.
- Smoke test (not filed as its own run; outputs deleted after verification): 2 goals x 1 start
  x 2 pushes, cem-pop=4/iters=1, two cells (lyap, crowd_floor); checkpoint/resume verified by
  killing mid-push and restarting; `run_queue.py`'s `run_queue()` verified end-to-end through
  `scripts/run_probe.py`; `analyse.py` verified on the smoke outputs.
- Main queue launched detached:
  `nohup /home/alon/anaconda3/envs/pme/bin/python -u code/run_queue.py main
  > runs/queue_main.log 2>&1 &` (see runs/queue_main.pid for the PID).
  Jobs (in order): `oracle_A_default` (64x4), `oracle_A_128x2`, `oracle_A_32x8`. Estimated
  ~3.8h each, ~11.4h total (see DESIGN.md Step 2). On completion the same process
  automatically continues into the extra queue (`run_queue(EXTRA, ...)` called in-process by
  `main()`), so a single nohup covers both main and extra -- no separate launch needed for the
  48h tail.
- Extra queue jobs (run automatically after main, same process): `oracle_A_256x1`,
  `oracle_A_16x16` (completing the (A) sweep), `oracle_B_mass1`, `oracle_B_mass3`,
  `oracle_B_crowd`, the 5 `_starts4445` extensions, `oracle_len_20_40`, `oracle_len_fixed20`.
- Each cell's own invocation is separately logged via `scripts/run_probe.py` to
  `runs/<tag>.log` / `<tag>.json` / `<tag>.pid`, and appended to `runs/COMMANDS.jsonl` (root)
  and copied into `experiments/COMMANDS.jsonl`.

## Relaunch 2026-09-25 22:35
First launch (22:16, unseeded, no early stop) stopped before its first push completed; its
partial files are in runs/aborted_unseeded_2216/. `code/run_queue.py` now passes
`--seed-base 0 --stop-solved` to every cell; relaunched with the same command
(`nohup python -u code/run_queue.py main > runs/queue_main.log 2>&1 &`, PID in runs/queue_main.pid).

## Paused 2026-09-26 12:54 (user)
Main queue complete (oracle_A_default, oracle_A_128x2, oracle_A_32x8). Extra queue stopped during
its first cell, oracle_A_256x1, checkpointed at push 5/20 (its run_probe ledger start has no end
event: interrupted by the pause). Resume: `cd experiments/EXP-0057-oracle-budget-ablation && nohup
/home/alon/anaconda3/envs/pme/bin/python -u code/run_queue.py main > runs/queue_main.log 2>&1 &`
(complete cells are skipped; 256x1 resumes at push 6, then the rest of the extra queue).
