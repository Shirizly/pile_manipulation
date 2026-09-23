# RUN-0002-within-bin

Follow-up written during EXP-0011/EXP-0012 promotion, per the promotion
task's instruction to test the REAL version of the cross-bin calibration
concern the source RESULTS.md left unresolved (and had stated incorrectly).

- Code: `experiments/EXP-0012-slates-binned-corpus-difficulty/code/
  binned-calibration__within_bin_test.py`
- Commit: 6ea03278 (dirty tree -- this promotion's own uncommitted work)
- Command: `OMP_NUM_THREADS=4 python -u experiments/temp/binned-calibration/within_bin_test.py`
- Data: `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`.
- Model: `weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt`
  (reused unmodified).
- Output: `results/results_within_bin.json`.
- Status: completed within budget (~10 min).
- Key design choice: grouped candidates by MODEL-0002's OWN operator-bin
  assignment (`Baselines.LinearForesight.model.bin_index` on `length_m`),
  not the corpus's own collection-time `bin_realized` field -- the two use
  different bin edges (MODEL-0002: 6 bins, 0-80mm; corpus: 5 bins, 20-70mm),
  and the concern under test is specifically about MODEL-0002's own operator
  switching, so grouping had to match that scheme. Verified the per-bin
  candidate counts (2675/5325/5347/5301/1352) exactly match the counts
  already cited in `docs/CODEMAP.md`, confirming correct bin assignment.
