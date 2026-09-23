# RUN-0001-calibration

Reproduction + characterisation diagnostic: is slates_binned's near-zero
switched-linear slateN real (corpus difficulty) or a scoring-path defect?

- Code: `experiments/EXP-0012-slates-binned-corpus-difficulty/code/
  binned-calibration__calibrate.py` (originally
  `experiments/temp/binned-calibration/calibrate.py`)
- Commit: 6ea03278 (dirty tree)
- Command (reconstructed): `python -u experiments/temp/binned-calibration/calibrate.py`
- Data: `Genesis/data/slates_multistep/n20_{L10mm,L20mm,L40mm}`,
  `Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm`.
- Model: `weights/MODEL-0002-descriptor-only-D-all-local/checkpoint.pt`
  (reused unmodified).
- Output: `results/results_calibration.json`.
- Status: completed within its declared 40min/~90k token budget.
