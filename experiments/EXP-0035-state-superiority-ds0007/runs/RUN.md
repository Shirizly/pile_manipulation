# RUN-n20 / RUN-n50

Via run_probe (exp0035_n20.json / exp0035_n50.json): experiments/EXP-0030-state-superiority-ds0005/code/analyse.py --corpus datasets/DS-0007-sean-same-state-pools/data/scattered_n{20,50}.pt --art artifacts/RUN-n* --res results/analysis_n*.json. First pass had NaN-propagation (pools whose true dv are all equal); patched (nan-aware means, undefined cells dropped and counted: 2 / 160,200 for n20, 0 for n50) and rerun; predictions were cached, only the analysis changed. Commit 3bae8cd7, dirty.
