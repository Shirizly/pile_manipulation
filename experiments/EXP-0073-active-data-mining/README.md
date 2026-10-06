# EXP-0073 active data mining (retrospective + proposed live loop)

Status: **PAUSED by user 2026-10-06**. Exploration mode, **pilot-grade** (narrow n20 exact-20 mm task, DS-0015/DS-0016/DS-0017, zoom-64 NFD,
3 retrain seeds, one scoring ensemble, one selection draw). There is deliberately **no EXPERIMENT.md and no REGISTER entry** yet; a proper record is
written later (see experiments/TODO.md, MID, "EXP-0073 active data mining: next validation steps"). Split out of EXP-0072 on 2026-10-06.

What it is: can hard-example mining pick better new training rows than random ones for the zoom-window NFD? Done at zero sim cost as a *retrospective*
active-learning loop on DS-0015 (seed set 2000 rows, reservoir = other 8653 rows with known outcomes, +500 rows per arm), plus a proposed live
collection loop (perturbed states, mined actions, `--actions-file` for Genesis/chain_collection.py) that was NOT run.

Headline (results/active_collection_plan.md, "v2 -- READ THIS FIRST"):
* Noise share: ~42 % of ft100's raw swept-region error is a simulator noise floor (16 % if only <= 0.01 mm re-run nondeterminism counts); reducible error e_red is 69 % of raw.
* Proxies vs reducible error: the model's own p(1-p) (rho 0.87 with e_red) and entropy are best; ensemble variance / jitter shift (0.71-0.73) track noise as much as e_red; heuristics and novelty are weak.
* Retrospective arms barely move slateN (window 0.945-0.954, pasted 0.741-0.747, all within seed noise) but cut swept-region error ~9-10 % vs base (random +500: -4 %).
* Mined 500 rows ~ or better than random 1000 rows (2.5-3.5x data efficiency on swept-region error / window accuracy).
* Verdict: NO-GO for the ~6 h, 15k-sim 3-arm live study; GO only for a small (2000 sims/arm) live validation round if the user wants error/accuracy gains.

Layout: `code/active_*.py` (run with `PYTHONPATH=.`, from repo root; they put EXP-0072/code and EXP-0073/code on sys.path), `results/active_*.json` + `active_collection_plan.md`,
`runs/active_pilot/` (models, caches, retro/ evals; large, untracked), `figures/active_perturb_examples.png`.
Cross-dependencies on EXP-0072 (kept there, referenced by path): `code/score_zoom.py`, `model/zoom_nfd/` window libs, `runs/window_cache.pt`, `runs/ft100|ft300|scratch100|world128_300/`,
`artifacts/ceiling/resim_*mm_r*.pt` (re-sim replicates read by `active_reducible.py`).
