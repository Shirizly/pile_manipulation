# RUN-0015 — image-metric scoring (accuracy + slateN, image-mask truth) on DS-0022 per count group, and accuracy-vs-slateN correlations

- what: `code/score_image_metrics.py` (`score` per model -> `artifacts/RUN-0015-image-metric-scoring/<model>.npz`,
  atomic, skip-if-present; `analyze` -> `results/nfd_lf_image_metrics.{json,md}` + `artifacts/.../_per_state.npz`).
  Pipeline = EXP-0061/0062's: `load_flex_cell(DS-0022, "all", occ_source="image_mask")` (loader drops escaped /
  out-of-grid / null rows: 16,417 of 17,761 kept), occ0 and truth = binary colour-image masks
  (= `eval_report --truth-scoring image`), predictors through `eval_report._predict` with `--ckpt`-style overrides,
  swept-region `accuracy` (EXP-0061 `final_eval.row_rms`, ratio of sums), slateN values from
  `eval_report._goals_for_slate` (random_quadrant seeded by state, ring_O, T) x lyapunov / mass_in_region / signed_mass.
- capture rule: the closed form of `goals.slate_n_capture`, except ties in the model's value are averaged over the
  tied set (expected value of a random tie-break) so a constant predictor scores exactly 0; K-equalised read: K = 50
  (smallest clean pool), 50 random subsets per state (rng 0). CIs: state bootstrap (2000; 1000 for rank correlations).
- models: lf13_switched / lf13_single (MODEL-0013), nfd14 (MODEL-0014, after RUN-0014), gnn12 (MODEL-0012 node
  displacements, FPS rep 0 seed = state_idx, carried onto the input mask: each occupied cell 4x4-supersampled, nearest
  node's table-frame displacement, binned like flex_predictor.render), gnn_truecap (same carry with the TRUE motion of the
  same nodes = renderer cap), field (EXP-0064's untrained action field through the same carry), out-of-domain refs
  nfd08_ds0020 (MODEL-0008) and lf09_switched_ds0020 (MODEL-0009) (trained on DS-0020 v2, same scene/grid), baselines
  persistence (pred = occ0) and random (uniform noise image, seed 0). Self-check: zero-displacement carry == occ0 (passed).
- commands (ledger `runs/COMMANDS.jsonl`, logs in `artifacts/RUN-0015-image-metric-scoring/`):
  `python scripts/run_probe.py --tag exp0064_run0015_score_cpu --threads 8 -- python experiments/EXP-0064-obj-count-effect-study/code/score_image_metrics.py score --self-check --models persistence,random,field,lf13_switched,lf13_single,gnn12,lf09_switched_ds0020,nfd08_ds0020`;
  `... --tag exp0064_run0015_score_truecap -- ... score --models gnn_truecap`; `... --tag exp0064_run0015_analyze_nonfd -- ... analyze`;
  nfd14 scored + re-analysed by `code/nfd_after_queue.sh` (tags exp0064_run0015_score_nfd14 / exp0064_run0015_analyze).
- all models except nfd14 scored on CPU (8-24 s each), 2026-10-04 00:15-00:30 CEST; nfd14 on cuda (3 s) 06:20 and the final analyze 06:21 (2000 boot reps); commit 3373e65b dirty.
