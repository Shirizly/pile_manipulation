# RUN-0002 — collect DS-0022 (test slates), source repo

- where: `~/Code/dyn-res-pile-manip` @ 5c9eca9, dirty (as RUN-0001); 2026-10-02/03.
- command (RECONSTRUCTED): `python collect_true_action_results_parallel.py --config config/data_gen/slates_objbiased_carrots_grouped.yaml` (n_envs 2). Copies: `code/collect_true_action_results_parallel.py`, `code/configs/slates_objbiased_carrots_grouped.yaml` (= `data/run_config.json`).
- output: DS-0022. Log: `artifacts/RUN-0002-collect-test/slates_objbiased_carrots_grouped_run.log`.
- status: complete, 200/200 states, 17,761/20,000 actions valid.
