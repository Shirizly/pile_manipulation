# RUN-0001 — collect DS-0021 (train transitions), source repo

- where: external FleX project `~/Code/dyn-res-pile-manip` @ 5c9eca9, **dirty** (collector + `env/flex_env_multi.py` count_target branch + PyFleX voxelize fix uncommitted there); 2026-10-02/03 (from log timestamps).
- command (RECONSTRUCTED from the source report, not recorded): 2 shards `bash run_transitions_shard.sh <shard>` → `python collect_transitions.py --config config/data_gen/transitions_carrots_grouped.yaml --shard <k>`, then `--merge`. Copies: `code/collect_transitions.py`, `code/run_transitions_shard.sh`, `code/configs/transitions_carrots_grouped.yaml`.
- output: DS-0021 (`datasets/DS-0021-flex-carrots-countgroups-train/data/`). Logs: `artifacts/RUN-0001-collect-train/transitions_carrots_grouped_shard{0,1}.log`.
- status: complete, 2000/2000 states, 20000 transitions (all flagged valid; see DS-0021 audit for escaped rows).
