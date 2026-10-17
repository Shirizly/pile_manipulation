---
id: MODEL-0015
status: active
---
# MODEL-0015 gated ensemble, narrow v2

Per-pixel softmax gate (3 conv layers, hid 24) over 5 occupancy members: `nfd_3ch_narrow_l20_v2`, `_seed1`, `_seed2`, `_sharp_w03`, `linear_narrow_l20_v2_res64`.
Input channels: occ0, 5 member predictions, swept-region mask. Fit on DS-0017 val_pools_v2 (flagged rows dropped), 60 epochs Adam 2e-3, seed 0, by
`experiments/EXP-0071-gated-ensemble/code/fit_gate.py` (needs the member-prediction cache from `experiments/temp/ensemble-proto/collect.py`).
Payload `checkpoint.pt` (members, hid, state_dict). Adapter: `simple_mpc.adapters.GatedEnsembleAdapter`, registry `gated_ens_narrow_v2`.
Pilot-grade (narrow-trained members). Caution: worst ensemble under CEM in EXP-0071.
