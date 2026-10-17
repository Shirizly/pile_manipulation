# EXP-0071 closed-loop results (paired over (goal, start) episodes)


## rank arm: 64 common episodes

| cell | mean | k8 | k16 |
|---|---|---|---|
| nfd_3ch_narrow_l20_v2 | 0.198 | 0.197 | 0.220 |
| nfd_3ch_narrow_l20_v2_seed1 | 0.145 | 0.141 | 0.154 |
| nfd_3ch_narrow_l20_v2_seed2 | 0.175 | 0.175 | 0.195 |
| linear_narrow_l20_v2_res64 | 0.153 | 0.148 | 0.171 |
| gated_ens_narrow_v2 | 0.203 | 0.195 | 0.229 |
| nfd3_meanblend_narrow_v2 | 0.194 | 0.194 | 0.212 |

| comparison | diff [95% paired-bootstrap CI] |
|---|---|
| gate - NFD3 mean (H1) | +0.0302 [+0.0195, +0.0409] |
| gate - best NFD seed (nfd_3ch_narrow_l20_v2; selected post hoc) | +0.0048 [-0.0079, +0.0169] |
| gate - linear res64 | +0.0501 [+0.0346, +0.0655] |
| gate - equal blend (H2) | +0.0091 [-0.0028, +0.0204] |
| equal blend - NFD3 mean | +0.0211 [+0.0118, +0.0305] |
- per goal letter_C: gate - NFD3 mean +0.030
- per goal letter_H: gate - NFD3 mean +0.025
- per goal letter_L: gate - NFD3 mean +0.058
- per goal letter_O: gate - NFD3 mean +0.018
- per goal letter_S: gate - NFD3 mean -0.001
- per goal letter_T: gate - NFD3 mean +0.015
- per goal letter_X: gate - NFD3 mean +0.034
- per goal letter_Z: gate - NFD3 mean +0.062
- NFD seed cell means [0.198, 0.145, 0.175]

## cem arm: 128 common episodes

| cell | mean | k8 | k16 |
|---|---|---|---|
| nfd_3ch_narrow_l20_v2 | 0.286 | 0.268 | 0.352 |
| nfd_3ch_narrow_l20_v2_seed1 | 0.233 | 0.215 | 0.286 |
| nfd_3ch_narrow_l20_v2_seed2 | 0.253 | 0.227 | 0.333 |
| linear_narrow_l20_v2_res64 | 0.205 | 0.189 | 0.252 |
| gated_ens_narrow_v2 | 0.208 | 0.209 | 0.223 |
| nfd3_meanblend_narrow_v2 | 0.325 | 0.298 | 0.428 |

| comparison | diff [95% paired-bootstrap CI] |
|---|---|
| gate - NFD3 mean (H1) | -0.0491 [-0.0643, -0.0345] |
| gate - best NFD seed (nfd_3ch_narrow_l20_v2; selected post hoc) | -0.0782 [-0.0968, -0.0591] |
| gate - linear res64 | +0.0030 [-0.0131, +0.0186] |
| gate - equal blend (H2) | -0.1172 [-0.1373, -0.0961] |
| equal blend - NFD3 mean | +0.0681 [+0.0532, +0.0829] |
- per goal letter_C: gate - NFD3 mean -0.082
- per goal letter_H: gate - NFD3 mean -0.060
- per goal letter_L: gate - NFD3 mean -0.035
- per goal letter_O: gate - NFD3 mean -0.127
- per goal letter_S: gate - NFD3 mean -0.049
- per goal letter_T: gate - NFD3 mean +0.010
- per goal letter_X: gate - NFD3 mean -0.015
- per goal letter_Z: gate - NFD3 mean -0.035
- NFD seed cell means [0.286, 0.233, 0.253]