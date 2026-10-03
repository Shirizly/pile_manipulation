## slateN averaged over 3 goals (family mean; seed range; slate-bootstrap 95% CI of the family mean)

| model | lyapunov | mass_in_region | signed_mass |
|---|---|---|---|
| NFD (3 seeds) | 0.952 [0.948-0.955] CI 0.944..0.960 | 0.876 [0.874-0.878] CI 0.862..0.891 | 0.901 [0.896-0.904] CI 0.885..0.916 |
| GNN (3 sampling seeds) | 0.762 [0.759-0.764] CI 0.715..0.800 | 0.644 [0.640-0.648] CI 0.604..0.682 | 0.642 [0.636-0.652] CI 0.599..0.683 |
| lf_flex_switched | 0.843 CI 0.818..0.867 | 0.711 CI 0.685..0.735 | 0.626 CI 0.582..0.667 |
| lf_flex_single | 0.677 CI 0.642..0.715 | 0.539 CI 0.500..0.576 | 0.516 CI 0.475..0.553 |
| random | 0.006 CI -0.007..0.018 | 0.002 CI -0.008..0.011 | 0.003 CI -0.007..0.012 |
| persistence (DEGENERATE) | 0.123 CI 0.075..0.169 | 0.011 CI -0.033..0.057 | 0.093 CI 0.054..0.134 |

## slateN per goal x vf (per model, mean +- slate sem)

| model | random_quadrant/lyapunov | random_quadrant/mass_in_region | random_quadrant/signed_mass | ring_O/lyapunov | ring_O/mass_in_region | ring_O/signed_mass | T/lyapunov | T/mass_in_region | T/signed_mass |
|---|---|---|---|---|---|---|---|---|---|
| nfd_s0 | 0.926+-0.010 | 0.832+-0.018 | 0.898+-0.012 | 0.952+-0.008 | 0.871+-0.015 | 0.885+-0.014 | 0.967+-0.008 | 0.929+-0.012 | 0.906+-0.014 |
| nfd_s1 | 0.937+-0.009 | 0.838+-0.018 | 0.903+-0.012 | 0.955+-0.008 | 0.873+-0.016 | 0.887+-0.014 | 0.972+-0.006 | 0.910+-0.014 | 0.915+-0.017 |
| nfd_s2 | 0.939+-0.009 | 0.831+-0.017 | 0.902+-0.012 | 0.950+-0.008 | 0.877+-0.016 | 0.883+-0.015 | 0.969+-0.007 | 0.925+-0.013 | 0.927+-0.012 |
| gnn_flex_drp_samp0 | 0.863+-0.024 | 0.719+-0.025 | 0.818+-0.023 | 0.589+-0.037 | 0.562+-0.035 | 0.509+-0.037 | 0.839+-0.025 | 0.650+-0.028 | 0.582+-0.037 |
| gnn_flex_drp_samp1 | 0.866+-0.023 | 0.719+-0.028 | 0.831+-0.023 | 0.603+-0.034 | 0.562+-0.033 | 0.515+-0.037 | 0.822+-0.029 | 0.640+-0.030 | 0.611+-0.036 |
| gnn_flex_drp_samp2 | 0.864+-0.024 | 0.722+-0.027 | 0.817+-0.023 | 0.596+-0.036 | 0.580+-0.034 | 0.518+-0.033 | 0.817+-0.031 | 0.642+-0.028 | 0.574+-0.036 |
| lf_flex_switched | 0.742+-0.030 | 0.561+-0.031 | 0.495+-0.038 | 0.880+-0.017 | 0.738+-0.027 | 0.695+-0.031 | 0.906+-0.013 | 0.833+-0.021 | 0.688+-0.029 |
| lf_flex_single | 0.602+-0.030 | 0.366+-0.034 | 0.362+-0.032 | 0.648+-0.037 | 0.549+-0.037 | 0.509+-0.037 | 0.781+-0.029 | 0.703+-0.031 | 0.678+-0.025 |
| gnn_flex_drp_n50 | 0.898+-0.013 | 0.760+-0.017 | 0.849+-0.013 | 0.678+-0.037 | 0.637+-0.030 | 0.612+-0.032 | 0.814+-0.029 | 0.691+-0.032 | 0.611+-0.040 |
| random | 0.008+-0.013 | 0.001+-0.009 | 0.011+-0.008 | 0.006+-0.010 | 0.009+-0.009 | 0.001+-0.010 | 0.003+-0.012 | -0.004+-0.008 | -0.004+-0.007 |
| persistence | 0.184+-0.047 | -0.031+-0.028 | 0.004+-0.029 | -0.117+-0.038 | -0.093+-0.037 | 0.061+-0.032 | 0.303+-0.042 | 0.158+-0.034 | 0.215+-0.030 |

## accuracy DS-0019 (ratio of population means)

| model | accuracy | row-boot CI | slate-cluster CI |
|---|---|---|---|
| nfd_s0 | 0.4895 | [0.4868642807006836, 0.4918474584817886] | [0.4721292446254783, 0.5077032664200996] |
| nfd_s1 | 0.4809 | [0.47838613092899324, 0.4834927171468735] | [0.4632237717820348, 0.49797686214756437] |
| nfd_s2 | 0.4842 | [0.48157499432563783, 0.48674342036247253] | [0.4657513798188508, 0.5025811399083866] |
| gnn_flex_drp_samp0 | 0.1908 | [0.18801034688949586, 0.19333648085594177] | [0.16748884782706894, 0.21275487572309426] |
| gnn_flex_drp_samp1 | 0.1913 | [0.18862752616405487, 0.19402469098567962] | [0.16836952739808012, 0.2133372658624258] |
| gnn_flex_drp_samp2 | 0.1913 | [0.18868162631988525, 0.19400570541620255] | [0.16824112351738643, 0.2140778667076375] |
| lf_flex_switched | 0.3284 | [0.3265013411641121, 0.33046034127473833] | [0.31225085785589174, 0.3440652023027768] |
| lf_flex_single | 0.2394 | [0.23664184510707856, 0.24238077253103257] | [0.2257239833996672, 0.25181101246834015] |
| gnn_flex_drp_n50 | 0.1936 | [0.19081965535879136, 0.19624608010053635] | [0.17104181964961213, 0.21579129769650732] |
| persistence | 0.0000 |  |  |
| NFD seed-mean | 0.4849 |  | 0.004313477677741305 |
| GNN sampling-seed-mean | 0.1911 |  | 0.0003128620059039992 |

### paired accuracy deltas

- nfd_s0 - lf_flex_switched: +0.1611 row CI [0.1596088409423828, 0.1624546766281128] slate CI [0.14938077935674812, 0.17213477578772088]
- nfd_s0 - lf_flex_single: +0.2501 row CI [0.2482861638069153, 0.25185921639204023] slate CI [0.24087205949576626, 0.2593096817692136]
- nfd_s0 - gnn_flex_drp_samp0: +0.2987 row CI [0.2965951278805733, 0.3008384644985199] slate CI [0.27891502865697465, 0.31764910140862834]
- lf_flex_switched - lf_flex_single: +0.0890 row CI [0.08731932193040848, 0.09081774652004242] slate CI [0.08472174991147746, 0.09321190534548734]
- lf_flex_switched - gnn_flex_drp_samp0: +0.1377 row CI [0.13576851636171341, 0.1395650640130043] slate CI [0.12389334002523394, 0.15309115832559939]
- lf_flex_single - gnn_flex_drp_samp0: +0.0486 row CI [0.04638224095106125, 0.050887937843799594] slate CI [0.034059954449860486, 0.06339046245593744]

## paired slateN (Holm)


**all_9_cells**

- NFD vs GNN: +0.2271 [+0.1942, +0.2631] p_holm 0.0000 wins 98/2 resolved=True
- NFD vs LF_sw: +0.1833 [+0.1596, +0.2069] p_holm 0.0000 wins 94/6 resolved=True
- NFD vs LF_single: +0.3321 [+0.3050, +0.3601] p_holm 0.0000 wins 100/0 resolved=True
- NFD vs random: +0.9061 [+0.8954, +0.9165] p_holm 0.0000 wins 100/0 resolved=True
- GNN vs LF_sw: -0.0439 [-0.0862, -0.0034] p_holm 0.0411 wins 46/54 resolved=True
- GNN vs LF_single: +0.1049 [+0.0665, +0.1424] p_holm 0.0000 wins 73/27 resolved=True
- GNN vs random: +0.6789 [+0.6420, +0.7119] p_holm 0.0000 wins 100/0 resolved=True
- LF_sw vs LF_single: +0.1488 [+0.1224, +0.1760] p_holm 0.0000 wins 86/14 resolved=True
- LF_sw vs random: +0.7228 [+0.6987, +0.7465] p_holm 0.0000 wins 100/0 resolved=True
- LF_single vs random: +0.5740 [+0.5444, +0.6025] p_holm 0.0000 wins 100/0 resolved=True

**lyapunov**

- NFD vs GNN: +0.1898 [+0.1496, +0.2356] p_holm 0.0000 wins 89/11 resolved=True
- NFD vs LF_sw: +0.1093 [+0.0878, +0.1319] p_holm 0.0000 wins 85/13 resolved=True
- NFD vs LF_single: +0.2749 [+0.2380, +0.3125] p_holm 0.0000 wins 98/2 resolved=True
- NFD vs random: +0.9462 [+0.9317, +0.9607] p_holm 0.0000 wins 100/0 resolved=True
- GNN vs LF_sw: -0.0805 [-0.1291, -0.0354] p_holm 0.0019 wins 38/61 resolved=True
- GNN vs LF_single: +0.0851 [+0.0342, +0.1346] p_holm 0.0019 wins 66/34 resolved=True
- GNN vs random: +0.7564 [+0.7114, +0.7953] p_holm 0.0000 wins 98/2 resolved=True
- LF_sw vs LF_single: +0.1656 [+0.1291, +0.2044] p_holm 0.0000 wins 78/18 resolved=True
- LF_sw vs random: +0.8368 [+0.8100, +0.8635] p_holm 0.0000 wins 100/0 resolved=True
- LF_single vs random: +0.6713 [+0.6299, +0.7113] p_holm 0.0000 wins 100/0 resolved=True

**mass_in_region**

- NFD vs GNN: +0.2326 [+0.1930, +0.2734] p_holm 0.0000 wins 92/8 resolved=True
- NFD vs LF_sw: +0.1659 [+0.1402, +0.1918] p_holm 0.0000 wins 86/14 resolved=True
- NFD vs LF_single: +0.3372 [+0.2973, +0.3776] p_holm 0.0000 wins 96/4 resolved=True
- NFD vs random: +0.8743 [+0.8568, +0.8915] p_holm 0.0000 wins 100/0 resolved=True
- GNN vs LF_sw: -0.0667 [-0.1124, -0.0218] p_holm 0.0042 wins 45/55 resolved=True
- GNN vs LF_single: +0.1046 [+0.0555, +0.1532] p_holm 0.0001 wins 74/26 resolved=True
- GNN vs random: +0.6418 [+0.5999, +0.6811] p_holm 0.0000 wins 99/1 resolved=True
- LF_sw vs LF_single: +0.1713 [+0.1347, +0.2089] p_holm 0.0000 wins 78/14 resolved=True
- LF_sw vs random: +0.7084 [+0.6811, +0.7353] p_holm 0.0000 wins 100/0 resolved=True
- LF_single vs random: +0.5372 [+0.4951, +0.5786] p_holm 0.0000 wins 98/2 resolved=True

**signed_mass**

- NFD vs GNN: +0.2590 [+0.2177, +0.3026] p_holm 0.0000 wins 96/4 resolved=True
- NFD vs LF_sw: +0.2746 [+0.2315, +0.3168] p_holm 0.0000 wins 86/14 resolved=True
- NFD vs LF_single: +0.3842 [+0.3459, +0.4234] p_holm 0.0000 wins 99/1 resolved=True
- NFD vs random: +0.8977 [+0.8802, +0.9145] p_holm 0.0000 wins 100/0 resolved=True
- GNN vs LF_sw: +0.0155 [-0.0458, +0.0754] p_holm 0.6183 wins 52/47 resolved=False
- GNN vs LF_single: +0.1251 [+0.0734, +0.1745] p_holm 0.0000 wins 71/29 resolved=True
- GNN vs random: +0.6387 [+0.5943, +0.6802] p_holm 0.0000 wins 97/3 resolved=True
- LF_sw vs LF_single: +0.1096 [+0.0648, +0.1551] p_holm 0.0000 wins 67/33 resolved=True
- LF_sw vs random: +0.6231 [+0.5779, +0.6663] p_holm 0.0000 wins 99/1 resolved=True
- LF_single vs random: +0.5135 [+0.4742, +0.5511] p_holm 0.0000 wins 99/1 resolved=True

**individual_models_all_9_cells**

- nfd_s0 vs nfd_s1: -0.0028 [-0.0105, +0.0050] p_holm 1.0000 wins 45/49 resolved=False
- nfd_s0 vs nfd_s2: -0.0043 [-0.0118, +0.0033] p_holm 1.0000 wins 45/52 resolved=False
- nfd_s0 vs gnn_flex_drp_samp0: +0.2261 [+0.1911, +0.2644] p_holm 0.0000 wins 96/4 resolved=True
- nfd_s0 vs gnn_flex_drp_samp1: +0.2219 [+0.1887, +0.2579] p_holm 0.0000 wins 95/5 resolved=True
- nfd_s0 vs gnn_flex_drp_samp2: +0.2263 [+0.1916, +0.2630] p_holm 0.0000 wins 96/4 resolved=True
- nfd_s0 vs lf_flex_switched: +0.1809 [+0.1568, +0.2045] p_holm 0.0000 wins 95/5 resolved=True
- nfd_s0 vs lf_flex_single: +0.3297 [+0.3027, +0.3569] p_holm 0.0000 wins 100/0 resolved=True
- nfd_s1 vs nfd_s2: -0.0015 [-0.0100, +0.0068] p_holm 1.0000 wins 45/52 resolved=False
- nfd_s1 vs gnn_flex_drp_samp0: +0.2289 [+0.1947, +0.2665] p_holm 0.0000 wins 96/4 resolved=True
- nfd_s1 vs gnn_flex_drp_samp1: +0.2247 [+0.1903, +0.2611] p_holm 0.0000 wins 96/4 resolved=True
- nfd_s1 vs gnn_flex_drp_samp2: +0.2291 [+0.1953, +0.2665] p_holm 0.0000 wins 95/5 resolved=True
- nfd_s1 vs lf_flex_switched: +0.1837 [+0.1600, +0.2074] p_holm 0.0000 wins 95/5 resolved=True
- nfd_s1 vs lf_flex_single: +0.3325 [+0.3044, +0.3607] p_holm 0.0000 wins 100/0 resolved=True
- nfd_s2 vs gnn_flex_drp_samp0: +0.2304 [+0.1968, +0.2669] p_holm 0.0000 wins 98/2 resolved=True
- nfd_s2 vs gnn_flex_drp_samp1: +0.2262 [+0.1927, +0.2617] p_holm 0.0000 wins 99/1 resolved=True
- nfd_s2 vs gnn_flex_drp_samp2: +0.2306 [+0.1965, +0.2672] p_holm 0.0000 wins 99/1 resolved=True
- nfd_s2 vs lf_flex_switched: +0.1852 [+0.1606, +0.2094] p_holm 0.0000 wins 93/7 resolved=True
- nfd_s2 vs lf_flex_single: +0.3340 [+0.3055, +0.3624] p_holm 0.0000 wins 100/0 resolved=True
- gnn_flex_drp_samp0 vs gnn_flex_drp_samp1: -0.0042 [-0.0159, +0.0073] p_holm 1.0000 wins 25/30 resolved=False
- gnn_flex_drp_samp0 vs gnn_flex_drp_samp2: +0.0002 [-0.0090, +0.0091] p_holm 1.0000 wins 29/25 resolved=False
- gnn_flex_drp_samp0 vs lf_flex_switched: -0.0452 [-0.0872, -0.0051] p_holm 0.3008 wins 42/58 resolved=True
- gnn_flex_drp_samp0 vs lf_flex_single: +0.1036 [+0.0650, +0.1412] p_holm 0.0000 wins 73/27 resolved=True
- gnn_flex_drp_samp1 vs gnn_flex_drp_samp2: +0.0044 [-0.0075, +0.0161] p_holm 1.0000 wins 30/25 resolved=False
- gnn_flex_drp_samp1 vs lf_flex_switched: -0.0410 [-0.0846, -0.0002] p_holm 0.4162 wins 43/57 resolved=True
- gnn_flex_drp_samp1 vs lf_flex_single: +0.1078 [+0.0689, +0.1456] p_holm 0.0000 wins 73/27 resolved=True
- gnn_flex_drp_samp2 vs lf_flex_switched: -0.0454 [-0.0887, -0.0057] p_holm 0.2939 wins 43/57 resolved=True
- gnn_flex_drp_samp2 vs lf_flex_single: +0.1034 [+0.0640, +0.1407] p_holm 0.0000 wins 74/26 resolved=True
- lf_flex_switched vs lf_flex_single: +0.1488 [+0.1230, +0.1763] p_holm 0.0000 wins 86/14 resolved=True

**nfd_seed_noise**

- nfd_s0 vs nfd_s1: -0.0028 [-0.0105, +0.0050] p_holm 0.9887 wins 45/49 resolved=False
- nfd_s0 vs nfd_s2: -0.0043 [-0.0118, +0.0033] p_holm 0.8253 wins 45/52 resolved=False
- nfd_s1 vs nfd_s2: -0.0015 [-0.0100, +0.0070] p_holm 0.9887 wins 45/52 resolved=False

**gnn_sampling_noise**

- gnn_flex_drp_samp0 vs gnn_flex_drp_samp1: -0.0042 [-0.0157, +0.0073] p_holm 1.0000 wins 25/30 resolved=False
- gnn_flex_drp_samp0 vs gnn_flex_drp_samp2: +0.0002 [-0.0091, +0.0093] p_holm 1.0000 wins 29/25 resolved=False
- gnn_flex_drp_samp1 vs gnn_flex_drp_samp2: +0.0044 [-0.0077, +0.0160] p_holm 1.0000 wins 30/25 resolved=False

**accuracy_per_slate**

- NFD vs GNN: +0.2979 [+0.2784, +0.3185] p_holm 0.0000 wins 100/0 resolved=True
- NFD vs LF_sw: +0.1587 [+0.1467, +0.1708] p_holm 0.0000 wins 100/0 resolved=True
- NFD vs LF_single: +0.2458 [+0.2357, +0.2560] p_holm 0.0000 wins 100/0 resolved=True
- GNN vs LF_sw: -0.1393 [-0.1550, -0.1248] p_holm 0.0000 wins 0/100 resolved=True
- GNN vs LF_single: -0.0521 [-0.0681, -0.0369] p_holm 0.0000 wins 29/71 resolved=True
- LF_sw vs LF_single: +0.0872 [+0.0830, +0.0913] p_holm 0.0000 wins 100/0 resolved=True

## strata


**init_pos=rand_blob**

- NFD (3 seeds): n_slates 50 slateN9 0.908 CI [0.8938071140635386, 0.9228534226653357] lyap 0.957 mass 0.863 sign 0.905 | acc 0.468 range [0.46283602714538574, 0.47332102060317993] CI0 [0.4405032040959406, 0.49335280550492416]
- GNN (3 sampling seeds): n_slates 50 slateN9 0.630 CI [0.569078594040512, 0.6882820478014704] lyap 0.737 mass 0.560 sign 0.594 | acc 0.113 range [0.1131325364112854, 0.11329835653305054] CI0 [0.08213915120910051, 0.14103566427348127]
- lf_flex_switched: n_slates 50 slateN9 0.807 CI [0.7765172947277513, 0.8343233131352956] lyap 0.909 mass 0.743 sign 0.770 | acc 0.263 range [0.26259517669677734, 0.26259517669677734] CI0 [0.24770950019185445, 0.27540211489954186]
- lf_flex_single: n_slates 50 slateN9 0.617 CI [0.574760472316748, 0.6573612299649977] lyap 0.710 mass 0.556 sign 0.586 | acc 0.193 range [0.1930432915687561, 0.1930432915687561] CI0 [0.17903260845777658, 0.2064243624739565]
- random: n_slates 50 slateN9 0.006 CI [-0.0034872298781336705, 0.014962191945347206] lyap 0.007 mass 0.007 sign 0.003
- paired: NFD-GNN +0.278 [+0.224, +0.343]; NFD-LF_sw +0.101 [+0.077, +0.127]; GNN-LF_sw -0.177 [-0.237, -0.125]; GNN-LF_single +0.013 [-0.043, +0.065]

**init_pos=rand_spread**

- NFD (3 seeds): n_slates 50 slateN9 0.911 CI [0.8982929463354414, 0.9229705951404757] lyap 0.947 mass 0.889 sign 0.897 | acc 0.499 range [0.4930883049964905, 0.507598876953125] CI0 [0.4829935415909761, 0.531090644142448]
- GNN (3 sampling seeds): n_slates 50 slateN9 0.735 CI [0.7081209475615698, 0.7623319701485981] lyap 0.788 mass 0.728 sign 0.689 | acc 0.255 range [0.2543831467628479, 0.2553994655609131] CI0 [0.23140966463306686, 0.27657502930595246]
- lf_flex_switched: n_slates 50 slateN9 0.645 CI [0.6216345111705286, 0.6688570488265061] lyap 0.776 mass 0.678 sign 0.482 | acc 0.382 range [0.3823545575141907, 0.3823545575141907] CI0 [0.36585421264039375, 0.39823470494468216]
- lf_flex_single: n_slates 50 slateN9 0.538 CI [0.5029451609934402, 0.5700957460159518] lyap 0.644 mass 0.522 sign 0.447 | acc 0.277 range [0.27740561962127686, 0.27740561962127686] CI0 [0.26210995623141087, 0.29112857929347496]
- random: n_slates 50 slateN9 0.001 CI [-0.00827629250810412, 0.010684615765366254] lyap 0.005 mass -0.003 sign 0.002
- paired: NFD-GNN +0.176 [+0.147, +0.207]; NFD-LF_sw +0.265 [+0.242, +0.288]; GNN-LF_sw +0.089 [+0.056, +0.124]; GNN-LF_single +0.197 [+0.161, +0.235]

**pieces_small**

- NFD (3 seeds): n_slates 21 slateN9 0.893 CI [0.8687653644723893, 0.9147035573730192] lyap 0.949 mass 0.832 sign 0.896 | acc 0.424 range [0.4204193353652954, 0.4275767207145691] CI0 [0.39020670816148845, 0.4546494252468782]
- GNN (3 sampling seeds): n_slates 21 slateN9 0.688 CI [0.6424481732918637, 0.735852456482311] lyap 0.792 mass 0.597 sign 0.676 | acc 0.107 range [0.10677671432495117, 0.10677671432495117] CI0 [0.08025956169712217, 0.13289198299939434]
- lf_flex_switched: n_slates 21 slateN9 0.812 CI [0.7747122266270268, 0.8480512267258993] lyap 0.932 mass 0.718 sign 0.786 | acc 0.243 range [0.24342679977416992, 0.24342679977416992] CI0 [0.22218205429716048, 0.26084540873791207]
- lf_flex_single: n_slates 21 slateN9 0.627 CI [0.5715631592303995, 0.6788593597852018] lyap 0.734 mass 0.565 sign 0.583 | acc 0.178 range [0.17768102884292603, 0.17768102884292603] CI0 [0.15623041496608675, 0.1962299809708394]
- random: n_slates 21 slateN9 0.004 CI [-0.012066286773770338, 0.019138294888448638] lyap 0.001 mass 0.016 sign -0.005
- paired: NFD-GNN +0.204 [+0.162, +0.243]; NFD-LF_sw +0.081 [+0.047, +0.115]; GNN-LF_sw -0.124 [-0.168, -0.072]; GNN-LF_single +0.061 [+0.014, +0.107]

**pieces_mid**

- NFD (3 seeds): n_slates 39 slateN9 0.920 CI [0.9056619911744194, 0.934571762381163] lyap 0.963 mass 0.888 sign 0.910 | acc 0.487 range [0.48172932863235474, 0.48923176527023315] CI0 [0.4572153932183948, 0.5185164459302186]
- GNN (3 sampling seeds): n_slates 39 slateN9 0.599 CI [0.5242871970305532, 0.666424927543488] lyap 0.693 mass 0.561 sign 0.542 | acc 0.126 range [0.12477463483810425, 0.12628918886184692] CI0 [0.08603472782947462, 0.1583760171624813]
- lf_flex_switched: n_slates 39 slateN9 0.745 CI [0.7028376696111999, 0.7858095117053637] lyap 0.861 mass 0.711 sign 0.661 | acc 0.301 range [0.3006504774093628, 0.3006504774093628] CI0 [0.2782814036864601, 0.32326826932728014]
- lf_flex_single: n_slates 39 slateN9 0.565 CI [0.5153023123630616, 0.6175456970203521] lyap 0.673 mass 0.501 sign 0.519 | acc 0.220 range [0.21962523460388184, 0.21962523460388184] CI0 [0.19831085506623577, 0.23895432441126196]
- random: n_slates 39 slateN9 0.003 CI [-0.005452560139068491, 0.012337870376129286] lyap 0.006 mass -0.008 sign 0.012
- paired: NFD-GNN +0.321 [+0.254, +0.400]; NFD-LF_sw +0.176 [+0.138, +0.213]; GNN-LF_sw -0.146 [-0.222, -0.075]; GNN-LF_single +0.034 [-0.042, +0.110]

**pieces_large**

- NFD (3 seeds): n_slates 40 slateN9 0.908 CI [0.8935264415520529, 0.922839800871981] lyap 0.943 mass 0.889 sign 0.893 | acc 0.511 range [0.5053275227546692, 0.5188950300216675] CI0 [0.49450538921117637, 0.5417929134188396]
- GNN (3 sampling seeds): n_slates 40 slateN9 0.761 CI [0.7348053123904738, 0.7856301236289455] lyap 0.814 mass 0.749 sign 0.720 | acc 0.284 range [0.2827802300453186, 0.284964382648468] CI0 [0.26481246993184016, 0.3003355203970775]
- lf_flex_switched: n_slates 40 slateN9 0.664 CI [0.6324995315653771, 0.6947795374527188] lyap 0.778 mass 0.706 sign 0.508 | acc 0.390 range [0.3897439241409302, 0.3897439241409302] CI0 [0.37403333843500103, 0.4046398320776894]
- lf_flex_single: n_slates 40 slateN9 0.564 CI [0.5235715330303659, 0.6001138436912528] lyap 0.651 mass 0.563 sign 0.479 | acc 0.284 range [0.2836328148841858, 0.2836328148841858] CI0 [0.2696669454557183, 0.2970532357300052]
- random: n_slates 40 slateN9 0.004 CI [-0.007312950701176096, 0.01502483388851362] lyap 0.009 mass 0.005 sign -0.002
- paired: NFD-GNN +0.147 [+0.120, +0.178]; NFD-LF_sw +0.244 [+0.215, +0.273]; GNN-LF_sw +0.097 [+0.059, +0.134]; GNN-LF_single +0.197 [+0.158, +0.236]

**gnn_fallback_states(<=200 voxels)**

- NFD (3 seeds): n_slates 43 slateN9 0.910 CI [0.8930584375534631, 0.924894522065416] lyap 0.957 mass 0.864 sign 0.908 | acc 0.475 range [0.47003424167633057, 0.48087698221206665] CI0 [0.4477464640664109, 0.5001205493762167]
- GNN (3 sampling seeds): n_slates 43 slateN9 0.702 CI [0.6689243962899784, 0.7348746174724162] lyap 0.832 mass 0.615 sign 0.661 | acc 0.145 range [0.14471060037612915, 0.14471060037612915] CI0 [0.1254427193467412, 0.16501774174679626]
- lf_flex_switched: n_slates 43 slateN9 0.823 CI [0.7981978722733808, 0.8486707969421688] lyap 0.921 mass 0.757 sign 0.791 | acc 0.266 range [0.2661629915237427, 0.2661629915237427] CI0 [0.2523082167660154, 0.27936360594785925]
- lf_flex_single: n_slates 43 slateN9 0.637 CI [0.596639852148692, 0.680080091374026] lyap 0.735 mass 0.574 sign 0.602 | acc 0.198 range [0.19752097129821777, 0.19752097129821777] CI0 [0.18271763635991273, 0.21075338374321487]
- random: n_slates 43 slateN9 0.006 CI [-0.004388879562022317, 0.015363408671146068] lyap 0.009 mass 0.007 sign 0.000
- paired: NFD-GNN +0.207 [+0.178, +0.239]; NFD-LF_sw +0.087 [+0.063, +0.112]; GNN-LF_sw -0.121 [-0.154, -0.086]; GNN-LF_single +0.065 [+0.023, +0.111]

**gnn_fps_states(>200 voxels)**

- NFD (3 seeds): n_slates 57 slateN9 0.910 CI [0.8961882639742124, 0.9219981529303247] lyap 0.948 mass 0.886 sign 0.895 | acc 0.491 range [0.4863243103027344, 0.4992298483848572] CI0 [0.47473935033895087, 0.5221823463815862]
- GNN (3 sampling seeds): n_slates 57 slateN9 0.667 CI [0.6111804867569319, 0.7210145575409123] lyap 0.710 mass 0.665 sign 0.627 | acc 0.221 range [0.22054272890090942, 0.2214943766593933] CI0 [0.1885973196465926, 0.2509157904516271]
- lf_flex_switched: n_slates 57 slateN9 0.653 CI [0.6295045918879898, 0.6767756948019586] lyap 0.784 mass 0.675 sign 0.502 | acc 0.369 range [0.3686671257019043, 0.3686671257019043] CI0 [0.34969768042914323, 0.3864337724454823]
- lf_flex_single: n_slates 57 slateN9 0.533 CI [0.4991075885436085, 0.5664861822283129] lyap 0.633 mass 0.513 sign 0.452 | acc 0.266 range [0.266493558883667, 0.266493558883667] CI0 [0.25013376767021367, 0.28197855732844596]
- random: n_slates 57 slateN9 0.002 CI [-0.006286992509244757, 0.01032496961611054] lyap 0.003 mass -0.002 sign 0.005
- paired: NFD-GNN +0.242 [+0.191, +0.299]; NFD-LF_sw +0.256 [+0.234, +0.280]; GNN-LF_sw +0.014 [-0.049, +0.073]; GNN-LF_single +0.135 [+0.077, +0.191]

## frac(dv_true==0)

- random_quadrant: lyapunov 0.0010, mass_in_region 0.1874, signed_mass 0.0061
- ring_O: lyapunov 0.0005, mass_in_region 0.0471, signed_mass 0.0083
- T: lyapunov 0.0005, mass_in_region 0.0674, signed_mass 0.0076

## val vs test accuracy

- nfd_s0: val 0.6095 [0.6060876006671699, 0.6128973521505535] test 0.4895 [0.4721292446254783, 0.5077032664200996] gap -0.1201
- nfd_s1: val 0.6117 [0.6084973141394244, 0.6148228040380318] test 0.4809 [0.4632237717820348, 0.49797686214756437] gap -0.1308
- nfd_s2: val 0.6098 [0.6064395981926824, 0.6130532085797231] test 0.4842 [0.4657513798188508, 0.5025811399083866] gap -0.1257
- gnn_flex_drp_samp0: val 0.2517 [0.24637539205214598, 0.2571128837721872] test 0.1908 [0.16748884782706894, 0.21275487572309426] gap -0.0609
- gnn_flex_drp_samp1: val 0.2517 [0.24658517304221472, 0.256829533714558] test 0.1913 [0.16836952739808012, 0.2133372658624258] gap -0.0604
- gnn_flex_drp_samp2: val 0.2517 [0.2462056003070981, 0.25701600992951434] test 0.1913 [0.16824112351738643, 0.2140778667076375] gap -0.0603
- lf_flex_switched: val 0.4824 [0.47794558808156856, 0.48653766771086876] test 0.3284 [0.31225085785589174, 0.3440652023027768] gap -0.1540
- lf_flex_single: val 0.3543 [0.3467621334375431, 0.3613032254158568] test 0.2394 [0.2257239833996672, 0.25181101246834015] gap -0.1149

ranking accuracy val ['nfd_s1', 'nfd_s2', 'nfd_s0', 'lf_flex_switched', 'lf_flex_single', 'gnn_flex_drp_samp0', 'gnn_flex_drp_samp1', 'gnn_flex_drp_samp2']
ranking accuracy test ['nfd_s0', 'nfd_s2', 'nfd_s1', 'lf_flex_switched', 'lf_flex_single', 'gnn_flex_drp_samp2', 'gnn_flex_drp_samp1', 'gnn_flex_drp_samp0']

ranking slateN test ['nfd_s2', 'nfd_s1', 'nfd_s0', 'gnn_flex_drp_n50', 'lf_flex_switched', 'gnn_flex_drp_samp1', 'gnn_flex_drp_samp0', 'gnn_flex_drp_samp2', 'lf_flex_single']
