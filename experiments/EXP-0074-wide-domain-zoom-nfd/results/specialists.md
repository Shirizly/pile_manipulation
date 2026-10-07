# Post-training specialisation of the single-step generalists (no new data) -- one-step acc1 / slateN [/ roll4 for n-domains], mean over the domain's test shards; control = generalist continued on all rows for ~the same number of steps


## zoom 128 ([8,16,32])
| domain | generalist | control (+all-data steps) | specialist | spec - gen | spec - control |
|---|---|---|---|---|---|
| n = 20 | 0.589 / 0.880 / 0.429 | 0.593 / 0.880 / 0.430 | 0.593 / 0.881 / 0.412 | +0.004 / +0.001 / -0.016 | +0.001 / +0.001 / -0.018 |
| n = 50 | 0.592 / 0.910 / 0.451 | 0.596 / 0.910 / 0.452 | 0.595 / 0.909 / 0.454 | +0.003 / -0.001 / +0.003 | -0.000 / -0.002 / +0.001 |
| n = 100 | 0.561 / 0.939 / 0.471 | 0.563 / 0.937 / 0.472 | 0.563 / 0.941 / 0.476 | +0.002 / +0.002 / +0.005 | -0.001 / +0.004 / +0.004 |

 push-length specialists routed by bin (edges 35/55 mm), one-step: acc1 0.587 (generalist 0.583, control 0.586); slateN 0.906 (generalist 0.906, control 0.906)
  - L<35: routed 0.621 vs generalist 0.617 vs control 0.621
  - L35-55: routed 0.595 vs generalist 0.592 vs control 0.595
  - L>=55: routed 0.557 vs generalist 0.554 vs control 0.557

## world 128 ([8,16,32])
| domain | generalist | control (+all-data steps) | specialist | spec - gen | spec - control |
|---|---|---|---|---|---|
| n = 20 | 0.574 / 0.905 / 0.415 | 0.579 / 0.906 / 0.416 | 0.580 / 0.909 / 0.378 | +0.006 / +0.003 / -0.037 | +0.000 / +0.002 / -0.038 |
| n = 50 | 0.576 / 0.925 / 0.428 | 0.580 / 0.923 / 0.432 | 0.582 / 0.924 / 0.435 | +0.006 / -0.001 / +0.007 | +0.001 / +0.001 / +0.003 |
| n = 100 | 0.540 / 0.935 / 0.433 | 0.545 / 0.933 / 0.437 | 0.546 / 0.934 / 0.442 | +0.006 / -0.001 / +0.008 | +0.001 / +0.000 / +0.005 |

 push-length specialists routed by bin (edges 35/55 mm), one-step: acc1 0.574 (generalist 0.566, control 0.571); slateN 0.925 (generalist 0.920, control 0.919)
  - L<35: routed 0.596 vs generalist 0.587 vs control 0.593
  - L35-55: routed 0.584 vs generalist 0.576 vs control 0.581
  - L>=55: routed 0.549 vs generalist 0.541 vs control 0.546
