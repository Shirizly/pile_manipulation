# MODEL-0013 test history

- DS-0021 val (1,741 rows, image-mask truth, swept-region `accuracy`): switched 0.360, single 0.266 -- lambda + bin
  scheme SELECTED on this split -> EXP-0064 / RUN-0013 -> `weights/MODEL-0013-*/fit.json`
- DS-0022 test, per count group: slateN (3 goals x 3 vf, binary image-mask truth, full pool and K-equalised), swept-region
  accuracy, paired vs GNN MODEL-0012 (node-carried onto the mask), `field`, MODEL-0009; accuracy-vs-slateN correlations
  -> EXP-0064 / RUN-0015 -> `experiments/EXP-0064-*/results/nfd_lf_image_metrics.{json,md}`
