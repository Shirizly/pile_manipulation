# EXP-0035 — state-dependent model superiority on DS-0007 (Sean pools, hundreds of states)

Written 2026-09-24 BEFORE running. Same analysis code and plan as EXP-0030 (A1-A4,
see that DESIGN.md), applied to DS-0007 shards scattered_n20 (801 states) and
scattered_n50 (800 states): physics matches training; pools of ~8 pushes, split
4/4 for A1/A2 (so per-state reliability will be LOW -- K=4 is tiny), full pool for
A3 (ensemble) and A4 (descriptors, advantage averaged over the whole pool).
Predictions:
- A1: median split-half r over NFD-family pairs < 0.2 (pools of 4 are dominated by
  pool noise; EXP-0029 found r 0.16 at K=128 on 20 states).
- A2: per-state switching gain interval includes 0.
- A3: the NFD ensemble beats the best single model (state-bootstrap CI excluding 0).
- A4: at least one descriptor is Holm-significant for at least one pair (with ~800
  states even small effects are detectable -- so also report effect sizes, rho).
The shards are analysed separately (n20, n50); n100 / inbetween / piled are not.
