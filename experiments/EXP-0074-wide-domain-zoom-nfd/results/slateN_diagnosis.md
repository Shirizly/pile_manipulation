# slateN diagnosis: z128_f8_ms4 (zoom) vs w128_f8_ms4 (vanilla 128), wide Sean test pools (2026-10-07)
Code: code/slaten_diag.py (+ slaten_diag2.py); data: results/slateN_diagnosis.json (+ ...2.json); cached predictions artifacts/slaten_diag/. Same test pools / goals / lyap / soft truth as eval_wide.run (reproduces .798/.852, .872/.928).
Scope: these two models, 8 test shards (60 pools x <=32 pushes each, 13 goals); no retraining.

## Cause: spurious, push-length-dependent MASS GAIN of the pasted prediction (mass-normalised lyap -> bias)
- Truth conserves mass (soft truth dM = 0; hard truth |dM|/M0 = 1-3 %). Both pasted predictions GAIN mass: |dM|/M0 (pool pushes) scattered_n20 zoom .168 vs vanilla .132; scattered_n50 .142 vs .090. Always positive (sigmoid leakage everywhere in the window + clamp at 0, no negative compensation).
- The zoom drift grows with push length (= window area; side = L+44 mm): dM/M0 by L bin (<20,20-35,35-50,>=50 mm) scattered_n20 zoom .05/.10/.14/.22 vs vanilla .08/.11/.13/.15; scattered_n50 zoom .03/.06/.12/.15 vs vanilla .04/.06/.09/.09. corr(dM, L) macro .30 (zoom) vs .23 (vanilla). Within-pool std of dM/M0: .071 vs .057 (n20), .063 vs .034 (n50). Pool candidates have mixed lengths, so this is a length-dependent bias that re-orders pushes in the pool.
- Mass drift explains (within-pool, linear) 29 % of the zoom dv-error variance vs 22 % vanilla (macro; scattered_n50 33 % vs 14 %).
- Ranking diagnostics (scattered_n20 / n50): Spearman(pred dv, true dv) zoom .794/.824 vs vanilla .820/.879; pick == true best 51/62 % vs 57/68 %; mean fraction of pushes truly better than the pick .111/.077 vs .085/.048. dv error outside the swept region is equal (rms .009-.010 vs .009-.011, true dv spread .025); inside (swept +4 px) zoom .018-.019 vs .016-.018. So the loss is not "outside the window"; it is the inside/mass term.
- Soft vs hard truth: hard truth lowers both (scattered_n20 .783 vs .829, n50 .858 vs .912) but the gap is unchanged => not a truth artefact. Pasting is not the cause per se: dv from the window only (zoom) gives .794/.808 (no better).
- Null-ish pushes: not testable; null pushes (<1 mm) are dropped from the test pools (soft motion < 1.5 % M0: 0 pushes). The 10 % lowest-motion rows: true motion .076 M0, predicted |delta| .18 M0 for both models (both over-predict motion on small moves equally; zoom dM +.11 vs +.08).
- Push-length bias of the mean centred dv error is small (<=.011) and similar in both models; the ranking damage is through the mass term above.

## Post-hoc fixes (no retraining): macro over 8 shards slateN / acc1; scattered_n20 / n50 slateN in brackets (zoom | vanilla)
| variant | zoom macro slateN / acc1 | vanilla macro | scat_n20 z|v | scat_n50 z|v |
|---|---|---|---|---|
| base | .899 / .582 | .924 / .572 | .798 \| .852 | .872 \| .928 |
| (a) zero |delta|<.1 | .903 / .581 | .930 / .571 | .809 \| .861 | .877 \| .942 |
| (a) zero |delta|<.2 | .903 / .579 | .932 / .568 | .813 \| .856 | .876 \| .945 |
| (b) global rescale to start mass | .899 / .594 | .924 / .582 | .798 \| .852 | .872 \| .928 |
| (b') balance delta (pos/neg parts scaled to equal mass = mean of the two) | .933 / .613 | .964 / .595 | .877 \| .926 | .908 \| .971 |
| (b'') zero |delta|<.1 then balance | .947 / .603 | .966 / .590 | .918 \| .941 | .939 \| .975 |
| (b''') balance to the negative part only (drop created mass) | .937 / .592 | .966 / .589 | .878 \| .931 | .906 \| .968 |
| (c) lyap on window only (zoom) | .903 / .582 | - | .794 | .808 |
| (d) lyap on union of swept regions | .889 / .582 | .911 / .572 | .816 \| .859 | .870 \| .923 |
| fixed start-mass denominator | .875 / .582 | .930 / .572 | .773 \| .888 | .807 \| .931 |
| (e) clip delta to swept region (+4 px) | .860 / .582 | .873 / .572 | .750 \| .780 | .817 \| .878 |
| (e) clip to swept exactly | .535 | .556 | .479 | .413 |
(acc1 = one-step accuracy on 1000 rows/shard, pasted frame, same fix applied to the image.)

## Verdict
- Cause: net mass created by the predicted change (zoom more than vanilla, and growing with push length through the window size), which the mass-normalised lyap turns into a push-length-dependent bias. Not a truth-type, outside-window or null-push effect.
- Works: enforcing a zero-net-mass delta (balance), best after a small threshold: slateN zoom .899 -> .947 (vanilla .924 -> .966), and acc1 does not drop (zoom .582 -> .603, vanilla .572 -> .590). scattered_n20 .798 -> .918, n50 .872 -> .939. It helps BOTH models, and the gap shrinks but remains: macro .025 -> .019, scattered_n20 .054 -> .023, scattered_n50 .056 -> .036 (vanilla balanced still ahead on slateN; zoom ahead on acc1 by +.013). Useful for any slateN/MPC use of these nets.
- Does not work: global rescale (no ranking change), window-only, union-region, clipping to the swept region (hurts a lot: motion beyond the push footprint matters), fixed-mass denominator.
- Retraining would be needed for the residual gap (inside-region dv errors, Spearman .79 vs .82): add a mass-conservation term (zero-sum delta / mass penalty, or predict a flow/conserving transport) and mass-balance loss over the window; length-stratified loss weighting for the zoom net. Untested hypothesis, not run.
