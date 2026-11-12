# AWS round v7 + v8: Compact planted worlds, natural-cue mechanisms, real cells

Protocols: `configs/protocols/pilot_asdpose_v7.yaml` (frozen dbff67f) and
`configs/protocols/pilot_asdpose_v8.yaml` (frozen b0f64bf). Analysis code
`experiments/analyze_v7.py` (frozen with v8 in b0f64bf). Runs 2026-10-30 to 2026-10-31 on one
on-demand g5.2xlarge (an EC2 instance, us-east-1c), 14.3 h, about USD 17: 71 v7 runs,
24 v8 runs, all scored by `experiments/score_v7.py` on the instance; no failed run, no failed
score. CompactCTRGCN, 25 epochs, three seeds per cell, length-controlled protocol. Host test 40
children. Child-clustered paired bootstrap, 2,000 resamples, seeds averaged inside resamples.
Analysis run locally on CPU 2026-10-31 (7 min). Numbers in `analysis/effects.csv`,
`crossed.csv`, `companions.csv`, `comparators.csv`, `length.csv`, `groups.csv`,
`criteria.json`.

Data location: this directory is a symlink to `results/aws/v7/results/v7`, the local mirror
of `s3://<bucket>/results/v7/` (mirror finished 2026-10-31 01:13 local, after
instance termination). Instance logs: `results/aws/v7/logs/`, `results/aws/v7/bootstrap.log`.

Scale: R and SE on the logit-AUC scale; SE = R(f_S) − R(f_I) for the same operator in
training and test. Margins (v8): ±0.05 for R, ±0.02 for clean AUC.

## Verdicts

| test | verdict | result |
|---|---|---|
| P1 validation on Compact, runs operator | **see note** | SE mid 0.756 [0.618, 0.920], low 0.218 [0.111, 0.326]; World I A_none R mid +0.021 [−0.053, +0.097], low +0.016 [−0.062, +0.077]; SE range over three test masks 0.011 (CI half-width 0.151) |
| P4 length not exploited | **PASS** | all 15 real-cell models; \|Spearman\| ≤ 0.09, ridge R² on log length ≤ 0.03 |
| P5 clean-AUC equivalence (A_none, runs, mid) | **FAIL** | −0.004 [−0.023, +0.014]; lower end outside −0.02 |
| M1 frozen | **FAIL** | SE 0.376 [0.259, 0.503] (passes); World I R −0.003 [−0.0512, +0.045], lower end outside −0.05 by 0.001 |
| M1 armruns | **FAIL** | SE 0.582 [0.482, 0.692] (passes); World I R **+0.051 [+0.013, +0.097]**, excludes zero |
| M3 remedy by mechanism | reported; **against expectation** | expected: whole-joint dropout acts on whole and runs, not on frozen or armruns. Observed SE(A_dropout) − SE(A_none): runs +0.092 [+0.000, +0.178], whole −0.028 [−0.095, +0.043], frozen +0.098 [+0.038, +0.161], armruns +0.075 [+0.017, +0.128] |
| P2, M2, M4, M5 | reported (no pass rule) | below |

**P1 note (decision pending).** The v7 text requires the World I reading to have a CI
*containing* 0; it does at both strengths, so P1 passes as written. The frozen analysis code
applies the v8 equivalence margin (CI *inside* ±0.05), under which the mid reading fails
(upper end +0.097). The SE(mid) − SE(low) contrast named in P1 is not computed by the code; the
two intervals do not overlap. Under either reading the paper may not say World I "reads zero"
for runs; it may say the reading is not distinguishable from zero.

## Remedies (P2, M3): decomposition of the dropout contrast

| mechanism | R(f_S) none → dropout | R(f_I) none → dropout | SE change |
|---|---|---|---|
| runs | 0.777 → 0.745 | +0.021 → −0.104 | +0.092 |
| whole | 0.478 → 0.406 | −0.005 → −0.049 | −0.028 |
| frozen | 0.373 → 0.451 | −0.003 → −0.023 | +0.098 |
| armruns | 0.633 → 0.691 | +0.051 → +0.034 | +0.075 |

Other runs-mechanism remedies, SE change against A_none: velocity +0.073 [−0.021, +0.163],
both +0.061 [−0.018, +0.130], coordinate dropout +0.094 [+0.015, +0.170]; whole mechanism,
coordinate dropout +0.032 [−0.041, +0.099]. Regularization interaction gain_S − gain_I (clean
AUC): within ±0.02 for dropout and coordinate dropout under whole; undetermined everywhere else
(all point estimates between −0.014 and +0.006).

No remedy lowers the shortcut effect under any mechanism. Under runs the dropout increase comes
from the World I reference moving negative, not from the cue-trained model; under frozen and
armruns the cue-trained model itself responds more.

## Operator specificity (M2, crossed table, World I A_none models of v7)

| trained under | test runs | test whole | test frozen | test armruns |
|---|---|---|---|---|
| runs | +0.021 [−0.053, +0.097] | +0.091 [−0.028, +0.209] | **+0.261 [+0.166, +0.366]** | **+0.093 [+0.026, +0.165]** |
| whole | **−0.148 [−0.260, −0.058]** | −0.005 [−0.082, +0.087] | **+0.387 [+0.253, +0.549]** | −0.045 [−0.138, +0.035] |

The v6 reading holds on Compact with three seeds: a model is invariant (to within the CI) to
the operator it was trained against and not to others; frozen frames move every invariant
model strongly.

## Real cells under the natural-form operators (M4)

Pooled R per real cell (three seeds; children with sign opposite to the pooled R, of 40):

| cell | runs | whole | frozen | armruns |
|---|---|---|---|---|
| A_none | −0.082 | +0.289 | +0.345 (8) | +0.001 |
| A_velocity | +0.155 | +0.280 | +0.368 (8) | +0.019 |
| A_dropout | −0.173 | +0.079 | +0.396 (5) | +0.022 |
| A_both | −0.079 | +0.047 | +0.346 (10) | +0.071 [+0.014, +0.131] |
| B_coorddrop | −0.012 | +0.240 | +0.326 (10) | −0.000 |

Every real cell responds to frozen frames by about +0.33 to +0.40, consistently across
children. That is above the frozen-trained reference (−0.003) but inside the range of World I
models trained against other operators (+0.26, +0.39). The frozen response of published
models therefore cannot be attributed to reliance on a learned frozen-frame cue rather than to
the generic sensitivity every model not trained against frozen frames shows. Under armruns
the real cells read near zero (only A_both excludes zero).

## Comparator measures (M5)

Spearman over the 15 real models: R_runs~R_whole 0.54, R_armruns~R_whole −0.42,
filtered_gap~sens_runs 0.45, filtered_gap~cpi_tracking_stats −0.38; every other pair
\|ρ\| ≤ 0.35, including the SAFE-SMM filtered gap against each R (0.10, 0.18, 0.24, −0.03). The
child-bootstrap intervals named in M5 are not written by the analysis code; with n = 15 the
listed values are not distinguishable from each other.

## Other observations

- Clean AUC, f_S − f_I: runs low +0.005 [−0.011, +0.024], whole −0.003 [−0.024, +0.019],
  armruns −0.001 [−0.015, +0.013] (within margin), frozen **+0.024 [+0.010, +0.042]** (the
  frozen cue-trained model is slightly better on clean data).
- Training-mask realizations (runs, mid): R(f_S) 0.777 (w0), 0.711 (w1), 0.762 (w2).
- In about a third of runs the best validation epoch is the last (25). The epoch budget is
  frozen; clean AUCs may be slightly below convergence.

## Reading

1. **The between-world shortcut effect is confirmed on the published architecture with three
   seeds, for all four mechanisms**, including the two natural QC cues (frozen frames, arm
   confidence). It scales with planted strength and is stable across mask realizations. This
   is the paper's validated quantity.
2. **The invariance reference is operator-specific and, for arm confidence, not invariant.**
   SE remains defined (it subtracts the reference), but R of a single model under armruns
   cannot be read against zero.
3. **The matched-form remedy hypothesis (§15 item 2) fails.** No tested remedy lowers the
   shortcut effect; whole-joint dropout raises it under three of four mechanisms. This is a
   negative result and is reported as such.
4. **Real-data reliance is not identified by these operators.** Published models respond
   strongly and uniformly to frozen frames, no more than invariant models trained against a
   different operator; they barely respond to arm-confidence rescaling.
5. **The SAFE-SMM filtered gap does not track any R.**

Pass rules: P4 passes; P5 and M1 (both mechanisms) fail; P1 is pending the decision above.
M3 has no pass rule; its expectation is contradicted. The P5 and M1-frozen failures miss the
margin by 0.003 and 0.001 and remain failures under the frozen rules.
