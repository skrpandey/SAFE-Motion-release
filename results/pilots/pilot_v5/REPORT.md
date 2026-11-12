# Pilot v5: the matched-world measure on real ASDPose models (test-time intervention)

Protocol: `configs/protocols/pilot_asdpose_v5.yaml` (frozen 734720d) and addendum 1 (da3896b).
Scoring only, 0.43 GPU-h plus 5 min for the addendum. 19 SAFE-SMM checkpoints (read-only, base
project commit e5217a5), 37 pilot TinyTCNs, and 12 planted-host models, each scored on the full
test split (4,751 segments, 40 children) clean and under 2 mechanisms x 2 strengths x 2 worlds.
R = logit(AUC_S) − logit(AUC_I); primary condition runs/mid; paired child-clustered bootstrap,
10,000 resamples. Analysis: `experiments/analyze_v5.py`; numbers in `groups.csv`, `summary.json`.

## Frozen verdicts

| test | verdict | result |
|---|---|---|
| H10 zero point, cue-free model `D_qcsym_none` | **FAIL** | R = 0.231 [0.149, 0.309]; raw logit gap 0.241 [0.197, 0.295] |
| H10b (addendum) zero point, degradation-invariant models | **PASS** | World-I-trained: TinyTCN −0.047 [−0.110, +0.014], Compact −0.024 [−0.078, +0.034]; World-S-trained: 0.619, 0.639, both CIs above 0.54 |
| H11 cue use | **PASS** | `A_none` 0.222 [0.153, 0.294]; SAFE-SMM v1 0.082 [0.017, 0.144] |
| H12 remedies on real data | **PASS** | `A_both` − `A_none` = −0.318 [−0.390, −0.251] |
| H13 ranking vs raw gap | descriptive | Spearman(R, logit gap) = 0.14 over 13 SAFE-SMM groups; (R, Δ) = 0.02 |

## What H10's failure and H10b's pass mean together

`D_qcsym_none` never saw heavy missingness in training (QC-passing segments only). Planting
missingness at test time moves its scores in a label-asymmetric direction: degraded segments are
scored as non-SMM, so degrading negatives more (World S) raises its AUC relative to World I. It
learned no cue, but its decisions depend on tracking quality at deployment. Models trained with
label-independent degradation (v3/v4 World I) learned to ignore it and read R ≈ 0 on the real
test split, on both architectures and on the QC-sym subset; models trained with the planted cue
read R ≈ 0.63. So R measures dependence of the decision on tracking quality, whatever its origin,
and its zero point is a degradation-invariant model, not merely a model trained without the cue.
The raw gap cannot make this distinction: `D_qcsym_none` and `A_both` have raw logit gaps of
0.24 and 0.47, yet R says the first depends on tracking quality (0.23) and the second does not
(−0.10). The paper should state R's definition this way and drop "learned shortcut" as its reading.

## Remedies on the R scale (TinyTCN, full test split, runs/mid)

| cell | seeds | clean AUC | raw logit gap | R [95% CI] | R − R(A_none) |
|---|---|---|---|---|---|
| A_none | 6 | 0.806 | 0.542 | 0.222 [0.153, 0.294] | |
| B_coorddrop (cue visible) | 6 | 0.818 | 0.555 | 0.222 [0.135, 0.302] | +0.000 [−0.032, +0.030] |
| A_dropout | 3 | 0.822 | 0.538 | −0.021 [−0.094, +0.044] | −0.242 [−0.302, −0.189] |
| A_velocity | 3 | 0.821 | 0.525 | −0.027 [−0.101, +0.039] | −0.248 [−0.312, −0.188] |
| A_both | 6 | 0.840 | 0.467 | −0.096 [−0.162, −0.033] | −0.318 [−0.390, −0.251] |
| D_qcsym_velocity | 3 | 0.749 | −0.029 | 0.126 [0.060, 0.191] | −0.096 [−0.157, −0.038] |

On R the picture is clean where the raw gap was muddled: regularization that keeps the cue
visible changes nothing (+0.000), each remedy alone brings R to zero, both together overshoot
below zero (degraded segments are then scored slightly toward SMM, probably through velocity
spikes at mask edges). On the raw gap the same cells differ by 0.01 to 0.08 with overlapping
intervals and B_coorddrop sits above A_none.

Permutation null (10 shuffled-label TinyTCNs): R from −0.23 to +0.30, `A_none` at 0.222 gives
p = 0.27. Shuffled-label models respond to degradation too, in either direction; R's null is as
wide as the raw gap's was in v2. A positive R for a single model is therefore not evidence by
itself; paired differences between models (as above) and the invariance-trained zero point are.

## SAFE-SMM checkpoints: R disagrees with the raw gap

| model | clean AUC | raw Δ | raw logit gap | R [95% CI] |
|---|---|---|---|---|
| v1 (`stage1_detector`) | 0.843 | 0.084 | 0.533 | 0.082 [0.017, 0.144] |
| v2 family (3 seeds) | 0.944 | 0.015 | 0.259 | 0.176 [0.078, 0.287] |
| v3 (`stage1_detector_v3`) | 0.953 | 0.010 | 0.212 | 0.164 [0.052, 0.299] |
| v3 no-confidence family (3) | 0.956 | 0.013 | 0.264 | 0.307 [0.169, 0.471] |
| ours base (no remedies) | 0.865 | 0.072 | 0.516 | 0.156 [0.085, 0.224] |
| ablation: dropout only | 0.938 | 0.012 | 0.183 | −0.052 [−0.159, +0.057] |
| ablation: velocity only | 0.931 | 0.013 | 0.184 | 0.060 [−0.052, 0.176] |
| ablation: velocity, v3 norm | 0.944 | 0.015 | 0.257 | 0.253 [0.119, 0.401] |
| ablation: dropout, v3 norm | 0.900 | 0.033 | 0.325 | −0.075 [−0.193, +0.037] |
| ST-GCN base / remedies | 0.837 / 0.903 | 0.072 / 0.027 | 0.455 / 0.280 | 0.200 / 0.159 |
| AGCN base / remedies | 0.754 / 0.791 | 0.153 / 0.103 | 0.709 / 0.540 | 0.393 / −0.032 |
| permuted controls v1 / v2 | 0.606 / 0.588 | 0.154 / 0.053 | 0.621 / 0.214 | 0.278 / 0.184 |

- The raw gap says v3 reduced reliance from v1 by a factor of eight (Δ 0.084 to 0.010). R says
  v3 depends on test-time tracking quality at least as much as v1 (0.164 vs 0.082; difference
  +0.082 [−0.052, +0.234]). The v3 recipe's gap reduction on the filtered protocol is not matched
  by a reduction in dependence under intervention.
- Removing the confidence channel raises R (0.307): the model then reads missingness from the
  zeroed coordinates and depends on it more, although the raw gap barely moves (0.264).
- Dropout alone, at either normalization, brings R to zero; velocity alone does not, and at v3
  normalization it raises R. This matches the TinyTCN pattern for dropout and differs for
  velocity; the overshoot of "both" on TinyTCN is not seen on Compact (v3: 0.164).
- AGCN remedies remove dependence (0.393 to −0.032); ST-GCN remedies do not (0.200 to 0.159).
- Both SAFE-SMM permuted controls carry R ≈ 0.2–0.3: they learned tracking quality through epoch
  selection, which R detects.
- Clean accuracy and R are unrelated across models (v3 0.953 clean with R 0.164; dropout
  ablation 0.938 with R −0.05), as on the planted hosts.

Secondary conditions agree in sign: R on the QC-sym subset tracks the full-split R within 0.05
for every group; the whole-joint mechanism gives larger R for the no-remedy models (A_none 0.43)
and keeps the remedy ordering.

## Consequences for the paper

1. Contribution A on real data: R applied by test-time intervention separates models that the
   raw gap cannot, in both directions. Two headline facts for the paper: the v3 recipe's gap
   reduction is not a reduction in tracking-quality dependence, and generic regularization leaves
   dependence unchanged while cue-hiding remedies remove it.
2. Definition to carry forward: R measures dependence of the decision on tracking quality under
   intervention; its zero point is a degradation-invariant model (validated with planted
   training on two architectures); "learned shortcut" is not what it measures.
3. R needs a reference. A single model's R has a wide null; the paper reports paired differences
   and the invariance-trained zero point, never a lone R with a p-value.
4. Open for the AWS matrix: the velocity remedy behaves differently on TinyTCN and Compact, and
   the v3-normalization interaction (dropout fine, velocity worse) is one seed per cell on
   Compact. Three Compact seeds for base / dropout / velocity / both at v3 normalization are the
   first paid cells.
