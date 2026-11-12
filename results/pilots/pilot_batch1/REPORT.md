# Pilot batch 1 (CPU): reweighted protocol (H4), d′ scale check, child confounding

Protocol: `configs/protocols/pilot_asdpose_v1.yaml` (frozen 81e0b5f, syntax fix c.f. REGISTRY.md).
Run: 2026-10-24, `experiments/pilot_batch1.py --n-boot 10000`. Inputs: the 29 SAFE-SMM test-score
files (read-only). Numbers: `models.csv`, `summary.json`. One test set, 4,751 segments, 40 children.

## H4: does the reweighted protocol agree with the QC filter? FAIL under the frozen rule

| check | frozen threshold | observed |
|---|---|---|
| propensity balance (weighted AUC of P(y\|q)) | 0.45–0.55 | 0.494, pass |
| Kish ESS of weighted negatives | report | 2,446 of 3,694 (QC filter keeps 1,648) |
| models whose paired CI of (reweighted − QC) contains 0 | ≥ 80% | **17%** (5 of 29) |
| Spearman(reweighted, QC) across models | ≥ 0.9 | **0.998** |

The two protocols rank the 29 models identically, but the reweighted AUC sits slightly above the
QC AUC for 28 of 29 models. The offset grows with reliance: +0.003 for v3, +0.023 for v1,
+0.045 for AGCN base, +0.023 for the label-permuted model that learned tracking quality only.
Reweighting on four linear tracking statistics removes less of the cue than the hard filter does.

Frozen consequence (`H4_fail_action`): the reweighted protocol is reported as a different
estimand, not a substitute for the QC filter. It keeps 66% of the negative information (ESS) and
preserves the ranking, so it remains useful as a no-discard companion measure.

## d′ as the primary ceiling-free scale: the binormal assumption does not hold

The equal-variance binormal model behind d′ needs a z-ROC slope near 1. Observed slopes on the
standard protocol: 0.50–0.66 for every model above 0.93 AUC, 0.82–0.87 for the mid models
(v1, ST-GCN base, ours base), above 1 only for the permuted controls and AGCN. The §14.3.1 choice
of d′ as the primary scale is therefore not supported for the strong models. Before batch 2 is
summarised for the paper, the primary scale must change to one that tolerates unequal variance
(binormal d_a, or logit-AUC with the binormal fit as a sensitivity). That choice goes into a new
frozen protocol before any paper cell is scored.

## Child-level confounding: the tracking cue lives mostly within children

Tracking statistics → label, logistic regression fit on train children, AUC on test children:

| features | AUC [95% child-clustered CI] |
|---|---|
| raw | 0.702 [0.672, 0.733] |
| within-child centred | 0.672 [0.638, 0.706] |

Removing every child's mean quality costs only 0.03. The cue is mostly segment-level within a
child, not cohort structure between children, which supports reading it as a recording artifact.

## Side observation (descriptive)

The SAFE-SMM permutation control (`stage1_permuted`) scores 0.606 standard and 0.453 QC. Its
training labels were shuffled, but its epoch was selected on unshuffled validation labels (the
"selection ceiling" in SAFE-SMM). The epoch that selection picks happens to read tracking quality,
and the score falls below chance once the QC filter removes the cue. Any permutation null in
SAFE-Motion must therefore shuffle the validation labels too, or use a fixed epoch.
