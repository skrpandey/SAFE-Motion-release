# Pilot v2 (GPU, TinyTCN, 3 seeds): H1, H5, symmetric QC pool, permutation null

Protocol: `configs/protocols/pilot_asdpose_v2.yaml` (frozen a5759d8). Runs 2026-10-24, 26 new runs
plus 5 reused v1 seed-0 runs; v2 GPU time 1.41 h (budget 2.5), local total so far 1.8 GPU-h.
Analysis: `experiments/analyze_v2.py --n-boot 10000`; numbers in `cells.csv`, `summary.json`.
Pilot numbers decide go/no-go only.

## Seed-averaged cells (test split; primary scale: logit gap, standard vs QC)

| cell | standard | QC | QC-sym | reweighted | logit gap [95% CI] | d_a gap |
|---|---|---|---|---|---|---|
| A_none | 0.805 | 0.709 | 0.751 | 0.730 | 0.532 [0.447, 0.640] | 0.484 |
| A_velocity | 0.821 | 0.731 | 0.773 | 0.753 | 0.525 [0.444, 0.621] | 0.449 |
| A_dropout | 0.822 | 0.729 | 0.773 | 0.750 | 0.538 [0.448, 0.646] | 0.476 |
| A_both | 0.838 | 0.761 | 0.804 | 0.776 | 0.484 [0.401, 0.583] | 0.421 |
| B_coorddrop | 0.818 | 0.724 | 0.768 | 0.745 | 0.536 [0.446, 0.648] | 0.478 |
| D_qcsym_none | 0.726 | 0.676 | 0.714 | 0.692 | 0.241 [0.197, 0.295] | 0.210 |
| D_qcsym_velocity | 0.749 | 0.754 | 0.773 | 0.759 | −0.029 [−0.086, 0.030] | 0.039 |

## H1: PASS, now including the reliance reduction

A_both − A_none: standard +0.032 [+0.011, +0.053], QC +0.052 [+0.020, +0.085], logit gap
−0.048 [−0.088, −0.014]. With three seeds the gap reduction is significant (it was not at one seed).

## H5 (regularization with the cue visible): INCONCLUSIVE under the frozen rule

B_coorddrop − A_none on QC: +0.016 [+0.004, +0.030], 30% of the A_both gain. A_both − B_coorddrop:
+0.036 [+0.009, +0.065]. The seed-0 result (zero gain) was not stable: generic regularization
recovers about a third of the QC gain. It is below the frozen half-of-gain bar for
"regularization", so the verdict is inconclusive.

Post hoc observation, not a frozen criterion: B_coorddrop raises standard and QC AUC by similar
amounts and leaves the logit gap where A_none has it (0.536 vs 0.532), whereas A_both lowers it
(0.484). Read on the gap scale, regularization adds accuracy without changing reliance and the
cue-hiding remedy adds both. That is the decomposition contribution B is meant to make; it needs
a frozen test in the next protocol before it can be claimed.

## Symmetric QC pool: a genuine weak-cue pool, and two informative results

Pool check: quality-only AUC on validation children 0.530, inside the frozen [0.40, 0.60] band.
11,869 training segments (3,800 SMM).

1. Velocity helps where the cue is absent: QC-sym +0.060 [+0.028, +0.091], standard +0.023
   [−0.003, +0.048]. At least part of the velocity remedy is better motion features, as the
   SAFE-SMM 86% observation suggested.
2. A model trained with no access to the cue still shows a gap: D_qcsym_none has logit gap 0.241
   [0.197, 0.295]. The QC protocol therefore measures more than learned reliance; poorly tracked
   test segments differ for reasons a cue-free model also responds to (for instance, missing
   joints read as little movement). The velocity version brings this to zero (−0.029).
   Implication for contribution A: a raw gap needs a cue-free reference model, and part of what
   SAFE-SMM called reliance is this baseline offset.

## Permutation null (train and validation labels shuffled, 10 runs): the gap is not separable

Null standard AUC 0.41–0.65, null QC AUC 0.45–0.58, null logit gap −0.71 to +0.72. Shuffled-label
models still score segments in ways correlated with tracking quality, and on the standard protocol
quality correlates with the label, so their gap swings widely in both directions.
Empirical p (one-sided, +1 correction; the smallest possible with 10 runs is 0.091):
- QC AUC: A_none 0.091, A_both 0.091 (both above every null run)
- logit gap: A_none 0.18, A_both 0.27 (not separable from the null)

The models are clearly better than chance on the QC protocol, but the size of the gap itself is
within what label-free models produce. The gap is not a usable reliance statistic on its own;
it has to be read against a reference (cue-free training, symmetric control or this null).

## What this means for the paper

- Holding: remedies raise QC accuracy and lower the gap on a third architecture (H1, 3 seeds).
- New and useful for contribution A: the gap has a substantial cue-free offset (0.24) and a wide
  permutation null. This is direct evidence for reviewer 5TrW's point that Δ is not a clean
  reliance measure, and it motivates measuring reliance against a matched reference rather than
  as a raw gap.
- Not yet supported: "the remedy removes a shortcut rather than regularizing". Regularization
  recovers about a third of the gain; the cleaner decomposition is on the gap scale and must be
  frozen and tested.
- Next decisive test: planted worlds (H2/H3), where ground truth separates the two readings.
