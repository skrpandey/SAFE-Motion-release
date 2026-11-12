# Pilot batch 2 (GPU, TinyTCN): H1, regularization-only control, QC-pool cells, probe

Protocol: `pilot_asdpose_v1.yaml` + `pilot_asdpose_v1_addendum1.yaml` (see REGISTRY.md).
Run 2026-10-24, one seed per cell, 0.39 GPU-h in total (`results/gpu_ledger.csv`).
Analysis: `experiments/analyze_batch2.py --n-boot 10000`, paired child-clustered bootstrap.
Pilot numbers decide go/no-go only; they are not paper cells.

## Cell results (test split, 4,751 segments)

| cell | standard | QC | reweighted | Δ (std − QC) | z-ROC slope |
|---|---|---|---|---|---|
| A_none | 0.802 | 0.720 | 0.736 | 0.082 | 0.93 |
| A_velocity | 0.823 | 0.743 | 0.762 | 0.080 | 0.88 |
| A_dropout | 0.825 | 0.734 | 0.755 | 0.091 | 0.91 |
| A_both | 0.842 | 0.775 | 0.787 | 0.066 | 0.85 |
| B_coorddrop | 0.818 | 0.719 | 0.742 | 0.099 | 0.94 |
| C_qc_none | 0.462 | 0.760 | 0.667 | −0.298 | 1.48 |
| C_qc_velocity | 0.523 | 0.805 | 0.728 | −0.281 | 1.36 |

TinyTCN is well below the GCNs (A_both 0.842 against CompactCTRGCN v3 0.953), so it is a weak
third architecture. Its z-ROC slopes (0.85–0.94) are closer to 1 than the strong GCNs' (batch 1).

## H1: remedies raise both AUCs on the TCN. PASS under the frozen rule, reliance reduction weak

| A_both − A_none | point | 95% paired CI |
|---|---|---|
| standard AUC | +0.039 | [+0.019, +0.059] |
| QC AUC | +0.055 | [+0.027, +0.083] |
| d′ gap | −0.030 | [−0.064, +0.003] |

The training-corruption pattern reproduces on an architecture with no graph. The QC gain is larger
than the standard gain, so reliance falls, but the d′-gap reduction is not significant at one seed.

## Regularization with the cue kept visible (B_coorddrop): recovers none of the gain

Coordinate dropout at the same rate as joint dropout, leaving the confidence channel (the real
missing-joint pattern) intact: QC AUC −0.002 [−0.016, +0.016] against A_none, i.e. −3% of the
A_none → A_both QC gain, and the largest Δ of any full-pool cell (0.099). Whole-joint dropout,
which hides missingness, recovers 25% alone; velocity 41%; both together 100% (super-additive).
This is the cleanest evidence so far that the dropout remedy works by removing the cue, not by
generic regularization. One seed; needs three.

## QC-pool cells: the frozen design did not produce a weak-cue pool. It inverted the cue

`pool_qc` (frozen in v1) filters negatives only and keeps every positive. In the training
children 58.6% of negatives but only 30.6% of positives fail QC, so the filter leaves poor tracking
present only among positives. The cue flips sign rather than weakening, and the model learns it:
standard AUC falls to 0.46 (below chance, because poorly tracked negatives now score as SMM) and
Δ is −0.30. This is a real-data demonstration that training-pool composition alone sets the
direction of the shortcut, but it does not answer the intended question (does velocity help when
the cue is weak). Within this inverted pool velocity still helps: QC +0.045 [+0.022, +0.068],
standard +0.061 [+0.045, +0.079].

Fix for protocol v2: a symmetric QC pool (both classes must pass the filter), where the cue
should be near zero; check quality-only AUC on that pool before training.

## Linear probe: uninformative, as anticipated in addendum 1

Probe R² for the four tracking statistics is 0.89–0.96 for every cell and 0.88–0.97 for the
random-init reference; trained minus random is within ±0.06 everywhere and mostly negative.
Tracking statistics are linearly readable from any features of these inputs. The probe does not
measure reliance and is dropped from the paper plan.

## Consequences for the plan

1. H1 holds directionally on a third architecture; the regularization-only control supports the
   shortcut-removal reading (Contribution B's question) on real data, before any planted world.
2. Protocol v2 needs: symmetric QC pool cells; a primary scale tolerant of unequal variance
   (batch 1); three seeds for A_none, A_both, B_coorddrop, A_dropout.
3. Drop the penultimate-layer probe. If a representation measure is still wanted, use the
   model's own output: within-class correlation of score with tracking statistics.
